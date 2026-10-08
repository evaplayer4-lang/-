# -*- coding: utf-8 -*-
"""
人口密度预测 - 方案A（随机分层）/ 方案B（空间分组） + 固定前50个重要特征
模型：XGBoost-Tweedie
已加入夜间灯光特征 ntl_log_300m
已加入遥感空间纹理特征
已加入遥感光谱特征
土地利用数据：RURBAN-Map 17类比例 + 香农熵
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')

# ==================== 全局设置 ====================
RANDOM_STATE = 42
GRID_SIZE = 0.002
SPLIT_METHOD = 'A'   # 'A' 随机分层划分，'B' 空间分组划分
K_FEATURES = 50      # 固定使用前50个重要特征

# ==================== 数据加载与合并 ====================
print("正在加载数据...")
df_main = pd.read_csv(r"D:\向日葵\远程文件传输\近邻分析结果_带行人计数_complete.csv")
if 'X' not in df_main.columns or 'Y' not in df_main.columns:
    raise KeyError("主数据缺少 X 或 Y 列")
main_cols = df_main.columns.tolist()

def load_aux_table(path):
    df_aux = pd.read_csv(path)
    df_aux = df_aux[['FID'] + [c for c in df_aux.columns if c not in main_cols and c != 'FID']]
    return df_aux

# 加载各类辅助数据
df_poi = load_aux_table(r"D:\向日葵\远程文件传输\近邻分析结果_带行人计数_complete_poi.csv")
df_climate = load_aux_table(r"E:\A研究生文献\beijing_climate.csv")
# ★ 修改1：土地利用数据替换为 RURBAN-Map
#df_landcover = load_aux_table(r"D:\向日葵\远程文件传输\landuse_runbanmap.csv")
df_pop = load_aux_table(r"E:\A研究生文献\population.csv")
df_road = load_aux_table(r"E:\A研究生文献\photo_road_density.csv")

df_building_raw = pd.read_csv(r"E:\A研究生文献\photo_building_coverage.csv")
if 'mean_height_m' not in df_building_raw.columns:
    raise ValueError("建筑数据缺少 'mean_height_m' 列")
df_building_raw['floor_area_ratio'] = (df_building_raw['building_coverage_pct'] * df_building_raw['mean_height_m']) / 300.0
df_building = df_building_raw[['FID', 'floor_area_ratio']]

# 加载夜间灯光数据
df_ntl = pd.read_csv(r"D:\向日葵\远程文件传输\NTL_log_300m_all_years.csv")
df_ntl = df_ntl.rename(columns={'year': 'Year'})
df_main['Year'] = df_main['Year'].astype(int)
df_ntl['Year'] = df_ntl['Year'].astype(int)

# 加载遥感空间纹理特征
df_texture = pd.read_csv(r"D:\向日葵\远程文件传输\遥感空间纹理特征.csv")
df_texture['Year'] = df_texture['Year'].astype(int)

# 加载遥感光谱特征
df_spectral = pd.read_csv(r"D:\向日葵\远程文件传输\遥感光谱特征.csv")
df_spectral['Year'] = df_spectral['Year'].astype(int)

# 合并所有数据
df = df_main.merge(df_poi, on='FID', how='left') \
            .merge(df_climate, on='FID', how='left') \
            .merge(df_pop, on='FID', how='left') \
            .merge(df_road, on='FID', how='left') \
            .merge(df_building, on='FID', how='left') \
            .merge(df_ntl, on=['FID', 'Year'], how='left') \
            .merge(df_texture, on=['FID', 'Year'], how='left') \
            .merge(df_spectral, on=['FID', 'Year'], how='left') 
         #   .merge(df_landcover, on='FID', how='left') 

# ★ 可选：将 17 类字段重命名为可读名称（若后续看特征重要性更直观）
lu_rename = {
    'LU_class_1_ratio':  'LU_Grassland_ratio',
    'LU_class_2_ratio':  'LU_Cropland_ratio',
    'LU_class_3_ratio':  'LU_Forest_ratio',
    'LU_class_4_ratio':  'LU_Orchard_ratio',
    'LU_class_5_ratio':  'LU_Other_ratio',
    'LU_class_6_ratio':  'LU_Water_ratio',
    'LU_class_7_ratio':  'LU_Wetland_ratio',
    'LU_class_8_ratio':  'LU_Industrial_ratio',
    'LU_class_9_ratio':  'LU_PublicMgmt_ratio',
    'LU_class_10_ratio': 'LU_PublicFacilities_ratio',
    'LU_class_11_ratio': 'LU_Road_ratio',
    'LU_class_12_ratio': 'LU_Transportation_ratio',
    'LU_class_13_ratio': 'LU_UrbanResidential_ratio',
    'LU_class_14_ratio': 'LU_RuralResidential_ratio',
    'LU_class_15_ratio': 'LU_ParkSquare_ratio',
    'LU_class_16_ratio': 'LU_Commercial_ratio',
    'LU_class_17_ratio': 'LU_SpecialLand_ratio',
}
df = df.rename(columns={k: v for k, v in lu_rename.items() if k in df.columns})

# 仅保留白天时段
df = df[(df['Time'] >= 7) & (df['Time'] <= 19)].reset_index(drop=True)
print(f"有效样本量: {len(df)}")

# 生成网格ID（用于空间划分方案B）
df['grid_x'] = (df['X'] // GRID_SIZE).astype(int)
df['grid_y'] = (df['Y'] // GRID_SIZE).astype(int)
df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)

# ==================== 特征工程 ====================
target_col = '行人数量'
base_features = ['X', 'Y', 'Year', 'Time', 'population', 'dist_to_metro',
                 'road_density_km_per_sqkm', 'floor_area_ratio', 'ntl_log_300m']
poi_cols = [c for c in df.columns if '_POI50' in c or '_POI100' in c or '_POI300' in c]
climate_cols = ['daily_temp', 'monthly_temp', 'daily_precip', 'daily_wind']
climate_cols = [c for c in climate_cols if c in df.columns]


restored_time_features = [c for c in ['Season', '日期类型'] if c in df.columns]

# 遥感纹理特征列
texture_cols = [c for c in df_texture.columns if c not in ['FID', 'Year']]
# 遥感光谱特征列
spectral_cols = [c for c in df_spectral.columns if c not in ['FID', 'Year']]

all_features_candidate = (base_features + poi_cols + climate_cols +
                          restored_time_features +
                          texture_cols + spectral_cols)
all_features_candidate = list(set([f for f in all_features_candidate if f in df.columns]))

print(f"候选特征总数（含夜间灯光、遥感纹理、遥感光谱、RURBAN-Map）: {len(all_features_candidate)}")

# ==================== 共线性处理 ====================
X_corr = df[all_features_candidate].select_dtypes(include=[np.number])
corr_matrix = X_corr.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > 0.95]

to_drop = set()
drop_details = []
if high_corr_pairs:
    for feat1, feat2 in high_corr_pairs:
        if feat1 in to_drop or feat2 in to_drop:
            continue
        corr_value = corr_matrix.loc[feat1, feat2]
        if 'daily' in feat1 and ('monthly' in feat2 or 'yearly' in feat2):
            drop, keep = feat2, feat1
        elif 'daily' in feat2 and ('monthly' in feat1 or 'yearly' in feat1):
            drop, keep = feat1, feat2
        else:
            var1, var2 = X_corr[feat1].var(), X_corr[feat2].var()
            if var1 <= var2:
                drop, keep = feat1, feat2
            else:
                drop, keep = feat2, feat1
        to_drop.add(drop)
        drop_details.append((drop, keep, corr_value))
    all_features_final = [f for f in all_features_candidate if f not in to_drop]
else:
    all_features_final = all_features_candidate

print("\n因共线性被删除的特征：")
for i, (drop, keep, corr) in enumerate(drop_details, 1):
    print(f"{i:3d}. 删除: {drop:45s} | 保留: {keep:45s} | 相关系数: {corr:.4f}")
print(f"共线性处理后特征数: {len(all_features_final)}")

# ==================== 数据集划分 ====================
def split_random_stratified(df, target, test_size=0.15, val_size=0.15):
    bins = [-1, 0, 2, 5, 10, np.inf]
    labels = [0, 1, 2, 3, 4]
    stratify_labels = pd.cut(df[target], bins=bins, labels=labels)
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_labels
    )
    stratify_labels_train_val = pd.cut(train_val_df[target], bins=bins, labels=labels)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size),
        random_state=RANDOM_STATE, stratify=stratify_labels_train_val
    )
    return train_df, val_df, test_df

def split_by_grid(df, target, test_size=0.15, val_size=0.15):
    grid_ids = df['grid_id'].unique()
    train_val_grids, test_grids = train_test_split(
        grid_ids, test_size=test_size, random_state=RANDOM_STATE
    )
    train_grids, val_grids = train_test_split(
        train_val_grids, test_size=val_size/(1-test_size), random_state=RANDOM_STATE
    )
    train_df = df[df['grid_id'].isin(train_grids)]
    val_df = df[df['grid_id'].isin(val_grids)]
    test_df = df[df['grid_id'].isin(test_grids)]
    return train_df, val_df, test_df

if SPLIT_METHOD == 'A':
    print("\n使用方案A：随机分层划分")
    train_df, val_df, test_df = split_random_stratified(df, target_col)
elif SPLIT_METHOD == 'B':
    print("\n使用方案B：空间分组划分")
    train_df, val_df, test_df = split_by_grid(df, target_col)
else:
    raise ValueError("SPLIT_METHOD 必须为 'A' 或 'B'")

print(f"训练集样本: {len(train_df)}")
print(f"验证集样本: {len(val_df)}")
print(f"测试集样本: {len(test_df)}")

# ==================== 数据准备函数 ====================
def prepare_data(df, features):
    X = df[features].copy()
    y = df[target_col].copy()
    for col in ['Season', '日期类型']:
        if col in X.columns:
            X[col] = X[col].astype('category')
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
    return X, y

# ==================== 模型参数 ====================
model_params = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.1,
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 5,
    'min_child_weight': 30,
    'reg_alpha': 1.0,
    'reg_lambda': 10.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

# ==================== 初始模型训练（获取特征重要性） ====================
print("\n[特征筛选] 训练初始模型以获取特征重要性...")
X_train_full, y_train_full = prepare_data(train_df, all_features_final)
X_val_full, y_val_full = prepare_data(val_df, all_features_final)
X_test_full, y_test_full = prepare_data(test_df, all_features_final)

start = time.time()
model_full = xgb.XGBRegressor(**model_params)
model_full.fit(
    X_train_full.values, y_train_full.values,
    eval_set=[(X_val_full.values, y_val_full.values)],
    verbose=False
)
print(f"初始模型训练完成，耗时 {time.time()-start:.2f}s")

# 特征重要性排序
importance = model_full.feature_importances_
feat_imp = pd.DataFrame({'feature': all_features_final, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
print("\n特征重要性 Top 20：")
print(feat_imp.head(20))

# ==================== 选择前 K_FEATURES 个重要特征 ====================
selected_features = feat_imp['feature'].head(K_FEATURES).tolist()
print(f"\n选择前 {K_FEATURES} 个重要特征，最终保留特征: {selected_features}")

# ==================== 使用筛选后的特征重新训练模型 ====================
X_train_sel, y_train_sel = prepare_data(train_df, selected_features)
X_val_sel, y_val_sel = prepare_data(val_df, selected_features)
X_test_sel, y_test_sel = prepare_data(test_df, selected_features)

print("\n[最终模型训练] 使用筛选后的特征训练 XGBoost-Tweedie ...")
start = time.time()
model_final = xgb.XGBRegressor(**model_params)
model_final.fit(
    X_train_sel.values, y_train_sel.values,
    eval_set=[(X_val_sel.values, y_val_sel.values)],
    verbose=False
)
print(f"最终模型训练完成，耗时 {time.time()-start:.2f}s")

# ==================== 评估函数 ====================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    threshold = np.quantile(y_true, 0.9)
    high_mask = y_true >= threshold
    high_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.sum() > 0 else np.nan
    spearman_corr, _ = spearmanr(y_true, y_pred)
    nonzero = y_true > 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.sum() > 0 else np.nan
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    return {
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'High-value MAE': high_mae,
        'Spearman': spearman_corr,
        'MAPE (%)': mape,
        'WAPE (%)': wape
    }

train_pred = np.clip(model_final.predict(X_train_sel.values), 0, None)
val_pred = np.clip(model_final.predict(X_val_sel.values), 0, None)
test_pred = np.clip(model_final.predict(X_test_sel.values), 0, None)

train_metrics = evaluate(y_train_sel.values, train_pred)
val_metrics = evaluate(y_val_sel.values, val_pred)
test_metrics = evaluate(y_test_sel.values, test_pred)

# ==================== 输出结果 ====================
print("\n" + "="*60)
print(f"XGBoost-Tweedie 评估结果（{SPLIT_METHOD}方案，固定前{K_FEATURES}特征）")
print("="*60)
for subset, metrics in [('训练集', train_metrics), ('验证集', val_metrics), ('测试集', test_metrics)]:
    print(f"\n{subset}:")
    for kk, v in metrics.items():
        print(f"  {kk}: {v:.4f}")