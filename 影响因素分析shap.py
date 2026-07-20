import pandas as pd
import numpy as np
import re
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ========== 0. 全局设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
RANDOM_STATE = 42
# 改为 True 就运行空间划分，False 运行时间划分
USE_SPACE_SPLIT = True
TEST_YEAR = 2023
SPACE_BLOCK_SIZE = 0.02      # 约 2km 网格

ANALYSIS_PARAMS_CLF = {
    'n_estimators': 400,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'objective': 'binary:logistic',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

ANALYSIS_PARAMS_REG = {
    'n_estimators': 600,
    'max_depth': 12,
    'learning_rate': 0.03,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'objective': 'reg:squarederror',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ========== 1. 加载与合并数据 ==========
print("正在加载数据...")
df_main = pd.read_csv(r"E:\A研究生文献\近邻分析结果_带行人计数_complete_with_time_features.csv")

# 提取坐标
coords_available = 'X' in df_main.columns and 'Y' in df_main.columns
if coords_available:
    df_coords = df_main[['FID', 'X', 'Y']].copy()
    print("✅ 读取到 X, Y 坐标")
else:
    df_coords = None
    print("⚠️ 无坐标列，空间划分不可用")

df_poi = pd.read_csv(r"E:\A研究生文献\beijing_POI.csv")
df_climate = pd.read_csv(r"E:\A研究生文献\beijing_climate.csv")
df_landcover = pd.read_csv(r"E:\A研究生文献\beijing_landcover.csv")
df_pop = pd.read_csv(r"E:\A研究生文献\population.csv")
df_road = pd.read_csv(r"E:\A研究生文献\photo_road_density.csv")
df_building_raw = pd.read_csv(r"E:\A研究生文献\photo_building_coverage.csv")

if 'mean_height_m' not in df_building_raw.columns:
    raise ValueError("缺少 mean_height_m 列")
df_building_raw['floor_area_ratio'] = (df_building_raw['building_coverage_pct'] * df_building_raw['mean_height_m']) / 300.0
df_building = df_building_raw[['FID', 'floor_area_ratio']]

# 合并
df = df_main.merge(df_poi, on='FID', how='left') \
           .merge(df_climate, on='FID', how='left') \
           .merge(df_landcover, on='FID', how='left') \
           .merge(df_pop[['FID', 'population']], on='FID', how='left') \
           .merge(df_road, on='FID', how='left') \
           .merge(df_building, on='FID', how='left')

# 把坐标加回来（确保用FID对齐）
if coords_available:
    df = df.merge(df_coords, on='FID', how='left')

# ========== 2. 特征工程 ==========
target_col = '行人数量'
poi_cols = [c for c in df.columns if '_POI50' in c or '_POI100' in c or '_POI300' in c]
core_poi_keywords = ['1', '7', '4', '19', '8', '6', '15', '22', '10']

spatiotemporal_features = []
if 'hour' in df.columns:
    for poi_col in poi_cols:
        for keyword in core_poi_keywords:
            if re.search(rf'(^|_){keyword}(_|$)', poi_col):
                interaction_col = f'hour_{poi_col}'
                df[interaction_col] = df['hour'] * df[poi_col]
                spatiotemporal_features.append(interaction_col)
                break

climate_cols = [c for c in df.columns if 'temp' in c or 'precip' in c or 'wind' in c]

base_features = ['Year', 'hour', 'population', 'dist_to_metro',
                 'road_density_km_per_sqkm', 'floor_area_ratio']

land_features = ['缓冲区LU多样性', '不透水面占比(%)']
land_features = [c for c in land_features if c in df.columns]

# 加入泛化坐标特征
if coords_available:
    df['X_round'] = df['X'].round(3)
    df['Y_round'] = df['Y'].round(3)
    base_features += ['X_round', 'Y_round']

all_features_candidate = base_features + poi_cols + spatiotemporal_features + climate_cols + land_features
all_features_candidate = list(set([f for f in all_features_candidate if f in df.columns]))
print(f"候选特征数量：{len(all_features_candidate)}")

# ---------- 共线性处理 ----------
X_corr = df[all_features_candidate].select_dtypes(include=[np.number])
corr_matrix = X_corr.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = []
for col in upper_tri.columns:
    for row in upper_tri.index:
        if upper_tri.loc[row, col] > 0.95:
            high_corr_pairs.append((row, col, upper_tri.loc[row, col]))

to_drop = set()
if high_corr_pairs:
    print(f"发现 {len(high_corr_pairs)} 对极高相关特征")
    for (feat1, feat2, r_val) in high_corr_pairs:
        if feat1 in to_drop or feat2 in to_drop:
            continue
        if feat1.startswith('hour_') and feat1[len('hour_'):] == feat2:
            drop = feat1
        elif feat2.startswith('hour_') and feat2[len('hour_'):] == feat1:
            drop = feat2
        else:
            var1 = X_corr[feat1].var()
            var2 = X_corr[feat2].var()
            drop = feat1 if var1 <= var2 else feat2
        to_drop.add(drop)
        print(f"  删除: {drop}")
    all_features_final = [f for f in all_features_candidate if f not in to_drop]
else:
    all_features_final = all_features_candidate
print(f"最终特征数：{len(all_features_final)}")

# ========== 3. 准备数据（保留原始坐标用于空间网格） ==========
X_all = df[all_features_final].copy()
y_all = df[target_col].copy()
# 坐标单独保存，并保证与最终样本索引一致
coords_raw = df[['X', 'Y']].copy() if coords_available else None

# 删除缺失值，同时对齐坐标
valid_mask = pd.concat([X_all, y_all], axis=1).notna().all(axis=1)
X_all = X_all[valid_mask].reset_index(drop=True)
y_all = y_all[valid_mask].reset_index(drop=True)
if coords_raw is not None:
    coords_raw = coords_raw[valid_mask].reset_index(drop=True)

for col in X_all.select_dtypes(include=['object', 'category']).columns:
    X_all[col] = X_all[col].astype('category').cat.codes

# ========== 4. 划分数据集 ==========
if USE_SPACE_SPLIT:
    if not coords_available:
        raise RuntimeError("无法空间划分：缺少坐标列。")
    # 创建网格ID
    block_x = (coords_raw['X'] / SPACE_BLOCK_SIZE).astype(int)
    block_y = (coords_raw['Y'] / SPACE_BLOCK_SIZE).astype(int)
    block_id = block_x.astype(str) + '_' + block_y.astype(str)
    unique_blocks = block_id.unique()
    np.random.seed(RANDOM_STATE)
    test_blocks = np.random.choice(unique_blocks, size=int(len(unique_blocks) * 0.2), replace=False)
    train_idx = ~block_id.isin(test_blocks).values
    test_idx = block_id.isin(test_blocks).values
    split_mode = "空间网格"
else:
    if 'Year' not in X_all.columns:
        raise ValueError("时间划分需要 'Year' 列。")
    train_idx = X_all['Year'] < TEST_YEAR
    test_idx = X_all['Year'] == TEST_YEAR
    split_mode = f"时间（测试年 {TEST_YEAR}）"

X_train, y_train = X_all[train_idx].copy(), y_all[train_idx].copy()
X_test, y_test = X_all[test_idx].copy(), y_all[test_idx].copy()
print(f"\n划分方式：{split_mode}")
print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")

# 从训练集再切出验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE,
    stratify=(y_train > 0).astype(int)
)

# ========== 5. 两阶段模型训练与评估 ==========
y_train_bin = (y_train > 0).astype(int)
y_val_bin = (y_val > 0).astype(int)
y_test_bin = (y_test > 0).astype(int)

# 第一阶段
clf = xgb.XGBClassifier(**ANALYSIS_PARAMS_CLF)
clf.fit(
    X_train, y_train_bin,
    eval_set=[(X_val, y_val_bin)],
    early_stopping_rounds=20,
    verbose=False
)
prob_train = clf.predict_proba(X_train)[:, 1]
prob_test = clf.predict_proba(X_test)[:, 1]

# 第二阶段
mask_train_pos = (y_train > 0)
mask_test_pos = (y_test > 0)
X_train_pos = X_train[mask_train_pos]
y_train_pos_log = np.log1p(y_train[mask_train_pos])
X_test_pos = X_test[mask_test_pos]
y_test_pos_log = np.log1p(y_test[mask_test_pos])

reg = xgb.XGBRegressor(**ANALYSIS_PARAMS_REG)
eval_set_reg = [(X_val[y_val > 0], np.log1p(y_val[y_val > 0]))] if (y_val > 0).sum() > 0 else None
reg.fit(
    X_train_pos, y_train_pos_log,
    eval_set=eval_set_reg,
    early_stopping_rounds=20,
    verbose=False
)

# 评估
auc_train = roc_auc_score(y_train_bin, prob_train)
auc_test = roc_auc_score(y_test_bin, prob_test)

y_train_pos_pred = np.expm1(reg.predict(X_train_pos))
y_test_pos_pred = np.expm1(reg.predict(X_test_pos))
r2_train_reg = r2_score(y_train[mask_train_pos], y_train_pos_pred)
r2_test_reg = r2_score(y_test[mask_test_pos], y_test_pos_pred)

y_train_full = prob_train * np.expm1(reg.predict(X_train))
y_test_full = prob_test * np.expm1(reg.predict(X_test))
r2_train_overall = r2_score(y_train, y_train_full)
r2_test_overall = r2_score(y_test, y_test_full)

print("\n" + "="*60)
print(f"【{split_mode}】模型评估")
print(f"第一阶段 AUC - 训练: {auc_train:.4f}  测试: {auc_test:.4f}")
print(f"第二阶段 R²  - 训练: {r2_train_reg:.4f}  测试: {r2_test_reg:.4f}")
print(f"整体 Hurdle R² - 训练: {r2_train_overall:.4f}  测试: {r2_test_overall:.4f}")
print("="*60)

# ========== 6. SHAP 分析（测试集） ==========
print("\n生成 SHAP 图（测试集）...")
try:
    import shap
    X_test_sample = X_test.sample(n=min(3000, len(X_test)), random_state=RANDOM_STATE)
    X_test_pos_sample = X_test_pos.sample(n=min(3000, len(X_test_pos)), random_state=RANDOM_STATE)

    explainer_clf = shap.TreeExplainer(clf)
    shap_v_clf = explainer_clf.shap_values(X_test_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_v_clf, X_test_sample, show=False)
    plt.title(f'第一阶段：有无行人影响因素（{split_mode}测试集）', fontsize=14)
    plt.tight_layout()
    plt.show()

    explainer_reg = shap.TreeExplainer(reg)
    shap_v_reg = explainer_reg.shap_values(X_test_pos_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_v_reg, X_test_pos_sample, show=False)
    plt.title(f'第二阶段：人流量级影响因素（{split_mode}测试集）', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 最重要的特征边际效应
    top_idx = np.argsort(np.abs(shap_v_reg).mean(0))[-1]
    top_name = X_test_pos.columns[top_idx]
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(top_name, shap_v_reg, X_test_pos_sample, show=False)
    plt.title(f'边际效应：{top_name}', fontsize=12)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"SHAP 异常: {e}")