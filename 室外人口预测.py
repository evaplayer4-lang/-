import sys
print(sys.executable)
print("单阶段 - 空间代理模型平滑版 (2024夏季综合多时段演变版 - 全局绝对可比色尺)")
import pandas as pd
import numpy as np
import warnings
import joblib
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import KNeighborsRegressor  
import geopandas as gpd

# 解决中文绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ========== 0. 全局设置 ==========
RANDOM_STATE = 42
VAL_SIZE = 0.2
K_NEIGHBORS = 8
GRID_SIZE = 0.002       
BUFFER_RADIUS = 50      
AREA_PER_POINT = np.pi * (BUFFER_RADIUS / 1000) ** 2   

MODEL_PATH = "pedestrian_density_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

# ========== 1. 加载与合并数据 ==========
print("正在加载数据...")
df_main = pd.read_csv(r"E:\A研究生文献\近邻分析结果_带行人计数_complete_with_time_features.csv")

if 'X' in df_main.columns and 'Y' in df_main.columns:
    df_coords = df_main[['FID', 'X', 'Y']].copy()
else:
    raise KeyError("主数据中缺少 'X' 或 'Y' 列")

df_poi   = pd.read_csv(r"E:\A研究生文献\beijing_POI.csv")
df_climate = pd.read_csv(r"E:\A研究生文献\beijing_climate.csv")
df_landcover = pd.read_csv(r"E:\A研究生文献\beijing_landcover.csv")
df_pop   = pd.read_csv(r"E:\A研究生文献\population.csv")
df_road  = pd.read_csv(r"E:\A研究生文献\photo_road_density.csv")
df_building_raw = pd.read_csv(r"E:\A研究生文献\photo_building_coverage.csv")

if 'mean_height_m' not in df_building_raw.columns:
    raise ValueError("建筑数据缺少 'mean_height_m' 列")

df_building_raw['floor_area_ratio'] = (df_building_raw['building_coverage_pct'] * df_building_raw['mean_height_m']) / 300.0
df_building = df_building_raw[['FID', 'floor_area_ratio']]

df = df_main.merge(df_poi, on='FID', how='left') \
            .merge(df_climate, on='FID', how='left') \
            .merge(df_landcover, on='FID', how='left') \
            .merge(df_pop[['FID', 'population']], on='FID', how='left') \
            .merge(df_road, on='FID', how='left') \
            .merge(df_building, on='FID', how='left') \
            .merge(df_coords, on='FID', how='left')

# 白天筛选
df = df[(df['hour'] >= 7) & (df['hour'] <= 19)].reset_index(drop=True)
print(f"清洗非白天数据后，剩余样本量: {len(df)}")

coords_full = df[['X', 'Y']].copy()

# ========== 2. 空间网格锚点构建 ==========
df['grid_x'] = (df['X'] // GRID_SIZE).astype(int)
df['grid_y'] = (df['Y'] // GRID_SIZE).astype(int)
df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)

# ========== 3. 空间邻居特征 ==========
print(f"正在构建空间邻居特征（KNN, k={K_NEIGHBORS}）...")
coords = df[['X', 'Y']].values
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, algorithm='ball_tree').fit(coords)
_, indices = nbrs.kneighbors(coords)
neighbor_indices = indices[:, 1:]

for feat in ['floor_area_ratio', 'road_density_km_per_sqkm', 'population']:
    if feat in df.columns:
        df[f'neighbor_{feat}_mean'] = df[feat].values[neighbor_indices].mean(axis=1)

# ========== 4. 特征工程 ==========
target_col = '行人数量'
base_features = ['Year', 'hour', 'population', 'dist_to_metro', 'road_density_km_per_sqkm', 'floor_area_ratio']
poi_cols = [c for c in df.columns if '_POI50' in c or '_POI100' in c or '_POI300' in c]
climate_cols = [c for c in df.columns if 'temp' in c or 'precip' in c or 'wind' in c]
land_features = ['缓冲区LU多样性', '不透水面占比(%)']
land_features = [c for c in land_features if c in df.columns]
restored_time_features = [c for c in ['season', '日期类型', '时段', 'hour_sin', 'hour_cos', 'is_covid', 'is_other_holiday'] if c in df.columns]

neighbor_cols = [c for c in df.columns if c.startswith('neighbor_')]
all_features_candidate = (base_features + poi_cols + climate_cols + land_features + restored_time_features + neighbor_cols)
all_features_candidate = list(set([f for f in all_features_candidate if f in df.columns]))

# 共线性处理
X_corr = df[all_features_candidate].select_dtypes(include=[np.number])
corr_matrix = X_corr.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > 0.95]

to_drop = set()
if high_corr_pairs:
    for feat1, feat2 in high_corr_pairs:
        if feat1 in to_drop or feat2 in to_drop:
            continue
        if 'daily' in feat1 and ('monthly' in feat2 or 'yearly' in feat2):
            drop = feat2
        elif 'daily' in feat2 and ('monthly' in feat1 or 'yearly' in feat1):
            drop = feat1
        else:
            var1, var2 = X_corr[feat1].var(), X_corr[feat2].var()
            drop = feat1 if var1 <= var2 else feat2
        to_drop.add(drop)
    all_features_final = [f for f in all_features_candidate if f not in to_drop]
else:
    all_features_final = all_features_candidate

# ========== 5. 准备数据 ==========
X_all = df[[*all_features_final, 'grid_id']].copy()
y_all = df[target_col].copy()
valid_mask = pd.concat([X_all, y_all], axis=1).notna().all(axis=1)
X_all = X_all[valid_mask].reset_index(drop=True)
y_all = y_all[valid_mask].reset_index(drop=True)

for col in ['season', '日期类型', '时段']:
    if col in X_all.columns:
        X_all[col] = X_all[col].astype('category')
for col in X_all.select_dtypes(include=['object']).columns:
    if col != 'grid_id':
        X_all[col] = X_all[col].astype('category')

bins = [-1, 0, 2, 5, 10, np.inf]
labels = [0, 1, 2, 3, 4]
stratify_labels = pd.cut(y_all, bins=bins, labels=labels)
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=stratify_labels)

grid_train = X_train['grid_id'].copy()
grid_val   = X_val['grid_id'].copy()
X_train = X_train.drop(columns=['grid_id'])
X_val   = X_val.drop(columns=['grid_id'])

# ========== 6. 空间网格目标编码 ==========
X_train['spatial_target_enc'] = 0.0
X_val['spatial_target_enc']   = 0.0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

for train_idx, holdout_idx in kf.split(X_train):
    fold_y   = y_train.iloc[train_idx].values
    fold_grid = grid_train.iloc[train_idx].values
    holdout_grid = grid_train.iloc[holdout_idx].values

    temp_df = pd.DataFrame({'grid': fold_grid, 'target': fold_y})
    grid_mean = temp_df.groupby('grid')['target'].mean()
    mapped = pd.Series(holdout_grid).map(grid_mean).fillna(y_train.mean()).values
    X_train.iloc[holdout_idx, X_train.columns.get_loc('spatial_target_enc')] = mapped

global_grid_mean = pd.DataFrame({'grid': grid_train, 'target': y_train}).groupby('grid')['target'].mean()
X_val['spatial_target_enc'] = grid_val.map(global_grid_mean).fillna(y_train.mean()).values

all_features_final_with_enc = all_features_final + ['spatial_target_enc']

# ========== 7. 模型训练 ==========
reg = lgb.LGBMRegressor(
    objective='tweedie', tweedie_variance_power=1.1, n_estimators=2000, learning_rate=0.015,
    max_depth=5, num_leaves=25, min_child_samples=100, min_split_gain=0.02,
    reg_alpha=1.0, reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='tweedie', callbacks=[lgb.early_stopping(50, verbose=False)])

train_pred = reg.predict(X_train)
val_pred = reg.predict(X_val)

# ========== 9. 保存模型与预处理器 ==========
preprocessor = {
    'nbrs': nbrs, 'all_features_final': all_features_final, 'global_grid_mean': global_grid_mean,
    'y_mean': y_train.mean(), 'valid_mask': valid_mask, 'coords': coords_full,
    'area_per_point': AREA_PER_POINT, 'category_cols': ['season', '日期类型', '时段']
}
joblib.dump(reg, MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)

# ========== 10. 预测功能 ==========
def predict_density(year, season, date_type=1, hour_start=7):
    reg = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESSOR_PATH)
    
    df_pred = df_main.copy()
    df_pred = df_pred.merge(df_poi, on='FID', how='left') \
                     .merge(df_climate, on='FID', how='left') \
                     .merge(df_landcover, on='FID', how='left') \
                     .merge(df_pop[['FID', 'population']], on='FID', how='left') \
                     .merge(df_road, on='FID', how='left') \
                     .merge(df_building, on='FID', how='left') \
                     .merge(df_coords, on='FID', how='left')
    
    df_temp = df_pred.copy()
    df_temp['Year'] = year
    df_temp['hour'] = hour_start
    df_temp['season'] = season
    df_temp['日期类型'] = date_type
    
    if 'hour_sin' in df_temp.columns:
        df_temp['hour_sin'] = np.sin(2 * np.pi * hour_start / 24)
    if 'hour_cos' in df_temp.columns:
        df_temp['hour_cos'] = np.cos(2 * np.pi * hour_start / 24)
    if 'is_covid' in df_temp.columns:
        df_temp['is_covid'] = 0
    if 'is_other_holiday' in df_temp.columns:
        df_temp['is_other_holiday'] = 0
    for col in climate_cols:
        if col in df_temp.columns:
            df_temp[col] = df[col].mean()  
    
    coords_temp = df_temp[['X', 'Y']].values
    nbrs = pre['nbrs']
    _, indices = nbrs.kneighbors(coords_temp)
    neighbor_indices = indices[:, 1:]
    for feat in ['floor_area_ratio', 'road_density_km_per_sqkm', 'population']:
        if feat in df_temp.columns:
            df_temp[f'neighbor_{feat}_mean'] = df_temp[feat].values[neighbor_indices].mean(axis=1)
    
    df_temp['grid_x'] = (df_temp['X'] // GRID_SIZE).astype(int)
    df_temp['grid_y'] = (df_temp['Y'] // GRID_SIZE).astype(int)
    df_temp['grid_id'] = df_temp['grid_x'].astype(str) + "_" + df_temp['grid_y'].astype(str)
    
    X_pred = df_temp[pre['all_features_final']].copy()
    for col in pre['category_cols']:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype('category')
    for col in X_pred.select_dtypes(include=['object']).columns:
        X_pred[col] = X_pred[col].astype('category')
    
    X_pred['spatial_target_enc'] = df_temp['grid_id'].map(pre['global_grid_mean']).fillna(pre['y_mean']).values
    
    pred_count = reg.predict(X_pred[all_features_final_with_enc])
    pred_density = pred_count / pre['area_per_point']
    
    res = df_temp[['X', 'Y']].copy()
    res['hour'] = hour_start
    res['pred_count'] = pred_count
    res['pred_density'] = pred_density
    return res

# ========== 11. 仿 ArcGIS 空间平滑预测绘图函数 ==========
def plot_density_surrogate_smooth_auto(density_df, season_label, hour_label, save_path=None, boundaries=None):
    print(f"\n[1/3] 正在加载五环边界并构建空间代理填补模型...")
    shp_path = r"D:\edge loaddown\postgraduate_data\投影坐标系\北京五环_polygon.shp"
    beijing_5th_ring = gpd.read_file(shp_path)
    if beijing_5th_ring.crs != "EPSG:4326":
        beijing_5th_ring = beijing_5th_ring.to_crs(epsg=4326)
    poly_geom = beijing_5th_ring.geometry.unary_union
    
    x, y, z = density_df['X'].values, density_df['Y'].values, density_df['pred_density'].values
    
    spatial_model = KNeighborsRegressor(n_neighbors=150, weights='distance', n_jobs=-1)
    spatial_model.fit(np.column_stack((x, y)), z)
    
    print("[2/3] 正在构建高清网格并进行五环裁剪掩膜...")
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    
    grid_z = spatial_model.predict(np.column_stack((grid_x.ravel(), grid_y.ravel()))).reshape(grid_x.shape)
    grid_z = np.clip(grid_z, 0, 1900.0)
    
    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()), crs="EPSG:4326")
    mask = grid_points.within(poly_geom).values.reshape(grid_x.shape)
    grid_z = np.where(mask, grid_z, np.nan)
    
    print("[3/3] 正在应用统一色尺渲染图像...")
    if boundaries is None:
        boundaries = np.linspace(0, 1900, 20)
        
    if boundaries[0] > 0:
        arcgis_colors = ['#e0e0e0', '#a1c1d6', '#73a2c6', '#4d89b3', '#fce853', '#e05a2b', '#9c0f0f']
    else:
        arcgis_colors = ['#e0e0e0', '#6b8ebf', '#a1c7a1', '#fce853', '#e05a2b', '#9c0f0f']
        
    cmap = mcolors.LinearSegmentedColormap.from_list('arcgis_premium_smooth', arcgis_colors, N=256)
    cmap.set_bad(color='#ffffff', alpha=1.0)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    
    plt.figure(figsize=(11, 9), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    
    im = ax.imshow(grid_z, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, norm=norm, aspect='auto', interpolation='bilinear')
    
    x_pad, y_pad = (x_max - x_min) * 0.06, (y_max - y_min) * 0.06
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    cbar = plt.colorbar(im, label='预测密度 (人/平方公里)', spacing='uniform', pad=0.04, shrink=0.7, aspect=28)
    tick_indices = np.linspace(0, len(boundaries)-1, 6, dtype=int)
    cbar.set_ticks(boundaries[tick_indices])
    tick_labels = [str(int(b)) for b in boundaries[tick_indices]]
    tick_labels[0], tick_labels[-1] = '0', '1900'
    cbar.set_ticklabels(tick_labels)
    
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.title(f'人口密度预测({season_label} {hour_label})', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('经度', fontsize=11)
    plt.ylabel('纬度', fontsize=11)
    
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像成功保存至: {save_path}")
    plt.show()

# ========== 12. 综合批量演变预测与共享色尺出图 ==========
target_year = 2024
target_season_val = 2       
target_season_name = '夏季'
output_directory = r"E:\A研究生文献\预测套图"
target_hours = [8, 10, 12, 14, 16, 18]

# --- 阶段一：后台并行预测并进行【全周加权融合】 ---
print("\n" + "="*30 + " 阶段一：执行全周综合权重时空预测 " + "="*30)
predict_results_cache = {}
all_combined_densities = []

for h in target_hours:
    print(f"正在计算 -> {target_season_name} 在 {h}:00 时的全城综合平均人口密度...")
    
    # 分别预测工作日(1)与非工作日(2)
    df_workday = predict_density(year=target_year, season=target_season_val, date_type=1, hour_start=h)
    df_weekend = predict_density(year=target_year, season=target_season_val, date_type=2, hour_start=h)
    
    # 按照 5天工作日 vs 2天非工作日 进行精细加权融合
    df_combined = df_workday.copy()
    df_combined['pred_count'] = (df_workday['pred_count'] * 5 + df_weekend['pred_count'] * 2) / 7
    df_combined['pred_density'] = (df_workday['pred_density'] * 5 + df_weekend['pred_density'] * 2) / 7
    
    predict_results_cache[h] = df_combined
    all_combined_densities.extend(df_combined['pred_density'].values)

# --- 阶段二：提取 6 张图跨时空共享的分位数母尺 ---
print("\n" + "="*30 + " 阶段二：合并全天时段，生成全局绝对统一色尺 " + "="*30)
quantiles = np.linspace(0, 100, 20)
GLOBAL_BOUNDARIES = np.percentile(all_combined_densities, quantiles)
GLOBAL_BOUNDARIES = np.unique(GLOBAL_BOUNDARIES)
GLOBAL_BOUNDARIES[0] = 0.0
GLOBAL_BOUNDARIES[-1] = 1900.0

print(f">>> 统一共享色尺划分节点: {np.round(GLOBAL_BOUNDARIES, 1)}")

# --- 阶段三：批量出图 ---
print("\n" + "="*30 + " 阶段三：套用全局统一色尺，精准渲染 6 张时空演变图 " + "="*30)
for h in target_hours:
    current_df = predict_results_cache[h]
    file_name = f"{target_year}年{target_season_name}{h}：00.png"
    full_save_path = os.path.join(output_directory, file_name)
    
    plot_density_surrogate_smooth_auto(
        density_df=current_df,
        season_label=target_season_name,
        hour_label=f"{h}:00",
        save_path=full_save_path,
        boundaries=GLOBAL_BOUNDARIES
    )

print(f"\n[完成] 2024年{target_season_name}共 {len(target_hours)} 张综合人口密度时空演变图已全部无误输出！")