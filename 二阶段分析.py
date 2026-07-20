import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# 解决中文绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ========== 0. 全局设置 ==========
RANDOM_STATE = 42
VAL_SIZE = 0.2          
K_NEIGHBORS = 8
GRID_SIZE = 0.002       # 网格大小

# ========== 1. 加载与合并数据 ==========
print("正在加载数据...")
# 修复路径换行错误
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


df_remote = pd.read_csv(r"D:\edge loaddown\Beijing_Sentinel2_Indices_50m.csv")
df_remote = df_remote[['FID', 'NDVI', 'NDBI', 'MNDWI']]   # 只保留需要的列

df = df_main.merge(df_poi, on='FID', how='left') \
           .merge(df_climate, on='FID', how='left') \
           .merge(df_landcover, on='FID', how='left') \
           .merge(df_pop[['FID', 'population']], on='FID', how='left') \
           .merge(df_road, on='FID', how='left') \
           .merge(df_building, on='FID', how='left') \
           .merge(df_coords, on='FID', how='left')\
           .merge(df_remote, on='FID', how='left')



# 白天筛选
df = df[(df['hour'] >= 7) & (df['hour'] <= 19)].reset_index(drop=True)
print(f"清洗非白天数据后，剩余样本量: {len(df)}")

# ========== 1.5. 创建年份和疫情特征 ==========
# 创建疫情期间特征（2020-2022为疫情期间）
if 'Year' in df.columns:
    df['is_epidemic_period'] = ((df['Year'] >= 2020) & (df['Year'] <= 2021)).astype(int)
    print(f"已创建疫情期间特征，疫情期间（2020-2021）样本数：{df['is_epidemic_period'].sum()}")
    print(f"非疫情期间样本数：{len(df) - df['is_epidemic_period'].sum()}")
else:
    print("警告：数据中没有年份信息，无法创建疫情特征")

# ========== 2. 空间网格锚点构建 ==========
df['grid_x'] = (df['X'] // GRID_SIZE).astype(int)
df['grid_y'] = (df['Y'] // GRID_SIZE).astype(int)
df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)
print(f"生成网格数量: {df['grid_id'].nunique()}")

# ========== 3. 空间邻居特征 ==========
print(f"正在构建空间邻居特征（KNN, k={K_NEIGHBORS}）...")
coords = df[['X', 'Y']].values
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, algorithm='ball_tree').fit(coords)
_, indices = nbrs.kneighbors(coords)
neighbor_indices = indices[:, 1:]

for feat in ['floor_area_ratio', 'road_density_km_per_sqkm', 'population']:
    if feat in df.columns:
        df[f'neighbor_{feat}_mean'] = df[feat].values[neighbor_indices].mean(axis=1)
        print(f"  已添加特征：neighbor_{feat}_mean")

# ========== 4. 特征工程 ==========
target_col = '行人数量'
#base_features = ['Year', 'hour', 'population', 'dist_to_metro',
#                 'road_density_km_per_sqkm', 'floor_area_ratio']
base_features = ['is_epidemic_period', 'hour', 'population', 'dist_to_metro',
                 'road_density_km_per_sqkm', 'floor_area_ratio']
poi_cols = [c for c in df.columns if '_POI50' in c or '_POI100' in c or '_POI300' in c]
climate_cols = [c for c in df.columns if 'temp' in c or 'precip' in c or 'wind' in c]
land_features = ['缓冲区LU多样性', '不透水面占比(%)']
land_features = [c for c in land_features if c in df.columns]
restored_time_features = [c for c in ['season', '日期类型', '时段', 'hour_sin', 'hour_cos',
'is_covid', 'is_other_holiday'] if c in df.columns]

# 遥感特征（已时间对齐）
remote_features = ['NDVI', 'NDBI', 'MNDWI']

neighbor_cols = [c for c in df.columns if c.startswith('neighbor_')]
all_features_candidate = (base_features + poi_cols + climate_cols + land_features +
                          restored_time_features + neighbor_cols + remote_features)
all_features_candidate = list(set([f for f in all_features_candidate if f in df.columns]))
print(f"候选特征数量：{len(all_features_candidate)}")

# 共线性处理
X_corr = df[all_features_candidate].select_dtypes(include=[np.number])
corr_matrix = X_corr.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index if
upper_tri.loc[row, col] > 0.8]

to_drop = set()
if high_corr_pairs:
    print(f"发现 {len(high_corr_pairs)} 对极高相关特征，将删除冗余项...")
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
        print(f"  删除: {drop}")
    all_features_final = [f for f in all_features_candidate if f not in to_drop]
else:
    all_features_final = all_features_candidate
print(f"最终特征数量（不含空间编码）: {len(all_features_final)}")

# ========== 5. 准备数据（临时保留 grid_id） ==========
X_all = df[[*all_features_final, 'grid_id']].copy()
y_all = df[target_col].copy()
valid_mask = pd.concat([X_all, y_all], axis=1).notna().all(axis=1)
X_all = X_all[valid_mask].reset_index(drop=True)
y_all = y_all[valid_mask].reset_index(drop=True)

# 类别特征编码
for col in ['season', '日期类型', '时段']:
    if col in X_all.columns:
        X_all[col] = X_all[col].astype('category')
for col in X_all.select_dtypes(include=['object']).columns:
    if col != 'grid_id':
        X_all[col] = X_all[col].astype('category')

# 分层划分训练/验证集
bins = [-1, 0, 2, 5, 10, np.inf]
labels = [0, 1, 2, 3, 4]
stratify_labels = pd.cut(y_all, bins=bins, labels=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=VAL_SIZE, random_state=RANDOM_STATE,
    stratify=stratify_labels
)
print(f"\n训练集: {len(X_train)}，验证集: {len(X_val)}")

# 分离 grid_id
grid_train = X_train['grid_id'].copy()
grid_val   = X_val['grid_id'].copy()
X_train = X_train.drop(columns=['grid_id'])
X_val   = X_val.drop(columns=['grid_id'])

# ========== 6. 空间网格目标编码（KFold） ==========
'''print("正在进行空间网格目标编码 (K-Fold)...")
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

# 验证集编码
global_grid_mean = pd.DataFrame({'grid': grid_train, 'target': y_train}).groupby('grid')['target'].mean()
X_val['spatial_target_enc'] = grid_val.map(global_grid_mean).fillna(y_train.mean()).values
print("空间目标编码完成。新特征 'spatial_target_enc' 已添加。")

#all_features_final_with_enc = all_features_final + ['spatial_target_enc']
# 不再使用 spatial_target_enc，直接使用原始特征'''
all_features_final_with_enc = all_features_final.copy()

# ========== 7. 模型训练 ==========
print("\n" + "="*60)
print("单阶段 LightGBM Tweedie 回归（含空间目标编码）")
print("="*60)

reg = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.7,
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=5,
    num_leaves=25,
    min_child_samples=100,
    min_split_gain=0.02,
    reg_alpha=1.0,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

# ========== 高值样本权重函数 ==========
def get_sample_weight(y):
    """给高值样本分配更高权重，缓解低估问题"""
    w = np.ones_like(y, dtype=float)
    w[y > 5]   = 2.0
    w[y > 10]  = 3.0
    w[y > 20]  = 5.0
    w[y > 50]  = 10.0   # 对极端高值进一步加码
    return w

sample_weight_train = get_sample_weight(y_train.values)
print("样本权重分布：")
print(pd.Series(sample_weight_train).value_counts().sort_index())

reg.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,          # 新增：高值加权
    eval_set=[(X_val, y_val)],
    eval_metric='tweedie',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# 预测
train_pred = reg.predict(X_train)
val_pred = reg.predict(X_val)

# ========== 8. 整体评估 ==========
tr_r2 = r2_score(y_train, train_pred)
tr_mae = mean_absolute_error(y_train, train_pred)
tr_mse = mean_squared_error(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)
val_mae = mean_absolute_error(y_val, val_pred)
val_mse = mean_squared_error(y_val, val_pred)

print(f"\n整体指标:")
print(f"训练集 R²: {tr_r2:.4f}, MAE: {tr_mae:.2f}, MSE: {tr_mse:.2f}")
print(f"验证集 R²: {val_r2:.4f}, MAE: {val_mae:.2f}, MSE: {val_mse:.2f}")

# ========== 9. 分位数分层对比 + SSE 贡献 ==========
print("\n" + "="*60)
print("分位数分层对比（训练集 vs 验证集） + SSE 贡献")
print("="*60)

bins_display = [-1, 0, 2, 5, 10, np.inf]
labels_display = ['0', '1-2', '3-5', '6-10', '>10']
train_binned = pd.cut(y_train, bins=bins_display, labels=labels_display)
val_binned = pd.cut(y_val, bins=bins_display, labels=labels_display)

train_sse_total = ((y_train - train_pred) ** 2).sum()
val_sse_total = ((y_val - val_pred) ** 2).sum()

print(f"{'区间':<8} {'训练样本':<8} {'训练R²':<9} {'训练MAE':<9} {'训练MSE':<9} {'训练SSE':<10} | {'验证样本':<8} {'验证R²':<9} {'验证MAE':<9} {'验证MSE':<9} {'验证SSE':<10} {'SSE贡献%':<10}")
print("-" * 120)

for interval in labels_display:
    mask_tr = train_binned == interval
    mask_val = val_binned == interval
    n_tr = mask_tr.sum()
    n_val = mask_val.sum()

    if n_tr >= 2:
        r2_tr = r2_score(y_train[mask_tr], train_pred[mask_tr])
        mae_tr = mean_absolute_error(y_train[mask_tr], train_pred[mask_tr])
        mse_tr = mean_squared_error(y_train[mask_tr], train_pred[mask_tr])
        sse_tr = mse_tr * n_tr
    else:
        r2_tr, mae_tr, mse_tr, sse_tr = np.nan, np.nan, np.nan, np.nan

    if n_val >= 2:
        r2_val = r2_score(y_val[mask_val], val_pred[mask_val])
        mae_val = mean_absolute_error(y_val[mask_val], val_pred[mask_val])
        mse_val = mean_squared_error(y_val[mask_val], val_pred[mask_val])
        sse_val = mse_val * n_val
        sse_contrib = (sse_val / val_sse_total) * 100 if val_sse_total > 0 else np.nan
    else:
        r2_val, mae_val, mse_val, sse_val, sse_contrib = np.nan, np.nan, np.nan, np.nan, np.nan

    print(f"{interval:<8} {n_tr:<8} {r2_tr:<9.4f} {mae_tr:<9.2f} {mse_tr:<9.2f} {sse_tr:<10.1f} | "
          f"{n_val:<8} {r2_val:<9.4f} {mae_val:<9.2f} {mse_val:<9.2f} {sse_val:<10.1f} {sse_contrib:<10.1f}")

print("\n--- 零值 vs 正样本（含 SSE 贡献） ---")
for desc, mask_tr, mask_val in [("零值", y_train == 0, y_val == 0), ("正样本", y_train > 0, y_val > 0)]:
    n_tr, n_val = mask_tr.sum(), mask_val.sum()
    r2_tr = r2_score(y_train[mask_tr], train_pred[mask_tr]) if n_tr > 1 else np.nan
    r2_val = r2_score(y_val[mask_val], val_pred[mask_val]) if n_val > 1 else np.nan
    mae_tr = mean_absolute_error(y_train[mask_tr], train_pred[mask_tr])
    mae_val = mean_absolute_error(y_val[mask_val], val_pred[mask_val])
    sse_tr = mean_squared_error(y_train[mask_tr], train_pred[mask_tr]) * n_tr
    sse_val = mean_squared_error(y_val[mask_val], val_pred[mask_val]) * n_val
    sse_contrib_val = (sse_val / val_sse_total) * 100 if val_sse_total > 0 else np.nan
    print(f"{desc}: 训练集 R²={r2_tr:.4f}, MAE={mae_tr:.2f}, SSE={sse_tr:.1f}; "
          f"验证集 R²={r2_val:.4f}, MAE={mae_val:.2f}, SSE={sse_val:.1f}, 验证SSE贡献%={sse_contrib_val:.1f}%")

# ========== 10. SHAP 分析 ==========
'''print("\n正在生成 SHAP 图...")
sample_size = min(2000, len(X_val))
X_sample = X_val.sample(n=sample_size, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_sample)

# SHAP蜂群图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=X_val.columns, show=False)
plt.title('SHAP Feature Importance (Validation Set)', fontsize=14)
plt.tight_layout()
plt.show()

# SHAP柱状图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=X_val.columns, plot_type="bar", show=False)
plt.title('SHAP Feature Importance Ranking', fontsize=14)
plt.tight_layout()
plt.show()
'''

# =============================================================================
#                       全面诊断与问题分析
# =============================================================================
print("\n\n" + "="*70)
print("                        全 面 诊 断 开 始")
print("="*70)

# ---------- 1. 残差分析 ----------
residuals_train = y_train - train_pred
residuals_val   = y_val - val_pred

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1.1 残差分布直方图
axes[0,0].hist(residuals_val, bins=50, alpha=0.7, color='dodgerblue', edgecolor='black')
axes[0,0].axvline(x=0, color='red', linestyle='--')
axes[0,0].set_title('验证集残差分布', fontsize=13)
axes[0,0].set_xlabel('残差 (实际 - 预测)')

# 1.2 残差 vs 预测值散点图
axes[0,1].scatter(val_pred, residuals_val, alpha=0.3, s=2)
axes[0,1].axhline(y=0, color='red', linestyle='--')
axes[0,1].set_title('残差 vs 预测值', fontsize=13)
axes[0,1].set_xlabel('预测值'); axes[0,1].set_ylabel('残差')

# 1.3 预测值 vs 实际值散点图（理想情况在对角线）
axes[0,2].scatter(y_val, val_pred, alpha=0.3, s=2)
axes[0,2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
axes[0,2].set_title('预测 vs 实际', fontsize=13)
axes[0,2].set_xlabel('实际值'); axes[0,2].set_ylabel('预测值')

# 1.4 残差 Q-Q 图（检查正态性）
import scipy.stats as stats
stats.probplot(residuals_val, dist="norm", plot=axes[1,0])
axes[1,0].set_title('残差 Q-Q 图', fontsize=13)

# 1.5 零值预测分布
zero_mask = y_val == 0
axes[1,1].hist(val_pred[zero_mask], bins=40, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].axvline(x=0, color='red', linestyle='--')
axes[1,1].set_title('对真实零值的预测分布', fontsize=13)
axes[1,1].set_xlabel('预测值')

# 1.6 正样本预测偏差按真实值分组
pos_mask = y_val > 0
bias_by_true = pd.DataFrame({'actual': y_val[pos_mask], 'pred': val_pred[pos_mask]})
bias_by_true['error'] = bias_by_true['actual'] - bias_by_true['pred']
bias_by_true['actual_bin'] = pd.cut(bias_by_true['actual'], bins=[1,3,5,10,20,50,100,200,500])
grouped_bias = bias_by_true.groupby('actual_bin')['error'].agg(['mean','std','count'])
print("\n正样本预测偏差（按实际值分组）：")
print(grouped_bias)

# 也可绘制偏差图
grouped_bias['mean'].plot(kind='bar', ax=axes[1,2], color='teal', yerr=grouped_bias['std'])
axes[1,2].axhline(y=0, color='red', linestyle='--')
axes[1,2].set_title('不同实际值区间的平均误差', fontsize=13)
axes[1,2].set_xlabel('实际行人数量区间')
axes[1,2].set_ylabel('平均误差 (实际-预测)')

plt.tight_layout()
plt.show()

# ---------- 2. 特征重要性 & 增益对比 ----------
print("\n----- LightGBM 特征重要性 Top 20 (按 Gain) -----")
gain_imp = pd.DataFrame({
    'feature': reg.feature_name_,
    'gain': reg.booster_.feature_importance(importance_type='gain')
}).sort_values('gain', ascending=False)
print(gain_imp.head(20))

print("\n----- LightGBM 特征重要性 Top 20 (按 Split) -----")
split_imp = pd.DataFrame({
    'feature': reg.feature_name_,
    'split': reg.booster_.feature_importance(importance_type='split')
}).sort_values('split', ascending=False)
print(split_imp.head(20))

# 绘制 Gain 重要性
plt.figure(figsize=(10,8))
top_gain = gain_imp.head(20)
plt.barh(top_gain['feature'][::-1], top_gain['gain'][::-1], color='steelblue')
plt.title('特征重要性 (Gain) Top 20', fontsize=14)
plt.xlabel('Gain')
plt.tight_layout()
plt.show()

# ---------- 3. 预测值的分位数对比 ----------
print("\n----- 预测值与实际值的分位数对比 -----")
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
actual_quant = y_val.quantile(quantiles)
pred_quant = pd.Series(val_pred).quantile(quantiles)
quant_df = pd.DataFrame({'实际分位数': actual_quant.values, '预测分位数': pred_quant.values}, index=quantiles)
print(quant_df)

# 可视化
plt.figure(figsize=(8,5))
plt.plot(quantiles, actual_quant.values, 'bo-', label='实际值')
plt.plot(quantiles, pred_quant.values, 'rs--', label='预测值')
plt.xlabel('分位数'); plt.ylabel('行人数量')
plt.title('实际 vs 预测分位数对比')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# ---------- 4. 空间残差检查（若坐标可用） ----------
if 'X' in df.columns and 'Y' in df.columns:
    # 获取验证集坐标
    val_indices = X_val.index
    df_val_plot = df.iloc[val_indices].copy()
    df_val_plot['residual'] = residuals_val.values
    df_val_plot['abs_residual'] = np.abs(residuals_val.values)

    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    sc = ax.scatter(df_val_plot['X'], df_val_plot['Y'], 
                    c=df_val_plot['residual'], cmap='RdBu_r', 
                    s=0.5, alpha=0.5, vmin=-5, vmax=5)
    plt.colorbar(sc, label='残差')
    ax.set_title('空间残差分布 (红=低估, 蓝=高估)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.show()

# ---------- 5. 零膨胀分析 ----------
print(f"\n----- 零膨胀分析 -----")
zero_count = (y_val == 0).sum()
total_count = len(y_val)
print(f"验证集中零值比例: {zero_count/total_count:.3f} ({zero_count}/{total_count})")
print(f"预测值的最小值: {val_pred.min():.3f}, 最大值: {val_pred.max():.3f}")
print(f"预测值中 ≤0.5 的占比: {(val_pred <= 0.5).sum()/len(val_pred):.3f}")

# 实际零值与预测零值对比
print(f"真实零样本中被预测<0.5的比例: { (val_pred[zero_mask] < 0.5).mean():.3f}")

# ---------- 6. 学习曲线（训练历史） ----------
if hasattr(reg, 'booster_') and hasattr(reg.booster_, 'evals_result_'):
    evals_result = reg.booster_.evals_result_
    if 'valid_0' in evals_result:
        plt.figure(figsize=(8,5))
        plt.plot(evals_result['valid_0']['tweedie'], label='Validation Tweedie')
        plt.xlabel('Iterations'); plt.ylabel('Tweedie Loss')
        plt.title('训练过程损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ---------- 7. 改进建议（自动输出） ----------
print("\n" + "="*70)
print("                        改 进 建 议")
print("="*70)
print("""
根据上述诊断结果，你可以按优先级尝试以下改进：

1. 【零膨胀处理】若零值占比>40%且预测值全部>0，强烈建议两阶段模型或零膨胀模型(ZIP)。
2. 【高值低估】观察分位数对比，若预测值90%分位数远低于实际，可：
   - 对高值样本加权训练（设置sample_weight）
   - 尝试 tweedie_variance_power 调大（1.3~1.5）
   - 对目标变量取对数变换后回归，再指数还原
3. 【特征工程】根据 Gain 重要性，若遥感特征排位极低且与建成环境高相关，尝试：
   - 引入纹理特征(NDVI_std 50m/100m)
   - 构造交互特征(NDVI×周末, NDVI×地铁距离)
4. 【空间依赖】若空间残差图呈现明显聚集，可增加空间滞后项或使用空间回归模型。
5. 【类别不平衡】若零值预测概率分布偏向高值，尝试 tweedie 回归时设置`min_child_samples`更大，或换用Gamma回归。
6. 【模型融合】若单模型无法同时兼顾零值和高值，可训练多个模型对不同区间分别预测再组合。
""")