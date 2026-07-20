import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
RANDOM_STATE = 42
TEST_SIZE = 0.2
MI_THRESHOLD = 0.01
CORR_THRESHOLD = 0.8

# 最优参数
BEST_PARAMS = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.05,
    'max_depth': 9,
    'n_estimators': 300,
    'subsample': 0.8,
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

# ========== 1. 加载数据 ==========
main_path = r"E:\A研究生文献\近邻分析结果_带行人计数_complete_with_time_features.csv"
poi_path = r"E:\A研究生文献\beijing_POI.csv"
climate_path = r"E:\A研究生文献\beijing_climate.csv"
landcover_path = r"E:\A研究生文献\beijing_landcover.csv"
pop_path = r"E:\A研究生文献\匹配结果.csv"          # 新增：人口数据文件
# 新增：路网密度/道路长度数据
road_path = r"E:\A研究生文献\photo_road_density.csv"  

df_main = pd.read_csv(main_path)
df_poi = pd.read_csv(poi_path)
df_climate = pd.read_csv(climate_path)
df_landcover = pd.read_csv(landcover_path)
df_pop = pd.read_csv(pop_path)                     # 新增：读取人口数据
df_road = pd.read_csv(road_path)  # 新增：读取路网特征

print("数据加载完成。")

# ========== 2. 合并数据 ==========
df = df_main.merge(df_poi, on='FID', how='left')
df = df.merge(df_climate, on='FID', how='left')
df = df.merge(df_landcover, on='FID', how='left')
df = df.merge(df_pop[['FID', 'population']], on='FID', how='left')   # 新增：合并人口特征
# 新增：合并路网特征（FID匹配）
df = df.merge(df_road, on='FID', how='left')  
print("合并后数据形状:", df.shape)

# ========== 2B. 剔除无效时段 ==========
# 假设 '时段' == 4 代表晚上/夜间（无效时段）
if '时段' in df.columns:
    before_count = len(df)
    df = df[df['时段'] != 4].copy()
    print(f"过滤掉夜间无效时段(时段==4)，样本数从 {before_count} 减少至 {len(df)}")
else:
    print("警告：数据中无 '时段' 列，跳过无效时段过滤")

# ========== 3. 剔除日尺度气候缺失 ==========
daily_cols = ['daily_temp', 'daily_precip', 'daily_wind']
if all(c in df.columns for c in daily_cols):
    before = len(df)
    df = df.dropna(subset=daily_cols, how='any')
    print(f"剔除日尺度缺失样本 {before - len(df)} 条，剩余 {len(df)} 条")
else:
    print("警告：缺少日尺度气候列，跳过剔除步骤。")

# ========== 3B. 重置 Hour 特征：确保仅在白天有效范围内 ==========
# 确保模型只在有数据的 7:00 - 18:00 范围内进行拟合
if 'hour' in df.columns:
    valid_hours_mask = (df['hour'] >= 7) & (df['hour'] <= 18)
    valid_hours_count = valid_hours_mask.sum()
    print(f"小时范围检查 (7:00-18:00)：符合条件的样本数 {valid_hours_count} / {len(df)}")
    
    # 过滤出有效白天时段的数据
    before_hour_filter = len(df)
    df = df[valid_hours_mask].copy()
    after_hour_filter = len(df)
    print(f"过滤白天有效时段 (7:00-18:00)，样本数从 {before_hour_filter} 减少至 {after_hour_filter}")
else:
    print("警告：数据中无 'hour' 列，跳过小时范围过滤")

# ========== 全局变量定义 ==========
target_col = '行人数量'

def find_col(df, pattern):
    matches = [c for c in df.columns if re.search(pattern, c, re.IGNORECASE)]
    return matches[0] if matches else None

year_col = find_col(df, r'^year$|^年份$') or 'year'
season_col = find_col(df, r'^season$|^季节$') or 'season'
date_type_col = find_col(df, r'日期类型|日期类别') or '日期类型'
period_col = find_col(df, r'^时段$|^时间段$') or '时段'

poi_cols = [c for c in df.columns if '_POI50' in c or '_POI100' in c or '_POI300' in c]
climate_cols = ['daily_temp', 'daily_precip', 'daily_wind',
                'monthly_temp', 'monthly_precip', 'monthly_wind',
                'yearly_temp', 'yearly_precip', 'yearly_wind']
climate_cols = [c for c in climate_cols if c in df.columns]

landcover_features = []
if '缓冲区LU多样性' in df.columns:
    landcover_features.append('缓冲区LU多样性')
if '缓冲区主导LU名称' in df.columns:
    landcover_features.append('缓冲区主导LU名称')
if '不透水面占比(%)' in df.columns:
    landcover_features.append('不透水面占比(%)')

spatial_features = ['X', 'Y'] if 'X' in df.columns else []
time_features = []
for col in ['hour', 'hour_sin', 'hour_cos', 'is_covid', 'is_other_holiday']:
    if col in df.columns:
        time_features.append(col)

# 新增：人口特征（连续数值，无缺失）
pop_feature = ['population'] if 'population' in df.columns else []

# 新增：距离地铁特征（连续数值）
metro_feature = ['dist_to_metro'] if 'dist_to_metro' in df.columns else []

# ====================== 新增：路网特征 ======================
road_features = []
if 'road_length_m' in df.columns:
    road_features.append('road_length_m')
if 'road_density_km_per_sqkm' in df.columns:
    road_features.append('road_density_km_per_sqkm')
# ===========================================================

# ========== 新增：时空交互特征 ==========
# 定义核心POI类别关键词（基于数字代码，可根据数据调整）
# 对应POI分类代码：餐饮(1)、购物(7)、娱乐/体育(4,19)、交通(8)、商务(6,15)、医疗(22)、教育(10)等
core_poi_keywords = [
    '1',  # 餐饮服务
    '7',  # 购物服务
    '4', '19',  # 风景名胜/体育休闲
    '8',  # 交通设施服务
    '6', '15',  # 公司企业/商务住宅
    '22',  # 医疗保健服务
    '10'  # 科教文化服务
]  # 示例关键词，可根据实际列名扩展

# 识别核心POI特征
core_poi_cols = []
for poi in poi_cols:
    for keyword in core_poi_keywords:
        if keyword in poi:
            core_poi_cols.append(poi)
            break

# 创建时空交互特征：Hour × 核心POI
spatiotemporal_features = []
if 'hour' in df.columns and core_poi_cols:
    for poi_col in core_poi_cols:
        interaction_col = f'hour_{poi_col}'
        df[interaction_col] = df['hour'] * df[poi_col]
        spatiotemporal_features.append(interaction_col)
    print(f"创建时空交互特征: {len(spatiotemporal_features)} 个 (Hour × 核心POI)")
else:
    print("警告：缺少 'hour' 列或核心POI特征，跳过时空交互特征创建")

# ====================== 新增：地铁距离 × 小时 交互特征 ======================
metro_hour_interaction = []
if 'hour' in df.columns and 'dist_to_metro' in df.columns:
    # 创建交互特征：小时 × 地铁距离
    df['hour_dist_to_metro'] = df['hour'] * df['dist_to_metro']
    metro_hour_interaction.append('hour_dist_to_metro')
    spatiotemporal_features.extend(metro_hour_interaction)  # 加入时空特征列表
    print(f"新增地铁-时间交互特征: {len(metro_hour_interaction)} 个 (Hour × 地铁距离)")
# ==========================================================================

# 合并所有特征（加入人口 + 地铁距离 + 时空交互）
#all_features = [year_col, season_col, date_type_col, period_col] + poi_cols + climate_cols + landcover_features + spatial_features + time_features + pop_feature + metro_feature + spatiotemporal_features
# 合并所有特征（加入人口 + 地铁距离 + 时空交互）
# 【删除了 year_col，不参与分析】
# 合并所有特征（加入人口 + 地铁距离 + 时空交互 + 路网特征）
all_features = [season_col, date_type_col, period_col] + poi_cols + climate_cols + landcover_features + spatial_features + time_features + pop_feature + metro_feature + spatiotemporal_features + road_features
all_features = [f for f in all_features if f in df.columns]

# FID 不作为特征进入分析
# if 'FID' in df.columns and 'FID' not in all_features:
#     all_features.insert(0, 'FID')  # 将 FID 放在首位

print("原始特征总数:", len(all_features))

# 构建特征矩阵 X 和目标 y
X = df[all_features].copy()
y = df[target_col].copy()

# ========== 5. 删除所有包含缺失值的行（而不是填充） ==========
print("\n检查缺失值...")
before = len(X)
data_temp = pd.concat([X, y], axis=1)
data_temp = data_temp.dropna()
after = len(data_temp)
print(f"删除缺失值行数: {before - after}，剩余样本数: {after}")
X = data_temp.drop(columns=[target_col])
y = data_temp[target_col]
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# ========== 6. 类别特征编码 ==========
categorical_cols = [season_col, date_type_col, period_col]
if '缓冲区主导LU名称' in X.columns:
    categorical_cols.append('缓冲区主导LU名称')
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('category').cat.codes

print(f"特征矩阵形状: {X.shape}")

# ========== 6A. 混合模型三元需求: 二分类目标 ==========
# 创建二分类目标: 1表示有人(>0)，0表示无人(==0)
y_binary = (y > 0).astype(int)
print(f"二分类目标的正样本比: {y_binary.mean():.2%}")

# ========== 7. 过滤式特征选择 ==========
print("\n开始过滤式特征选择 (基于训练集)...")
X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
    X, y, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

variances = X_train.var()
zero_var_features = variances[variances == 0].index.tolist()
print(f"方差为0的特征（共{len(zero_var_features)}个）: {zero_var_features}")
X_train_filtered = X_train.drop(columns=zero_var_features)

mi_scores = mutual_info_regression(X_train_filtered, y_train, random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=X_train_filtered.columns).sort_values(ascending=False)
print("互信息得分前10:")
print(mi_series.head(10))
mi_selected = mi_series[mi_series > MI_THRESHOLD].index.tolist()
print(f"保留互信息 > {MI_THRESHOLD} 的特征，共 {len(mi_selected)} 个")

X_mi = X_train_filtered[mi_selected] if mi_selected else X_train_filtered
numeric_cols = [c for c in mi_selected if c not in categorical_cols]
if numeric_cols:
    corr_matrix = X_train[numeric_cols].corr(method='spearman').abs()
else:
    corr_matrix = pd.DataFrame(index=[], columns=[])

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, idx) for col in upper.columns for idx in upper.index if upper.loc[idx, col] > CORR_THRESHOLD]
print(f"高相关特征对（|r|>{CORR_THRESHOLD}）数量: {len(high_corr_pairs)}")
to_drop = set()
for col1, col2 in high_corr_pairs:
    if mi_series[col1] > mi_series[col2]:
        to_drop.add(col2)
    else:
        to_drop.add(col1)
print(f"建议剔除的高相关特征: {list(to_drop)}")
X_redundant_removed = X_mi.drop(columns=to_drop)
final_features_filter = X_redundant_removed.columns.tolist()
print(f"过滤式筛选后特征数: {len(final_features_filter)}")

# ========== 8. 嵌入式特征选择 ==========
print("\n开始嵌入式特征选择...")
# 对目标变量进行Log转换
y_train_log = np.log1p(y_train)
temp_model = xgb.XGBRegressor(**BEST_PARAMS)
temp_model.fit(X_train[final_features_filter], y_train_log)
importance = temp_model.feature_importances_
importance_df = pd.DataFrame({'feature': final_features_filter, 'importance': importance}).sort_values('importance', ascending=False)
print("特征重要性排序（前20）：")
print(importance_df.head(20))
# 使用中位数门槛保留更多特征，避免筛掉过多空间特征
thr = importance_df['importance'].median()
selected_features = importance_df[importance_df['importance'] >= thr]['feature'].tolist()
if 'hour' in X_train.columns and 'hour' not in selected_features:
    selected_features.append('hour')
# FID 已不作为特征
# if 'FID' in selected_features:
#     selected_features.remove('FID')
embedded_features = selected_features
print(f"嵌入式筛选后特征数: {len(embedded_features)}")
print("保留的特征列表：")
print(embedded_features)

# ========== 9. 定义两阶段模型一体评估函数 ==========
def evaluate_hurdle_model(X_tr, X_te, y_tr, y_te, y_bin_tr, y_bin_te, feature_list, description):
    """
    Hurdle Model: 第一阶段事务二传分类判断是否有人，
                   第二阶段第龙回归预测有人时的数量
    """
    # 第一阶段: 二分类器
    print(f"\n{description} - 第一阶段：训练二分类器...")
    scale_pos_weight = (y_bin_tr == 0).sum() / max((y_bin_tr == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective='binary:logistic',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    clf.fit(X_tr[feature_list], y_bin_tr)
    
    # 评估二分类器
    proba_has_people = clf.predict_proba(X_te[feature_list])[:, 1]
    bin_preds = (proba_has_people > 0.5).astype(int)
    clf_accuracy = accuracy_score(y_bin_te, bin_preds)
    prob_has_people = proba_has_people
    print(f"  二分类器准确率: {clf_accuracy:.4f}")
    
    # 第二阶段: 回归模型 (仅使用正值数据)
    print(f"  第二阶段：训练正值回归模型...")
    mask_tr = (y_tr > 0)
    X_tr_pos = X_tr[mask_tr][feature_list]
    y_tr_pos = np.log1p(y_tr[mask_tr])
    
    if len(X_tr_pos) > 0:
        # 添加样本权重：普通流量(≤20人)=1.0，长尾爆发点(>20人)=3.0
        sample_weights = np.where(y_tr[mask_tr] > 20, 3.0, 1.0)
        
        reg = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            objective='reg:squarederror',  # 回到原来的均方误差
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        reg.fit(X_tr_pos, y_tr_pos, sample_weight=sample_weights)
        
        # 屏宽二阶段预测: 是否有人的概率 * 预测的人数
        raw_reg_preds = np.expm1(reg.predict(X_te[feature_list]))
        y_pred = prob_has_people * raw_reg_preds
    else:
        y_pred = np.zeros(len(y_te))
    
    # 评估评指标
    r2 = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    
    print(f"\n{description} 两阶段模型评估结果:")
    print(f"  特征数量: {len(feature_list)}")
    print(f"  二分类准确率: {clf_accuracy:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")
    
    return {
        'Model': description,
        'NumFeatures': len(feature_list),
        'Clf_Accuracy': clf_accuracy,
        'R²': r2,
        'MSE': mse,
        'MAE': mae,
        'y_pred': y_pred,
        'y_test': y_te
    }

# ========== 9A. 定义两阶段模型对比残差分析函数 ==========
def plot_comparative_hurdle_analysis(results_hurdle):
    num_models = len(results_hurdle)
    fig, axes = plt.subplots(3, num_models, figsize=(6 * num_models, 15), squeeze=False)
    row_titles = ['True vs Predicted', 'Residual Distribution', 'Predicted vs Residuals']

    for col_idx, result in enumerate(results_hurdle):
        y_test = result['y_test']
        y_pred = result['y_pred']
        residuals = y_test - y_pred
        model_name = result['Model']

        ax = axes[0, col_idx]
        sns.scatterplot(ax=ax, x=y_test, y=y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(model_name)

        ax = axes[1, col_idx]
        sns.histplot(ax=ax, x=residuals, kde=True, color='g')
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Residual')
        ax.set_title('Residual Distribution')

        ax = axes[2, col_idx]
        sns.scatterplot(ax=ax, x=y_pred, y=residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Predicted vs Residuals')

    for row_idx, row_title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(row_title, fontsize=12)

    plt.tight_layout()
    plt.show()

# ========== 9B. 定义空间维度误差分析函数 ==========
def analyze_spatial_errors(X_test, y_test, y_pred):
    # 1. 组装数据
    analysis_df = X_test.copy()
    analysis_df['true_val'] = y_test
    analysis_df['pred_val'] = y_pred
    analysis_df['abs_error'] = np.abs(y_test - y_pred)
    
    # 由于FID不作为特征，跳过按FID聚合
    print("--- 空间误差分析：整体误差统计 ---")
    print(f"平均绝对误差: {analysis_df['abs_error'].mean():.4f}")
    print(f"最大绝对误差: {analysis_df['abs_error'].max():.4f}")
    print(f"最小绝对误差: {analysis_df['abs_error'].min():.4f}")
    
    # 4. 可视化：真实 vs 预测 的空间分布对比 (如果有 X, Y 坐标)
    if 'X' in analysis_df.columns and 'Y' in analysis_df.columns:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(analysis_df['X'], analysis_df['Y'], c=analysis_df['true_val'], cmap='YlOrRd', s=10)
        plt.title('True Pedestrian Flow Distribution')
        plt.colorbar(label='Count')
        
        plt.subplot(1, 2, 2)
        plt.scatter(analysis_df['X'], analysis_df['Y'], c=analysis_df['pred_val'], cmap='YlOrRd', s=10)
        plt.title('Predicted Pedestrian Flow Distribution')
        plt.colorbar(label='Count')
        plt.show()
    else:
        print("无X,Y坐标，无法进行空间可视化")

    # 返回整体误差统计
    return {
        'mean_abs_error': analysis_df['abs_error'].mean(),
        'max_abs_error': analysis_df['abs_error'].max(),
        'min_abs_error': analysis_df['abs_error'].min()
    }

# ========== 9C. 定义时间维度分析函数 ==========
def analyze_temporal_patterns(X_test, y_test, y_pred):
    # 1. 组装数据
    temp_df = pd.DataFrame({
        'hour': X_test['hour'],  # 确保你的 X_test 里有 hour 这一列
        'true_val': y_test,
        'pred_val': y_pred
    })
    
    # 2. 按小时计算平均值
    hourly_stats = temp_df.groupby('hour').agg({
        'true_val': 'mean',
        'pred_val': 'mean'
    }).reset_index()
    
    # 3. 绘制 24 小时对比曲线
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_stats['hour'], hourly_stats['true_val'], label='Actual Average', marker='o', linewidth=2)
    plt.plot(hourly_stats['hour'], hourly_stats['pred_val'], label='Predicted Average', marker='x', linestyle='--', linewidth=2)
    
    plt.title('Hourly Pedestrian Flow: Actual vs Predicted')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Pedestrian Count')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # 4. 计算相关性
    correlation = hourly_stats['true_val'].corr(hourly_stats['pred_val'])
    print(f"时间维度上的趋势相关性: {correlation:.4f}")

# ========== 11B. 使用两阶段模型 (Hurdle Model) 评估 ==========
print("\n" + "="*80)
print("使用两阶段模型 (Hurdle Model) 进行评估")
print("="*80)

results_hurdle = []
hurdle_result1 = evaluate_hurdle_model(X_train, X_test, y_train, y_test, y_bin_train, y_bin_test, all_features, "全特征模型")
results_hurdle.append(hurdle_result1)
hurdle_result2 = evaluate_hurdle_model(X_train, X_test, y_train, y_test, y_bin_train, y_bin_test, final_features_filter, "过滤式特征模型")
results_hurdle.append(hurdle_result2)
hurdle_result3 = evaluate_hurdle_model(X_train, X_test, y_train, y_test, y_bin_train, y_bin_test, embedded_features, "嵌入式特征模型")
results_hurdle.append(hurdle_result3)

# ========== 12B. 输出两阶段模型对比表格 ==========
print("\n" + "="*80)
print("两阶段模型: 最终对比总结")
print("="*80)
results_hurdle_display = [{k: v for k, v in r.items() if k not in ['y_pred', 'y_test']} for r in results_hurdle]
df_results_hurdle = pd.DataFrame(results_hurdle_display)
print(df_results_hurdle.to_string(index=False))

# ========== 13. 残差分析 (两阶段模型) ==========
print("\n" + "="*80)
print("两阶段模型残差分析对比")
print("="*80)
plot_comparative_hurdle_analysis(results_hurdle)

# ========== 13B. 空间维度误差分析 ==========
print("\n" + "="*80)
print("空间维度误差分析（两阶段模型 - 嵌入式特征）")
print("="*80)
spatial_results = analyze_spatial_errors(X_test, hurdle_result3['y_test'], hurdle_result3['y_pred'])

# ========== 13C. 时间维度分析 ==========
print("\n" + "="*80)
print("时间维度分析（两阶段模型 - 嵌入式特征）")
print("="*80)
analyze_temporal_patterns(X_test, hurdle_result3['y_test'], hurdle_result3['y_pred'])

# ========== 14. SHAP分析（两阶段模型 - 嵌入式特征） ==========
print("\n" + "="*80)
print("SHAP分析（两阶段模型-嵌入式特征，基于测试集抽样2000）")
print("="*80)

try:
    import shap
    import matplotlib.pyplot as plt
    
    # 两阶段模型-第二阶段回归的SHAP分析
    X_train_emb = X_train[embedded_features].copy()
    X_test_emb = X_test[embedded_features].copy()
    
    # 仅在正值数据上训练回归模型
    mask_train_pos = (y_train > 0)
    X_train_emb_pos = X_train_emb[mask_train_pos]
    y_train_log = np.log1p(y_train[mask_train_pos])
    
    model_emb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    model_emb.fit(X_train_emb_pos, y_train_log)
    
    sample_size = min(2000, X_test_emb.shape[0])
    X_sample = X_test_emb.sample(n=sample_size, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model_emb)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=embedded_features,
        show=False,
        max_display=10
    )
    plt.title('SHAP Analysis - Hurdle Model (Stage2: Regression on Positive Values)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("未安装shap库。请运行: pip install shap")
except Exception as e:
    print(f"SHAP分析出错: {e}")





