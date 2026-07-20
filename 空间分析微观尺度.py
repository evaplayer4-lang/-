#yolov5环境
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
file_cotype = r"E:\A研究生文献\近邻分析结果_带行人计数_complete_with_time_features.csv"
file_landuse = r"E:\A研究生文献\beijing_landcover.csv"
file_poi = r"E:\A研究生文献\beijing_POI.csv"
output_dir = r"E:\A研究生文献\微观分析结果"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# POI类别名称映射
poi_names = {
    1: '餐饮服务', 2: '道路附属设施', 3: '地名地址信息', 4: '风景名胜',
    5: '公共设施', 6: '公司企业', 7: '购物服务', 8: '交通设施服务',
    9: '金融保险服务', 10: '科教文化服务', 11: '摩托车服务', 12: '汽车服务',
    13: '汽车维修', 14: '汽车销售', 15: '商务住宅', 16: '生活服务',
    17: '事件活动', 18: '室内设施', 19: '体育休闲服务', 20: '通行设施',
    21: '虚拟数据', 22: '医疗保健服务', 23: '政府机关及社会群体'
}

# ==================== 数据读取与合并 ====================
print("正在读取数据...")
df_cotype = pd.read_csv(file_cotype, encoding='utf-8-sig')
df_landuse = pd.read_csv(file_landuse, encoding='utf-8-sig')
df_poi = pd.read_csv(file_poi, encoding='utf-8-sig')

df_cotype['FID'] = df_cotype['FID'].astype(str)
df_landuse['FID'] = df_landuse['FID'].astype(str)
df_poi['FID'] = df_poi['FID'].astype(str)

print("合并数据...")
df_merged = pd.merge(df_cotype, df_landuse, on='FID', how='inner')
df_merged = pd.merge(df_merged, df_poi, on='FID', how='inner')
print(f"合并后行数: {len(df_merged)}")

if 'COType' in df_merged.columns:
    df_merged['COType'] = df_merged['COType'].fillna('Not Significant')
else:
    print("警告：未找到COType列")
    exit()

# ==================== 数据筛选 ====================
df_lh = df_merged[df_merged['COType'] == 'LH'].copy()
df_hl = df_merged[df_merged['COType'] == 'HL'].copy()
df_hh = df_merged[df_merged['COType'] == 'HH'].copy()
df_ll = df_merged[df_merged['COType'] == 'LL'].copy()
print(f"LH点数: {len(df_lh)}, HL点数: {len(df_hl)}, HH点数: {len(df_hh)}, LL点数: {len(df_ll)}")

# ==================== 土地利用数值统计（表格） ====================
print("\n=== 提取土地利用数值统计 ===")
lu_columns = ['耕地占比(%)', '林地占比(%)', '灌木占比(%)', '草地占比(%)', 
              '水域占比(%)', '不透水面占比(%)', '荒地占比(%)', '湿地占比(%)']
group_names = ['LH', 'HL', 'HH', 'LL']
dfs = [df_lh, df_hl, df_hh, df_ll]

stats_dict = {}
for name, dfg in zip(group_names, dfs):
    stats = {}
    for col in lu_columns:
        stats[col] = {
            'mean': dfg[col].mean(),
            'std': dfg[col].std(),
            'median': dfg[col].median()
        }
    stats_dict[name] = stats

rows = []
for col in lu_columns:
    row = {'LandUse': col.replace('占比(%)', '')}
    for name in group_names:
        row[f'{name}_mean'] = stats_dict[name][col]['mean']
        row[f'{name}_std'] = stats_dict[name][col]['std']
        row[f'{name}_median'] = stats_dict[name][col]['median']
    rows.append(row)
df_lu_stats = pd.DataFrame(rows)
df_lu_stats.to_csv(os.path.join(output_dir, 'landuse_statistics.csv'), index=False, encoding='utf-8-sig')
print("土地利用统计表已保存: landuse_statistics.csv")

print("\n土地利用均值（%）：")
print(df_lu_stats[['LandUse'] + [f'{n}_mean' for n in group_names]].round(2))

# ==================== POI分析（四组分组条形图，顺序：HH, LH, HL, LL） ====================
print("\n=== 绘制POI分组条形图（顺序：HH, LH, HL, LL） ===")
poi_50_cols = [f'{i}_POI50' for i in range(1, 24+1)]
existing_poi_50 = [c for c in poi_50_cols if c in df_merged.columns]

def get_poi_stats(df_group):
    means = df_group[existing_poi_50].mean()
    stds = df_group[existing_poi_50].std()
    return means, stds

lh_means, lh_stds = get_poi_stats(df_lh)
hl_means, hl_stds = get_poi_stats(df_hl)
hh_means, hh_stds = get_poi_stats(df_hh)
ll_means, ll_stds = get_poi_stats(df_ll)

# 确定前10个POI类别（基于四组中最大均值排序）
all_means = pd.DataFrame({
    'HH': hh_means,
    'LH': lh_means,
    'HL': hl_means,
    'LL': ll_means
})
all_means['max'] = all_means.max(axis=1)
top_cats = all_means.nlargest(23, 'max').index.tolist()

# 按最大均值降序排列（用于x轴顺序）
top_cats = sorted(top_cats, key=lambda x: all_means.loc[x, 'max'], reverse=True)

# 设置绘图参数
x = np.arange(len(top_cats))
width = 0.2  # 每组柱子宽度

# 定义顺序：HH, LH, HL, LL（从左到右）
order = ['HH', 'LH', 'HL', 'LL']
# 对应的均值和标准差字典
means_dict = {'HH': hh_means, 'LH': lh_means, 'HL': hl_means, 'LL': ll_means}
stds_dict = {'HH': hh_stds, 'LH': lh_stds, 'HL': hl_stds, 'LL': ll_stds}
# 颜色映射
colors = {'HH': 'crimson', 'LH': 'goldenrod', 'HL': 'steelblue', 'LL': 'purple'}

fig, ax = plt.subplots(figsize=(16, 7))

# 计算偏移量：四个柱子均匀分布在 x 周围，偏移量分别为 -1.5*width, -0.5*width, 0.5*width, 1.5*width
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

for i, group in enumerate(order):
    means = means_dict[group][top_cats]
    stds = stds_dict[group][top_cats]
    ax.bar(x + offsets[i], means, width, label=group, yerr=stds, capsize=3,
           color=colors[group], error_kw={'linewidth': 1})

# 设置对数y轴
ax.set_yscale('log')
ax.set_ylabel('平均 POI 数量 (对数刻度)', fontsize=12)
ax.set_xlabel('POI 类别', fontsize=12)
ax.set_title('50m 缓冲区内主要 POI 类别（不同 COType 对比）', fontsize=14)
ax.set_xticks(x)
labels = [poi_names.get(int(col.split('_')[0]), col) for col in top_cats]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 保存POI统计表（四组，顺序按HH, LH, HL, LL）
poi_stats = pd.DataFrame({
    'POI_Code': top_cats,
    'POI_Name': [poi_names.get(int(c.split('_')[0]), c) for c in top_cats],
    'HH_mean': hh_means[top_cats].values,
    'HH_std': hh_stds[top_cats].values,
    'LH_mean': lh_means[top_cats].values,
    'LH_std': lh_stds[top_cats].values,
    'HL_mean': hl_means[top_cats].values,
    'HL_std': hl_stds[top_cats].values,
    'LL_mean': ll_means[top_cats].values,
    'LL_std': ll_stds[top_cats].values
})
poi_stats.to_csv(os.path.join(output_dir, 'all_groups_poi_top10.csv'), index=False, encoding='utf-8-sig')
print("POI统计表已保存: all_groups_poi_top10.csv")

print("\n所有分析完成！POI图表已显示在屏幕上。")