#yolov5环境
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 数据文件路径（请根据你的实际路径修改）
file_path = r"E:\A研究生文献\近邻分析结果_带行人计数_complete_with_time_features.csv"
# 输出结果保存路径（请根据你的实际路径修改）
output_dir = r"E:\A研究生文献\输出结果"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 字段定义（与原始数据列名对应）
col_season = 'season'
col_pedestrian = '行人数量'
col_date_type = '日期类型'
col_time_period = '时段'
col_cotype = 'COType'

# 映射与顺序（数字编码转文字标签）
season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
date_type_map = {1: 'Weekday', 2: 'Weekend', 3: 'Holiday'}
time_period_map = {1: 'Morning peak', 2: 'Off-peak', 3: 'Evening peak'}
cotype_order = ['HH', 'HL', 'LH', 'LL', 'Not Significant']

# ==================== 数据读取与预处理 ====================
print("正在读取数据...")
df = pd.read_csv(file_path)
df = df[df[col_time_period] != 4].copy()  # 过滤夜间时段（时段=4）

# 添加文字标签列
df['Season'] = df[col_season].map(season_map)
df['DateType'] = df[col_date_type].map(date_type_map)
df['TimePeriod'] = df[col_time_period].map(time_period_map)
df['LogPed'] = np.log1p(df[col_pedestrian])  # 行人数量log变换（避免0值问题）

# 设置分类变量顺序（保证绘图时类别不乱序）
season_order = list(season_map.values())
date_order = list(date_type_map.values())
time_order = list(time_period_map.values())

df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)
df['DateType'] = pd.Categorical(df['DateType'], categories=date_order, ordered=True)
df['TimePeriod'] = pd.Categorical(df['TimePeriod'], categories=time_order, ordered=True)

# 处理COType列（确保分类顺序）
if col_cotype in df.columns:
    df[col_cotype] = df[col_cotype].astype(str)
    present = df[col_cotype].unique()
    final_cotype_order = [c for c in cotype_order if c in present] + [c for c in present if c not in cotype_order]
    df[col_cotype] = pd.Categorical(df[col_cotype], categories=final_cotype_order, ordered=True)

# ==================== 统计检验 ====================
print("\n执行统计检验...")
# Kruskal-Wallis检验（多组非参数差异检验）
# 季节组检验
groups_season = [df[df['Season'] == s][col_pedestrian] for s in season_order]
h_season, p_season = kruskal(*groups_season)
# 日期类型组检验
groups_date = [df[df['DateType'] == d][col_pedestrian] for d in date_order]
h_date, p_date = kruskal(*groups_date)
# 时段组检验
groups_time = [df[df['TimePeriod'] == t][col_pedestrian] for t in time_order]
h_time, p_time = kruskal(*groups_time)

print(f"季节 K-W p = {p_season:.3e}, 日期类型 p = {p_date:.3e}, 时段 p = {p_time:.3e}")

# 保存描述性统计结果
for name, col, order in [('Season', 'Season', season_order),
                          ('DateType', 'DateType', date_order),
                          ('TimePeriod', 'TimePeriod', time_order)]:
    desc = df.groupby(col)[col_pedestrian].agg(['mean', 'median', 'std', 'count']).round(2)
    desc.columns = ['Mean', 'Median', 'Std', 'N']
    desc.to_csv(os.path.join(output_dir, f'{name}_stats.csv'), encoding='utf-8-sig')

# 保存Kruskal-Wallis检验结果
kw_df = pd.DataFrame({
    'Group': ['Season', 'DateType', 'TimePeriod'],
    'H': [h_season, h_date, h_time],
    'p': [p_season, p_date, p_time]
})
kw_df.to_csv(os.path.join(output_dir, 'kruskal_wallis.csv'), index=False, encoding='utf-8-sig')

# 夏季LH类型针对性检验（Mann-Whitney U检验）
if col_cotype in df.columns:
    summer = df[df['Season'] == 'Summer']
    lh = summer[summer[col_cotype] == 'LH'][col_pedestrian]
    hh = summer[summer[col_cotype] == 'HH'][col_pedestrian]
    ll = summer[summer[col_cotype] == 'LL'][col_pedestrian]
    if len(lh) > 0 and len(hh) > 0:
        u1, p1 = mannwhitneyu(lh, hh, alternative='two-sided')
        u2, p2 = mannwhitneyu(lh, ll, alternative='two-sided')
        with open(os.path.join(output_dir, 'summer_LH_tests.txt'), 'w') as f:
            f.write(f"Summer LH vs HH: p = {p1:.6f}\n")
            f.write(f"Summer LH vs LL: p = {p2:.6f}\n")
print("统计结果已保存至文件夹。")

# ==================== 绘图样式配置 ====================
def set_style():
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = 'black'

set_style()  # 应用绘图样式
palette = sns.color_palette("viridis", 8)  # 调色板

# ==================== 图1：时间三因素分布（合并为单一大框 + 组间间隔区分） ====================
import matplotlib.gridspec as gridspec

# 1. 创建单一大框画布，预留右侧图例空间
fig = plt.figure(figsize=(22, 8))
gs = gridspec.GridSpec(1, 10)  # 10列网格，用于灵活分配空间
ax_main = fig.add_subplot(gs[0, 0:8])  # 主图占前8列，右侧2列留作图例
plt.subplots_adjust(right=0.8)  # 进一步确保右侧空白

# 定义10个变量的不重复颜色（tab10调色板）
all_colors = sns.color_palette("tab10", 10)
season_colors = all_colors[:4]    # Spring, Summer, Autumn, Winter
date_colors = all_colors[4:7]     # Weekday, Weekend, Holiday
time_colors = all_colors[7:10]    # Morning peak, Off-peak, Evening peak

# -------------------- 数据重组：x轴标签完全等距 --------------------
# 10个变量等距分配x轴位置（0-9，每个间隔1，完全等距）
x_positions = list(range(10))  # [0,1,2,3,4,5,6,7,8,9] 等距位置
group_labels = [
    'Spring', 'Summer', 'Autumn', 'Winter',  # 季节4个
    'Weekday', 'Weekend', 'Holiday',          # 日期类型3个
    'Morning peak', 'Off-peak', 'Evening peak'# 时段3个
]  # 10个标签与等距位置一一对应
group_colors = all_colors  # 10种颜色与标签一一对应

# 重组数据为长格式（适配等距x轴）
plot_data = []
# 季节组（对应x=0,1,2,3）
for i, s in enumerate(season_order):
    for val in df[df['Season'] == s]['LogPed']:
        plot_data.append({'x': x_positions[i], 'LogPed': val})
# 日期类型组（对应x=4,5,6）
for i, d in enumerate(date_order):
    for val in df[df['DateType'] == d]['LogPed']:
        plot_data.append({'x': x_positions[4+i], 'LogPed': val})
# 时段组（对应x=7,8,9）
for i, t in enumerate(time_order):
    for val in df[df['TimePeriod'] == t]['LogPed']:
        plot_data.append({'x': x_positions[7+i], 'LogPed': val})
plot_df = pd.DataFrame(plot_data)

# -------------------- 绘制单一大框小提琴图 --------------------
violin = sns.violinplot(
    data=plot_df, x='x', y='LogPed',
    palette=group_colors,  # 10种不重复颜色
    inner='box', linewidth=1.2, ax=ax_main
)

# -------------------- 美化坐标轴与标签 --------------------
# 设置x轴标签（对应三组变量）
ax_main.set_xticks(x_positions)
ax_main.set_xticklabels(group_labels, fontsize=14)
ax_main.set_xlabel('')  # 去除x轴总标签，用分组结构体现
ax_main.set_ylabel('log(Pedestrian + 1)', fontsize=16)
ax_main.tick_params(axis='both', labelsize=14)

# 去除原分组标题（无标题，靠组间间隔区分三组）
ax_main.set_title('')

# 添加组间分隔线（可选，增强区分度）
ax_main.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)  # 季节与日期类型之间
ax_main.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)  # 日期类型与时段之间

# -------------------- 全局图例（放在右侧空白区域，完全不遮挡） --------------------
all_labels = [
    'Spring', 'Summer', 'Autumn', 'Winter',
    'Weekday', 'Weekend', 'Holiday',
    'Morning peak', 'Off-peak', 'Evening peak'
]
all_handles = [plt.Rectangle((0,0),1,1, color=all_colors[i]) for i in range(len(all_labels))]

fig.legend(
    handles=all_handles,
    labels=all_labels,
    title='Categories',
    loc='center left',
    bbox_to_anchor=(0.82, 0.5),  # 定位到右侧空白区，不遮挡主图
    fontsize=14,
    title_fontsize=16,
    frameon=True,
    borderpad=1.2
)



plt.tight_layout()
plt.show()
# 可选：保存图片
# plt.savefig(os.path.join(output_dir, 'time_factors_combined.png'), dpi=300, bbox_inches='tight')


# ==================== 图2：季节 × COType 分面小提琴图 ====================
if col_cotype in df.columns:
    set_style()
    # 核心修改1：设置sharey=True，所有分面共享y轴（仅左侧显示）
    g = sns.catplot(
        data=df, x=col_cotype, y='LogPed', col='Season', kind='violin',
        palette='viridis', inner='box', linewidth=1.2,
        height=4, aspect=0.8, order=final_cotype_order, sharey=True  # sharey=True：共享y轴
    )
    g.set_axis_labels('COType', 'log(Pedestrian + 1)')
    g.set_titles('{col_name}')  # 保留季节标题，便于区分分面
    
    # 核心修改2：隐藏非左侧分面的y轴标签和刻度（仅保留第一个分面的y轴）
    for i, ax in enumerate(g.axes.flat):
        if i > 0:  # 除了第一个分面（Spring），其他分面隐藏y轴
            ax.set_ylabel('')  # 隐藏y轴标签
            ax.tick_params(axis='y', left=False, labelleft=False)  # 隐藏y轴刻度和刻度标签
    
    # 核心修改3：调整画布布局，右侧预留空间放图例，避免遮挡
    plt.subplots_adjust(right=0.85)  # 右侧预留15%空白区域
    
    # 核心修改4：将图例移到画布右侧空白处（不遮挡小提琴图）
    cotype_labels = final_cotype_order
    handles_cotype = [plt.Rectangle((0,0),1,1, color=sns.color_palette('viridis', len(cotype_labels))[i]) 
                      for i in range(len(cotype_labels))]
    # 图例定位：bbox_to_anchor=(0.9, 0.5) 表示图例左边缘在画布90%宽度位置，垂直居中
    g.fig.legend(
        handles=handles_cotype,
        labels=cotype_labels,
        title='COType',
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),  # 放在右侧预留区域
        fontsize=10,
        title_fontsize=12,
        frameon=True
    )
    
    plt.tight_layout()
    plt.show()
    # 可选：保存图2（取消注释即可）
    # plt.savefig(os.path.join(output_dir, 'season_cotype_distribution_final.png'), dpi=300, bbox_inches='tight')
else:
    print("COType列不存在，跳过图2。")

print("\n所有图表已生成完成！")