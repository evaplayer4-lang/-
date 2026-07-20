import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.enums import Resampling
from matplotlib.font_manager import FontProperties
import warnings
warnings.simplefilter("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 设置路径 ==================
data_dir = r"E:\Python空间数据分析\Python-期中大作业题目-2026春\期中大作业2\data"

# ================== 2. 读取站点数据 ==================
stations = pd.read_csv(f"{data_dir}/AQI_station_en.csv")
if 'Unnamed: 0' in stations.columns:
    stations = stations.drop(columns=['Unnamed: 0'])
elif stations.columns[0] == '0':
    stations = pd.read_csv(f"{data_dir}/AQI_station_en.csv", header=None, names=['station_id', 'lon', 'lat'])
else:
    stations.columns = ['station_id', 'lon', 'lat']
stations['station_id'] = stations['station_id'].astype(str)
print(f"站点数: {len(stations)}")

# ================== 3. 读取AQI数据并提取12月PM2.5 ==================
aqi = pd.read_csv(f"{data_dir}/AQI_monthly_2014.csv")
type_col = aqi.columns[0]
date_col = aqi.columns[1]
station_cols = aqi.columns[2:]

df_long = pd.melt(aqi, id_vars=[type_col, date_col], value_vars=station_cols,
                  var_name='station_id', value_name='value')
df_long.columns = ['type', 'datetime', 'station_id', 'value']
df_long = df_long.dropna(subset=['value'])

pm25_mask = df_long['type'].str.strip() == 'PM2.5'
df_pm25 = df_long[pm25_mask]

# 筛选12月
if df_pm25['datetime'].dtype == 'object':
    try:
        df_pm25['datetime_dt'] = pd.to_datetime(df_pm25['datetime'])
        dec_mask = df_pm25['datetime_dt'].dt.month == 12
    except:
        dec_mask = df_pm25['datetime'].str.contains('Dec', na=False)
else:
    dec_mask = df_pm25['datetime'].dt.month == 12

df_pm25_dec = df_pm25[dec_mask].copy()
print(f"12月PM2.5记录数: {len(df_pm25_dec)}")

if len(df_pm25_dec) == 0:
    print("未找到12月PM2.5数据")
    exit()

df_pm25_dec['station_id'] = df_pm25_dec['station_id'].astype(str)
df = pd.merge(df_pm25_dec, stations, on='station_id', how='inner')
df.rename(columns={'value': 'pm25'}, inplace=True)
df = df.dropna(subset=['pm25'])
print(f"合并后有效站点数: {len(df)}")

# ================== 4. 读取所有栅格数据（自然+人文） ==================
mask = rxr.open_rasterio(f"{data_dir}/mask5k.tif").squeeze()
if mask.rio.crs is None:
    mask = mask.rio.set_crs("EPSG:4326")
study_mask = mask > 0

# 所有栅格文件列表（9个特征）
raster_files = {
    'prec': 'prec_201412.tif',      # 降水
    'temp': 'temp_201412.tif',      # 气温
    'dew': 'dew_201412.tif',        # 露点
    'wind': 'wind_speed_201412.tif',# 风速
    'aod': 'AOD_201412.tif',        # 气溶胶
    'roadden': 'roadden.tif',       # 路网密度
    'pop': 'pop2014.tif',           # 人口
    'ntl': 'ntl_2014.tif',          # 夜间灯光
    'dem': 'dem5k.tif'              # 高程
}

raster_data = {}
for name, fname in raster_files.items():
    da = rxr.open_rasterio(f"{data_dir}/{fname}").squeeze()
    if da.rio.crs is None:
        da = da.rio.set_crs("EPSG:4326")
    da = da.rio.reproject_match(mask, resampling=Resampling.bilinear)
    da = da.where(study_mask)
    raster_data[name] = da
    print(f"{name} 栅格已加载，形状: {da.shape}")

# ================== 5. 提取站点处所有栅格值 ==================
print("\n=== 提取站点特征 ===")
for name, da in raster_data.items():
    vals = da.interp(x=xr.DataArray(df['lon'].values, dims='point'),
                     y=xr.DataArray(df['lat'].values, dims='point'),
                     method='linear')
    df[name] = vals.values
    missing = df[name].isna().sum()
    print(f"{name}: 缺失值数量 = {missing}")

# 删除任何特征缺失的站点
before = len(df)
df = df.dropna(subset=list(raster_files.keys()))
print(f"删除缺失值后有效站点数: {len(df)} (原{before})")
if len(df) == 0:
    raise ValueError("无有效站点")

# ================== 6. 准备特征和目标 ==================
feature_cols = list(raster_files.keys())
X = df[feature_cols]
y = df['pm25']
print(f"\n参与建模的站点数: {len(X)}")
print(f"特征列表: {feature_cols}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== 7. 多模型比较 ==================
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Linear Regression': LinearRegression()
}

# 对SVR和KNN进行特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
best_model = None
best_r2 = -np.inf
best_model_name = None

print("\n" + "="*60)
print("开始训练模型...")
for name, model in models.items():
    print(f"\n训练模型: {name}")
    if name in ['SVR', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f"  RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name
        best_scaler = scaler if name in ['SVR', 'KNN'] else None

# 输出对比结果
print("\n" + "="*60)
print("模型性能对比表")
print("-"*60)
print(f"{'模型名称':<20} {'RMSE':<10} {'R²':<10}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['RMSE']:>8.2f}   {metrics['R2']:>8.3f}")
print("-"*60)
print(f"最佳模型: {best_model_name} (R² = {best_r2:.3f})")
print(f"参与建模的站点数: {len(X)}")
print("="*60)

# ================== 8. 使用最佳模型进行空间预测 ==================
print("\n=== 使用最佳模型进行空间预测 ===")
# 将所有特征栅格堆叠
X_stack = xr.concat([raster_data[name] for name in feature_cols], dim='var')
X_stack = X_stack.assign_coords(var=feature_cols)

X_flat = X_stack.values.reshape(len(feature_cols), -1).T
valid_mask = ~np.isnan(X_flat).any(axis=1)
X_valid = X_flat[valid_mask]

if best_scaler is not None:
    X_valid = best_scaler.transform(X_valid)
    y_pred_flat = best_model.predict(X_valid)
else:
    y_pred_flat = best_model.predict(X_valid)

pred_grid = np.full(X_flat.shape[0], np.nan)
pred_grid[valid_mask] = y_pred_flat
pred_2d = pred_grid.reshape(X_stack.shape[1], X_stack.shape[2])
coords = {d: X_stack.coords[d] for d in ['y', 'x']}
pred_da = xr.DataArray(pred_2d, coords=coords, dims=('y', 'x'))
pred_da = pred_da.where(study_mask)

# ================== 9. 绘图（叠加shp底图、站点、网格、鼠标坐标） ==================
# 读取中国省级行政区划
shp_path = r"E:\Python空间数据分析\Python-期中大作业题目-2026春\期中大作业1\China_adm\CHN_adm1.shp"
china = gpd.read_file(shp_path)
if china.crs is None:
    china = china.set_crs("EPSG:4326")

# 设置显示范围（比预测栅格实际范围稍大，留出空白）
raster_bounds = pred_da.rio.bounds()  # (left, bottom, right, top)
lon_min, lon_max = 107, 125
lat_min, lat_max = 32, 45

fig, ax = plt.subplots(figsize=(10, 8))

# 1. 省级边界
china.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

# 2. 预测栅格（使用最佳模型）
im = ax.imshow(pred_da, origin='upper', cmap='jet',
               extent=(raster_bounds[0], raster_bounds[2], raster_bounds[1], raster_bounds[3]),
               alpha=1.0)
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar.set_label('PM2.5(μg/m³)', fontsize=10)

# 3. 站点实测值（只显示在显示范围内的）
mask_plot = (df['lon'] >= lon_min) & (df['lon'] <= lon_max) & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
df_plot = df[mask_plot].copy()
sc = ax.scatter(df_plot['lon'], df_plot['lat'], c='red', s=30, cmap='coolwarm',
                edgecolor='k', linewidth=0.5, zorder=5)

# 设置显示范围
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# 刻度设置
lon_ticks = np.arange(lon_min, lon_max + 1, 3)
lat_ticks = np.arange(lat_min, lat_max + 1, 2)
ax.set_xticks(lon_ticks)
ax.set_yticks(lat_ticks)

eng_font = FontProperties(family='DejaVu Sans', size=9)
ax.set_xticklabels([f"{int(x)}\u00b0E" for x in lon_ticks], fontproperties=eng_font)
ax.set_yticklabels([f"{int(y)}\u00b0N" for y in lat_ticks], fontproperties=eng_font)

# 网格
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax.tick_params(top=False, right=False, labeltop=False, labelright=False)

ax.set_title(f"PM2.5 Concentration and AOI stations, Northen China", fontsize=12)

plt.tight_layout()
plt.show()