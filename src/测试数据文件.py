import pandas as pd
import ast
from collections import defaultdict

# 读取 train_data.csv 文件
df_train = pd.read_csv('data/train_data.csv')

# 初始化一个默认字典来存储 grid_id 和 brand_type 的计数
grid_brand_counts = defaultdict(lambda: defaultdict(int))

# 已知的 brand_type 前两个字符集合
brand_types = {'住宿', '摩托', '公司', '餐饮', '体育', '购物', '生活', '汽车', '医疗', '科教'}

# 遍历 train_data.csv 的每一行
for _, row in df_train.iterrows():
    brand = row['brand_type'][:2]  # 取 brand_type 的前两个字符
    # 将 grid_id_list 从字符串解析为列表
    grid_ids = ast.literal_eval(row['grid_id_list'])

    # 为每个 grid_id 统计 brand_type 的数量
    for grid_id in grid_ids:
        grid_brand_counts[grid_id][brand] += 1

df_grid = pd.read_csv('data/grid_coordinates.csv')

# 为每个 brand_type 添加一列，初始化为 0
for brand in brand_types:
    df_grid[brand] = 0

# 更新 df_grid 中每个 grid_id 对应的 brand_type 数量
for grid_id in grid_brand_counts:
    # 确保 grid_id 存在于 df_grid 中
    if grid_id in df_grid['grid_id'].values:
        for brand, count in grid_brand_counts[grid_id].items():
            df_grid.loc[df_grid['grid_id'] == grid_id, brand] = count

# 保存更新后的表格到新的 Excel 文件
df_grid.to_csv(
    "data/grid_coordinates-2.csv", encoding='gbk'
)