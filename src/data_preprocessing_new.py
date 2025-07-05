"""
数据预处理模块
该模块用于处理位置预测任务的数据，包括网格信息加载、样本生成和数据集划分。
主要实现了基于密度的店铺序列排序和序列到预测样本的转换。
"""
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def parse_list(s):
    """
    将字符串形式的列表解析为Python列表对象
    
    参数:
        s: 字符串表示的列表，如 "[1, 2, 3]"
        
    返回:
        解析后的Python列表，若解析失败则返回空列表
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def load_grid_info(grid_csv_path):
    """
    加载网格信息，包括坐标和POI特征，并进行归一化处理
    
    归一化的目的是将不同尺度的特征转换到同一尺度，提高模型训练效果。
    坐标归一化到[0,1]区间，使模型对不同区域的预测更加公平。
    POI特征归一化避免某些数值较大的特征主导模型训练。
    
    参数:
        grid_csv_path: 网格信息CSV文件路径
        
    返回:
        coords_map: 网格ID到归一化坐标的映射字典
        poi_feat_map: 网格ID到归一化POI特征的映射字典
    """
    grid_df = pd.read_csv(grid_csv_path, encoding='gbk')
    coords_map = {}
    poi_feat_map = {}
    poi_columns = ['医疗', '住宿', '摩托', '体育', '餐饮', '公司', '购物', '生活', '科教', '汽车']
    for _, row in grid_df.iterrows():
        gid = int(row["grid_id"])
        x = (row["grid_lon_min"] + row["grid_lon_max"]) / 2.0
        y = (row["grid_lat_min"] + row["grid_lat_max"]) / 2.0
        coords_map[gid] = (x, y)
        poi_feat_map[gid] = row[poi_columns].values.astype(float)
    # 归一化空间和poi
    xs, ys = zip(*coords_map.values())
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    for gid in coords_map:
        x, y = coords_map[gid]
        coords_map[gid] = [(x - x_min) / (x_max - x_min + 1e-8), (y - y_min) / (y_max - y_min + 1e-8)]
    all_poi = np.stack(list(poi_feat_map.values()))
    poi_min, poi_max = all_poi.min(axis=0), all_poi.max(axis=0)
    for gid in poi_feat_map:
        poi_feat_map[gid] = (poi_feat_map[gid] - poi_min) / (poi_max - poi_min + 1e-8)
    return coords_map, poi_feat_map

def sort_by_density(gid_list, coords_map):
    """
    基于密度对网格ID进行排序
    
    算法原理：
    计算每个点到其他K个最近邻点的平均距离，距离越小表示该点周围密度越大。
    通过这种排序方式，我们可以从高密度区域到低密度区域构建序列，
    这种顺序更符合商业扩张的实际规律（先在核心区域布局，再扩展到周边）。
    
    参数:
        gid_list: 网格ID列表
        coords_map: 网格ID到坐标的映射字典
        
    返回:
        按照密度排序后的网格ID列表
    """
    if len(gid_list) <= 1: return gid_list[:]
    k = 3  # 考虑最近的3个邻居点
    locs = [coords_map[g] for g in gid_list]
    scores = []
    for i, g in enumerate(gid_list):
        xi, yi = locs[i]
        dists = [float(np.linalg.norm([xi-xj, yi-yj])) for j, (xj, yj) in enumerate(locs) if i != j]
        # 计算到K个最近邻点的平均距离，如果点不足K个，则取所有点
        # 若没有其他点，则设置一个极大值表示最低密度
        avg = np.mean(sorted(dists)[:min(k, len(dists))]) if dists else 1e5
        scores.append((avg, g))
    scores.sort()  # 按平均距离排序，距离小的（密度大的）排前面
    return [g for _, g in scores]

def make_samples(data_csv_path, coords_map, poi_feat_map, grid2idx, max_seq_len=10):
    """
    构建序列预测样本
    
    采用滑动窗口方法，从每个品牌的店铺序列中生成多个训练样本：
    例如序列[A,B,C,D]会生成样本：
    - 已有[A]，预测B
    - 已有[A,B]，预测C
    - 已有[A,B,C]，预测D
    
    这种处理方法能够:
    1. 最大化利用有限的数据
    2. 学习不同长度序列的预测模式
    3. 模拟品牌实际扩张过程中的决策链
    
    参数:
        data_csv_path: 数据CSV文件路径
        coords_map: 网格ID到坐标的映射字典
        poi_feat_map: 网格ID到POI特征的映射字典
        grid2idx: 网格ID到索引的映射字典
        max_seq_len: 最大序列长度，若超过则截断
        
    返回:
        brand_samples: 样本列表，每个样本是(前缀索引, 前缀坐标, 前缀POI特征, 品牌文本, 目标索引)的元组
    """
    data_df = pd.read_csv(data_csv_path)
    brand_samples = []
    for _, row in data_df.iterrows():
        brand_name = row['brand_name']
        brand_type = row['brand_type']
        # 将品牌名称和类型组合成一个文本
        brand_text = f"{brand_name} {brand_type}"
        
        gid_list = parse_list(row['grid_id_list'])
        seq = sort_by_density(gid_list, coords_map)
        if len(seq) < 2: continue  # 至少需要两个点才能形成序列预测样本
        for l in range(1, len(seq)):
            prefix = seq[:l]
            target = seq[l]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]  # 保留最近的max_seq_len个点
            prefix_idx = [grid2idx[g] for g in prefix]
            prefix_coords = [coords_map[g] for g in prefix]
            prefix_poi = [poi_feat_map[g] for g in prefix]
            target_idx = grid2idx[target]
            brand_samples.append((prefix_idx, prefix_coords, prefix_poi, brand_text, target_idx))
    return brand_samples

def load_all_data(train_csv, test_csv, grid_csv, val_size=0.2):
    """
    加载并处理所有数据，划分为训练、验证和测试集
    
    该函数是数据处理的主入口，完成以下步骤：
    1. 加载网格信息（坐标和POI特征）
    2. 构建全局网格ID到索引的映射
    3. 生成训练和测试样本（包含品牌信息）
    4. 从训练样本中再划分出验证集
    
    数据划分的合理性：
    - 使用固定的随机种子确保实验可复现
    - 验证集从训练集中划分，保证测试集的纯净性
    - 按品牌-网格对划分，避免信息泄露
    
    参数:
        train_csv: 训练数据CSV文件路径
        test_csv: 测试数据CSV文件路径
        grid_csv: 网格信息CSV文件路径
        val_size: 验证集比例，默认为0.2
        
    返回:
        train_set: 训练集样本，每个样本包含(前缀索引, 前缀坐标, 前缀POI特征, 品牌文本, 目标索引)
        val_set: 验证集样本，格式同训练集
        test_samples: 测试集样本，格式同训练集
        num_classes: 类别数量（网格总数）
        grid2idx: 网格ID到索引的映射字典
    """
    coords_map, poi_feat_map = load_grid_info(grid_csv)
    # 构造全网格字典
    all_grids = set()
    for csvf in [train_csv, test_csv]:
        df = pd.read_csv(csvf)
        for _, row in df.iterrows():
            all_grids.update(parse_list(row['grid_id_list']))
    grid2idx = {gid: idx for idx, gid in enumerate(sorted(all_grids))}
    num_classes = len(grid2idx)
    train_samples = make_samples(train_csv, coords_map, poi_feat_map, grid2idx)
    test_samples = make_samples(test_csv, coords_map, poi_feat_map, grid2idx)
    # 再在train_samples中拆分出val
    train_idx, val_idx = train_test_split(np.arange(len(train_samples)), test_size=val_size, random_state=42)
    train_set = [train_samples[i] for i in train_idx]
    val_set = [train_samples[i] for i in val_idx]
    return train_set, val_set, test_samples, num_classes, grid2idx

def preprocess_brand_texts(brand_texts, tokenizer, max_length=64):
    """
    预先对品牌文本进行tokenization，避免在训练过程中重复tokenize
    
    参数:
        brand_texts: 品牌文本列表
        tokenizer: BERT tokenizer
        max_length: 最大文本长度
        
    返回:
        tokenized_data: 包含input_ids和attention_mask的字典
    """
    encoded = tokenizer(
        brand_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }
