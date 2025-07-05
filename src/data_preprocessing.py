import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

def parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def load_grid_info(grid_csv_path):
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
    if len(gid_list) <= 1: return gid_list[:]
    k = 3
    locs = [coords_map[g] for g in gid_list]
    scores = []
    for i, g in enumerate(gid_list):
        xi, yi = locs[i]
        dists = [np.linalg.norm([xi-xj, yi-yj]) for j, (xj, yj) in enumerate(locs) if i != j]
        avg = np.mean(sorted(dists)[:min(k, len(dists))]) if dists else 1e5
        scores.append((avg, g))
    scores.sort()
    return [g for _, g in scores]

def make_samples(data_csv_path, coords_map, poi_feat_map, grid2idx, max_seq_len=10):
    data_df = pd.read_csv(data_csv_path)
    brand_samples = []
    for _, row in data_df.iterrows():
        brand = row['brand_name']
        gid_list = parse_list(row['grid_id_list'])
        seq = sort_by_density(gid_list, coords_map)
        if len(seq) < 2: continue
        for l in range(1, len(seq)):
            prefix = seq[:l]
            target = seq[l]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]
            prefix_idx = [grid2idx[g] for g in prefix]
            prefix_coords = [coords_map[g] for g in prefix]
            prefix_poi = [poi_feat_map[g] for g in prefix]
            target_idx = grid2idx[target]
            brand_samples.append((prefix_idx, prefix_coords, prefix_poi, target_idx))
    return brand_samples

def load_all_data(train_csv, test_csv, grid_csv, val_size=0.2):
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
