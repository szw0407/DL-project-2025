import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast


#拿到一个字符串类型的数据
def parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def load_data(train_csv, test_csv, grid_csv, method='density'):
    # 读取 grid 坐标（将中心店作为坐标来进行处理）
    grid_df = pd.read_csv(grid_csv)
    coords_map = {}
    for _, row in grid_df.iterrows():
        gid = row["grid_id"]
        # 用网格中心点
        x_center = (row["grid_lon_min"] + row["grid_lon_max"]) / 2.0
        y_center = (row["grid_lat_min"] + row["grid_lat_max"]) / 2.0
        coords_map[gid] = (x_center, y_center)
    # 将数据归一化
    xs, ys = zip(*coords_map.values())
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    for gid in coords_map:
        x, y = coords_map[gid]
        coords_map[gid] = ((x - x_min) / (x_max - x_min + 1e-9), (y - y_min) / (y_max - y_min + 1e-9))

    """
     这部分的功能是将测试集和训练集的brand + grid_list对应起来塞到一个集合里面
    """
    # 读取训练集，按 brand_name分组
    train_df = pd.read_csv(train_csv)
    brand2gids = {}
    for _, row in train_df.iterrows():
        brand = row["brand_name"]
        grid_list = parse_list(row["grid_id_list"]) #将grid_id_list转化为字符串
        brand2gids[brand] = grid_list #将这里面的brand_name跟grid_id_list对照起来
    # 测试集同理
    test_df = pd.read_csv(test_csv)
    test_brand2gids = {}
    for _, row in test_df.iterrows():
        brand = row["brand_name"]
        grid_list = parse_list(row["grid_id_list"])
        test_brand2gids[brand] = grid_list
    # 构建类别空间
    all_grid_ids = set()#设置的是一个集合，可以去重，其实就是得到一共多少种grid，额，我认为直接用grid这个方法就可以
    for gl in brand2gids.values():
        all_grid_ids.update(gl)
    for gl in test_brand2gids.values():
        all_grid_ids.update(gl)
    all_grid_ids = sorted(all_grid_ids)
    grid_to_index = {gid: idx for idx, gid in enumerate(all_grid_ids)}#建立映射，后面会用到
    num_classes = len(all_grid_ids)

    """
      功能是应对输入的grid_list里面，对所有的序列都计算一遍每个元素的密集程度，
      认为是从中心发散的设置店面的模式
    """
    def sort_by_density(gids):
        #这个防止错误的片段就是很谨慎的，但是用不到的
        if len(gids) <= 1: return gids[:]
        k = 3
        #将每个区间的左边进行表示出来
        locations = [coords_map[g] for g in gids]
        scores = []
        for i, g in enumerate(gids):
            xi, yi = locations[i]
            dists = [np.linalg.norm([xi-xj, yi-yj]) for j, (xj, yj) in enumerate(locations) if i != j]
            #如果没有邻居就设置为无穷大
            if not dists: avg = float('inf')
            else:
                dists.sort()
                avg = np.mean(dists[:min(k, len(dists))])
            scores.append((avg, g))
        scores.sort()
        #经过排序之后，我们认为这就是我们的时间序列了
        return [g for _, g in scores]

    """
      不同于上面的sort_by_density纯粹的看聚集程度，
      cluster_then_density是认为店面的生成是一簇一簇的，
      即先有主体，然后围绕主体生成剩下的店面，成规模后再次在其他地区展开分店，
      然后这个分店继续扩张，重复。
    """
    def cluster_then_density(gids):
        #如果序列很短，就直接扔给sort_by_density，此时这两个方法等价
        if len(gids) <= 2:
            return sort_by_density(gids)
        #仍旧是先把每个grid都转化为他对应的坐标
        coords = np.array([coords_map[g] for g in gids])
        best_k, best_score, best_labels = None, -1, None
        #这里是设置最大的k是10或序列长减1
        max_K = min(10, len(gids)-1)
        #这里是用来计算最优的k值，最好的结果（用silhouette_score评估），
        #最好的结果是什么（用km.fit_predict(coords)评估）
        for k in range(2, max_K+1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(coords)
            if len(set(labels)) < 2: continue
            score = silhouette_score(coords, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        if best_labels is None:
            return sort_by_density(gids)
        clusters = {}
        #将网格分配到各自所属的簇中，形成一个字典 clusters
        for g, l in zip(gids, best_labels):
            clusters.setdefault(l, []).append(g)
        # 按聚类规模（数量大小）排序
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        seq = []
        #然后对每个簇，都使用sort_by_density再排序一次作为新的序列
        for cluster_gids in sorted_clusters:
            seq.extend(sort_by_density(cluster_gids))
        return seq

    # 生成训练样本（前缀->下一个）
    train_samples = []
    for brand, gids in brand2gids.items():
        #将那些序列太短的部分直接给舍弃掉，这也是为什么说认为前面的太过保守
        if len(gids) < 2: continue
        #根据method来确定使用哪种方法，但是目前只是用了density
        seq = sort_by_density(gids) if method=='density' else cluster_then_density(gids)
        #依据seq构建训练样本：将序列拆分为前缀 + 下一个目标
        # for t in range(1, len(seq)):
        #     #也就是这部分
        #     prefix = seq[:t]
        #     target = seq[t]
        #     #数据再处理，获得每个位置的坐标索引
        #     prefix_idx = [grid_to_index[g] for g in prefix]
        #     target_idx = grid_to_index[target]
        #     #这里获得的是实际坐标
        #     prefix_coords = [coords_map[g] for g in prefix]
        #     #添加到训练集里面
        #     train_samples.append((prefix_idx, prefix_coords, target_idx))
        prefix = seq[:-1]
        target = seq[-1]
        # 数据再处理，获得每个位置的坐标索引
        prefix_idx = [grid_to_index[g] for g in prefix]
        target_idx = grid_to_index[target]
        # 这里获得的是实际坐标
        prefix_coords = [coords_map[g] for g in prefix]
        # 添加到训练集里面
        train_samples.append((prefix_idx, prefix_coords, target_idx))
    #设置随机种子以保证实验可重复性
    np.random.seed(42)
    # 对训练样本进行随机打乱顺序
    np.random.shuffle(train_samples)
    # 按照 9:1 的比例划分训练集和验证集
    split_idx = int(0.9 * len(train_samples))
    #90%训练集，10%验证集
    train_data = train_samples[:split_idx]
    val_data = train_samples[split_idx:]

    # 测试集构建
    test_data = []
    for brand, gids in test_brand2gids.items():
        # 舍弃门店数量太少的品牌（更上面一样，同时，还可以提高正确率
        if len(gids) < 2: continue
        # 使用指定排序方法对门店网格ID进行排序（密度排序或聚类后排序）
        seq = sort_by_density(gids) if method=='density' else cluster_then_density(gids)
        # 将最后一个门店作为目标预测点
        prefix = seq[:-1]
        target = seq[-1]
        # 将网格ID转换为模型使用的类别索引
        prefix_idx = [grid_to_index[g] for g in prefix]
        target_idx = grid_to_index[target]
        #获得实际坐标
        prefix_coords = [coords_map[g] for g in prefix]
        #加入数据
        test_data.append((prefix_idx, prefix_coords, target_idx))

    return train_data, val_data, test_data, num_classes, coords_map, grid_to_index



