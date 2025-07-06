"""
模型评估模块
该模块提供了评估模型性能的函数和用于批量处理数据的工具函数。
主要实现了模型在验证集/测试集上的评估指标计算，包括Top-K准确率和MRR。
"""

import torch
import numpy as np

@torch.no_grad()
def evaluate_model(model, dataset, device, k_list=[1, 5, 10]):
    """
    评估模型在给定数据集上的性能
    
    该函数计算两个关键指标：
    1. Top-K准确率：预测结果的前K个选项中包含正确答案的比例
    2. 平均倒数排名(MRR)：正确答案排名的倒数的平均值
    
    使用@torch.no_grad()装饰器可以在评估时禁用梯度计算，节省内存并加速评估过程
    
    参数:
        model: 要评估的模型
        dataset: 评估数据集
        device: 计算设备(CPU/GPU)
        k_list: Top-K准确率中K值的列表，默认为[1, 5, 10]
        
    返回:
        acc_k: 包含各个K值对应准确率的字典
        mrr: 平均倒数排名(Mean Reciprocal Rank)
    """
    model.eval()  # 设置模型为评估模式
    acc_k = {k: 0 for k in k_list}  # 初始化各K值的准确计数
    mrr_sum = 0  # MRR累加值
    total = 0  # 样本总数
    for seq_ids, seq_coords, seq_poi, targets in batcher(dataset, 64):
        # 将数据移至指定设备
        seq_ids, seq_coords, seq_poi, targets = (
            seq_ids.to(device), seq_coords.to(device), seq_poi.to(device), targets.to(device)
        )
        # 获取模型输出（预测分数）
        logits = model(seq_ids, seq_coords, seq_poi)
        # 获取预测分数最高的K个位置的索引
        topk = torch.topk(logits, max(k_list), dim=1).indices.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        for i, target in enumerate(targets_np):
            # 找出目标位置在预测的topk结果中的排名
            rank = np.where(topk[i] == target)[0]
            if len(rank) > 0:  # 如果目标在topk中
                rank = rank[0] + 1  # 将索引转换为排名（从1开始）
                mrr_sum += 1.0 / rank  # 计算倒数排名并累加
                for k in k_list:
                    if rank <= k:  # 如果排名在k以内，对应的Top-K准确率计数加1
                        acc_k[k] += 1
            total += 1
    
    # 计算平均值
    mrr = mrr_sum / total if total else 0
    acc_k = {k: acc_k[k]/total for k in k_list}
    return acc_k, mrr

def batcher(samples, batch_size=64):
    """
    将样本数据转换为批次形式的生成器函数
    
    该函数实现了以下功能：
    1. 将样本数据按批次划分
    2. 对每个批次内的序列进行填充，使其长度一致
    3. 构建张量格式的数据，用于模型输入
    
    填充策略是从序列右侧对齐，这种方式在处理序列数据时更为合理，
    因为序列的最新部分（右侧）通常包含更重要的信息。
    
    参数:
        samples: 样本列表，每个样本是(前缀索引, 前缀坐标, 前缀POI特征, 目标索引)的元组
        batch_size: 批次大小，默认为64
        
    生成:
        每次生成一个批次的数据，包含:
        - seq_ids: 序列ID张量，形状为(batch_size, max_seq_len)
        - seq_coords: 序列坐标张量，形状为(batch_size, max_seq_len, 2)
        - seq_poi: 序列POI特征张量，形状为(batch_size, max_seq_len, 10)
        - targets: 目标索引张量，形状为(batch_size,)
    """
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        # 找出当前批次中最长的序列长度
        maxlen = max(len(x[0]) for x in batch)
        # 初始化批次数据数组
        seq_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
        seq_coords = np.zeros((len(batch), maxlen, 2), dtype=np.float32)
        seq_poi = np.zeros((len(batch), maxlen, 10), dtype=np.float32)
        
        for j, (pidx, pcoord, ppoi, _) in enumerate(batch):
            L = len(pidx)
            # 右对齐填充：将数据放在数组的右侧，左侧用0填充
            seq_ids[j, -L:] = pidx
            seq_coords[j, -L:, :] = pcoord
            seq_poi[j, -L:, :] = ppoi
        
        # 提取目标值
        targets = np.array([x[3] for x in batch], dtype=np.int64)
        
        # 将NumPy数组转换为PyTorch张量并返回
        yield (
            torch.from_numpy(seq_ids),
            torch.from_numpy(seq_coords),
            torch.from_numpy(seq_poi),
            torch.from_numpy(targets)
        )
