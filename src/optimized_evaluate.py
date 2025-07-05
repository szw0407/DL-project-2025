"""
优化版评估模块
"""
import torch
import numpy as np
from optimized_dataloader import create_optimized_dataloader

@torch.no_grad()
def optimized_evaluate_model(model, dataset, device, k_list=[1, 5, 10]):
    """
    优化版评估函数，使用优化的数据加载器
    """
    model.eval()
    acc_k = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0
    
    # 使用优化版数据加载器
    dataloader = create_optimized_dataloader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    for batch in dataloader:
        # 将数据移至设备
        seq_ids = batch['seq_ids'].to(device, non_blocking=True)
        seq_coords = batch['seq_coords'].to(device, non_blocking=True)
        seq_poi = batch['seq_poi'].to(device, non_blocking=True)
        brand_input_ids = batch['brand_input_ids'].to(device, non_blocking=True)
        brand_attention_mask = batch['brand_attention_mask'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        # 获取模型输出
        logits = model(seq_ids, seq_coords, seq_poi, brand_input_ids, brand_attention_mask)
        
        # 获取预测分数最高的K个位置的索引
        topk = torch.topk(logits, max(k_list), dim=1).indices.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        for i, target in enumerate(targets_np):
            # 找出目标位置在预测的topk结果中的排名
            rank = np.where(topk[i] == target)[0]
            if len(rank) > 0:
                rank = rank[0] + 1
                mrr_sum += 1.0 / rank
                for k in k_list:
                    if rank <= k:
                        acc_k[k] += 1
            total += 1
    
    # 计算平均指标
    mrr = mrr_sum / total
    for k in k_list:
        acc_k[k] = acc_k[k] / total
    
    return acc_k, mrr
