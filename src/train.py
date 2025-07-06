"""
模型训练模块
该模块提供了模型训练的相关函数，包括训练数据批处理和模型训练流程。
实现了带有早停机制的模型训练，以及基于验证集性能的模型选择。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evaluate import evaluate_model

def batch_iter(samples, batch_size=32, shuffle=True):
    """
    训练数据批次迭代器
    
    与evaluate模块中的batcher函数类似，但增加了数据打乱功能，
    适用于训练过程中需要随机打乱数据的场景。
    
    参数:
        samples: 样本列表，每个样本是(前缀索引, 前缀坐标, 前缀POI特征, 目标索引)的元组
        batch_size: 批次大小，默认为32
        shuffle: 是否打乱数据顺序，默认为True
        
    生成:
        每次生成一个批次的数据，包含:
        - seq_ids: 序列ID张量，形状为(batch_size, max_seq_len)
        - seq_coords: 序列坐标张量，形状为(batch_size, max_seq_len, 2)
        - seq_poi: 序列POI特征张量，形状为(batch_size, max_seq_len, 10)
        - targets: 目标索引张量，形状为(batch_size,)
    """
    idxs = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(idxs)  # 随机打乱索引，增加训练随机性
    for i in range(0, len(samples), batch_size):
        batch = [samples[j] for j in idxs[i:i+batch_size]]
        maxlen = max(len(x[0]) for x in batch)
        # pad to maxlen
        seq_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
        seq_coords = np.zeros((len(batch), maxlen, 2), dtype=np.float32)
        seq_poi = np.zeros((len(batch), maxlen, 10), dtype=np.float32)
        for i, (pidx, pcoord, ppoi, _) in enumerate(batch):
            L = len(pidx)
            # 右对齐填充：将数据放在数组的右侧，左侧用0填充
            seq_ids[i, -L:] = pidx
            seq_coords[i, -L:, :] = pcoord
            seq_poi[i, -L:, :] = ppoi
        targets = np.array([x[3] for x in batch], dtype=np.int64)
        yield (
            torch.from_numpy(seq_ids),
            torch.from_numpy(seq_coords),
            torch.from_numpy(seq_poi),
            torch.from_numpy(targets)
        )

def train_model(model, train_set, val_set, device, num_epochs=40, batch_size=32, lr=1e-3, patience=5):
    """
    训练模型的主函数
    
    实现了完整的模型训练流程，包括：
    1. 模型训练与参数优化
    2. 验证集评估
    3. 早停机制
    4. 最佳模型保存
    
    使用MRR(平均倒数排名)作为模型选择的指标，
    当验证集上的MRR不再提升时，触发早停机制。
    
    参数:
        model: 待训练的模型
        train_set: 训练数据集
        val_set: 验证数据集
        device: 计算设备(CPU/GPU)
        num_epochs: 最大训练轮数，默认为40
        batch_size: 批次大小，默认为32
        lr: 学习率，默认为1e-3
        patience: 早停耐心值，当验证指标连续多少轮未改善时停止训练，默认为5
        
    返回:
        训练完成的模型（已加载最佳状态）
    """
    model = model.to(device)  # 将模型移至指定设备
    opt = optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    best_mrr = -1  # 最佳MRR值
    best_state = None  # 最佳模型状态
    no_improve = 0  # 未改善轮数计数器
    
    for epoch in range(1, num_epochs+1):
        # 训练阶段
        model.train()  # 设置模型为训练模式
        losses = []  # 记录每个批次的损失
        
        for seq_ids, seq_coords, seq_poi, targets in batch_iter(train_set, batch_size):
            # 将数据移至指定设备
            seq_ids, seq_coords, seq_poi, targets = (
                seq_ids.to(device), seq_coords.to(device), seq_poi.to(device), targets.to(device)
            )
            opt.zero_grad()  # 梯度清零
            logits = model(seq_ids, seq_coords, seq_poi)  # 前向传播
            loss = criterion(logits, targets)  # 计算损失
            loss.backward()  # 反向传播
            opt.step()  # 参数更新
            losses.append(loss.item())  # 记录损失值
        
        # 验证阶段
        val_acc_k, val_mrr = evaluate_model(model, val_set, device)
        
        # 打印当前训练状态
        print(f"Epoch {epoch} | loss={np.mean(losses):.4f} | Val_MRR={val_mrr:.4f} | Acc@1={val_acc_k[1]:.3f} Acc@5={val_acc_k[5]:.3f} Acc@10={val_acc_k[10]:.3f}")
        
        # 模型选择与早停机制
        if val_mrr > best_mrr:  # 如果验证集MRR有提升
            best_mrr = val_mrr  # 更新最佳MRR
            best_state = model.state_dict()  # 保存当前模型状态
            no_improve = 0  # 重置未改善计数器
        else:
            no_improve += 1  # 未改善计数器加1
            if no_improve >= patience:  # 如果连续patience轮未改善
                print("Early stop triggered.")  # 触发早停
                break
    
    # 加载最佳模型状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model
