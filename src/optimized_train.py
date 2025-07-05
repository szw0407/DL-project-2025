"""
优化版训练脚本 - 提高GPU利用率
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from optimized_dataloader import create_optimized_dataloader
from optimized_evaluate import optimized_evaluate_model as evaluate_model

def optimized_train_model(model, train_set, val_set, device, num_epochs=40, batch_size=32, 
                         lr=1e-3, patience=5, use_amp=True, num_workers=10):
    """
    优化版训练函数
    
    主要优化:
    1. 使用混合精度训练 (AMP)
    2. 优化数据加载器
    3. 减少CPU-GPU数据传输
    4. 更好的内存管理
    
    参数:
        model: 待训练的模型
        train_set: 训练数据集
        val_set: 验证数据集
        device: 计算设备(CPU/GPU)
        num_epochs: 最大训练轮数
        batch_size: 批次大小
        lr: 学习率
        patience: 早停耐心值
        use_amp: 是否使用混合精度训练
        num_workers: 数据加载的进程数
        
    返回:
        训练完成的模型
    """
    print(f"优化训练配置:")
    print(f"- 设备: {device}")
    print(f"- 混合精度训练: {use_amp}")
    print(f"- 数据加载进程数: {num_workers}")
    print(f"- 批次大小: {batch_size}")
    
    model = model.to(device)
    
    # 创建优化版数据加载器
    train_loader = create_optimized_dataloader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # 使用AdamW优化器
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练设置
    scaler = GradScaler('cuda') if use_amp else None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    
    best_mrr = -1
    best_state = None
    no_improve = 0
    
    for epoch in range(1, num_epochs+1):
        # 训练阶段
        model.train()
        losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移至设备
            seq_ids = batch['seq_ids'].to(device, non_blocking=True)
            seq_coords = batch['seq_coords'].to(device, non_blocking=True)
            seq_poi = batch['seq_poi'].to(device, non_blocking=True)
            brand_input_ids = batch['brand_input_ids'].to(device, non_blocking=True)
            brand_attention_mask = batch['brand_attention_mask'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            
            opt.zero_grad()
            
            # 混合精度前向传播
            if use_amp:
                with autocast('cuda'):
                    logits = model(seq_ids, seq_coords, seq_poi, brand_input_ids, brand_attention_mask)
                    loss = criterion(logits, targets)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(seq_ids, seq_coords, seq_poi, brand_input_ids, brand_attention_mask)
                loss = criterion(logits, targets)
                loss.backward()
                opt.step()
            
            losses.append(loss.item())
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 验证阶段
        val_acc_k, val_mrr = evaluate_model(model, val_set, device)
        
        # 学习率调度
        scheduler.step(val_mrr)
        current_lr = opt.param_groups[0]['lr']
        
        # 打印训练状态
        print(f"Epoch {epoch} | loss={np.mean(losses):.4f} | lr={current_lr:.6f} | Val_MRR={val_mrr:.4f} | Acc@1={val_acc_k[1]:.3f} Acc@5={val_acc_k[5]:.3f} Acc@10={val_acc_k[10]:.3f}")
        
        # 模型选择与早停
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_state = model.state_dict()
            no_improve = 0
            print(f"*** 新的最佳模型 (MRR: {best_mrr:.4f}) ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop triggered after {patience} epochs without improvement.")
                break
    
    # 加载最佳模型状态
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已加载最佳模型 (MRR: {best_mrr:.4f})")
    
    return model
