"""
GPU常驻训练脚本 - 所有数据都在GPU上，最大化GPU利用率
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from gpu_resident_dataloader import create_gpu_resident_dataloader, check_gpu_memory_usage
from optimized_evaluate import optimized_evaluate_model as evaluate_model

def gpu_resident_train_model(model, train_set, val_set, device, num_epochs=40, batch_size=32, 
                           lr=1e-3, patience=5, use_amp=True, bert_model_name='bert-base-chinese'):
    """
    GPU常驻训练函数 - 所有数据预先加载到GPU
    
    主要优化:
    1. 所有数据预先加载到GPU，避免CPU-GPU传输
    2. 使用混合精度训练 (AMP)
    3. 消除tokenization开销
    4. 最大化GPU利用率
    
    参数:
        model: 待训练的模型
        train_set: 训练数据集
        val_set: 验证数据集
        device: 计算设备(必须是GPU)
        num_epochs: 最大训练轮数
        batch_size: 批次大小
        lr: 学习率
        patience: 早停耐心值
        use_amp: 是否使用混合精度训练
        bert_model_name: BERT模型名称
        
    返回:
        训练完成的模型
    """
    if not torch.cuda.is_available():
        raise ValueError("GPU常驻训练需要CUDA设备")
    
    print(f"GPU常驻训练配置:")
    print(f"- 设备: {device}")
    print(f"- 混合精度训练: {use_amp}")
    print(f"- 批次大小: {batch_size}")
    print(f"- BERT模型: {bert_model_name}")
    
    # 检查初始GPU内存
    print("\n初始GPU内存状态:")
    check_gpu_memory_usage()
    
    model = model.to(device)
    
    # 创建GPU常驻数据加载器
    print("\n创建GPU常驻数据加载器...")
    train_loader = create_gpu_resident_dataloader(
        train_set, device=device, batch_size=batch_size, shuffle=True, bert_model_name=bert_model_name
    )
    
    val_loader = create_gpu_resident_dataloader(
        val_set, device=device, batch_size=batch_size, shuffle=False, bert_model_name=bert_model_name
    )
    
    # 检查数据加载后的GPU内存
    print("\n数据加载后GPU内存状态:")
    check_gpu_memory_usage()
    
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=2
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n开始训练 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(
                        seq_ids=batch['seq_ids'],
                        seq_coords=batch['seq_coords'],
                        seq_poi=batch['seq_poi'],
                        brand_input_ids=batch['brand_input_ids'],
                        brand_attention_mask=batch['brand_attention_mask']
                    )
                    loss = criterion(outputs, batch['targets'])
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    seq_ids=batch['seq_ids'],
                    seq_coords=batch['seq_coords'],
                    seq_poi=batch['seq_poi'],
                    brand_input_ids=batch['brand_input_ids'],
                    brand_attention_mask=batch['brand_attention_mask']
                )
                loss = criterion(outputs, batch['targets'])
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 验证阶段
        val_acc = _validate_gpu_resident(model, val_loader, use_amp)
        
        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_loss:.4f}")
        print(f"  验证准确率: {val_acc:.4f}")
        print(f"  学习率: {current_lr:.2e}")
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_gpu_resident_model.pth')
            print(f"  ✓ 新的最佳验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  验证准确率未提升 ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        print()
    
    # 加载最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load('best_gpu_resident_model.pth'))
        print(f"已加载最佳模型 (验证准确率: {best_val_acc:.4f})")
    
    return model


def _validate_gpu_resident(model, val_loader, use_amp=True):
    """GPU常驻验证函数"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(
                        seq_ids=batch['seq_ids'],
                        seq_coords=batch['seq_coords'],
                        seq_poi=batch['seq_poi'],
                        brand_input_ids=batch['brand_input_ids'],
                        brand_attention_mask=batch['brand_attention_mask']
                    )
            else:
                outputs = model(
                    seq_ids=batch['seq_ids'],
                    seq_coords=batch['seq_coords'],
                    seq_poi=batch['seq_poi'],
                    brand_input_ids=batch['brand_input_ids'],
                    brand_attention_mask=batch['brand_attention_mask']
                )
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch['targets'].size(0)
            correct += (predicted == batch['targets']).sum().item()
    
    return correct / total


def gpu_resident_evaluate_model(model, test_set, device, bert_model_name='bert-base-chinese', batch_size=32):
    """
    GPU常驻模型评估函数
    
    参数:
        model: 训练好的模型
        test_set: 测试数据集
        device: 计算设备
        bert_model_name: BERT模型名称
        batch_size: 批次大小
        
    返回:
        acc_k: 各k值的准确率字典
        mrr: MRR分数
    """
    print("创建测试数据的GPU常驻加载器...")
    test_loader = create_gpu_resident_dataloader(
        test_set, device=device, batch_size=batch_size, shuffle=False, bert_model_name=bert_model_name
    )
    
    model.eval()
    all_scores = []
    all_targets = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in test_loader:
            with autocast(device_type='cuda'):
                outputs = model(
                    seq_ids=batch['seq_ids'],
                    seq_coords=batch['seq_coords'],
                    seq_poi=batch['seq_poi'],
                    brand_input_ids=batch['brand_input_ids'],
                    brand_attention_mask=batch['brand_attention_mask']
                )
            
            all_scores.append(outputs.cpu())
            all_targets.append(batch['targets'].cpu())
    
    # 合并所有结果
    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算排名指标
    acc_k = {}
    mrr_scores = []
    
    for i in range(len(all_targets)):
        scores = all_scores[i]
        target = all_targets[i].item()
        
        # 获取排序后的索引
        sorted_indices = torch.argsort(scores, descending=True)
        target_rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
        
        # 计算MRR
        mrr_scores.append(1.0 / target_rank)
        
        # 计算Acc@k
        for k in [1, 5, 10]:
            if k not in acc_k:
                acc_k[k] = 0
            if target_rank <= k:
                acc_k[k] += 1
    
    # 计算平均值
    for k in acc_k:
        acc_k[k] /= len(all_targets)
    
    mrr = np.mean(mrr_scores)
    
    return acc_k, mrr
