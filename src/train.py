import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from data_preprocessing import load_data
from model import StorePredictionModel
from evaluate import compute_metrics

DEFAULT_EMBED_DIM = 72
DEFAULT_COORD_DIM = 18
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LSTM_LAYERS = 24
DEFAULT_DROPOUT = 0.15
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 8

def train_model(
    train_samples, 
    val_samples, 
    num_total_classes, 
    embed_dim=DEFAULT_EMBED_DIM, 
    coord = None,
    coord_dim_config=DEFAULT_COORD_DIM, 
    hidden_dim=DEFAULT_HIDDEN_DIM, 
    lstm_layers=DEFAULT_LSTM_LAYERS,
    dropout=DEFAULT_DROPOUT,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    device_name='cuda',
    model_save_path=None,
    use_bert=True
):
    """
    训练门店选址预测模型。

    参数:
        train_samples: 训练样本列表，每个样本为 (prefix_idx, prefix_coords, target_idx)
        val_samples: 验证样本列表
        num_total_classes: 网格类别总数
        coords_info: 坐标信息字典，键为网格ID，值为对应的坐标元组 (x, y)
        embed_dim: 嵌入维度
        coord_dim_config: 坐标嵌入维度，如果为0则不使用坐标
        hidden_dim: LSTM隐藏层维度
        lstm_layers: LSTM层数
        dropout: Dropout比例
        lr: 学习率
        weight_decay: 权重衰减率
        epochs: 训练轮数
        patience: 早停耐心值
        device_name: 设备名称，'cuda'或'cpu'
        model_save_path: 模型保存路径
        use_bert: 是否使用BERT特征提取，默认为False

    返回:
        训练好的模型
    """
    print("\n=== 配置训练参数 ===")
    print(f"嵌入维度: {embed_dim}")
    print(f"坐标嵌入维度: {coord_dim_config}")
    print(f"LSTM隐藏层维度: {hidden_dim}")
    print(f"LSTM层数: {lstm_layers}")
    print(f"Dropout比例: {dropout}")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"训练轮数: {epochs}")
    print(f"早停耐心值: {patience}")
    print(f"使用BERT: {use_bert}")
    
    # 确定设备
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = StorePredictionModel(
        num_classes=num_total_classes, 
        embed_dim=embed_dim, 
        coord_dim=coord_dim_config, 
        lstm_hidden=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        use_bert=use_bert
    )
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 定义早停相关变量
    best_val_mrr = -1
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    print("\n=== 开始训练 ===")
    for epoch in range(epochs):
        # 训练模式
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # 随机打乱训练样本顺序
        np.random.shuffle(train_samples)
        
        # 批处理训练
        batch_size = 32
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i:i + batch_size]
            
            # 处理批次数据
            batch_seq_ids = []
            batch_seq_coords = []
            batch_targets = []
              # 解析批次样本，现在每个样本包含品牌名称和类型
            batch_brand_names = []
            batch_brand_types = []
            
            # 获取当前批次中最长序列的长度
            max_seq_len = max(len(seq_ids) for seq_ids, _, _, _, _ in batch)
            
            for seq_ids, seq_coords, target_idx, brand_name, brand_type in batch:
                # 填充序列到最大长度
                padded_ids = seq_ids + [0] * (max_seq_len - len(seq_ids))
                batch_seq_ids.append(padded_ids)
                
                # 如果使用坐标，也进行相应的填充
                if coord_dim_config > 0:
                    # 确保每个坐标是一个二维元组 (x, y)
                    padded_coords = seq_coords + [(0.0, 0.0)] * (max_seq_len - len(seq_coords))
                    batch_seq_coords.append(padded_coords)
                
                # 保存目标和品牌信息
                batch_targets.append(target_idx)
                batch_brand_names.append(brand_name)
                batch_brand_types.append(brand_type)
            
            # 转换为张量
            seq_ids_tensor = torch.tensor(batch_seq_ids, dtype=torch.long).to(device)
            targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(device)
            # 处理坐标数据
            seq_coords_tensor = None
            if coord_dim_config > 0:
                # 确保坐标数据形状正确 (batch, seq_len, 2)
                seq_coords_tensor = torch.tensor(batch_seq_coords, dtype=torch.float).to(device)
                # 确保最后一个维度是2（x和y坐标）
                if seq_coords_tensor.shape[-1] != 2:
                    seq_coords_tensor = seq_coords_tensor.view(seq_coords_tensor.shape[0], seq_coords_tensor.shape[1], 2)            # 前向传播
            optimizer.zero_grad()
            
            # 使用真实的品牌名称和类型数据
            brand_names = None
            brand_types = None
            if use_bert:
                # 使用实际的品牌信息
                brand_names = batch_brand_names
                brand_types = batch_brand_types
            
            # 传递品牌信息到模型
            outputs = model(seq_ids_tensor, seq_coords_tensor, brand_names, brand_types)
            
            # 计算损失
            loss = criterion(outputs, targets_tensor)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # 计算平均训练损失
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0        # 在验证集上评估 - 使用我们自己的实现而不是依赖compute_metrics函数
        model.eval()
        correct_at_k = {1: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        
        with torch.no_grad():
            for val_sample in val_samples:
                prefix_idx, prefix_coords, true_idx_val, brand_name, brand_type = val_sample
                
                # 处理序列数据
                seq_ids = torch.tensor(prefix_idx, dtype=torch.long).unsqueeze(0).to(device)
                
                # 处理坐标数据
                seq_coords_tensor = None
                if coord_dim_config > 0 and prefix_coords:
                    seq_coords_tensor = torch.tensor(prefix_coords, dtype=torch.float).unsqueeze(0).to(device)
                
                # 处理品牌数据
                brand_names = None
                brand_types = None
                if use_bert:
                    brand_names = [brand_name]
                    brand_types = [brand_type]
                
                # 前向传播
                outputs = model(seq_ids, seq_coords_tensor, brand_names, brand_types)
                
                # 计算预测
                probs = outputs.softmax(dim=1)
                _, pred_indices = probs.sort(descending=True)
                pred_indices = pred_indices.squeeze(0).tolist()
                
                # 计算Top-K准确率
                for k in [1, 5, 10]:
                    if true_idx_val in pred_indices[:k]:
                        correct_at_k[k] += 1
                
                # 计算MRR
                try:
                    rank = pred_indices.index(true_idx_val) + 1
                    reciprocal_ranks.append(1.0 / rank)
                except ValueError:
                    reciprocal_ranks.append(0.0)
        
        # 计算最终指标
        val_acc_k = {k: correct_at_k[k] / len(val_samples) for k in correct_at_k}
        val_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 验证 MRR: {val_mrr:.4f}, 验证 Top-1: {val_acc_k[1]:.4f} Top-5: {val_acc_k[5]:.4f}, Top-10: {val_acc_k[10]:.4f}")
        
        # 早停检查
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # 如果指定了保存路径，保存模型
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mrr': val_mrr,
                    'embed_dim': embed_dim,
                    'coord_dim': coord_dim_config,
                    'hidden_dim': hidden_dim,
                    'lstm_layers': lstm_layers,
                    'dropout': dropout,
                    'num_classes': num_total_classes
                }, model_save_path)
                print(f"模型已保存至: {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发: {patience}轮未改善")
                break
    
    # 加载最佳模型权重
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (验证 MRR: {best_val_mrr:.4f})")
    
    return model

if __name__ == "__main__":
    USE_CUDA_IF_AVAILABLE = True
    MODEL_SAVE_FILENAME = "store_predictor_best.pth"

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    data_dir = os.path.join(project_root_dir, 'data')
    model_output_path = os.path.join(project_root_dir, MODEL_SAVE_FILENAME)
    train_csv_file = os.path.join(data_dir, "train_data.csv")
    test_csv_file = os.path.join(data_dir, "test_data.csv")
    grid_csv_file = os.path.join(data_dir, "grid_coordinates.csv")

    print("--- 门店选址预测模型训练脚本 ---")
    print(f"数据文件路径:")
    print(f"  训练集: {train_csv_file}")
    print(f"  测试集: {test_csv_file}")
    print(f"  网格坐标: {grid_csv_file}")
    print(f"模型将保存至: {model_output_path}")

    print("\n正在加载和预处理数据...")
    data_load_result = load_data(train_csv_file, test_csv_file, grid_csv_file)

    train_samples, val_samples, test_samples, num_total_classes, coords_info, grid_to_index_map, brand_type_map = data_load_result
    print(f"数据加载完成。类别总数: {num_total_classes}")
    print(f"训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}, 测试样本数: {len(test_samples)}")
    print(f"品牌总数: {len(brand_type_map)}")

    trained_model = train_model(
        train_samples,
        val_samples,
        num_total_classes,
        
        embed_dim=DEFAULT_EMBED_DIM,
        coord=coords_info,
        coord_dim_config=DEFAULT_COORD_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lstm_layers=DEFAULT_LSTM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        device_name='cuda' if USE_CUDA_IF_AVAILABLE else 'cpu',
        model_save_path=model_output_path
    )
    
    if trained_model and test_samples:
        print("\n在测试集上评估最终模型...")
        device_for_eval = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
        trained_model.to(device_for_eval)
        
        # 自己实现测试评估，确保正确处理品牌信息
        trained_model.eval()
        correct_at_k = {1: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        
        with torch.no_grad():
            for test_sample in test_samples:
                prefix_idx, prefix_coords, true_idx_val, brand_name, brand_type = test_sample
                
                # 处理序列数据
                seq_ids = torch.tensor(prefix_idx, dtype=torch.long).unsqueeze(0).to(device_for_eval)
                
                # 处理坐标数据
                seq_coords_tensor = None
                if DEFAULT_COORD_DIM > 0 and prefix_coords:
                    seq_coords_tensor = torch.tensor(prefix_coords, dtype=torch.float).unsqueeze(0).to(device_for_eval)
                
                # 处理品牌数据
                brand_names = None
                brand_types = None
                if trained_model.use_bert:
                    brand_names = [brand_name]
                    brand_types = [brand_type]
                
                # 前向传播
                outputs = trained_model(seq_ids, seq_coords_tensor, brand_names, brand_types)
                
                # 计算预测
                probs = outputs.softmax(dim=1)
                _, pred_indices = probs.sort(descending=True)
                pred_indices = pred_indices.squeeze(0).tolist()
                
                # 计算Top-K准确率
                for k in [1, 5, 10]:
                    if true_idx_val in pred_indices[:k]:
                        correct_at_k[k] += 1
                
                # 计算MRR
                try:
                    rank = pred_indices.index(true_idx_val) + 1
                    reciprocal_ranks.append(1.0 / rank)
                except ValueError:
                    reciprocal_ranks.append(0.0)
        
        # 计算最终指标
        test_acc_k = {k: correct_at_k[k] / len(test_samples) for k in correct_at_k}
        test_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        print(f"\n--- 测试集评估结果 ---")
        print(f"  测试集 MRR: {test_mrr:.4f}")
        for k_val in [1, 5, 10]:
            print(f"  测试集 Acc@{k_val}: {test_acc_k[k_val]:.4f}")
    elif not test_samples:
        print("\n无测试数据可供评估。")
    else:
        print("\n模型训练失败，无法在测试集上评估。")

    print("\n--- 脚本执行完毕 ---")


