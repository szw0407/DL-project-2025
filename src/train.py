import torch_geometric

import torch
import torch.nn as nn
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import load_data
from model import StorePredictionModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import random

plt.rcParams['font.family'] = 'Noto Sans SC'
DEFAULT_EMBED_DIM = 16
DEFAULT_COORD_DIM = 4
DEFAULT_HIDDEN_DIM = 64  # 提升隐藏层维度
DEFAULT_LSTM_LAYERS = 8
DEFAULT_DROPOUT = 0.2  # 适当降低Dropout比例以适应更大的模型
DEFAULT_LR = 8e-4  # 适当降低学习率，适应更大的模型
DEFAULT_WEIGHT_DECAY = 1e-4  
DEFAULT_EPOCHS = 300  
DEFAULT_TRANS_DIM = 32
DEFAULT_N_LAYERS = 1    # 增加层数以提升特征学习能力
DEFAULT_N_HEADS = 2     # 增加注意力头数
DEFAULT_MLP_RATIO = 4.0  # 增加MLP扩展比
DEFAULT_DROP_PATH = 0.1 # 适当降低DropPath
DEFAULT_BATCH_SIZE = 32  # 适当降低batch size以适应更大的模型
DEFAULT_OPTIMIZER = 'adamw'
DEFAULT_PATIENCE = 15  # 增加patience，给更深模型更多训练时间
DEFAULT_CONTRASTIVE = True
DEFAULT_CONTRASTIVE_WEIGHT = 0.1
DEFAULT_CONTRASTIVE_TEMPERATURE = 0.5
DEFAULT_USE_SPATIAL_STATS = True

class StochasticDepthScheduler:
    """训练过程中动态调整DropPath概率，提升泛化能力"""
    def __init__(self, model, max_drop_path=0.3, min_drop_path=0.05, total_epochs=100):
        self.model = model
        self.max_drop_path = max_drop_path
        self.min_drop_path = min_drop_path
        self.total_epochs = total_epochs
    def step(self, epoch):
        cur_prob = self.min_drop_path + (self.max_drop_path - self.min_drop_path) * (epoch / self.total_epochs)
        for m in self.model.modules():
            if hasattr(m, 'drop_prob'):
                m.drop_prob = cur_prob

class EarlyStopping:
    """带恢复功能的早停机制，防止过拟合"""
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f"早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class AdaptiveLRScheduler:
    """自适应学习率调度器，基于验证集MRR自动调整学习率"""
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True):
        self.verbose = verbose
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, min_lr=min_lr
        )
    def step(self, val_score):
        prev_lr = self.scheduler.optimizer.param_groups[0]['lr']
        self.scheduler.step(val_score)
        new_lr = self.scheduler.optimizer.param_groups[0]['lr']
        if self.verbose and new_lr != prev_lr:
            print(f"[AdaptiveLR] 学习率调整: {prev_lr:.6f} -> {new_lr:.6f}")

def augment_sequence(seq_ids, seq_coords, mask_prob=0.15, do_flip=True, do_rotate=True):
    """对序列和坐标进行mask、翻转、旋转等增强"""
    
    # mask部分id
    seq_ids_aug = seq_ids.copy()
    for i in range(len(seq_ids_aug)):
        if random.random() < mask_prob:
            seq_ids_aug[i] = 0  # 假设0为mask token
    # 坐标增强
    seq_coords_aug = seq_coords.copy() if seq_coords is not None else None
    if seq_coords_aug is not None:
        # 随机翻转
        if do_flip and random.random() < 0.5:
            seq_coords_aug = seq_coords_aug[::-1]
        # 随机旋转90/180/270度
        if do_rotate and random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            import math
            angle_rad = math.radians(angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            seq_coords_aug = [
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a) for (x, y) in seq_coords_aug
            ]
    return seq_ids_aug, seq_coords_aug

def get_optimizer(model, lr, weight_decay, optimizer_type='adamw'):
    base_opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == 'adamw':
        print('[优化器] 使用AdamW')
    return base_opt

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def get_loss_fn(num_classes, smoothing=0.1, use_contrastive=False):
    if use_contrastive:
        def loss_fn(outputs, targets, contrastive_loss=None, contrastive_weight=0.1):
            ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)(outputs, targets)
            if contrastive_loss is not None:
                return ce_loss + contrastive_weight * contrastive_loss
            return ce_loss
        return loss_fn
    else:
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

def train_model(
    train_samples, 
    val_samples, 
    num_total_classes, 
    embed_dim=DEFAULT_EMBED_DIM, 
    coord = None,
    coord_dim_config=DEFAULT_COORD_DIM, 
    trans_dim=DEFAULT_TRANS_DIM,
    n_layers=DEFAULT_N_LAYERS,
    n_heads=DEFAULT_N_HEADS,
    dropout=DEFAULT_DROPOUT,
    mlp_ratio=1.5,
    drop_path=DEFAULT_DROP_PATH,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    device_name='cuda',
    model_save_path=None,
    use_bert=True,
    use_mixup=True,
    mixup_alpha=0.4,
    grad_clip=2.0,
    batch_size=DEFAULT_BATCH_SIZE,
    optimizer_type=DEFAULT_OPTIMIZER,
    brand_type_map=None,
    use_contrastive=DEFAULT_CONTRASTIVE,
    contrastive_weight=DEFAULT_CONTRASTIVE_WEIGHT,
    contrastive_temperature=DEFAULT_CONTRASTIVE_TEMPERATURE,
    contrastive_pair_func=None,
    use_spatial_stats=DEFAULT_USE_SPATIAL_STATS,
):
    """
    训练门店选址预测模型。

        train_samples: 训练样本列表，每个样本为 (prefix_idx, prefix_coords, target_idx)
        val_samples: 验证样本列表
        num_total_classes: 网格类别总数
        coords_info: 坐标信息字典，键为网格ID，值为对应的坐标元组 (x, y)
        embed_dim: 嵌入维度
        coord_dim_config: 坐标嵌入维度，如果为0则不使用坐标
        trans_dim: Transformer维度
        n_layers: Transformer层数
        n_heads: 多头数
        mlp_ratio: MLP扩展比
        drop_path: DropPath比例
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
        use_mixup: 是否使用mixup数据增强，默认为True
        mixup_alpha: mixup的α超参数
        grad_clip: 梯度裁剪的阈值，默认为2.0
        optimizer_type: 优化器类型，'adamw'或'lookahead'
        brand_type_map: 品牌类型映射字典，键为品牌名称，值为类型ID

    返回:
        训练好的模型
    """
    print("\n=== 配置训练参数 ===")
    print(f"嵌入维度: {embed_dim}")
    print(f"坐标嵌入维度: {coord_dim_config}")
    print(f"Transformer维度: {trans_dim}")
    print(f"Transformer层数: {n_layers}")
    print(f"多头数: {n_heads}")
    print(f"MLP扩展比: {mlp_ratio}")
    print(f"DropPath比例: {drop_path}")
    print(f"Dropout比例: {dropout}")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"训练轮数: {epochs}")
    print(f"早停耐心值: {patience}")
    print(f"使用BERT: {use_bert}")
    print(f"Mixup: {use_mixup}")
    print(f"梯度裁剪: {grad_clip}")
    print(f"对比学习: {use_contrastive}, 对比损失权重: {contrastive_weight}, 温度: {contrastive_temperature}")
    
    # 确定设备
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 统计brand_type类别数及编码映射
    if brand_type_map is not None:
        brand_type_list = list(set(brand_type_map.values()))
        brand_type_to_id = {t: i for i, t in enumerate(brand_type_list)}
        brand_type_num = len(brand_type_list)
    else:
        brand_type_to_id = {}
        brand_type_num = 64
    # 初始化模型
    model = StorePredictionModel(
        num_classes=num_total_classes, 
        embed_dim=embed_dim, 
        coord_dim=coord_dim_config, 
        gnn_dim=32,
        gnn_layers=2,
        trans_dim=trans_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        drop_path=drop_path,
        use_bert=use_bert,
        brand_type_num=brand_type_num,
        brand_type_embed_dim=8,
        use_spatial_stats=use_spatial_stats
    )
    model = model.to(device)
    # 特色功能1：动态DropPath调度器
    drop_path_scheduler = StochasticDepthScheduler(model, max_drop_path=drop_path, min_drop_path=0.05, total_epochs=epochs)
    
    # 定义损失函数和优化器
    criterion = get_loss_fn(num_total_classes, smoothing=0.1, use_contrastive=use_contrastive)
    optimizer = get_optimizer(model, lr, weight_decay, optimizer_type)
    total_steps = epochs * max(1, len(train_samples) // batch_size)
    warmup_steps = int(0.1 * total_steps)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    early_stopper = EarlyStopping(patience=patience)
      # 定义早停相关变量
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    best_score = -float('inf')
    
    # 自适应学习率调度器
    lr_scheduler = AdaptiveLRScheduler(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    
    # 用于记录训练过程的指标
    train_losses = []
    val_mrrs = []
    val_acc1s = []
    val_acc5s = []
    val_acc10s = []
    
    # 训练循环    
    print("\n=== 开始训练 ===")
    for epoch in range(epochs):
        # 动态调整DropPath
        drop_path_scheduler.step(epoch)
        # 训练模式
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # 随机打乱训练样本顺序
        np.random.shuffle(train_samples)
        
        # 批处理训练
        # batch_size = 32  # 移除硬编码
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            range(0, len(train_samples), batch_size), 
            desc=f"Epoch {epoch+1}/{epochs}", 
            ncols=100, 
            ascii=False,
            leave=True
        )
        
        for i, batch_start in enumerate(progress_bar):
            batch = train_samples[batch_start:batch_start + batch_size]
            batch_seq_ids = []
            batch_seq_coords = []
            batch_targets = []
            batch_brand_names = []
            batch_brand_types = []
            batch_brand_type_ids = []
            max_seq_len = max(len(seq_ids) for seq_ids, _, _, _, _ in batch)
            for seq_ids, seq_coords, target_idx, brand_name, brand_type in batch:
                seq_ids_aug, seq_coords_aug = augment_sequence(seq_ids, seq_coords)
                padded_ids = seq_ids_aug + [0] * (max_seq_len - len(seq_ids_aug))
                batch_seq_ids.append(padded_ids)
                if coord_dim_config > 0:
                    if seq_coords_aug is not None:
                        padded_coords = seq_coords_aug + [(0.0, 0.0)] * (max_seq_len - len(seq_coords_aug))
                    else:
                        padded_coords = [(0.0, 0.0)] * max_seq_len
                    batch_seq_coords.append(padded_coords)
                batch_targets.append(target_idx)
                batch_brand_names.append(brand_name)
                batch_brand_types.append(brand_type)
                # 新增brand_type编码
                if brand_type in brand_type_to_id:
                    batch_brand_type_ids.append(brand_type_to_id[brand_type])
                else:
                    batch_brand_type_ids.append(0)
            # CutMix增强
            batch_seq_ids, batch_seq_coords, batch_targets = cutmix_batch(batch_seq_ids, batch_seq_coords if coord_dim_config > 0 else None, batch_targets)
            
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
            brand_names = batch_brand_names if use_bert else None
            brand_types = batch_brand_types if use_bert else None
            brand_type_ids = torch.tensor(batch_brand_type_ids, dtype=torch.long).to(device)
            # 支持对比学习
            contrastive_pairs = None
            if use_contrastive and contrastive_pair_func is not None:
                contrastive_pairs = contrastive_pair_func(batch)
            # 支持mixup
            if use_mixup:
                outputs, y_a, y_b, lam, contrastive_loss = model(
                    seq_ids_tensor, seq_coords_tensor, brand_names, brand_types, brand_type_ids=brand_type_ids, mixup=True, targets=targets_tensor, mixup_alpha=mixup_alpha,
                    contrastive=use_contrastive, contrastive_pairs=contrastive_pairs, contrastive_temperature=contrastive_temperature
                )
                loss = lam * criterion(outputs, y_a, contrastive_loss, contrastive_weight) + (1 - lam) * criterion(outputs, y_b, contrastive_loss, contrastive_weight)
            elif use_contrastive:
                outputs, contrastive_loss = model(
                    seq_ids_tensor, seq_coords_tensor, brand_names, brand_types, brand_type_ids=brand_type_ids, contrastive=True, contrastive_pairs=contrastive_pairs, contrastive_temperature=contrastive_temperature
                )
                loss = criterion(outputs, targets_tensor, contrastive_loss, contrastive_weight)
            else:
                outputs = model(seq_ids_tensor, seq_coords_tensor, brand_names, brand_types, brand_type_ids=brand_type_ids)
                loss = criterion(outputs, targets_tensor)
            # 反向传播和优化
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            warmup_scheduler.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            batch_count += 1
            
            # 更新进度条显示当前批次的损失
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")
        
        # 计算平均训练损失
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0# 在验证集上评估 - 使用我们自己的实现而不是依赖compute_metrics函数
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
        # 统一综合指标
        composite_score = 20 * val_mrr + val_acc_k[1] + val_acc_k[5] + val_acc_k[10]
        
        # 记录指标用于绘图
        train_losses.append(avg_train_loss)
        val_mrrs.append(val_mrr)
        val_acc1s.append(val_acc_k[1])
        val_acc5s.append(val_acc_k[5])
        val_acc10s.append(val_acc_k[10])
        print(f"\nEpoch {epoch + 1}/{epochs} - "
              f"训练损失: {avg_train_loss:.4f}, " +
              f"验证 MRR: {val_mrr:.4f}, " +
              f"验证 Acc@1: {val_acc_k[1]:.4f}, " +
              f"验证 Acc@5: {val_acc_k[5]:.4f}, " +
              f"验证 Acc@10: {val_acc_k[10]:.4f}, " +
              f"综合指标: {composite_score:.4f}")
        # 以综合指标为唯一标准
        if composite_score > best_score:
            best_score = composite_score
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            if model_save_path:
                while True:
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mrr': val_mrr,
                            'val_acc1': val_acc_k[1],
                            'val_acc5': val_acc_k[5],
                            'val_acc10': val_acc_k[10],
                            'composite_score': composite_score,
                            'embed_dim': embed_dim,
                            'coord_dim': coord_dim_config,
                            'trans_dim': trans_dim,
                            'n_layers': n_layers,
                            'n_heads': n_heads,
                            'dropout': dropout,
                            'mlp_ratio': mlp_ratio,
                            'drop_path': drop_path,
                            'num_classes': num_total_classes
                        }, model_save_path)
                        print(f"模型已保存至: {model_save_path}")
                        break
                    except Exception as e:
                        print(f"保存模型时出错: {e}")
                        print("请检查文件路径和权限，稍后重试...")
                        time.sleep(5)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发: {patience}轮未改善")
                break
    # 训练循环后，早停判断
    early_stopper(val_mrr, model)
    lr_scheduler.step(val_mrr)
    if early_stopper.early_stop:
        print(f"[EarlyStopping] 训练提前终止，最佳MRR: {early_stopper.best_score:.4f}")
        if early_stopper.best_state is not None:
            model.load_state_dict(early_stopper.best_state)
        return model
      # 加载最佳模型权重
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (综合指标: {best_score:.4f})")
    
    # 绘制训练过程中的指标变化
    if len(train_losses) > 0:
        print("\n=== 绘制训练指标图表 ===")
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='训练损失')
        plt.title('训练损失随时间变化')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.axvline(best_epoch, color='r', linestyle='--', label='Early Stop')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(val_mrrs, label='MRR', marker='o')
        plt.plot(val_acc1s, label='Top-1准确率', marker='s')
        plt.plot(val_acc5s, label='Top-5准确率', marker='^')
        plt.plot(val_acc10s, label='Top-10准确率', marker='*')
        plt.title('验证集评估指标')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True)
        plt.axvline(best_epoch, color='r', linestyle='--', label='Early Stop')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if model_save_path:
            plot_save_path = os.path.splitext(model_save_path)[0] + '_training_plot.svg'
            plt.savefig(plot_save_path)
            print(f"训练过程指标图表已保存至: {plot_save_path}")
        
        # 显示图表（注意：在无界面的环境中需要注释掉）
        plt.show()
    
    return model

def cutmix_batch(batch_seq_ids, batch_seq_coords, batch_targets, cutmix_prob=0.3):
    import random
    if random.random() > cutmix_prob:
        return batch_seq_ids, batch_seq_coords, batch_targets
    idx = list(range(len(batch_seq_ids)))
    random.shuffle(idx)
    lam = np.random.beta(1.0, 1.0)
    new_seq_ids = []
    new_seq_coords = []
    new_targets = []
    for i, j in enumerate(idx):
        # 拼接前半部分i和后半部分j
        split = len(batch_seq_ids[i]) // 2
        seq = batch_seq_ids[i][:split] + batch_seq_ids[j][split:]
        if batch_seq_coords is not None:
            coords = batch_seq_coords[i][:split] + batch_seq_coords[j][split:]
        else:
            coords = None
        new_seq_ids.append(seq)
        new_seq_coords.append(coords)
        # 目标采用混合
        new_targets.append(batch_targets[i] if lam > 0.5 else batch_targets[j])
    return new_seq_ids, new_seq_coords, new_targets

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

    # 数据增强扩展训练集
    aug_times = 4
    augmented = []
    for sample in train_samples:
        seq_ids, seq_coords, target_idx, brand_name, brand_type = sample
        for _ in range(aug_times):
            aug_ids, aug_coords = augment_sequence(seq_ids, seq_coords, mask_prob=0.3, do_flip=True, do_rotate=True)
            augmented.append((aug_ids, aug_coords, target_idx, brand_name, brand_type))
    train_samples = train_samples + augmented
    print(f"扩展后训练样本数: {len(train_samples)}")

    trained_model = train_model(
        train_samples,
        val_samples,
        num_total_classes,
        embed_dim=DEFAULT_EMBED_DIM,
        coord=coords_info,
        coord_dim_config=DEFAULT_COORD_DIM,
        trans_dim=DEFAULT_TRANS_DIM,
        n_layers=DEFAULT_N_LAYERS,
        n_heads=DEFAULT_N_HEADS,
        dropout=DEFAULT_DROPOUT,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        device_name='cuda' if USE_CUDA_IF_AVAILABLE else 'cpu',
        model_save_path=model_output_path,
        batch_size=DEFAULT_BATCH_SIZE,
        optimizer_type=DEFAULT_OPTIMIZER,
        brand_type_map=brand_type_map
    )
    
    if trained_model and test_samples:
        print("\n在测试集上评估最终模型...")
        device_for_eval = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
        trained_model.to(device_for_eval)
        
        # 自己实现测试评估，确保正确处理品牌信息        trained_model.eval()
        correct_at_k = {1: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        
        test_progress_bar = tqdm(test_samples, desc="测试集评估", ncols=100)
        
        with torch.no_grad():
            for test_sample in test_progress_bar:
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

    elif not test_samples:
        print("\n无测试数据可供评估。")
    else:
        print("\n模型训练失败，无法在测试集上评估。")

print(f"\n--- 测试集评估结果 ---")
print(f"  测试集 MRR: {test_mrr:.4f}")
for k_val in [1, 5, 10]:
    print(f"  测试集 Acc@{k_val}: {test_acc_k[k_val]:.4f}")