import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import load_data
from model import StorePredictionModel
from evaluate import compute_metrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# try:
from torch_optimizer import Lookahead
LOOKAHEAD_AVAILABLE = True


plt.rcParams['font.family'] = 'Noto Sans SC'
DEFAULT_EMBED_DIM = 48  # 适当提升表达力
DEFAULT_COORD_DIM = 8   # 降低坐标嵌入维度
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LSTM_LAYERS = 8
DEFAULT_DROPOUT = 0.2   # 增大Dropout
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 5e-3  # 增大权重衰减
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_TRANS_DIM = 96  # 降低Transformer维度
DEFAULT_N_LAYERS = 2    # 降低层数
DEFAULT_N_HEADS = 4     # 降低头数
DEFAULT_MLP_RATIO = 2.0
DEFAULT_DROP_PATH = 0.3 # 增大DropPath
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = 'adamw'

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
    import random
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
    if optimizer_type == 'lookahead' and LOOKAHEAD_AVAILABLE:
        print('[优化器] 使用Lookahead+AdamW')
        return Lookahead(base_opt)
    if optimizer_type == 'adamw':
        print('[优化器] 使用AdamW')
    return base_opt

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

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
    mlp_ratio=DEFAULT_MLP_RATIO,
    drop_path=DEFAULT_DROP_PATH,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    device_name='cuda',
    model_save_path=None,
    use_bert=True,
    use_mixup=True,
    mixup_alpha=0.2,
    grad_clip=2.0,
    batch_size=DEFAULT_BATCH_SIZE,
    optimizer_type=DEFAULT_OPTIMIZER
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
    
    # 确定设备
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = StorePredictionModel(
        num_classes=num_total_classes, 
        embed_dim=embed_dim, 
        coord_dim=coord_dim_config, 
        trans_dim=trans_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        use_bert=use_bert
    )
    model = model.to(device)
    # 特色功能1：动态DropPath调度器
    drop_path_scheduler = StochasticDepthScheduler(model, max_drop_path=drop_path, min_drop_path=0.05, total_epochs=epochs)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr, weight_decay, optimizer_type)
    total_steps = epochs * max(1, len(train_samples) // batch_size)
    warmup_steps = int(0.1 * total_steps)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    early_stopper = EarlyStopping(patience=patience)
      # 定义早停相关变量
    best_val_mrr = -1
    best_val_acc1 = -1
    best_val_acc5 = -1
    best_val_acc10 = -1
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # 自适应学习率调度器
    lr_scheduler = AdaptiveLRScheduler(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    
    # 用于记录训练过程的指标
    train_losses = []
    val_mrrs = []
    val_acc1s = []
    val_acc5s = []
    val_acc10s = []
    
    # 训练循环    print("\n=== 开始训练 ===")
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
            # 支持mixup
            if use_mixup:
                outputs, y_a, y_b, lam = model(seq_ids_tensor, seq_coords_tensor, brand_names, brand_types, mixup=True, targets=targets_tensor, mixup_alpha=mixup_alpha)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(seq_ids_tensor, seq_coords_tensor, brand_names, brand_types)
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
                f"验证 Acc@10: {val_acc_k[10]:.4f}")
        
        # 综合判断指标提升/下降数量
        improved_count = 0
        declined_count = 0
        if val_mrr > best_val_mrr:
            improved_count += 1
        elif val_mrr < best_val_mrr:
            declined_count += 1
        if val_acc_k[1] > best_val_acc1:
            improved_count += 1
        elif val_acc_k[1] < best_val_acc1:
            declined_count += 1
        if val_acc_k[5] > best_val_acc5:
            improved_count += 1
        elif val_acc_k[5] < best_val_acc5:
            declined_count += 1
        if val_acc_k[10] > best_val_acc10:
            improved_count += 1
        elif val_acc_k[10] < best_val_acc10:
            declined_count += 1
        improved = (improved_count >= 2 and declined_count <= 2)

        if improved:
            # 修复类型问题，直接赋值而不是max（因为improved时必然是更优）
            best_val_mrr = val_mrr
            best_val_acc1 = val_acc_k[1]
            best_val_acc5 = val_acc_k[5]
            best_val_acc10 = val_acc_k[10]
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            # 如果指定了保存路径，保存模型
            if model_save_path:
                while True:
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mrr': val_mrr,
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
            break
      # 加载最佳模型权重
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (验证 MRR: {best_val_mrr:.4f})")
    
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
        optimizer_type=DEFAULT_OPTIMIZER
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