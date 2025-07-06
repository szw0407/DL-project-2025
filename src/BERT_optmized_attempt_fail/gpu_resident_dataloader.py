"""
GPU常驻数据加载器 - 将所有数据预先加载到GPU，避免重复的CPU-GPU传输
"""
import torch
import numpy as np
from transformers import BertTokenizer

class GPUResidentDataset:
    """
    GPU常驻数据集 - 所有数据都预先加载到GPU内存中
    """
    def __init__(self, samples, device, bert_model_name='bert-base-chinese', max_length=64):
        self.device = device
        self.max_length = max_length
        
        print(f"正在将 {len(samples)} 个样本预处理并加载到 {device}...")
        
        # 预处理所有数据
        self._preprocess_all_data(samples, bert_model_name)
        
        print(f"所有数据已加载到GPU，占用显存: {self._estimate_memory():.2f} MB")
        
    def _preprocess_all_data(self, samples, bert_model_name):
        """预处理所有数据并加载到GPU"""
        # 1. 预处理品牌文本
        print("  - 预处理品牌文本...")
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        brand_texts = [sample[3] for sample in samples]
        
        encoded = tokenizer(
            brand_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 直接移动到GPU
        self.brand_input_ids = encoded['input_ids'].to(self.device)
        self.brand_attention_mask = encoded['attention_mask'].to(self.device)
        
        # 2. 预处理序列数据
        print("  - 预处理序列数据...")
        max_seq_len = max(len(sample[0]) for sample in samples)
        
        # 预分配GPU张量
        self.seq_ids = torch.zeros((len(samples), max_seq_len), dtype=torch.long, device=self.device)
        self.seq_coords = torch.zeros((len(samples), max_seq_len, 2), dtype=torch.float, device=self.device)
        self.seq_poi = torch.zeros((len(samples), max_seq_len, 10), dtype=torch.float, device=self.device)
        self.targets = torch.zeros(len(samples), dtype=torch.long, device=self.device)
        self.seq_lengths = torch.zeros(len(samples), dtype=torch.long, device=self.device)
        
        # 填充数据
        for i, (prefix_idx, prefix_coords, prefix_poi, _, target_idx) in enumerate(samples):
            seq_len = len(prefix_idx)
            
            # 右对齐填充（保持与原来的collate_fn一致）
            self.seq_ids[i, -seq_len:] = torch.tensor(prefix_idx, device=self.device)
            self.seq_coords[i, -seq_len:] = torch.tensor(prefix_coords, device=self.device)
            self.seq_poi[i, -seq_len:] = torch.tensor(np.array(prefix_poi), device=self.device)
            self.targets[i] = target_idx
            self.seq_lengths[i] = seq_len
            
        print(f"  - 序列数据形状: {self.seq_ids.shape}")
        print(f"  - 品牌文本形状: {self.brand_input_ids.shape}")
    
    def _estimate_memory(self):
        """估算占用的GPU内存（MB）"""
        memory = 0
        memory += self.seq_ids.element_size() * self.seq_ids.numel()
        memory += self.seq_coords.element_size() * self.seq_coords.numel()
        memory += self.seq_poi.element_size() * self.seq_poi.numel()
        memory += self.brand_input_ids.element_size() * self.brand_input_ids.numel()
        memory += self.brand_attention_mask.element_size() * self.brand_attention_mask.numel()
        memory += self.targets.element_size() * self.targets.numel()
        memory += self.seq_lengths.element_size() * self.seq_lengths.numel()
        return memory / (1024 * 1024)  # 转换为MB
    
    def __len__(self):
        return len(self.targets)
    
    def get_batch(self, indices):
        """获取指定索引的批次数据"""
        return {
            'seq_ids': self.seq_ids[indices],
            'seq_coords': self.seq_coords[indices],
            'seq_poi': self.seq_poi[indices],
            'brand_input_ids': self.brand_input_ids[indices],
            'brand_attention_mask': self.brand_attention_mask[indices],
            'targets': self.targets[indices],
            'seq_lengths': self.seq_lengths[indices]
        }


class GPUResidentDataLoader:
    """
    GPU常驻数据加载器 - 从GPU内存中直接获取批次数据
    """
    def __init__(self, gpu_dataset, batch_size=32, shuffle=True):
        self.dataset = gpu_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(gpu_dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
    def __iter__(self):
        # 生成索引
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.dataset.device)
        else:
            indices = torch.arange(self.num_samples, device=self.dataset.device)
        
        # 按批次返回数据
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)
    
    def __len__(self):
        return self.num_batches


def create_gpu_resident_dataloader(samples, device, batch_size=32, shuffle=True, 
                                 bert_model_name='bert-base-chinese', max_length=64):
    """
    创建GPU常驻数据加载器
    
    参数:
        samples: 原始样本数据
        device: GPU设备
        batch_size: 批次大小
        shuffle: 是否打乱数据
        bert_model_name: BERT模型名称
        max_length: 文本最大长度
        
    返回:
        GPUResidentDataLoader实例
    """
    gpu_dataset = GPUResidentDataset(samples, device, bert_model_name, max_length)
    return GPUResidentDataLoader(gpu_dataset, batch_size, shuffle)


def check_gpu_memory_usage():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"GPU内存使用情况:")
        print(f"  - 已分配: {allocated:.2f} GB")
        print(f"  - 已保留: {reserved:.2f} GB") 
        print(f"  - 总容量: {total:.2f} GB")
        print(f"  - 使用率: {allocated/total*100:.1f}%")
        
        return allocated, reserved, total
    else:
        print("CUDA不可用")
        return 0, 0, 0
