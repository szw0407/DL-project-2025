"""
优化版数据加载 - 预先tokenize并优化批处理
"""
import torch
import numpy as np
from transformers import BertTokenizer

class OptimizedDataset(torch.utils.data.Dataset):
    """
    优化版数据集，预先tokenize所有品牌文本
    """
    def __init__(self, samples, bert_model_name='bert-base-chinese', max_length=64):
        self.samples = samples
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_length = max_length
        
        # 预先tokenize所有品牌文本
        self._preprocess_brand_texts()
        
    def _preprocess_brand_texts(self):
        """预先处理所有品牌文本"""
        print("预处理品牌文本...")
        brand_texts = [sample[3] for sample in self.samples]  # 第4个元素是品牌文本
        
        # 批量tokenize所有文本
        encoded = self.tokenizer(
            brand_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        self.brand_input_ids = encoded['input_ids']
        self.brand_attention_mask = encoded['attention_mask']
        print(f"完成预处理 {len(brand_texts)} 个品牌文本")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        prefix_idx, prefix_coords, prefix_poi, _, target_idx = self.samples[idx]
        
        return {
            'prefix_idx': prefix_idx,
            'prefix_coords': prefix_coords,
            'prefix_poi': prefix_poi,
            'brand_input_ids': self.brand_input_ids[idx],
            'brand_attention_mask': self.brand_attention_mask[idx],
            'target_idx': target_idx
        }

def optimized_collate_fn(batch):
    """
    优化版批处理函数
    """
    # 找出当前批次中最长的序列长度
    maxlen = max(len(item['prefix_idx']) for item in batch)
    
    # 初始化批次数据数组
    seq_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
    seq_coords = np.zeros((len(batch), maxlen, 2), dtype=np.float32)
    seq_poi = np.zeros((len(batch), maxlen, 10), dtype=np.float32)
    
    brand_input_ids = []
    brand_attention_mask = []
    targets = []
    
    for i, item in enumerate(batch):
        L = len(item['prefix_idx'])
        # 右对齐填充
        seq_ids[i, -L:] = item['prefix_idx']
        seq_coords[i, -L:, :] = item['prefix_coords']
        seq_poi[i, -L:, :] = item['prefix_poi']
        
        brand_input_ids.append(item['brand_input_ids'])
        brand_attention_mask.append(item['brand_attention_mask'])
        targets.append(item['target_idx'])
    
    return {
        'seq_ids': torch.from_numpy(seq_ids),
        'seq_coords': torch.from_numpy(seq_coords),
        'seq_poi': torch.from_numpy(seq_poi),
        'brand_input_ids': torch.stack(brand_input_ids),
        'brand_attention_mask': torch.stack(brand_attention_mask),
        'targets': torch.tensor(targets, dtype=torch.long)
    }

def create_optimized_dataloader(samples, batch_size=32, shuffle=True, num_workers=4):
    """
    创建优化版数据加载器
    """
    dataset = OptimizedDataset(samples)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=optimized_collate_fn,
        num_workers=num_workers,
        pin_memory=True,  # 加速数据传输到GPU
        persistent_workers=True if num_workers > 0 else False
    )
