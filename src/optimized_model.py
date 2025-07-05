"""
优化版商业选址预测模型 - 解决GPU利用率问题

主要优化：
1. 移除forward中的tokenization，改为预先tokenize
2. 添加混合精度训练支持
3. 优化数据加载和批处理
4. 减少CPU-GPU数据传输
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class OptimizedBrandEncoder(nn.Module):
    """
    优化版品牌编码器：接受预先tokenized的输入，避免forward时tokenization
    """
    def __init__(self, bert_model_name='bert-base-chinese', out_dim=64, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # 将BERT的输出维度映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播 - 接受预先tokenized的输入
        
        参数:
            input_ids: tokenized的输入ID张量，形状为(batch_size, seq_len)
            attention_mask: 注意力掩码张量，形状为(batch_size, seq_len)
            
        返回:
            编码后的品牌特征，形状为(batch_size, out_dim)
        """
        # 通过BERT获取文本表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        # 使用[CLS]标记的表示作为整个文本的表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # 通过投影层映射到目标维度
        brand_features = self.projection(cls_output)  # (batch_size, out_dim)
        
        return brand_features

class SeqEncoder(nn.Module):
    """序列编码器"""
    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
                            
    def forward(self, seq_ids):
        emb = self.embed(seq_ids)  # (batch_size, seq_len, embed_dim)
        out, (h, _) = self.lstm(emb)  # out: (batch_size, seq_len, lstm_hidden)
        return out[:, -1, :]  # 返回最后一个时间步的输出作为序列表示

class MLPEncoder(nn.Module):
    """多层感知机编码器"""
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)

class OptimizedNextGridPredictor(nn.Module):
    """
    优化版下一个网格预测器
    
    主要优化：
    1. 使用优化版BrandEncoder，避免forward时tokenization
    2. 支持混合精度训练
    3. 减少不必要的数据传输
    """
    def __init__(self, num_classes, embed_dim=32, lstm_hidden=64, lstm_layers=1,
                 coord_dim=2, poi_dim=10, coord_out_dim=16, poi_out_dim=16, 
                 brand_out_dim=64, fusion_dim=64, dropout=0.1, 
                 bert_model_name='bert-base-chinese', freeze_bert=False):
        super().__init__()
        # 序列编码器：处理历史选址序列
        self.seq_encoder = SeqEncoder(num_classes, embed_dim, lstm_hidden, lstm_layers, dropout)
        # 坐标编码器：处理地理坐标信息
        self.coord_encoder = MLPEncoder(coord_dim, coord_out_dim)
        # POI编码器：处理兴趣点特征信息
        self.poi_encoder = MLPEncoder(poi_dim, poi_out_dim)
        # 优化版品牌编码器：处理预先tokenized的品牌信息
        self.brand_encoder = OptimizedBrandEncoder(bert_model_name=bert_model_name, 
                                                 out_dim=brand_out_dim, 
                                                 freeze_bert=freeze_bert)
        # 特征融合网络：整合四种编码后的特征
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + coord_out_dim + poi_out_dim + brand_out_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 分类器：预测下一个网格的概率分布
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, seq_ids, seq_coords, seq_poi, brand_input_ids, brand_attention_mask):
        """
        优化版前向传播
        
        参数:
            seq_ids: 网格ID序列，形状为(batch_size, seq_len)
            seq_coords: 坐标序列，形状为(batch_size, seq_len, 2)
            seq_poi: POI特征序列，形状为(batch_size, seq_len, 10)
            brand_input_ids: 预先tokenized的品牌input_ids，形状为(batch_size, text_seq_len)
            brand_attention_mask: 预先tokenized的注意力掩码，形状为(batch_size, text_seq_len)
            
        返回:
            logits: 下一个网格的预测概率分布，形状为(batch_size, num_classes)
        """
        # 序列编码: (batch, seq_len) -> (batch, lstm_hidden)
        seq_out = self.seq_encoder(seq_ids)               
        
        # 坐标编码: 取序列平均值，简化处理同时保留整体分布信息
        coords_out = self.coord_encoder(seq_coords.mean(dim=1)) # (batch, coord_out_dim)
        
        # POI编码: 取特征序列的平均值，同样简化处理
        poi_out = self.poi_encoder(seq_poi.mean(dim=1))   # (batch, poi_out_dim)
        
        # 品牌编码: 使用预先tokenized的输入
        brand_out = self.brand_encoder(brand_input_ids, brand_attention_mask)  # (batch, brand_out_dim)
        
        # 特征拼接：将四种编码后的特征向量拼接起来
        x = torch.cat([seq_out, coords_out, poi_out, brand_out], dim=-1)
        
        # 特征融合：通过MLP进一步提取联合特征
        f = self.fusion(x)
        
        # 分类预测：预测下一个网格的概率分布
        logits = self.classifier(f)
        return logits
