"""
商业选址预测模型

整体模型架构:
┌──────────────────────────────────────────────────────────────┐
│                    NextGridPredictor                         │
└──────────────────────────────────────────────────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ SeqEncoder   │   │ CoordEncoder │   │ POIEncoder   │
│ (LSTM-based) │   │  (MLP-based) │   │  (MLP-based) │
└──────────────┘   └──────────────┘   └──────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Fusion Layer   │
                    │     (MLP with    │
                    │     Dropout)     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Classifier    │
                    │  (Linear Layer)  │
                    └──────────────────┘
                              │
                              ▼
                    预测下一个网格的概率分布

该模型采用多模态融合架构，综合利用三种信息进行商业选址预测:
1. 历史选址序列信息 (通过LSTM编码)
2. 地理坐标信息 (通过MLP编码)
3. 兴趣点(POI)特征信息 (通过MLP编码)

这三种信息经过各自的编码器处理后，在特征层面融合，
再通过多层感知机进行进一步特征提取，最终用于预测下一个最佳选址位置。
"""
import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    """
    序列编码器：用于编码历史选址序列的特征
    
    该编码器首先将网格ID映射到低维嵌入空间，然后通过LSTM网络捕获序列中的
    时序依赖关系和选址模式。使用LSTM的优势在于能够记忆长期依赖，
    适合捕捉商业选址中的空间扩张规律。
    
    架构:
    输入序列 → 嵌入层 → LSTM → 最后时刻的隐藏状态
    
    参数:
        vocab_size: 词汇表大小，即网格总数
        embed_dim: 嵌入维度
        lstm_hidden: LSTM隐藏层维度
        lstm_layers: LSTM层数
        dropout: Dropout比率，用于防止过拟合
    """
    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
                            
    def forward(self, seq_ids):
        """
        前向传播
        
        参数:
            seq_ids: 网格ID序列，形状为(batch_size, seq_len)
            
        返回:
            序列的最终表示，形状为(batch_size, lstm_hidden)
        """
        emb = self.embed(seq_ids)  # (batch_size, seq_len, embed_dim)
        out, (h, _) = self.lstm(emb)  # out: (batch_size, seq_len, lstm_hidden)
        return out[:, -1, :]  # 返回最后一个时间步的输出作为序列表示

class MLPEncoder(nn.Module):
    """
    多层感知机编码器：用于编码坐标和POI特征
    
    这是一个简单的两层感知机，通过非线性变换将输入特征映射到固定维度的
    输出空间。适用于处理非序列性的数值特征，如坐标和POI统计数据。
    
    架构:
    输入特征 → 线性层 → ReLU → 线性层 → ReLU
    
    参数:
        in_dim: 输入特征维度
        out_dim: 输出特征维度
        hidden_dim: 隐藏层维度，默认为32
    """
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, in_dim)
            
        返回:
            编码后的特征，形状为(batch_size, out_dim)
        """
        return self.model(x)

class NextGridPredictor(nn.Module):
    """
    下一个网格预测器：预测品牌下一个最佳选址位置
    
    这是整个模型的主体部分，采用多模态融合方法，将三种不同类型的信息
    (历史选址序列、地理坐标、POI特征)整合起来，共同预测下一个最佳选址位置。
    
    模型处理流程:
    1. 通过各自的编码器分别编码三种信息
    2. 将编码后的特征向量拼接起来
    3. 通过多层感知机进行特征融合和提取
    4. 最终通过线性分类器预测下一个网格的概率分布
    
    多模态融合的优势:
    - 序列信息: 捕捉品牌扩张的时序模式和依赖关系
    - 坐标信息: 考虑地理位置的连续性和空间分布
    - POI信息: 考虑周边设施和商业环境的影响
    
    参数:
        num_classes: 类别数量，即网格总数
        embed_dim: 网格ID的嵌入维度，默认为32
        lstm_hidden: LSTM隐藏层维度，默认为64
        lstm_layers: LSTM层数，默认为1
        coord_dim: 坐标特征维度，默认为2 (经度、纬度)
        poi_dim: POI特征维度，默认为10 (10种POI类型)
        coord_out_dim: 坐标编码后的维度，默认为16
        poi_out_dim: POI编码后的维度，默认为16
        fusion_dim: 特征融合后的维度，默认为64
        dropout: Dropout比率，用于防止过拟合，默认为0.1
    """
    def __init__(self, num_classes, embed_dim=32, lstm_hidden=64, lstm_layers=1,
                 coord_dim=2, poi_dim=10, coord_out_dim=16, poi_out_dim=16, fusion_dim=64, dropout=0.1):
        super().__init__()
        # 序列编码器：处理历史选址序列
        self.seq_encoder = SeqEncoder(num_classes, embed_dim, lstm_hidden, lstm_layers, dropout)
        # 坐标编码器：处理地理坐标信息
        self.coord_encoder = MLPEncoder(coord_dim, coord_out_dim)
        # POI编码器：处理兴趣点特征信息
        self.poi_encoder = MLPEncoder(poi_dim, poi_out_dim)
        # 特征融合网络：整合三种编码后的特征
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + coord_out_dim + poi_out_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 分类器：预测下一个网格的概率分布
        self.classifier = nn.Linear(fusion_dim, num_classes)
    def forward(self, seq_ids, seq_coords, seq_poi):
        """
        前向传播
        
        数据流程:
        1. 各编码器独立处理对应的输入特征
        2. 对坐标和POI特征取序列平均值，降维处理
        3. 拼接所有特征向量
        4. 通过融合网络进一步提取联合特征
        5. 通过分类器预测下一个网格的概率分布
        
        参数:
            seq_ids: 网格ID序列，形状为(batch_size, seq_len)
            seq_coords: 坐标序列，形状为(batch_size, seq_len, 2)
            seq_poi: POI特征序列，形状为(batch_size, seq_len, 10)
            
        返回:
            logits: 下一个网格的预测概率分布，形状为(batch_size, num_classes)
        """
        # 输入: (batch, seq_len, dim)
        seq_out = self.seq_encoder(seq_ids)               # (batch, lstm_hidden)
        
        # 取坐标序列的平均值，简化处理同时保留整体分布信息
        coords_out = self.coord_encoder(seq_coords.mean(dim=1)) # (batch, coord_out_dim)
        
        # 取POI特征序列的平均值，同样简化处理
        poi_out = self.poi_encoder(seq_poi.mean(dim=1))   # (batch, poi_out_dim)
        
        # 特征拼接：将三种编码后的特征向量拼接起来
        x = torch.cat([seq_out, coords_out, poi_out], dim=-1)
        
        # 特征融合：通过MLP进一步提取联合特征
        f = self.fusion(x)
        
        # 分类预测：预测下一个网格的概率分布
        logits = self.classifier(f)
        return logits
