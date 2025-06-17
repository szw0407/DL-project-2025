import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class DropPath(nn.Module):
    """Stochastic Depth/DropPath: 随机丢弃残差分支"""

    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """LayerScale: 可学习缩放残差分支"""

    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=4.0,  # 增加MLP扩展比例
        drop=0.1,
        drop_path=0.1,
        layer_scale_init=1e-2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # 提升数值稳定性
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        # 增强MLP：添加中间层和更好的激活函数
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),  # 使用SiLU激活函数，比GELU更强
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 添加中间层
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, dim),
            nn.Dropout(drop),
        )
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        # x: (B, T, C)
        # Pre-norm attention with stronger residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path1(self.ls1(attn_out))
        
        # Pre-norm MLP with stronger residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path2(self.ls2(mlp_out))
        return x


def mixup_data(x, y, alpha=0.2):
    """Mixup: 数据增强, x: (B, ...), y: (B,)"""
    if alpha > 0:
        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha])).max().item()
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class SEBlock(nn.Module):
    """Squeeze-and-Excitation通道注意力模块"""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C) or (B, T, C)
        if x.dim() == 3:
            # (B, T, C) -> (B, C, T)
            x_perm = x.permute(0, 2, 1)
            y = self.avg_pool(x_perm).squeeze(-1)  # (B, C)
        else:
            y = x
        w = self.fc(y)
        if x.dim() == 3:
            w = w.unsqueeze(1)  # (B, 1, C)
        else:
            w = w  # (B, C)
        return x * w


class FeatureDropBlock(nn.Module):
    """特征DropBlock，训练时随机遮蔽部分特征，提升泛化能力"""

    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        mask = (torch.rand_like(x) > self.drop_prob).float()
        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_score = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = (x * attn_score).sum(dim=1)  # (B, C)
        return pooled


class StorePredictionModel(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=64,      # 增加嵌入维度 32->64
        coord_dim=12,      # 增加坐标维度 8->12  
        trans_dim=64,      # 增加主干维度 32->64
        n_layers=4,        # 增加层数 2->4
        n_heads=8,         # 增加注意力头数 4->8
        dropout=0.3,       # 适当降低dropout
        bert_model_name="bert-base-chinese",
        use_bert=True,
        bert_feature_dim=768,
        mlp_ratio=4.0,     # 增加MLP扩展比例 2.0->4.0
        drop_path=0.3,     # 适当降低drop_path
        brand_type_num=64,
        brand_type_embed_dim=12,  # 增加brand_type嵌入维度
        brand_type_to_id=None,
        use_multiscale_conv=True,
        use_spatial_stats=True,
        use_attn_pool=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.trans_dim = trans_dim
        self.use_bert = use_bert
        self.bert_feature_dim = bert_feature_dim
        self.brand_type_num = brand_type_num
        self.brand_type_embed_dim = brand_type_embed_dim
        self.brand_type_to_id = brand_type_to_id if brand_type_to_id is not None else {}
        self.use_multiscale_conv = use_multiscale_conv
        self.use_spatial_stats = use_spatial_stats
        self.use_attn_pool = use_attn_pool
        # Embedding
        self.id_embedding = nn.Embedding(num_classes, embed_dim)
        if coord_dim > 0:
            self.coord_embedding_layer = nn.Linear(2, coord_dim)
            self.coord_bn = nn.BatchNorm1d(coord_dim)
        else:
            self.coord_embedding_layer = None        # brand_type embedding + MLP (增强)
        self.brand_type_embedding = nn.Embedding(brand_type_num, brand_type_embed_dim)
        self.brand_type_mlp = nn.Sequential(
            nn.Linear(brand_type_embed_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 24),  # 输出24维
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # BERT
        if use_bert:
            try:
                self.bert_model = BertModel.from_pretrained(bert_model_name)
                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
                # 允许BERT参数finetune
                for p in self.bert_model.parameters():
                    p.requires_grad = True  # 允许finetune
            except:
                print(f"警告: 无法加载BERT模型 {bert_model_name}，将禁用BERT特征")
                self.use_bert = False
        if self.use_bert:
            self.bert_proj = nn.Sequential(
                nn.Linear(self.bert_feature_dim, embed_dim * 2),
                nn.BatchNorm1d(embed_dim * 2),
                nn.SiLU(),  # GELU -> SiLU
                nn.Linear(embed_dim * 2, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )        # 增强多尺度卷积特征融合（可选）
        if use_multiscale_conv:
            # 增加更多卷积核尺寸和深度
            self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=1, padding=0)
            self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
            self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)  # 新增7x7卷积
            # 深度卷积分支
            self.depth_conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.SiLU(),
                nn.Conv1d(embed_dim, 64, kernel_size=1),
            )
            self.conv_bn = nn.BatchNorm1d(64 * 5)  # 5个分支
            self.conv_act = nn.SiLU()  # 使用SiLU激活
            # 通道注意力
            self.conv_se = SEBlock(64 * 5)        # 增强空间统计特征分支（可选）
        if use_spatial_stats:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(8, 32),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 24),  # 输出24维
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        # AttentionPooling（可选）
        if use_attn_pool:
            self.attn_pool = AttentionPooling(trans_dim)        # 融合后投影到Transformer输入维度
        in_dim = (
            embed_dim
            + (coord_dim if coord_dim > 0 else 0)
            + (embed_dim if use_bert else 0)
            + 24  # brand_type_mlp输出维度增加到24
        )
        if use_multiscale_conv:
            in_dim += 64 * 5  # 更新为5个分支，每个64维
        if use_spatial_stats:
            in_dim += 24  # spatial_mlp输出维度增加到24
        self.input_proj = nn.Linear(in_dim, trans_dim)
        self.input_norm = nn.LayerNorm(trans_dim)  # 新增输入归一化
        # TransformerEncoder
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(trans_dim, n_heads, mlp_ratio, dropout, drop_path)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(trans_dim)
        # 全局注意力层
        self.global_attn = nn.MultiheadAttention(
            trans_dim, n_heads, dropout=dropout, batch_first=True
        )        # 增强MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.BatchNorm1d(trans_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, trans_dim // 2),
            nn.BatchNorm1d(trans_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim // 2, trans_dim // 4),
            nn.BatchNorm1d(trans_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(trans_dim // 4),
        )
        self.output_fc = nn.Linear(trans_dim // 4, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        self.se_block = SEBlock(trans_dim)
        self.feature_dropblock = FeatureDropBlock(drop_prob=0.4)  # DropBlock极致提升

    def extract_bert_features(self, brand_names, brand_types, seq_len):
        if not self.use_bert:
            return None
        device = next(self.parameters()).device
        combined_texts = [f"{n} {t}" for n, t in zip(brand_names, brand_types)]
        with torch.no_grad():
            encoded = self.bert_tokenizer(
                combined_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            bert_outputs = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            bert_features = bert_outputs.last_hidden_state[:, 0, :]
        bert_proj = self.bert_proj(bert_features)
        bert_seq_features = bert_proj.unsqueeze(1).expand(-1, seq_len, -1)
        return bert_seq_features

    def forward(
        self,
        seq_ids,
        seq_coords=None,
        brand_names=None,
        brand_types=None,
        brand_type_ids=None,
        mixup=False,
        targets=None,
        mixup_alpha=0.2,
    ):
        batch_size, seq_len = seq_ids.shape[:2]
        id_emb = self.id_embedding(seq_ids)  # (B, T, D)
        features = [id_emb]        # 增强多尺度卷积特征（可选）
        if self.use_multiscale_conv:
            id_emb_conv = id_emb.permute(0, 2, 1)  # (B, D, T)
            conv1 = self.conv1(id_emb_conv)
            conv3 = self.conv3(id_emb_conv)
            conv5 = self.conv5(id_emb_conv)
            conv7 = self.conv7(id_emb_conv)  # 新增7x7卷积
            depth_conv = self.depth_conv(id_emb_conv)  # 深度卷积
            conv_feat = torch.cat([conv1, conv3, conv5, conv7, depth_conv], dim=1)  # (B, 64*5, T)
            conv_feat = self.conv_bn(conv_feat)
            conv_feat = self.conv_act(conv_feat)
            conv_feat = self.conv_se(conv_feat.permute(0, 2, 1)).permute(0, 2, 1)  # 通道注意力
            conv_feat = conv_feat.permute(0, 2, 1)  # (B, T, 64*5)
            features.append(conv_feat)
        if self.coord_embedding_layer is not None and seq_coords is not None:
            coords_flat = seq_coords.reshape(batch_size * seq_len, 2)
            coord_emb_flat = self.coord_embedding_layer(coords_flat)
            coord_emb_flat = self.coord_bn(coord_emb_flat)
            coord_emb_activated = torch.tanh(coord_emb_flat)
            coord_emb = coord_emb_activated.view(batch_size, seq_len, self.coord_dim)
            features.append(coord_emb)
        if self.use_bert and brand_names is not None and brand_types is not None:
            bert_features = self.extract_bert_features(
                brand_names, brand_types, seq_len
            )
            if bert_features is not None:
                features.append(bert_features)
        if (
            brand_type_ids is None
            and brand_types is not None
            and hasattr(self, "brand_type_to_id")
        ):
            brand_type_ids = [self.brand_type_to_id.get(bt, 0) for bt in brand_types]
            brand_type_ids = torch.tensor(
                brand_type_ids, dtype=torch.long, device=seq_ids.device
            )
        if brand_type_ids is not None:
            brand_type_emb = self.brand_type_embedding(brand_type_ids)
            brand_type_feat = self.brand_type_mlp(brand_type_emb)  # (B, 24)  更新输出维度
            brand_type_feat = brand_type_feat.unsqueeze(1).expand(-1, seq_len, -1)
            features.append(brand_type_feat)
        # 空间统计特征分支（可选）
        if self.use_spatial_stats and seq_coords is not None:
            x = seq_coords[:, :, 0]
            y = seq_coords[:, :, 1]
            stat_feat = torch.stack(
                [
                    x.mean(dim=1),
                    y.mean(dim=1),
                    x.std(dim=1),
                    y.std(dim=1),
                    x.min(dim=1).values,
                    x.max(dim=1).values,
                    y.min(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )  # (B, 8)
            stat_feat = self.spatial_mlp(stat_feat)  # (B, 24)  更新输出维度
            stat_feat = stat_feat.unsqueeze(1).expand(-1, seq_len, -1)
            features.append(stat_feat)
        x = torch.cat(features, dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.feature_dropblock(x)
        x = self.transformer(x)
        x = self.norm(x)
        # AttentionPooling（可选）
        if self.use_attn_pool:
            last_out = self.attn_pool(x)
        else:
            global_query = x[:, :1, :]
            global_out, _ = self.global_attn(global_query, x, x)
            last_out = global_out.squeeze(1)
        last_out = self.se_block(last_out)
        last_out = self.dropout_layer(last_out)
        # Mixup（可选）
        if mixup and targets is not None:
            last_out, y_a, y_b, lam = mixup_data(last_out, targets, alpha=mixup_alpha)
        else:
            y_a = y_b = lam = None
        mlp_out = self.mlp(last_out)
        logits = self.output_fc(mlp_out)
        # logits = torch.softmax(logits, dim=-1)  # 如为分类任务可解注释
        if mixup and targets is not None:
            return logits, y_a, y_b, lam
        return logits
