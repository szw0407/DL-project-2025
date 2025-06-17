import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_heads=2,  # 简化：减少注意力头数
        mlp_ratio=1.5,  # 简化：减小MLP扩展比例
        drop=0.1,
        drop_path=0.05,
        layer_scale_init=1e-2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path1(self.ls1(attn_out))
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


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent对比损失，z_i/z_j: (B, D)"""
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    similarity_matrix = torch.matmul(representations, representations.T)  # (2B, 2B)
    batch_size = z_i.size(0)
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(2 * batch_size, device=z_i.device).bool()
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size)
    ], dim=0)
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
    loss = F.cross_entropy(logits, labels)
    return loss


class StorePredictionModel(nn.Module):
    
    def __init__(
        self,
        num_classes,
        embed_dim=16,
        coord_dim=4,
        trans_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        bert_model_name="bert-base-chinese",
        use_bert=True,
        bert_feature_dim=768,
        mlp_ratio=1.5,
        drop_path=0.05,
        brand_type_num=16,
        brand_type_embed_dim=4,
        brand_type_to_id=None,
        use_multiscale_conv=False,
        use_spatial_stats=False,
        use_attn_pool=False,
        gnn_dim=32,           # 新增: GNN输出维度
        gnn_layers=1,         # 新增: GNN层数
        gnn_heads=2,          # 新增: GAT头数
        gnn_dropout=0.1,      # 新增: GNN dropout
        gnn_fuse="cat",      # 新增: GNN与Transformer融合方式: cat/mean/sum
        *args, **kwargs
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
        self.gnn_dim = gnn_dim
        self.gnn_layers = gnn_layers
        self.gnn_heads = gnn_heads
        self.gnn_dropout = gnn_dropout
        self.gnn_fuse = gnn_fuse
        # Embedding
        self.id_embedding = nn.Embedding(num_classes, embed_dim)
        # 坐标特征编码：MLP+LayerNorm
        if coord_dim > 0:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, coord_dim * 2),
                nn.SiLU(),
                nn.Linear(coord_dim * 2, coord_dim),
                nn.LayerNorm(coord_dim),
            )
        else:
            self.coord_encoder = None
        # brand_type embedding + MLP (增强)
        self.brand_type_embedding = nn.Embedding(brand_type_num, brand_type_embed_dim)
        self.brand_type_mlp = nn.Sequential(
            nn.Linear(brand_type_embed_dim, 8),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
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
                nn.LayerNorm(embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.bert_vote_proj = nn.Linear(embed_dim, trans_dim)
        # GNN分支
        self.use_gnn = HAS_PYG
        if self.use_gnn:
            gnn_blocks = []
            gnn_in = (
                embed_dim
                + (coord_dim if coord_dim > 0 else 0)
                + (embed_dim if use_bert else 0)
                + 4
            )
            for i in range(gnn_layers):
                gnn_blocks.append(SimpleGNNBlock(gnn_in if i == 0 else gnn_dim, gnn_dim, heads=gnn_heads, dropout=gnn_dropout))
            self.gnn_blocks = nn.ModuleList(gnn_blocks)
        # 融合后输入维度
        in_dim = (
            embed_dim
            + (coord_dim if coord_dim > 0 else 0)
            + (embed_dim if use_bert else 0)
            + 4
        )
        if self.use_gnn:
            if gnn_fuse == "cat":
                self.fused_dim = trans_dim + gnn_dim
            else:
                self.fused_dim = trans_dim  # mean/sum
        else:
            self.fused_dim = trans_dim
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
        # MLP Head输入输出均应为trans_dim（加权融合后特征维度）
        self.mlp = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.trans_dim, self.trans_dim // 2),
            nn.LayerNorm(self.trans_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_fc = nn.Linear(self.trans_dim // 2, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        # 注意：SEBlock的输入通道应与加权融合后特征维度一致（trans_dim）
        self.se_block = SEBlock(self.trans_dim)
        self.feature_dropblock = FeatureDropBlock(drop_prob=0.1)
        # ===== 新增：轻量级时序分支（GRU） =====
        self.rnn_dim = trans_dim  # 保持与transformer输出一致，便于融合
        self.rnn = nn.GRU(
            in_dim, self.rnn_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        # ===== 新增：BERT特征投票权重参数（与主分支分离） =====
        # 假设每个类别/品牌类型有独立权重，或全局权重
        self.bert_vote_weight = nn.Parameter(torch.ones(1))  # 可扩展为更复杂结构
        # ===== 主分支融合权重参数 =====
        n_fuse = 2 if self.use_gnn else 1  # 只融合transformer和gru（gnn可选）
        self.fuse_weights = nn.Parameter(torch.ones(n_fuse))

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
        contrastive=False,
        contrastive_pairs=None,
        contrastive_temperature=0.5,
        edge_index=None,  # 新增: GNN需要的edge_index
    ):
        batch_size, seq_len = seq_ids.shape[:2]
        id_emb = self.id_embedding(seq_ids)  # (B, T, D)
        features = [id_emb]
        if self.coord_encoder is not None and seq_coords is not None:
            coords_flat = seq_coords.reshape(batch_size * seq_len, 2)
            coord_emb_flat = self.coord_encoder(coords_flat)
            coord_emb = coord_emb_flat.view(batch_size, seq_len, self.coord_dim)
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
            brand_type_feat = self.brand_type_mlp(brand_type_emb)
            brand_type_feat = brand_type_feat.unsqueeze(1).expand(-1, seq_len, -1)
            features.append(brand_type_feat)
        # ...特征拼接...
        x = torch.cat(features, dim=-1)  # (B, T, F)
        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = self.feature_dropblock(x_proj)
        # Transformer分支
        x_trans = self.transformer(x_proj)
        x_trans = self.norm(x_trans)
        # GNN分支
        if self.use_gnn:
            x_gnn_in = x.reshape(batch_size * seq_len, -1)
            if edge_index is None:
                idx = torch.arange(seq_len, device=x.device)
                edge_index = torch.combinations(idx, r=2).T
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index.to(x.device)
            x_gnn = x_gnn_in
            for gnn_block in self.gnn_blocks:
                x_gnn = gnn_block(x_gnn, edge_index)
            x_gnn = x_gnn.view(batch_size, seq_len, self.gnn_dim)
        # 时序分支（GRU）
        x_rnn, _ = self.rnn(x)
        # ===== 主分支加权融合（不含BERT） =====
        fuse_list = [x_trans]
        if self.use_gnn:
            if self.gnn_dim != self.trans_dim:
                x_gnn = nn.Linear(self.gnn_dim, self.trans_dim).to(x_gnn.device)(x_gnn)
            fuse_list.append(x_gnn)
        fuse_list.append(x_rnn)
        fuse_weights = torch.softmax(self.fuse_weights, dim=0)
        x_fused_main = sum(w * f for w, f in zip(fuse_weights, fuse_list))
        # ===== BERT特征投票融合 =====
        if self.use_bert and brand_names is not None and brand_types is not None:
            bert_features = self.extract_bert_features(brand_names, brand_types, seq_len)
            if bert_features is None or not isinstance(bert_features, torch.Tensor):
                bert_features = torch.zeros_like(x_proj)
            bert_features_proj = self.bert_vote_proj(bert_features)
            bert_vote = torch.sigmoid(self.bert_vote_weight)
            x_fused = x_fused_main + bert_vote * bert_features_proj
        else:
            if isinstance(x_fused_main, torch.Tensor):
                bert_features_proj = torch.zeros_like(x_fused_main)
            else:
                bert_features_proj = torch.zeros_like(x_proj)
            x_fused = x_fused_main + torch.sigmoid(self.bert_vote_weight) * bert_features_proj
        # 池化
        if hasattr(x_fused, 'dim') and x_fused.dim() == 3:
            last_out = x_fused[:, 0, :]
        else:
            last_out = x_fused
        last_out = self.se_block(last_out)
        last_out = self.dropout_layer(last_out)
        # Mixup（可选）
        if mixup and targets is not None:
            last_out, y_a, y_b, lam = mixup_data(last_out, targets, alpha=mixup_alpha)
        else:
            y_a = y_b = lam = None
        mlp_out = self.mlp(last_out)
        logits = self.output_fc(mlp_out)
        # 对比学习分支
        contrastive_loss = None
        if contrastive and contrastive_pairs is not None:
            z_i_list, z_j_list = [], []
            for (seq1, seq2) in contrastive_pairs:
                out1 = self.forward(seq1, seq_coords=None, brand_names=None, brand_types=None, contrastive=False)
                out2 = self.forward(seq2, seq_coords=None, brand_names=None, brand_types=None, contrastive=False)
                z_i_list.append(out1)
                z_j_list.append(out2)
            z_i = torch.cat(z_i_list, dim=0)
            z_j = torch.cat(z_j_list, dim=0)
            contrastive_loss = nt_xent_loss(z_i, z_j, temperature=contrastive_temperature)
        if mixup and targets is not None:
            return logits, y_a, y_b, lam, contrastive_loss
        if contrastive:
            return logits, contrastive_loss
        return logits


# ===== GNN模块（GCN/GAT） =====
from torch_geometric.nn import GCNConv, GATConv
HAS_PYG = True
class SimpleGNNBlock(nn.Module):
    """简单的GCN+GAT混合GNN块，适合小数据特征提取"""
    def __init__(self, in_dim, out_dim, use_gat=True, heads=2, dropout=0.1):
        super().__init__()
        self.use_gat = use_gat and HAS_PYG
        if HAS_PYG:
            self.gcn = GCNConv(in_dim, out_dim)
            if self.use_gat:
                self.gat = GATConv(out_dim, out_dim, heads=heads, dropout=dropout, concat=False)
        else:
            self.gcn = nn.Linear(in_dim, out_dim)
            self.gat = nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        # x: (N, F), edge_index: (2, E)
        x = self.gcn(x, edge_index) if HAS_PYG else self.gcn(x)
        x = self.norm(x)
        x = self.act(x)
        if self.use_gat and HAS_PYG:
            x = self.gat(x, edge_index)
            x = self.norm(x)
            x = self.act(x)
        x = self.dropout(x)
        return x
