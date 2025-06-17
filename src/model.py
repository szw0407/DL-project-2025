import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GCNConv, GATConv


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


class SimpleGNNBlock(nn.Module):
    """简单的GCN+GAT混合GNN块，适合小数据特征提取"""
    def __init__(self, in_dim, out_dim, use_gat=True, heads=2, dropout=0.1):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.use_gat = use_gat
        if use_gat:
            self.gat = GATConv(out_dim, out_dim, heads=heads, dropout=dropout, concat=False)
        else:
            self.gat = nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        if self.use_gat:
            x = self.gat(x, edge_index)
            x = self.norm(x)
            x = self.act(x)
        x = self.dropout(x)
        return x


class StorePredictionModel(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=8,
        coord_dim=4,
        trans_dim=12,
        n_layers=1,
        n_heads=2,
        dropout=0.25,
        bert_model_name="bert-base-chinese",
        use_bert=True,  # 默认开启BERT
        bert_feature_dim=768,
        mlp_ratio=1.2,
        brand_type_num=16,
        brand_type_embed_dim=4,
        brand_type_to_id=None,
        use_gnn=True,  # 新增：是否使用GNN
        gnn_dim=16,
        gnn_layers=1,
        gnn_heads=2,
        gnn_dropout=0.1,
        *args, **kwargs
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.trans_dim = trans_dim
        self.use_bert = use_bert
        self.bert_feature_dim = bert_feature_dim
        self.brand_type_num = brand_type_num
        self.brand_type_embed_dim = brand_type_embed_dim
        self.brand_type_to_id = brand_type_to_id if brand_type_to_id is not None else {}
        self.use_gnn = use_gnn
        # Embedding
        self.id_embedding = nn.Embedding(num_classes, embed_dim)
        # 坐标特征编码
        if coord_dim > 0:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, coord_dim * 2),
                nn.SiLU(),
                nn.Linear(coord_dim * 2, coord_dim),
                nn.LayerNorm(coord_dim),
            )
        else:
            self.coord_encoder = None
        # brand_type embedding + MLP
        self.brand_type_embedding = nn.Embedding(brand_type_num, brand_type_embed_dim)
        self.brand_type_mlp = nn.Sequential(
            nn.Linear(brand_type_embed_dim, 8),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # BERT（可选，默认关闭）
        if use_bert:
            try:
                self.bert_model = BertModel.from_pretrained(bert_model_name)
                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
                for p in self.bert_model.parameters():
                    p.requires_grad = True
            except:
                print(f"警告: 无法加载BERT模型 {bert_model_name}，将禁用BERT特征")
                self.use_bert = False
        if self.use_bert:
            self.bert_proj = nn.Sequential(
                nn.Linear(self.bert_feature_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        # 融合后输入维度
        in_dim = embed_dim + (coord_dim if coord_dim > 0 else 0) + (embed_dim if use_bert else 0) + 4
        self.input_proj = nn.Linear(in_dim, trans_dim)
        self.input_norm = nn.LayerNorm(trans_dim)
        # TransformerEncoder
        self.transformer = nn.Sequential(
            *[
                nn.Sequential(
                    TransformerBlock(trans_dim, n_heads, mlp_ratio, dropout, drop_path=0.05),
                    nn.LayerNorm(trans_dim)
                )
                for _ in range(n_layers)
            ]
        )
        # GRU分支
        self.rnn_dim = trans_dim
        # 修正：GRU输入应为多尺度卷积输出的维度（trans_dim），而非原始in_dim
        self.rnn = nn.GRU(
            trans_dim, self.rnn_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.rnn_norm = nn.LayerNorm(self.rnn_dim)
        # GNN模块
        if self.use_gnn:
            gnn_blocks = []
            # 修正：GNN分支输入应为trans_dim
            for i in range(gnn_layers):
                gnn_blocks.append(SimpleGNNBlock(trans_dim if i == 0 else gnn_dim, gnn_dim, heads=gnn_heads, dropout=gnn_dropout))
            self.gnn_blocks = nn.ModuleList(gnn_blocks)
            self.gnn_norm = nn.LayerNorm(gnn_dim)
        # 融合方式：主干分支加权平均（含GNN）
        n_fuse = 2 + int(self.use_gnn)
        self.fuse_weights = nn.Parameter(torch.ones(n_fuse))
        # 融合方式升级：concat+MLP，融合Transformer、GRU、GNN、BERT、brand_type等特征，提升表达能力
        # bert_proj输出维度为embed_dim，brand_type_embedding为brand_type_embed_dim
        concat_dim = trans_dim + trans_dim + (gnn_dim if use_gnn else 0) + (embed_dim if use_bert else 0) + brand_type_embed_dim
        self.fuse_mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim * 2),
            nn.LayerNorm(concat_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim * 2, concat_dim),
            nn.LayerNorm(concat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.LayerNorm(concat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim, concat_dim // 2),
            nn.LayerNorm(concat_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_fc = nn.Linear(concat_dim // 2, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        self.se_block = SEBlock(concat_dim)  # 修正为concat_dim
        self.feature_dropblock = FeatureDropBlock(drop_prob=0.15)
        # BERT分支更好利用：解冻后半层参数
        if self.use_bert:
            bert_layers = list(self.bert_model.encoder.layer)
            for i, layer in enumerate(bert_layers):
                if i >= len(bert_layers) // 2:
                    for p in layer.parameters():
                        p.requires_grad = True
        # 移除全局池化Attention层
        # self.global_attn = nn.MultiheadAttention(concat_dim, n_heads, dropout=dropout, batch_first=True)
        # 增加DropBlock到融合特征
        self.fuse_dropblock = FeatureDropBlock(drop_prob=0.15)
        # 增加随机特征置零（stochastic feature zeroing）
        self.fuse_feature_zero_prob = 0.1

        # 多尺度1D卷积特征提取（专注序列信息）
        self.conv_kernels = [1, 3, 5]
        conv_in_dim = embed_dim + (coord_dim if coord_dim > 0 else 0) + (embed_dim if use_bert else 0) + 4
        self.seq_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(conv_in_dim, trans_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(trans_dim),
                nn.SiLU()
            ) for k in self.conv_kernels
        ])
        self.seq_conv_rescale = nn.Linear(len(self.conv_kernels) * trans_dim, trans_dim)

    def add_bert_input_noise(self, input_ids, noise_prob=0.08):
        # 随机将部分token替换为[MASK]，以增强泛化
        if self.training and noise_prob > 0:
            mask_token_id = self.bert_tokenizer.mask_token_id
            rand_mask = (torch.rand_like(input_ids.float()) < noise_prob)
            input_ids = input_ids.clone()
            input_ids[rand_mask] = mask_token_id
        return input_ids

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
                max_length=64,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        # 在输入层加噪声
        input_ids = self.add_bert_input_noise(input_ids, noise_prob=0.08)
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
        id_emb = self.id_embedding(seq_ids)
        features = [id_emb]
        if self.coord_encoder is not None and seq_coords is not None:
            coords_flat = seq_coords.reshape(batch_size * seq_len, 2)
            coord_emb_flat = self.coord_encoder(coords_flat)
            coord_emb = coord_emb_flat.view(batch_size, seq_len, self.coord_dim)
            features.append(coord_emb)
        bert_features = None
        if self.use_bert and brand_names is not None and brand_types is not None:
            bert_features = self.extract_bert_features(brand_names, brand_types, seq_len)
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
        # 多尺度卷积特征提取
        x_conv_in = x.permute(0, 2, 1)  # (B, F, T)
        conv_outs = [conv(x_conv_in) for conv in self.seq_convs]
        x_conv = torch.cat(conv_outs, dim=1)  # (B, C*k, T)
        x_conv = x_conv.permute(0, 2, 1)  # (B, T, C*k)
        x_conv = self.seq_conv_rescale(x_conv)
        # 残差连接
        x = x_conv + self.input_proj(x)
        x_proj = self.input_norm(x)
        x_proj = self.feature_dropblock(x_proj)
        # Transformer分支
        x_trans = self.transformer(x_proj)
        if isinstance(x_trans, (list, tuple)):
            x_trans = x_trans[-1]
        # GRU分支
        x_rnn, _ = self.rnn(x)
        x_rnn = self.rnn_norm(x_rnn)
        # GNN分支
        x_gnn = None
        if self.use_gnn and edge_index is not None:
            # 修正：GNN分支输入应为x_proj，维度为trans_dim
            x_gnn_in = x_proj.reshape(-1, x_proj.shape[-1])
            for gnn_block in self.gnn_blocks:
                x_gnn_in = gnn_block(x_gnn_in, edge_index)
            x_gnn = x_gnn_in.view(x_proj.shape[0], x_proj.shape[1], -1)
            x_gnn = self.gnn_norm(x_gnn)
        # 融合（concat+MLP，含GNN/BERT/brand_type）
        concat_list = [
            x_trans[:, 0, :] if x_trans.dim() == 3 else x_trans,
            x_rnn[:, 0, :] if x_rnn.dim() == 3 else x_rnn
        ]
        if self.use_gnn and x_gnn is not None:
            concat_list.append(x_gnn[:, 0, :] if x_gnn.dim() == 3 else x_gnn)
        if bert_features is not None:
            concat_list.append(bert_features[:, 0, :] if bert_features.dim() == 3 else bert_features)
        if brand_type_ids is not None:
            brand_type_emb = self.brand_type_embedding(brand_type_ids)
            concat_list.append(brand_type_emb)
        # ...特征融合...
        x_fused = torch.cat(concat_list, dim=-1)
        # 融合特征加DropBlock
        x_fused = self.fuse_dropblock(x_fused)
        # 融合特征随机置零部分通道
        if self.training and self.fuse_feature_zero_prob > 0:
            mask = (torch.rand_like(x_fused) > self.fuse_feature_zero_prob).float()
            x_fused = x_fused * mask
        # 融合后MLP
        x_fused = self.fuse_mlp(x_fused)
        last_out = self.se_block(x_fused)
        last_out = nn.LayerNorm(last_out.shape[-1]).to(last_out.device)(last_out)
        last_out = self.dropout_layer(last_out)
        # Mixup
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
