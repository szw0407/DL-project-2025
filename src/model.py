import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
    def forward(self, seq_ids):
        emb = self.embed(seq_ids)
        out, (h, _) = self.lstm(emb)
        return out[:, -1, :]

class MLPEncoder(nn.Module):
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

class NextGridPredictor(nn.Module):
    def __init__(self, num_classes, embed_dim=32, lstm_hidden=64, lstm_layers=1,
                 coord_dim=2, poi_dim=10, coord_out_dim=16, poi_out_dim=16, fusion_dim=64, dropout=0.1):
        super().__init__()
        self.seq_encoder = SeqEncoder(num_classes, embed_dim, lstm_hidden, lstm_layers, dropout)
        self.coord_encoder = MLPEncoder(coord_dim, coord_out_dim)
        self.poi_encoder = MLPEncoder(poi_dim, poi_out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + coord_out_dim + poi_out_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
    def forward(self, seq_ids, seq_coords, seq_poi):
        # 输入: (batch, seq_len, dim)
        seq_out = self.seq_encoder(seq_ids)               # (batch, lstm_hidden)
        coords_out = self.coord_encoder(seq_coords.mean(dim=1)) # (batch, coord_out_dim)
        poi_out = self.poi_encoder(seq_poi.mean(dim=1))   # (batch, poi_out_dim)
        x = torch.cat([seq_out, coords_out, poi_out], dim=-1)
        f = self.fusion(x)
        logits = self.classifier(f)
        return logits
