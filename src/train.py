import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evaluate import evaluate_model

def batch_iter(samples, batch_size=32, shuffle=True):
    idxs = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(samples), batch_size):
        batch = [samples[j] for j in idxs[i:i+batch_size]]
        maxlen = max(len(x[0]) for x in batch)
        # pad to maxlen
        seq_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
        seq_coords = np.zeros((len(batch), maxlen, 2), dtype=np.float32)
        seq_poi = np.zeros((len(batch), maxlen, 10), dtype=np.float32)
        for i, (pidx, pcoord, ppoi, _) in enumerate(batch):
            L = len(pidx)
            seq_ids[i, -L:] = pidx
            seq_coords[i, -L:, :] = pcoord
            seq_poi[i, -L:, :] = ppoi
        targets = np.array([x[3] for x in batch], dtype=np.int64)
        yield (
            torch.from_numpy(seq_ids),
            torch.from_numpy(seq_coords),
            torch.from_numpy(seq_poi),
            torch.from_numpy(targets)
        )

def train_model(model, train_set, val_set, device, num_epochs=40, batch_size=32, lr=1e-3, patience=5):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_mrr = -1
    best_state = None
    no_improve = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        losses = []
        for seq_ids, seq_coords, seq_poi, targets in batch_iter(train_set, batch_size):
            seq_ids, seq_coords, seq_poi, targets = (
                seq_ids.to(device), seq_coords.to(device), seq_poi.to(device), targets.to(device)
            )
            opt.zero_grad()
            logits = model(seq_ids, seq_coords, seq_poi)
            loss = criterion(logits, targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_acc_k, val_mrr = evaluate_model(model, val_set, device)
        print(f"Epoch {epoch} | loss={np.mean(losses):.4f} | Val_MRR={val_mrr:.4f} | Acc@1={val_acc_k[1]:.3f} Acc@5={val_acc_k[5]:.3f} Acc@10={val_acc_k[10]:.3f}")
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stop triggered.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
