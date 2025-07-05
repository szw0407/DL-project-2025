import torch
import numpy as np

@torch.no_grad()
def evaluate_model(model, dataset, device, k_list=[1, 5, 10]):
    model.eval()
    acc_k = {k: 0 for k in k_list}
    mrr_sum = 0
    total = 0
    for seq_ids, seq_coords, seq_poi, targets in batcher(dataset, 64):
        seq_ids, seq_coords, seq_poi, targets = (
            seq_ids.to(device), seq_coords.to(device), seq_poi.to(device), targets.to(device)
        )
        logits = model(seq_ids, seq_coords, seq_poi)
        topk = torch.topk(logits, max(k_list), dim=1).indices.cpu().numpy()
        targets_np = targets.cpu().numpy()
        for i, target in enumerate(targets_np):
            rank = np.where(topk[i] == target)[0]
            if len(rank) > 0:
                rank = rank[0] + 1
                mrr_sum += 1.0 / rank
                for k in k_list:
                    if rank <= k:
                        acc_k[k] += 1
            total += 1
    mrr = mrr_sum / total if total else 0
    acc_k = {k: acc_k[k]/total for k in k_list}
    return acc_k, mrr

def batcher(samples, batch_size=64):
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        maxlen = max(len(x[0]) for x in batch)
        seq_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
        seq_coords = np.zeros((len(batch), maxlen, 2), dtype=np.float32)
        seq_poi = np.zeros((len(batch), maxlen, 10), dtype=np.float32)
        for j, (pidx, pcoord, ppoi, _) in enumerate(batch):
            L = len(pidx)
            seq_ids[j, -L:] = pidx
            seq_coords[j, -L:, :] = pcoord
            seq_poi[j, -L:, :] = ppoi
        targets = np.array([x[3] for x in batch], dtype=np.int64)
        yield (
            torch.from_numpy(seq_ids),
            torch.from_numpy(seq_coords),
            torch.from_numpy(seq_poi),
            torch.from_numpy(targets)
        )
