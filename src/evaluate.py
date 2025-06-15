import torch
import numpy as np

def compute_metrics(model, data_loader, device, k_values=[1, 5, 10]):
    """
    计算评估指标：Top-K准确率和MRR。
    参数:
        model (torch.nn.Module): 已训练的模型。
        data_loader (iterable): 数据加载器，每次迭代返回 (seq_ids, seq_coords, true_idx)。
        device (torch.device): 'cuda' 或 'cpu'。
        k_values (list of int): 计算Top-K准确率的K值列表。
    返回:
        tuple: (acc_at_k, mrr)
            acc_at_k (dict): Top-K准确率，键为K值。
            mrr (float): 平均倒数排名。
    """
    model.eval()
    correct_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            prefix_idx, prefix_coords, true_idx_val = batch
            seq_ids = torch.tensor(prefix_idx, dtype=torch.long).unsqueeze(0).to(device)

            if getattr(model, "coord_dim", 0) > 0 and prefix_coords:
                seq_coords_tensor = torch.tensor(prefix_coords, dtype=torch.float).unsqueeze(0).to(device)
            else:
                seq_coords_tensor = None

            logits = model(seq_ids, seq_coords_tensor) if seq_coords_tensor is not None else model(seq_ids)
            probs = logits.softmax(dim=1)
            prob_values, pred_indices_tensor = probs.squeeze(0).sort(descending=True)
            pred_indices_list = pred_indices_tensor.tolist()

            for k in k_values:
                if true_idx_val in pred_indices_list[:k]:
                    correct_at_k[k] += 1

            try:
                rank = pred_indices_list.index(true_idx_val) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

            total_samples += 1

    if total_samples == 0:
        print("警告 (compute_metrics): 数据加载器为空，无法计算指标。")
        return {k: 0.0 for k in k_values}, 0.0

    acc_at_k_final = {k: correct_at_k[k] / total_samples for k in k_values}
    mrr_final = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return acc_at_k_final, mrr_final


if __name__ == '__main__':
    print("evaluate.py: 此脚本主要提供 compute_metrics 函数。")
    print("要进行独立测试，您需要：")
    print("1. 加载一个已训练的模型。")
    print("2. 准备一个数据加载器 (例如，验证集或测试集)。")
    print("3. 调用 compute_metrics(model, data_loader, device)。")

    # 示例：创建一个虚拟模型和数据进行测试
    from model import StorePredictionModel  # 假设 model.py 在同一目录下或PYTHONPATH中

    # 虚拟参数
    num_classes_mock = 10
    embed_dim_mock = 8
    coord_dim_mock = 4  # 使用坐标
    lstm_hidden_mock = 16
    device_mock = torch.device("cpu")

    # 虚拟模型
    mock_model = StorePredictionModel(num_classes_mock, embed_dim_mock, coord_dim_mock, lstm_hidden_mock).to(
        device_mock)

    # 虚拟数据加载器 (简单列表模拟)
    mock_data = [
        ([1, 2], [[0.1, 0.1], [0.2, 0.2]], 3),  # 序列 [1,2], 目标 3
        ([0, 3, 5], [[0.0, 0.0], [0.3, 0.3], [0.5, 0.5]], 6),  # 序列 [0,3,5], 目标 6
        ([7], [[0.7, 0.7]], 1)  # 序列 [7], 目标 1
    ]

    print(f"\n使用虚拟模型和数据测试 compute_metrics (K={[1, 5, 10]})...")

    # 运行评估
    try:
        acc_k_results, mrr_result = compute_metrics(mock_model, mock_data, device_mock, k_values=[1, 5, 10])

        print("\n评估结果:")
        for k_val, acc_val in acc_k_results.items():
            print(f"  Top-{k_val} Accuracy: {acc_val:.4f}")
        print(f"  MRR: {mrr_result:.4f}")
    except Exception as e:
        print(f"compute_metrics 测试时发生错误: {e}")
        import traceback

        traceback.print_exc()

