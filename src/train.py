import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

try:
    from data_preprocessing import load_data
    from model import StorePredictionModel
    from evaluate import compute_metrics
except ImportError:
    print("确保 data_preprocessing.py, model.py, 和 evaluate.py 在同一目录下或 PYTHONPATH 中。")
    raise

DEFAULT_EMBED_DIM = 32
DEFAULT_COORD_DIM = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LSTM_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5

def train_model(train_data, val_data, num_classes, coords_map_for_model,
                embed_dim=DEFAULT_EMBED_DIM, coord_dim_config=DEFAULT_COORD_DIM,
                hidden_dim=DEFAULT_HIDDEN_DIM, lstm_layers=DEFAULT_LSTM_LAYERS,
                dropout=DEFAULT_DROPOUT, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, device_name='cpu',
                model_save_path="best_model.pth"):
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"使用设备: {device}")

    actual_coord_dim = coord_dim_config if coords_map_for_model and coord_dim_config > 0 else 0
    if actual_coord_dim == 0 and coord_dim_config > 0:
        print(f"警告: 配置的 coord_dim 为 {coord_dim_config}，但坐标数据缺失或配置禁用，实际将不使用坐标嵌入。")

    model = StorePredictionModel(num_classes, embed_dim, actual_coord_dim, hidden_dim, lstm_layers, dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_mrr = -1.0
    best_model_state = None
    epochs_no_improve = 0

    print(f"\n开始训练...")
    print(f"  总 Epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  学习率: {lr}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  模型参数: embed_dim={embed_dim}, actual_coord_dim={actual_coord_dim}, hidden_dim={hidden_dim}, lstm_layers={lstm_layers}, dropout={dropout}")

    if not train_data:
        print("错误: 训练数据为空，无法开始训练。")
        return model

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        current_train_data = list(train_data)
        np.random.shuffle(current_train_data)

        total_loss = 0.0
        processed_samples = 0

        for i, (prefix_idx, prefix_coords, target_idx_val) in enumerate(current_train_data):
            seq_ids = torch.tensor(prefix_idx, dtype=torch.long).unsqueeze(0).to(device)
            if actual_coord_dim > 0 and prefix_coords:
                seq_coords_tensor = torch.tensor(prefix_coords, dtype=torch.float).unsqueeze(0).to(device)
            else:
                seq_coords_tensor = None
            target = torch.tensor([target_idx_val], dtype=torch.long).to(device)
            optimizer.zero_grad()
            logits = model(seq_ids, seq_coords_tensor) if seq_coords_tensor is not None else model(seq_ids)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            processed_samples += 1

        avg_loss = total_loss / processed_samples if processed_samples > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time

        # 修改为 [1, 5, 10]
        if val_data:
            acc_k, mrr = compute_metrics(model, val_data, device, k_values=[1, 5, 10])
            print(
                f"Epoch {epoch + 1}/{epochs} - Duration: {epoch_duration:.2f}s - Loss: {avg_loss:.4f} - Val MRR: {mrr:.4f} - Val Acc@1: {acc_k[1]:.4f}, Acc@5: {acc_k[5]:.4f}, Acc@10: {acc_k[10]:.4f}"
            )
            if mrr > best_val_mrr:
                best_val_mrr = mrr
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
                if model_save_path:
                    torch.save(best_model_state, model_save_path)
                    print(f"  验证集MRR提升，最佳模型已保存至: {model_save_path}")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  连续 {patience} 个 epochs 验证集MRR未提升，触发早停。")
                break
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Duration: {epoch_duration:.2f}s - Loss: {avg_loss:.4f} - (无验证数据)")
            best_model_state = model.state_dict().copy()
            if model_save_path:
                torch.save(best_model_state, model_save_path)

    print("训练完成。")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载验证集上MRR最佳的模型 (MRR: {best_val_mrr:.4f})。")
    else:
        print("警告: 未能确定最佳模型状态 (可能因为没有验证数据或训练提前结束)。返回当前模型。")
    return model

if __name__ == "__main__":
    USE_CUDA_IF_AVAILABLE = True
    MODEL_SAVE_FILENAME = "store_predictor_best.pth"

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    data_dir = os.path.join(project_root_dir, 'data')
    model_output_path = os.path.join(project_root_dir, MODEL_SAVE_FILENAME)
    train_csv_file = os.path.join(data_dir, "train_data.csv")
    test_csv_file = os.path.join(data_dir, "test_data.csv")
    grid_csv_file = os.path.join(data_dir, "grid_coordinates.csv")

    print("--- 门店选址预测模型训练脚本 ---")
    print(f"数据文件路径:")
    print(f"  训练集: {train_csv_file}")
    print(f"  测试集: {test_csv_file}")
    print(f"  网格坐标: {grid_csv_file}")
    print(f"模型将保存至: {model_output_path}")

    print("\n正在加载和预处理数据...")
    data_load_result = load_data(train_csv_file, test_csv_file, grid_csv_file)
    if data_load_result[0] is None:
        print("数据加载失败，请检查之前的错误信息。脚本终止。")
        exit()

    train_samples, val_samples, test_samples, num_total_classes, coords_info, grid_to_index_map = data_load_result

    if not train_samples and not val_samples:
        print("错误：没有可用的训练或验证数据。脚本终止。")
        exit()
    if num_total_classes == 0:
        print("错误：类别数量为0。脚本终止。")
        exit()

    print(f"数据加载完成。类别总数: {num_total_classes}")
    print(f"训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}, 测试样本数: {len(test_samples)}")

    trained_model = train_model(
        train_samples,
        val_samples,
        num_total_classes,
        coords_info,
        embed_dim=DEFAULT_EMBED_DIM,
        coord_dim_config=DEFAULT_COORD_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lstm_layers=DEFAULT_LSTM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        device_name='cuda' if USE_CUDA_IF_AVAILABLE else 'cpu',
        model_save_path=model_output_path
    )

    if trained_model and test_samples:
        print("\n在测试集上评估最终模型...")
        device_for_eval = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
        trained_model.to(device_for_eval)
        test_acc_k, test_mrr = compute_metrics(trained_model, test_samples, device_for_eval, k_values=[1, 5, 10])
        print(f"\n--- 测试集评估结果 ---")
        print(f"  测试集 MRR: {test_mrr:.4f}")
        for k_val in [1, 5, 10]:
            print(f"  测试集 Acc@{k_val}: {test_acc_k[k_val]:.4f}")
    elif not test_samples:
        print("\n无测试数据可供评估。")
    else:
        print("\n模型训练失败，无法在测试集上评估。")

    print("\n--- 脚本执行完毕 ---")


