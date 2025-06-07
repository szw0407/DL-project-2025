import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

# 确保可以从同级目录导入模块
try:
    from data_preprocessing import load_data
    from model import StorePredictionModel
    from evaluate import compute_metrics
    from train import train_model  # 导入完整的 train_model 函数
except ImportError:
    print("确保 data_preprocessing.py, model.py, evaluate.py 和 train.py 在同一目录下或 PYTHONPATH 中。")
    raise

# --- 全局配置 (Optuna搜索用) ---
N_TRIALS = 30  # Optuna 试验次数
TIMEOUT_SECONDS = 1800  # Optuna 搜索总超时时间 (例如30分钟)
EPOCHS_PER_TRIAL = 15  # 每次试验中训练的epoch数 (可以少于完整训练，以加速搜索)
PATIENCE_PER_TRIAL = 3  # 每次试验中的早停耐心值
USE_CUDA_IF_AVAILABLE_OPTUNA = True  # Optuna 搜索时是否尝试使用GPU

# 获取路径信息
current_script_dir_optuna = os.path.dirname(os.path.abspath(__file__))
project_root_dir_optuna = os.path.dirname(current_script_dir_optuna)
data_dir_optuna = os.path.join(project_root_dir_optuna, 'data')
optuna_best_model_path = os.path.join(project_root_dir_optuna, "optuna_best_model.pth")
optuna_final_model_path = os.path.join(project_root_dir_optuna, "optuna_trained_final_model.pth")

# --- 1. 加载数据 (一次性加载，供所有trial使用) ---
print("--- Optuna 超参数搜索脚本 ---")
print("正在加载和预处理数据 (仅一次)...")
train_csv_optuna = os.path.join(data_dir_optuna, "train_data.csv")
test_csv_optuna = os.path.join(data_dir_optuna, "test_data.csv")
grid_csv_optuna = os.path.join(data_dir_optuna, "grid_coordinates.csv")

data_load_result_optuna = load_data(train_csv_optuna, test_csv_optuna, grid_csv_optuna)
if data_load_result_optuna[0] is None:
    print("数据加载失败，Optuna搜索无法进行。脚本终止。")
    exit()

train_data_optuna, val_data_optuna, test_data_optuna, num_classes_optuna, coords_map_optuna, grid_to_index_optuna = data_load_result_optuna

if not train_data_optuna or not val_data_optuna:
    print("错误：Optuna搜索需要训练数据和验证数据。脚本终止。")
    exit()
if num_classes_optuna == 0:
    print("错误：类别数量为0。脚本终止。")
    exit()

print(f"数据加载完成。类别: {num_classes_optuna}, 训练样本: {len(train_data_optuna)}, 验证样本: {len(val_data_optuna)}")


# --- 2. 定义 Optuna 的优化目标函数 ---
def objective(trial):
    """
    Optuna 的目标函数，用于评估一组超参数。
    """
    device_optuna = torch.device('cuda' if USE_CUDA_IF_AVAILABLE_OPTUNA and torch.cuda.is_available() else 'cpu')

    # --- 超参数搜索空间 ---
    embed_dim = trial.suggest_int("embed_dim", 16, 64, step=8)  # 嵌入维度
    # coord_dim: 0 (不用), 或 4 到 16
    use_coords_flag = trial.suggest_categorical("use_coords", [True, False])
    coord_dim = trial.suggest_int("coord_dim_value", 4, 16, step=4) if use_coords_flag and coords_map_optuna else 0

    hidden_dim = trial.suggest_int("hidden_dim", 32, 128, step=16)  # LSTM隐藏层大小
    lstm_layers = trial.suggest_int("lstm_layers", 1, 2)  # LSTM层数
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)  # Dropout比例
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # 学习率 (对数均匀)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)  # 权重衰减 (对数均匀)

    trial_model_save_path = f"temp_trial_{trial.number}_model.pth"  # 临时模型保存路径

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(
        f"  参数: embed={embed_dim}, coord={coord_dim}, hidden={hidden_dim}, layers={lstm_layers}, dropout={dropout:.3f}, lr={lr:.5f}, wd={weight_decay:.6f}")
    print(f"  使用设备: {device_optuna}")

    # --- 内部训练循环 (与 train.py 中的 train_model 类似，但更精简用于快速评估) ---
    # 此处我们直接调用修改后的 train_model，它已经包含了早停和模型保存逻辑
    # 注意：为了 Optuna 的剪枝，train_model 内部需要能访问 trial 对象或返回中间结果
    # 为了简化，我们这里让 train_model 完成训练，然后返回最佳验证MRR
    # 如果需要更细粒度的剪枝，需要修改 train_model 或在此处重写训练循环

    # 实际 coord_dim 取决于 coords_map_optuna 是否存在以及配置
    actual_coord_dim_for_trial = coord_dim if coords_map_optuna and coord_dim > 0 else 0

    model_for_trial = StorePredictionModel(
        num_classes_optuna, embed_dim, actual_coord_dim_for_trial, hidden_dim, lstm_layers, dropout
    )
    model_for_trial.to(device_optuna)
    optimizer_trial = optim.Adam(model_for_trial.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_trial = nn.CrossEntropyLoss()

    best_trial_val_mrr = -1.0
    epochs_no_improve_trial = 0

    if not train_data_optuna:  # 以防万一
        print("Optuna Trial: 训练数据为空，返回最差结果。")
        return -1.0

    for epoch in range(EPOCHS_PER_TRIAL):
        epoch_start_time_trial = time.time()
        model_for_trial.train()

        current_train_data_trial = list(train_data_optuna)
        np.random.shuffle(current_train_data_trial)
        total_loss_trial = 0.0

        for prefix_idx, prefix_coords, target_idx_val in current_train_data_trial:
            seq_ids = torch.tensor(prefix_idx, dtype=torch.long).unsqueeze(0).to(device_optuna)

            if actual_coord_dim_for_trial > 0 and prefix_coords:
                seq_coords_t = torch.tensor(prefix_coords, dtype=torch.float).unsqueeze(0).to(device_optuna)
            else:
                seq_coords_t = None

            target_t = torch.tensor([target_idx_val], dtype=torch.long).to(device_optuna)

            optimizer_trial.zero_grad()
            logits_t = model_for_trial(seq_ids, seq_coords_t) if seq_coords_t is not None else model_for_trial(seq_ids)
            loss_t = criterion_trial(logits_t, target_t)
            loss_t.backward()
            optimizer_trial.step()
            total_loss_trial += loss_t.item()

        avg_loss_trial = total_loss_trial / len(current_train_data_trial) if current_train_data_trial else 0

        # --- 验证评估 ---
        if val_data_optuna:
            acc_k_trial, mrr_trial = compute_metrics(model_for_trial, val_data_optuna, device_optuna)
            print(
                f"  Trial {trial.number}, Epoch {epoch + 1}/{EPOCHS_PER_TRIAL} - Loss: {avg_loss_trial:.4f} - Val MRR: {mrr_trial:.4f} (Best: {best_trial_val_mrr:.4f})")

            if mrr_trial > best_trial_val_mrr:
                best_trial_val_mrr = mrr_trial
                epochs_no_improve_trial = 0
                # 可选: 保存此trial的最佳模型状态，但不一定必要，因为Optuna关心的是最终指标
            else:
                epochs_no_improve_trial += 1

            # --- Optuna 剪枝 ---
            trial.report(mrr_trial, epoch)  # 报告中间值给Optuna
            if trial.should_prune():
                if os.path.exists(trial_model_save_path): os.remove(trial_model_save_path)  # 清理
                print(f"  Trial {trial.number} pruned at epoch {epoch + 1}.")
                raise optuna.TrialPruned()  # 触发剪枝

            if epochs_no_improve_trial >= PATIENCE_PER_TRIAL:
                print(f"  Trial {trial.number} early stopping at epoch {epoch + 1}.")
                break  # 自定义早停
        else:  # 无验证数据，无法进行有意义的超参搜索
            print("警告: Optuna Trial 中无验证数据，无法评估。返回最差结果。")
            return -1.0  # 或者抛出错误

    if os.path.exists(trial_model_save_path): os.remove(trial_model_save_path)  # 清理临时文件
    print(f"--- Trial {trial.number} Finished. Best Val MRR for this trial: {best_trial_val_mrr:.4f} ---")
    return best_trial_val_mrr  # 返回此trial的最佳验证MRR


# --- 3. 创建Optuna研究并开始调参 ---
if __name__ == "__main__":
    # 使用中位数剪枝器
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=EPOCHS_PER_TRIAL // 3, interval_steps=1)

    study = optuna.create_study(direction="maximize", pruner=pruner)

    print(f"\n开始 Optuna 超参数搜索 (Trials: {N_TRIALS}, Timeout: {TIMEOUT_SECONDS}s)...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        print("\nOptuna 搜索被用户中断。")
    except Exception as e:
        print(f"\nOptuna 搜索过程中发生错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Optuna 超参数搜索完成 ---")
    if study.best_trial:
        print("最佳 Trial:")
        print(f"  Value (Maximized Val MRR): {study.best_value:.4f}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        best_params_from_optuna = study.best_params

        # --- 4. 使用最佳超参数重新训练模型 ---
        # 可以选择合并训练集和验证集进行最终训练，或仅用训练集训练更久
        print("\n使用 Optuna 找到的最佳超参数重新训练最终模型...")

        # 准备参数给 train_model
        final_embed_dim = best_params_from_optuna.get("embed_dim")
        final_use_coords = best_params_from_optuna.get("use_coords", False)
        final_coord_dim_val = best_params_from_optuna.get("coord_dim_value", 0)
        final_coord_dim_config = final_coord_dim_val if final_use_coords else 0

        final_hidden_dim = best_params_from_optuna.get("hidden_dim")
        final_lstm_layers = best_params_from_optuna.get("lstm_layers")
        final_dropout = best_params_from_optuna.get("dropout")
        final_lr = best_params_from_optuna.get("lr")
        final_weight_decay = best_params_from_optuna.get("weight_decay")

        # 合并训练和验证数据进行最终训练
        full_train_data_optuna = train_data_optuna + val_data_optuna
        print(f"将使用 {len(full_train_data_optuna)} 个样本 (原训练集+验证集) 进行最终模型训练。")

        final_model = train_model(
            full_train_data_optuna,
            [],  # 不再使用单独的验证集进行此轮训练的早停，而是训练固定或更多epochs
            num_classes_optuna,
            coords_map_optuna,
            embed_dim=final_embed_dim,
            coord_dim_config=final_coord_dim_config,
            hidden_dim=final_hidden_dim,
            lstm_layers=final_lstm_layers,
            dropout=final_dropout,
            lr=final_lr,
            weight_decay=final_weight_decay,
            epochs=max(EPOCHS_PER_TRIAL, 30),  # 训练更久一些，例如50轮
            patience=10,  # 这里的patience可能不起作用，因为val_data为空
            device_name='cuda' if USE_CUDA_IF_AVAILABLE_OPTUNA else 'cpu',
            model_save_path=optuna_final_model_path  # 保存最终训练的模型
        )

        # --- 5. 在测试集上评估最终训练的模型 ---
        if final_model and test_data_optuna:
            print("\n在测试集上评估 Optuna 调优后的最终模型...")
            final_device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE_OPTUNA and torch.cuda.is_available() else 'cpu')
            final_model.to(final_device)

            test_acc_k_final, test_mrr_final = compute_metrics(final_model, test_data_optuna, final_device)
            print(f"\n--- Optuna 最终模型测试结果 ---")
            print(f"  测试集 MRR: {test_mrr_final:.4f}")
            for k_val, acc_val in test_acc_k_final.items():
                print(f"  测试集 Acc@{k_val}: {acc_val:.4f}")
            print(f"最终模型已保存至: {optuna_final_model_path}")
        elif not test_data_optuna:
            print("\n无测试数据可供评估最终模型。")
        else:
            print("\n最终模型训练失败或未生成，无法在测试集上评估。")

    else:
        print("Optuna 未能找到任何成功的 Trial。")

    print("\n--- Optuna 脚本执行完毕 ---")

