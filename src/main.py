import torch
from data_preprocessing import load_all_data
from model import NextGridPredictor
from train import train_model
from evaluate import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_csv = 'data/train_data.csv'
test_csv = 'data/test_data.csv'
grid_csv = 'data/grid_coordinates-2.csv'

print("加载并处理数据...")
train_set, val_set, test_set, num_classes, grid2idx = load_all_data(
    train_csv, test_csv, grid_csv, val_size=0.2
)

print(f"训练样本数: {len(train_set)}, 验证样本数: {len(val_set)}, 测试样本数: {len(test_set)}")
model = NextGridPredictor(num_classes=num_classes)

print("开始训练...")
model = train_model(model, train_set, val_set, device, num_epochs=40, batch_size=32, lr=1e-3, patience=5)

print("在测试集上评估...")
acc_k, mrr = evaluate_model(model, test_set, device)
print(f"Test_MRR: {mrr:.4f}")
for k in [1, 5, 10]:
    print(f"Test_Acc@{k}: {acc_k[k]:.3f}")
