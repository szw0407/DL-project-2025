"""
优化版主程序 - GPU常驻数据，最大化GPU利用率
"""
import torch
from data_preprocessing_new import load_all_data
from optimized_model import OptimizedNextGridPredictor
from gpu_resident_train import gpu_resident_train_model, gpu_resident_evaluate_model

def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据路径
    train_csv = 'data/train_data.csv'
    test_csv = 'data/test_data.csv'
    grid_csv = 'data/grid_coordinates-2.csv'
    
    print("加载并处理数据...")
    train_set, val_set, test_set, num_classes, grid2idx = load_all_data(
        train_csv, test_csv, grid_csv, val_size=0.2
    )
    
    print(f"数据统计:")
    print(f"- 训练样本数: {len(train_set)}")
    print(f"- 验证样本数: {len(val_set)}")
    print(f"- 测试样本数: {len(test_set)}")
    print(f"- 网格总数: {num_classes}")
    
    # 创建优化版模型
    model = OptimizedNextGridPredictor(
        num_classes=num_classes,
        embed_dim=32,
        lstm_hidden=64,
        lstm_layers=2,
        coord_out_dim=16,
        poi_out_dim=16,
        brand_out_dim=64,
        fusion_dim=72,  # 增大融合维度
        dropout=0.1,
        freeze_bert=False  # 可以尝试设置为True来减少显存使用
    )
    
    model = model.to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化训练配置 - GPU常驻模式
    training_config = {
        'num_epochs': 400,
        'batch_size': 128,  # GPU常驻模式可以使用更大的batch_size
        'lr': 2e-5,  # BERT建议使用较小的学习率
        'patience': 5,
        'use_amp': True,  # 混合精度训练
        'bert_model_name': 'bert-base-chinese'  # 指定BERT模型
    }
    
    print("\nGPU常驻训练配置:")
    for key, value in training_config.items():
        print(f"- {key}: {value}")
    
    print("\n开始GPU常驻训练...")
    model = gpu_resident_train_model(
        model, train_set, val_set, device, **training_config
    )
    
    print("\n在测试集上评估...")
    acc_k, mrr = gpu_resident_evaluate_model(
        model, test_set, device, 
        bert_model_name=training_config['bert_model_name'],
        batch_size=training_config['batch_size']
    )
    
    print("\n=== 最终测试结果 ===")
    print(f"Test MRR: {mrr:.4f}")
    for k in [1, 5, 10]:
        print(f"Test Acc@{k}: {acc_k[k]:.3f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'grid2idx': grid2idx,
        'config': training_config
    }, 'gpu_resident_model.pth')
    print("\n模型已保存为 gpu_resident_model.pth")

if __name__ == '__main__':
    main()
