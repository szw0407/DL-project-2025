# 基于深度学习的网格位置预测系统

## 项目概述

这是一个基于深度学习的空间位置预测项目，旨在根据品牌的历史访问网格序列来预测下一个最可能访问的网格位置。项目使用PyTorch框架实现，融合了序列信息、空间坐标和兴趣点(POI)特征，构建了一个多模态的神经网络模型。

## 项目特点

- **多模态融合**：结合序列编码、空间坐标和POI特征
- **端到端训练**：从原始数据到模型预测的完整流程
- **空间感知**：考虑地理位置的空间密度排序
- **性能优化**：支持GPU加速训练和早停机制

## 项目结构

```
DL/
├── data/                           # 数据目录
│   ├── train_data.csv             # 训练数据
│   ├── test_data.csv              # 测试数据
│   ├── grid_coordinates.csv       # 原始网格坐标数据
│   └── grid_coordinates-2.csv     # 增强的网格坐标数据（含POI特征）
├── src/                           # 源代码目录
│   ├── main.py                    # 主程序入口
│   ├── model.py                   # 神经网络模型定义
│   ├── data_preprocessing.py      # 数据预处理模块
│   ├── train.py                   # 模型训练模块
│   ├── evaluate.py                # 模型评估模块
│   └── 测试数据文件.py             # 数据预处理脚本
├── store_predictor_best.pth       # 最佳模型权重
└── README.md                      # 项目说明文档
```

## 数据格式

### 训练/测试数据格式 (train_data.csv, test_data.csv)
```csv
brand_name,brand_type,longitude_list,latitude_list,grid_id_list
四季啤酒屋,体育休闲服务;娱乐场所;酒吧,[116.928...],[36.697...],[71,77,72,25]
```

### 网格坐标数据格式 (grid_coordinates-2.csv)
包含每个网格的：
- 地理边界坐标 (经纬度范围)
- POI特征统计 (医疗、住宿、摩托、体育、餐饮、公司、购物、生活、科教、汽车)

## 模型架构

### NextGridPredictor
主要由以下组件构成：

1. **序列编码器 (SeqEncoder)**
   - 嵌入层：将网格ID转换为向量表示
   - LSTM层：捕获序列时序依赖关系

2. **空间坐标编码器 (MLPEncoder)**
   - 多层感知机：处理归一化的地理坐标信息

3. **POI特征编码器 (MLPEncoder)**
   - 多层感知机：处理兴趣点特征

4. **特征融合层**
   - 全连接网络：融合多模态特征
   - Dropout：防止过拟合

5. **分类器**
   - 输出层：预测下一个网格的概率分布

## 核心功能

### 数据预处理
- **空间密度排序**：根据网格间的空间距离对序列进行重排序
- **特征归一化**：对坐标和POI特征进行标准化处理
- **序列生成**：从原始轨迹数据生成训练样本

### 模型训练
- **早停机制**：基于验证集MRR指标防止过拟合
- **批量处理**：支持变长序列的批量训练
- **GPU加速**：自动检测并使用CUDA设备

### 模型评估
支持多种评估指标：
- **Accuracy@K**：Top-K准确率 (K=1,5,10)
- **MRR**：平均倒数排名

## 安装和使用

### 1. 克隆项目
```bash
git clone <repository-url> DL
cd DL
```

### 2. 安装依赖
```bash
pip install torch pandas numpy scikit-learn pyopenxl 
```

### 3. 准备数据
确保数据文件放置在 `data/` 目录下：
- `train_data.csv`
- `test_data.csv`
- `grid_coordinates-2.csv`

### 4. 运行训练
```bash
python3 src/测试数据文件.py
python3 src/main.py
```

## 使用示例

### 快速开始
```python
from main import *

# 加载数据和模型
train_set, val_set, test_set, num_classes, grid2idx = load_all_data(
    'data/train_data.csv', 
    'data/test_data.csv', 
    'data/grid_coordinates-2.csv'
)

# 创建模型
model = NextGridPredictor(num_classes=num_classes)

# 训练模型
trained_model = train_model(model, train_set, val_set, device)

# 评估模型
acc_k, mrr = evaluate_model(trained_model, test_set, device)
```

### 自定义参数
```python
# 自定义模型参数
model = NextGridPredictor(
    num_classes=num_classes,
    embed_dim=64,        # 嵌入维度
    lstm_hidden=128,     # LSTM隐藏层大小
    lstm_layers=2,       # LSTM层数
    dropout=0.2          # Dropout率
)

# 自定义训练参数
trained_model = train_model(
    model, train_set, val_set, device,
    num_epochs=50,       # 训练轮数
    batch_size=64,       # 批大小
    lr=1e-4,            # 学习率
    patience=10          # 早停耐心值
)
```

## 技术亮点

1. **多模态特征融合**：有效结合序列、空间和语义信息
2. **空间感知处理**：基于密度的智能序列排序
3. **可扩展架构**：模块化设计便于功能扩展
4. **高效训练**：支持GPU加速和早停优化

## 性能指标

模型在测试集上的典型性能(具体数值取决于数据和参数设置，且有随机因素影响)：

```text
Test_MRR: 0.3257
Test_Acc@1: 0.270
Test_Acc@5: 0.395
Test_Acc@10: 0.474
```

---

其他有关讯息请查看对应的文档，在`report paper/`中。

*最后更新：2025年7月6日*
