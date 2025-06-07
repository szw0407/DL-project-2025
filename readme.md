# 门店选址预测项目

该项目采用 PyTorch 实现了基于 LSTM-合成路径 Attention 的门店选址预测模型，支持空间坐标的合成。包括数据加工，模型训练，效果评估和 Optuna 超参调优化。

---

## 目录结构

```
project_root/
|— data/
|   |— train_data.csv
|   |— test_data.csv
|   |— grid_coordinates.csv
|— data_preprocessing.py
|— model.py
|— train.py
|— evaluate.py
|— optuna_search.py
|— README.md
```

---

## 环境依赖

```bash
pip install torch numpy pandas scikit-learn optuna
```

---

## 数据格式

### `train_data.csv`

| brand\_name | grid\_id\_list   |
| ----------- | ---------------- |
| 品牌A         | \[101, 102, 105] |

### `test_data.csv`

格式同上，作为測试数据

### `grid_coordinates.csv`

\| grid\_id | grid\_lon\_min | grid\_lon\_max | grid\_lat\_min | grid\_lat\_max |

---

## 1. 模型训练

使用 `train.py` 进行训练：

```bash
python train.py
```

默认参数：

* 基于 grid ID + 坐标合成输入
* LSTM 隐藏层维度 = 64
* Dropout = 0.1
* Epoch = 20
* 早停系统判断 MRR 无提升 5 轮

训练完成后会保存为 `store_predictor_best.pth`，并进行測试集评估

---

## 2. 评估指标

使用 `evaluate.py`内的 `compute_metrics`：

* Top-K 准确率：Acc\@1, Acc\@5, Acc\@10
* MRR（平均倒数排名）

---

## 3. 超参优化 (Optuna)

执行:

```bash
python optuna_search.py
```

设置：

* 试验次数 = 30
* 单次 trial 最多运行 15 epochs
* 支持 GPU 检测
* 使用 MedianPruner 接口削枝

调优结束后会重训一个最优模型，并进行測试评估

---

## 模型特点

* 网格ID嵌入 + 2维坐标编码
* LSTM 对应间距间间店面排序
* Dropout + weight decay 防止过拟合
* 可选坐标合成或否

---

## 设计思路

* 采用 **空间密度排序** 或 **KMeans+密度排序** 构造假时间序列
* 以 (系列 -> 下一个目标) 构造训练样本

---

## 联系



---

## License


