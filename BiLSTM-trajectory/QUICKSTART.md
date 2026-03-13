# BiLSTM Trajectory Classification - Quick Start Guide

## 快速开始

### 1. 安装依赖

```bash
cd /home/research/Agri-MBT/BiLSTM-trajectory
pip install -r requirements.txt
```

### 2. 测试代码

```bash
python test_code.py
```

应该看到所有测试通过：
```
✅ All core components working correctly!
```

### 3. 运行实验

#### 方式 A：交互式运行（推荐）

```bash
bash run_experiments.sh
```

脚本会：
1. 先运行测试
2. 询问运行哪些实验
3. 执行训练
4. 显示结果

#### 方式 B：直接运行单个实验

**实验1：仅轨迹分类**
```bash
python train.py --mode trajectory_only --epochs 50
```

**实验2：多模态分类（轨迹+图像）**
```bash
python train.py --mode multimodal --epochs 50
```

**运行所有实验**
```bash
python train.py --mode all --epochs 50
```

### 4. 查看结果

训练完成后，结果保存在：
```
experiments/results/
├── results_trajectory_only.json
├── results_multimodal.json
└── weights/
    ├── BiLSTM_Trajectory_best.pth
    └── BiLSTM_Multimodal_best.pth
```

查看结果：
```bash
# 轨迹实验结果
cat experiments/results/results_trajectory_only.json | python -m json.tool

# 多模态实验结果
cat experiments/results/results_multimodal.json | python -m json.tool
```

## 预期输出

### 训练过程

```
Epoch 1/50 [Train]: 100%|██████████| 322/322 [00:15<00:00, loss: 2.1234]
Epoch 1/50 [Val]: 100%|██████████| 81/81 [00:02<00:00]
Epoch 1/50: train_acc=45.23%, val_acc=42.18%

...

Epoch 50/50: train_acc=85.67%, val_acc=73.52%

Test Accuracy: 72.84%
F1 Macro: 65.32%
F1 Weighted: 71.56%
```

### 结果JSON示例

```json
{
  "mode": "trajectory_only",
  "best_val_acc": 75.23,
  "test_acc": 72.84,
  "test_metrics": {
    "accuracy": 72.84,
    "precision_macro": 64.21,
    "recall_macro": 66.45,
    "f1_macro": 65.32,
    "f1_weighted": 71.56
  }
}
```

## 自定义参数

### 修改配置文件

编辑 `opt.py`：

```python
# 模型参数
rnn_size = 256      # LSTM隐藏层大小
rnn_layers = 2      # LSTM层数
dropout = 0.3       # Dropout率

# 训练参数
LEARNING_RATE = 3e-4
batch_size = 64
epochs = 50
```

### 命令行参数

```bash
python train.py \
  --mode trajectory_only \
  --data /path/to/aligned_data.csv \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cuda:0
```

## 实验对比

| 实验 | 输入 | 预期准确率 | 训练时间 |
|------|------|-----------|---------|
| Trajectory Only | 36维轨迹特征 | 65-75% | ~10分钟 |
| Multimodal | 轨迹+图像特征 | 70-80% | ~15分钟 |

**说明：**
- 轨迹特征包含GPS、IMU、环境传感器数据
- 图像特征来自预训练ViT-B16模型（768维）
- 多模态融合能提升3-5%的准确率

## 常见问题

### Q1: CUDA out of memory

**解决：** 减小batch_size
```bash
python train.py --mode trajectory_only --batch-size 32
```

### Q2: 数据文件不存在

**检查：**
```bash
ls -lh /home/research/Agri-MBT/data/aligned_output/aligned_data.csv
```

**解决：** 确保数据对齐脚本已运行
```bash
cd /home/research/Agri-MBT
python scripts/align_agri_data.py --trajectory data/trajectory/... --video-dir data/video/...
```

### Q3: 训练不收敛

**尝试：**
1. 降低学习率：`--lr 1e-4`
2. 增加epochs：`--epochs 100`
3. 使用CE损失：编辑 `opt.py`，设置 `loss = "CE"`

### Q4: 类别不平衡严重

**检查类别分布：**
```bash
python -c "
import pandas as pd
df = pd.read_csv('/home/research/Agri-MBT/data/aligned_output/aligned_data.csv')
print(df['分类'].value_counts().sort_index())
"
```

**解决：** 使用Focal Loss（默认已启用）

## 与MBT模型对比

| 模型 | 准确率 | 训练时间 | 参数量 |
|------|--------|---------|--------|
| BiLSTM Trajectory | ~70% | 10分钟 | ~2M |
| BiLSTM Multimodal | ~75% | 15分钟 | ~5M |
| MBT Multimodal | 94.72% | 30分钟 | ~86M |

**说明：**
- BiLSTM更轻量，训练更快
- MBT使用ViT-B16作为图像编码器，性能最优
- BiLSTM适合快速实验和baseline对比

## 下一步

1. **调参实验**
   - 尝试不同的rnn_size（128, 256, 512）
   - 尝试不同的rnn_layers（1, 2, 3）
   - 尝试不同的dropout（0.1, 0.3, 0.5）

2. **消融实验**
   - 不同序列长度（4, 8, 16帧）
   - 不同特征子集（仅GPS、仅IMU、全部）

3. **可视化分析**
   - 训练曲线
   - 混淆矩阵
   - 特征重要性

## 支持

如有问题，请检查：
1. 数据路径是否正确
2. GPU内存是否充足
3. 依赖是否全部安装

或查看详细文档：`README.md`
