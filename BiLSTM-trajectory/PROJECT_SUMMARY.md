# BiLSTM轨迹分类项目总结

## 项目概述

**目标：** 将GAN-BiLSTM二分类代码修改为11分类农业活动识别

**原始代码：** GAN-BiLSTM (Field-road Classification, 2-class)
**修改后：** BiLSTM-Trajectory (Agricultural Activity Classification, 11-class)

---

## 主要修改

### 1. 分类任务变更

| 项目 | 原代码 | 新代码 |
|------|--------|--------|
| 分类数 | 2 (道路/田地) | 11 (农业活动) |
| 特征维度 | 25 | 36 |
| 序列长度 | 512 | 8 (匹配MBT) |

### 2. 架构修改

#### 去除的模块：
- ❌ GAN数据增强 (CTGAN)
- ❌ WeightedRandomSampler
- ❌ 2分类输出层

#### 保留的模块：
- ✅ BiLSTM + Attention
- ✅ Focal Loss (处理类别不平衡)
- ✅ AdamW优化器
- ✅ Warmup + Cosine学习率调度

#### 新增的模块：
- ✅ 多模态融合 (轨迹+图像)
- ✅ 11分类输出
- ✅ 标准数据加载器

---

## 文件结构

```
BiLSTM-trajectory/
├── opt.py                   # ✅ 配置文件 (修改为11分类)
├── train.py                 # ✅ 主训练脚本 (新增)
├── trainer.py               # ✅ 训练器 (适配11分类)
├── test_code.py             # ✅ 测试脚本 (新增)
├── run_experiments.sh       # ✅ 实验脚本 (新增)
├── requirements.txt         # ✅ 依赖文件 (新增)
├── README.md                # ✅ 详细文档 (新增)
├── QUICKSTART.md            # ✅ 快速开始 (新增)
├── models/
│   ├── lstm.py              # ✅ BiLSTM模型 (修改为11分类)
│   └── lossFun.py           # ✅ 损失函数 (新增Focal Loss)
└── utils/
    ├── loader.py            # ✅ 数据加载器 (适配36特征)
    └── metrics.py           # ✅ 评估指标 (新增per-class)
```

---

## 关键代码修改

### 1. 模型定义 (`models/lstm.py`)

**原代码 (2分类):**
```python
Model = AttBiLSTM(2, opt.emb_size, opt.rnn_size, opt.rnn_layers, opt.dropout)
```

**新代码 (11分类):**
```python
Model = AttBiLSTM(n_classes=11, emb_size=36, rnn_size=256, rnn_layers=2, dropout=0.3)
```

### 2. 数据加载 (`utils/loader.py`)

**原代码 (25特征):**
```python
train_data = train.iloc[:, [0, 1, 2, ..., 24, 26, 27]].to_numpy()
```

**新代码 (36特征):**
```python
TRAJ_COLS = ['经度', '纬度', '间距(米)', '深度', '速度', ...]  # 36个特征
train_data = df[TRAJ_COLS].values
```

### 3. 损失函数 (`trainer.py`)

**原代码:**
```python
onehot_target = torch.eye(2)[y.long().cpu(), :].to(self.device)
```

**新代码:**
```python
onehot_target = torch.eye(opt.n_classes)[y.long().cpu(), :].to(self.device)
```

### 4. GAN模块 (`utils/loader.py`)

**原代码:**
```python
if opt.useGAN == True:
    # CTGAN数据增强
    model = CTGANSynthesizer(...)
    model.fit(data_train)
```

**新代码:**
```python
opt.useGAN = False  # 禁用GAN
```

---

## 实验设置

### 实验1：BiLSTM Trajectory Only

**输入：**
- 轨迹特征：36维 × 8帧 = (batch, 8, 36)

**模型：**
- BiLSTM (256 hidden, 2 layers)
- Attention mechanism
- Classifier: 256 → 512 → 512 → 11

**训练：**
- Epochs: 50
- Batch size: 64
- Learning rate: 3e-4
- Optimizer: AdamW
- Loss: Focal Loss

### 实验2：BiLSTM Multimodal

**输入：**
- 轨迹特征：36维 × 8帧 = (batch, 8, 36)
- 图像特征：768维 = (batch, 768) [ViT-B16]

**模型：**
- Trajectory encoder: BiLSTM + Attention
- Image encoder: Linear projection
- Fusion: Concatenation + Linear
- Classifier: 512 → 512 → 512 → 11

---

## 预期结果

| 模型 | 准确率 | Macro F1 | 训练时间 |
|------|--------|----------|---------|
| BiLSTM Trajectory | 65-75% | 55-65% | ~10分钟 |
| BiLSTM Multimodal | 70-80% | 60-70% | ~15分钟 |
| MBT Multimodal (对比) | 94.72% | 80.00% | ~30分钟 |

**说明：**
- BiLSTM作为轻量级baseline
- 多模态融合能提升3-5%
- MBT性能最优但参数量大

---

## 使用指南

### 快速开始

```bash
# 1. 进入项目目录
cd /home/research/Agri-MBT/BiLSTM-trajectory

# 2. 安装依赖
pip install -r requirements.txt

# 3. 测试代码
python test_code.py

# 4. 运行实验
bash run_experiments.sh
```

### 单独运行实验

```bash
# 仅轨迹实验
python train.py --mode trajectory_only --epochs 50

# 多模态实验
python train.py --mode multimodal --epochs 50
```

### 查看结果

```bash
# 轨迹实验结果
cat experiments/results/results_trajectory_only.json

# 多模态实验结果
cat experiments/results/results_multimodal.json
```

---

## 与原始代码对比

### 保留的优点

1. ✅ **BiLSTM + Attention**: 有效捕获时序依赖
2. ✅ **Focal Loss**: 处理类别不平衡
3. ✅ **Warmup + Cosine**: 稳定训练过程
4. ✅ **AdamW**: 权重衰减正则化

### 改进的地方

1. ✅ **去除GAN**: 数据充足，无需增强
2. ✅ **标准化特征**: StandardScaler归一化
3. ✅ **完整评估**: per-class metrics
4. ✅ **模块化设计**: 易于扩展

### 新增功能

1. ✅ **多模态支持**: 轨迹+图像融合
2. ✅ **详细文档**: README + QUICKSTART
3. ✅ **测试脚本**: 自动验证代码
4. ✅ **结果可视化**: JSON格式输出

---

## 技术细节

### 数据预处理

1. **特征选择**: 36个轨迹特征
2. **标准化**: StandardScaler (训练集拟合)
3. **序列构建**: 滑动窗口 (8帧)
4. **数据划分**: 80% train / 20% test

### 模型架构

```
Input (batch, 8, 36)
    ↓
BatchNorm1d
    ↓
BiLSTM (256 hidden, 2 layers)
    ↓
LayerNorm
    ↓
Attention (temporal aggregation)
    ↓
FC (256 → 512 → 512 → 11)
    ↓
Output (batch, 11)
```

### 训练策略

- **学习率**: 3e-4 (初始)
- **调度器**: Warmup (10% epochs) + Cosine annealing
- **早停**: 基于验证集准确率
- **保存**: 最佳模型 + 定期checkpoint

---

## 常见问题

### Q1: 为什么去除GAN？

**A:**
- 原始GAN用于处理类别不平衡 (道路多、田地少)
- 我们的数据集样本充足 (32,000+样本)
- Focal Loss已能处理不平衡
- GAN训练不稳定，增加复杂度

### Q2: 为什么用8帧序列？

**A:**
- 匹配MBT模型的clip设置 (8帧/clip)
- 平衡时序信息和计算效率
- 1秒数据足够捕获活动特征

### Q3: 如何提升性能？

**A:**
1. 增加序列长度 (16, 32帧)
2. 增加LSTM层数 (3, 4层)
3. 使用更强的图像编码器 (ResNet, EfficientNet)
4. 添加注意力融合机制 (类似MBT)
5. 数据增强 (时序抖动、噪声注入)

---

## 下一步工作

### 短期

- [ ] 运行实验验证代码
- [ ] 调参优化性能
- [ ] 可视化训练曲线

### 中期

- [ ] 消融实验 (特征子集)
- [ ] 对比实验 (BiLSTM vs MBT)
- [ ] 错误分析 (混淆矩阵)

### 长期

- [ ] 模型压缩 (知识蒸馏)
- [ ] 在线学习 (持续更新)
- [ ] 部署优化 (边缘设备)

---

## 总结

✅ **已完成：**
- 代码从2分类改为11分类
- 特征从25维改为36维
- 去除GAN数据增强
- 新增多模态融合版本
- 完整文档和测试脚本

✅ **核心改动：**
- `opt.py`: 配置更新
- `models/lstm.py`: 模型适配11分类
- `utils/loader.py`: 数据加载器适配36特征
- `trainer.py`: 训练器适配多分类

✅ **新增文件：**
- `train.py`: 主训练脚本
- `test_code.py`: 测试脚本
- `run_experiments.sh`: 实验运行脚本
- `README.md`, `QUICKSTART.md`: 文档

🎯 **准备就绪：** 代码已准备就绪，可以开始训练！

```bash
cd /home/research/Agri-MBT/BiLSTM-trajectory
bash run_experiments.sh
```

---

**作者：** 基于GAN-BiLSTM修改
**日期：** 2026-03-14
**项目：** Agri-MBT (Agricultural Multimodal Bottleneck Transformer)
