# BiLSTM优化实验计划

**日期**: 2026-03-18
**目标**: 提升BiLSTM在稀有类上的性能（当前class_1: 0% recall, class_6: 11.11% recall）
**基线**: BiLSTM + Weighted CE = 80.37% accuracy, 49.70% Macro F1

---

## 🔬 实验方案

### 方案1: Focal Loss (进行中 🔄)

**动机**:
- Weighted CE对所有样本同等对待，只调整类别权重
- Focal Loss额外降低易分类样本的权重，让模型更关注难样本（稀有类）

**公式**:
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

其中:
- α_t = 类别权重 (与Weighted CE相同)
- γ = focusing parameter (控制难样本权重)
- p_t = 正确类别的预测概率
```

**参数设置**:
- `γ = 2.0`: 标准Focal Loss (推荐值)
- `γ = 1.5`: 更温和的关注（避免过度关注噪声样本）

**预期效果**:
- ✅ 稀有类样本（class_1, class_6）获得更高权重
- ✅ 易分类的class_7样本权重降低
- ⚠️ 可能降低整体准确率（权衡）

**实验状态**:
- ❌ **Focal Loss (γ=2.0)**: 卡住在 epoch 1 (3小时未完成)，可能是数值不稳定导致
- ❌ **Focal Loss (γ=1.5)**: 同样卡住，可能是Focal Loss实现有问题
- 🔄 **增加BiLSTM层数 (3层)**: 训练中 (PID: 2978746)
  - 参数量: 8.9M (vs 基线 5.3M)
  - 日志: `experiments/train_bilstm_3layers_*.log`
  - 预计完成时间: ~1.5小时 (20 epochs × 4.5 min/epoch)

---

### 方案2: 增加模型容量 (计划中 📋)

**动机**:
- 当前2层BiLSTM (8.6M参数)可能容量不足
- 更深/更宽的网络能捕捉更复杂的轨迹模式

**配置选项**:

| 配置 | BiLSTM层数 | Hidden Size | 参数量 | 预期效果 |
|------|-----------|------------|--------|---------|
| **当前** | 2 | 384 (×2 dirs = 768) | 8.6M | 基线 80.37% |
| 配置A | 3 | 384 (×2 dirs = 768) | ~12M | +1-2% accuracy |
| 配置B | 2 | 512 (×2 dirs = 1024) | ~15M | +2-3% accuracy |
| 配置C | 4 | 384 (×2 dirs = 768) | ~16M | 可能过拟合 |

**实现方式**:
修改 `/home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/models/visual_model.py`:
```python
# 当前:
BILSTM_HIDDEN = 384
num_layers = 2

# 修改为可配置:
parser.add_argument('--bilstm_hidden', type=int, default=384)
parser.add_argument('--bilstm_layers', type=int, default=2)
```

**风险**:
- ⚠️ 训练时间增加 (3层 vs 2层: +50%)
- ⚠️ 可能过拟合 (32k样本 vs 12-16M参数)

---

### 方案3: Multi-Head Attention (计划中 📋)

**动机**:
- 当前Single-head attention只学习一种重要性分布
- Multi-head能同时捕捉多种模式（快速运动、缓慢转弯、静止等）

**架构**:
```
BiLSTM输出: (batch, 512, 768)
↓
Multi-Head Attention (4 heads, 192 dims each)
  Head 1: 关注快速加速/减速
  Head 2: 关注转弯方向
  Head 3: 关注全局轨迹形状
  Head 4: 关注局部细节
↓
Concatenate heads: (batch, 768)
↓
Linear projection: (batch, 768)
```

**实现复杂度**: 中等（需要修改模型代码）

**预期效果**: +3-5% Macro F1（改善稀有类）

---

### 方案4: SMOTE过采样 (计划中 📋)

**动机**:
- Class_1只有393样本 (1.2%)，模型难以学习
- Class_6只有549样本 (1.7%)
- SMOTE生成合成样本，平衡训练集

**配置**:
```python
from imblearn.over_sampling import SMOTE

# 对训练集应用SMOTE
smote = SMOTE(
    sampling_strategy={
        1: 2000,  # class_1: 393 → 2000
        6: 2000,  # class_6: 549 → 2000
    },
    k_neighbors=5,
    random_state=42
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)
```

**风险**:
- ⚠️ 合成样本可能不够真实
- ⚠️ 训练时间增加（样本数从25775 → ~30000）

---

### 方案5: 两阶段训练 (计划中 📋)

**动机**:
- 先在平衡子集上预训练（学习基本特征）
- 再在全数据上微调（适应真实分布）

**流程**:
```
Stage 1: 平衡预训练
  - 对每个类随机采样500样本
  - 11 classes × 500 = 5500 samples
  - 训练10 epochs, lr=1e-3

Stage 2: 全数据微调
  - 使用全部25775训练样本
  - 训练10 epochs, lr=3e-5 (低学习率)
  - Focal Loss (γ=2.0)
```

**预期效果**: +5-8% Macro F1

---

## 📊 实验优先级

| 方案 | 难度 | 预期收益 | 优先级 | 状态 |
|------|------|---------|--------|------|
| **Focal Loss (γ=2.0)** | ⭐ 简单 | +3-5% Macro F1 | 🔥 **最高** | 🔄 进行中 |
| **Focal Loss (γ=1.5)** | ⭐ 简单 | +2-4% Macro F1 | 🔥 最高 | 📋 待开始 |
| SMOTE过采样 | ⭐⭐ 中等 | +4-6% Macro F1 | ⬆️ 高 | 📋 计划中 |
| 增加BiLSTM层数 | ⭐⭐ 中等 | +1-2% accuracy | ➡️ 中 | 📋 计划中 |
| Multi-Head Attention | ⭐⭐⭐ 困难 | +3-5% Macro F1 | ➡️ 中 | 📋 计划中 |
| 两阶段训练 | ⭐⭐ 中等 | +5-8% Macro F1 | ⬆️ 高 | 📋 计划中 |

---

## 🎯 成功标准

**必须达到** (才能认为优化成功):
- ✅ Class_1 recall > 0% (当前0%)
- ✅ Class_6 recall > 20% (当前11.11%)
- ✅ Macro F1 > 55% (当前49.70%)
- ✅ Overall accuracy ≥ 78% (当前80.37%，允许轻微下降)

**理想目标**:
- 🌟 Class_1 recall > 30%
- 🌟 Class_6 recall > 40%
- 🌟 Macro F1 > 60%
- 🌟 Overall accuracy ≥ 82%

---

## 📝 实验记录

### 2026-03-18 15:38 - Focal Loss (γ=2.0) 开始
- **配置**: BiLSTM + Focal Loss (γ=2.0) + Weighted
- **训练数据**: 25775 samples
- **验证数据**: 6444 samples
- **参数量**: 5.3M
- **预计完成时间**: 16:40 (约1小时)

---

## 📚 参考资料

1. **Focal Loss for Dense Object Detection** (Lin et al., 2017)
   - 原始论文: https://arxiv.org/abs/1708.02002
   - 关键insight: γ=2.0在大多数任务上效果最好

2. **Class-Balanced Loss Based on Effective Number of Samples** (Cui et al., 2019)
   - 推荐结合Focal Loss使用
   - 比简单的inverse frequency权重更有效

3. **SMOTE: Synthetic Minority Over-sampling Technique** (Chawla et al., 2002)
   - 经典的过采样方法
   - 适用于类别极度不平衡（>10:1）的情况

---

**下一步**: 等待Focal Loss (γ=2.0)结果，如果效果好则尝试γ=1.5对比；如果效果不明显则转向SMOTE过采样。
