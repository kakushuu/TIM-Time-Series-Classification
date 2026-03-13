# 性能差异分析：为什么 BiLSTM Multimodal 只有 15.23%？

## 问题总结

| 模型 | 架构 | 图像特征来源 | 测试准确率 | 差距 |
|------|------|------------|-----------|-----|
| **MBT Multimodal** | ViT-B16 + BiLSTM | ✅ **真实 ViT 特征** | **94.65%** | 基准 |
| **BiLSTM Multimodal** | BiLSTM + MLP | ❌ **随机噪声** | **15.23%** | **-79.4%** |
| **BiLSTM Trajectory-only** | BiLSTM + MLP | - | 18.22% | - |

## 根本原因

**BiLSTM Multimodal 使用的是随机占位符特征，而不是真实的视觉特征！**

### 代码证据

在 `BiLSTM-trajectory/utils/loader.py` 第 151-161 行：

```python
# Placeholder: random image features (replace with actual features)
img_feat_size = 768  # ViT-B16 feature dimension

def generate_img_features(n_samples):
    """Placeholder: generate random image features"""
    return torch.randn(n_samples, img_feat_size).to(device)  # ❌ 随机噪声！

X_img_train = generate_img_features(X_train.shape[0])
X_img_valid = generate_img_features(X_valid.shape[0])
X_img_test = generate_img_features(X_test.shape[0])
```

**问题**：`torch.randn()` 生成的是**标准正态分布的随机噪声**，完全没有视觉信息！

## 为什么性能下降这么严重？

### 1. 随机噪声 vs 真实特征

| 特征类型 | 信息量 | 与标签相关性 |
|---------|--------|------------|
| 真实 ViT 特征 | 高（包含物体、场景、动作信息） | 强（94.65% 准确率） |
| 随机噪声 | 无（纯随机） | 无（独立于标签） |

### 2. 模型混淆

- **真实特征**：模型学习到"图像中有拖拉机 → 耕地"
- **随机噪声**：模型无法学习，因为每次"看到"的图像特征都不同

### 3. 负面影响

添加随机噪声不仅没有帮助，反而：
- **干扰轨迹特征学习**：噪声与轨迹特征不匹配
- **增加过拟合风险**：模型试图拟合随机噪声
- **降低泛化能力**：验证集性能崩溃

## 实验对比

### MBT 模型（之前）
- **架构**：ViT-B16 视觉编码器 + MBT 跨模态注意力
- **数据**：真实视频帧 → ViT → 768-dim 特征
- **结果**：
  - 训练准确率：99.06%
  - 验证准确率：94.72%
  - **测试准确率：94.65%**
  - F1 Macro: 80.0%

### BiLSTM Multimodal（现在）
- **架构**：BiLSTM 轨迹编码器 + 简单拼接融合
- **数据**：轨迹特征 + **随机噪声**
- **结果**：
  - 训练准确率：16.66%
  - 验证准确率：14.94%
  - **测试准确率：15.23%**
  - F1 Macro: 2.42%

## 为什么 Trajectory-only (18.22%) 比 Multimodal (15.23%) 更好？

**因为没有噪声干扰！**

- Trajectory-only：只用轨迹特征，干净清晰
- Multimodal：轨迹特征 + 随机噪声 = 更差

这证明：**错误的融合比不融合更糟糕**

## 解决方案

### 方案 1：提取真实 ViT 特征（推荐）

使用 `extract_visual_features.py` 从 MBT 模型的 ViT 编码器提取特征：

```bash
cd /home/research/Agri-MBT/BiLSTM-trajectory
python extract_visual_features.py \
    --csv /home/research/Agri-MBT/data/aligned_output/aligned_data.csv \
    --output data/visual_features.npz
```

然后修改 `utils/loader.py` 加载真实特征。

### 方案 2：直接使用 MBT 模型

如果目标是多模态融合，MBT 模型已经达到 94.65%，效果很好。

### 方案 3：使用预训练 ResNet/ViT

从 ImageNet 预训练模型提取特征（不需要 MBT 模型）。

## 结论

1. ✅ **MBT 模型已经很好**：94.65% 准确率
2. ❌ **BiLSTM Multimodal 使用了错误的特征**：随机噪声
3. ✅ **Trajectory-only 是干净的基线**：18.22%
4. 🔧 **需要集成真实视觉特征才能公平比较**

## 下一步

**如果要比较 BiLSTM vs MBT**：
1. 提取真实 ViT 特征
2. 重新训练 BiLSTM Multimodal
3. 对比：MBT (94.65%) vs BiLSTM+ViT (?)

**如果 MBT 已经足够好**：
- 直接使用 MBT 模型
- 专注于改进数据质量和类别平衡
