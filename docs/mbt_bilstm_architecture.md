# MBT + BiLSTM 架构详解

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        AVmodel (3 种模式)                         │
├─────────────────────────────────────────────────────────────────┤
│  1. trajectory_only:  BiLSTM → MLP → 分类                         │
│  2. multimodal:       BiLSTM + ViT → MBT融合 → 分类               │
│  3. image_only:       ViT → 分类 (无轨迹)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. BiLSTM 轨迹编码器（所有模式共用）

### 输入数据
```python
输入形状: (batch_size, T=8, 6)
- T=8: 滑动窗口长度（8个连续帧）
- 6: 轨迹特征维度
  [经度, 纬度, 间距(米), 深度, 速度, 方向角]
```

### BiLSTM 结构
```python
self.traj_bilstm = nn.LSTM(
    input_size=6,           # 6个轨迹特征
    hidden_size=384,        # 每个方向384维
    num_layers=2,           # 2层LSTM
    bidirectional=True,     # 双向
    batch_first=True,
    dropout=0.3,
)
# 输出: (bs, 8, 768)  ← 384*2=768 (双向拼接)
```

**工作原理**：
- **前向LSTM**：从第1帧→第8帧，捕获历史信息
- **后向LSTM**：从第8帧→第1帧，捕获未来上下文
- **拼接**：每个时间步的表示 = [前向;后向] = 768维

### 添加 CLS Token
```python
# 添加可学习的全局表示 token
self.traj_cls_token = nn.Parameter(torch.zeros(1, 1, 768))

# 前向传播中:
cls = self.traj_cls_token.expand(B, -1, -1)  # (bs, 1, 768)
x = torch.cat([cls, rnn_out], dim=1)          # (bs, 9, 768)
# 输出: [cls_token, frame1, frame2, ..., frame8]
```

**为什么要 CLS Token？**
- 类似 BERT 的 [CLS] token
- 聚合整个序列的全局信息
- MBT encoder 最后只取 `x[:, 0]`（cls token）作为序列表示

### Layer Normalization
```python
self.traj_layernorm = nn.LayerNorm(768)
rnn_out = self.traj_layernorm(rnn_out)  # 稳定训练
```

---

## 2. Multimodal 模式（BiLSTM + ViT + MBT融合）

### 完整数据流

```
输入:
  - 轨迹序列: (bs, 8, 6)
  - RGB帧:    (bs, 1, 3, 224, 224)

┌─────────────────────────────────────────────────────────────┐
│ 步骤1: BiLSTM 轨迹编码                                        │
├─────────────────────────────────────────────────────────────┤
│  (bs, 8, 6)                                                 │
│    ↓ BiLSTM(6→384*2)                                       │
│  (bs, 8, 768)                                               │
│    ↓ LayerNorm                                             │
│  (bs, 8, 768)                                               │
│    ↓ 添加 cls_token                                         │
│  (bs, 9, 768)  ← [cls, t1, t2, ..., t8]                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 步骤2: ViT 图像编码                                          │
├─────────────────────────────────────────────────────────────┤
│  (bs, 1, 3, 224, 224)                                       │
│    ↓ Conv2d (patch_embed)                                  │
│  (bs, 768, 14, 14)  ← 一帧切成14×14个patch                  │
│    ↓ Flatten                                               │
│  (bs, 196, 768)  ← 196个patch tokens                        │
│    ↓ 添加 cls_token                                         │
│  (bs, 197, 768)  ← [cls, p1, p2, ..., p196]                │
│    ↓ + positional embedding                                │
│  (bs, 197, 768)                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 步骤3: MBT 跨模态融合 (12个 AdaptFormer blocks)              │
├─────────────────────────────────────────────────────────────┤
│  输入:                                                      │
│    - traj_tokens: (bs, 9, 768)                             │
│    - rgb_tokens:  (bs, 197, 768)                           │
│                                                             │
│  每个 AdaptFormer block:                                    │
│  ┌────────────────────────────────────────┐                │
│  │ 1. Latent Bottleneck Fusion            │                │
│  │    - 拼接: concat = [traj; rgb]        │                │
│  │    - 4个latent tokens 作为桥梁         │                │
│  │    - traj ↔ latents ↔ rgb 交叉注意力   │                │
│  ├────────────────────────────────────────┤                │
│  │ 2. Self-Attention (ViT blocks)         │                │
│  │    - traj: 自注意力增强                │                │
│  │    - rgb:  自注意力增强                │                │
│  ├────────────────────────────────────────┤                │
│  │ 3. Feed-Forward + Adapter              │                │
│  │    - MLP层                             │                │
│  │    - AdaptFormer adapter (8维瓶颈)     │                │
│  └────────────────────────────────────────┘                │
│                                                             │
│  重复 × 12 次                                                │
│                                                             │
│  输出:                                                      │
│    - traj_tokens: (bs, 9, 768)   ← 融合了视觉信息           │
│    - rgb_tokens:  (bs, 197, 768) ← 融合了轨迹信息           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 步骤4: 提取 CLS Token 并分类                                 │
├─────────────────────────────────────────────────────────────┤
│  traj_cls = traj_tokens[:, 0]  # (bs, 768)                 │
│  rgb_cls  = rgb_tokens[:, 0]    # (bs, 768)                │
│                                                             │
│  fused = (traj_cls + rgb_cls) * 0.5  # (bs, 768)           │
│                                                             │
│  logits = classifier(fused)  # (bs, 11)                    │
└─────────────────────────────────────────────────────────────┘

输出: (bs, 11)  ← 11个类别的概率分布
```

---

## 3. Trajectory-Only 模式（仅 BiLSTM）

### 数据流

```
输入: (bs, 8, 6)

┌─────────────────────────────────────────────────────────────┐
│ BiLSTM 编码 (同上)                                           │
├─────────────────────────────────────────────────────────────┤
│  (bs, 8, 6)                                                 │
│    ↓ BiLSTM + LayerNorm + cls_token                         │
│  (bs, 9, 768)                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 提取 CLS Token                                               │
├─────────────────────────────────────────────────────────────┤
│  traj_cls = x[:, 0]  # (bs, 768)                            │
│  ↑ 只取第一个token（全局序列表示）                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MLP 分类器                                                   │
├─────────────────────────────────────────────────────────────┤
│  (bs, 768)                                                  │
│    ↓ Linear(768→512) + ReLU + Dropout(0.3)                  │
│  (bs, 512)                                                  │
│    ↓ Linear(512→256) + ReLU + Dropout(0.3)                  │
│  (bs, 256)                                                  │
│    ↓ Linear(256→11)                                         │
│  (bs, 11)                                                   │
└─────────────────────────────────────────────────────────────┘

输出: (bs, 11)
```

---

## 4. 关键设计对比

### 4.1 轨迹编码方式对比

| 方面 | 旧版（Linear） | 新版（BiLSTM） |
|------|---------------|---------------|
| **输入** | `(bs, 6)` 单帧 | `(bs, 8, 6)` 8帧序列 |
| **编码器** | `Linear(6→768)` | `BiLSTM(6→384*2)` |
| **时序建模** | ❌ 无 | ✅ 双向LSTM |
| **参数量** | 6×768 = 4,608 | ~4.7M（2层双向） |
| **语义** | 单点特征 | 时序上下文特征 |

### 4.2 Multimodal vs Trajectory-Only

| 模块 | Trajectory-Only | Multimodal |
|------|----------------|------------|
| **轨迹编码** | BiLSTM | BiLSTM |
| **图像编码** | ❌ 无 | ✅ ViT-B16 |
| **跨模态融合** | ❌ 无 | ✅ MBT (12层) |
| **分类器** | 3层MLP | 1层Linear |
| **可训练参数** | ~5.3M | ~5.1M |
| **推理速度** | 快 | 慢（ViT开销） |
| **性能** | 待测试 | 92.66% (epoch 2) |

---

## 5. AdaptFormer 详解（MBT融合核心）

### 5.1 Latent Bottleneck Fusion

```python
# 4个可学习的 latent tokens 作为桥梁
self.latents = nn.Parameter(torch.empty(1, 4, 768))

def fusion(traj_tokens, rgb_tokens):
    # 1. 拼接所有 tokens
    concat = torch.cat([traj_tokens, rgb_tokens], dim=1)
    # concat: (bs, 9+197=206, 768)

    # 2. Latents ← concat (cross-attention)
    # latents 从所有模态中提取关键信息
    latents_fused = attention(
        q=latents,      # (bs, 4, 768)
        k=concat,       # (bs, 206, 768)
        v=concat
    )  # → (bs, 4, 768)

    # 3. Traj ← latents (cross-attention)
    # 轨迹 tokens 从 latents 中获取视觉信息
    traj_enhanced = traj_tokens + attention(
        q=traj_tokens,  # (bs, 9, 768)
        k=latents_fused,
        v=latents_fused
    )

    # 4. RGB ← latents (cross-attention)
    # 视觉 tokens 从 latents 中获取轨迹信息
    rgb_enhanced = rgb_tokens + attention(
        q=rgb_tokens,   # (bs, 197, 768)
        k=latents_fused,
        v=latents_fused
    )

    return traj_enhanced, rgb_enhanced
```

**为什么用 Latent Bottleneck？**
- **信息瓶颈**：4个 latent tokens 强制模型压缩信息
- **避免过拟合**：限制信息流动，提高泛化
- **高效融合**：latent tokens 小（4个），注意力计算快

### 5.2 AdaptFormer Adapter

```python
# 参数高效微调
self.spec_down = nn.Linear(768, 8)  # 降维到8
self.spec_up   = nn.Linear(8, 768)  # 升维回768

def forward_audio_AF(x):
    # (bs, seq, 768)
    x_down = self.spec_down(x)    # (bs, seq, 8)
    x_down = QuickGELU(x_down)
    x_up   = self.spec_up(x_down) # (bs, seq, 768)
    return x_up

# 残差连接
x = x + mlp(x) + adapter(x) * scale
```

**优势**：
- ViT backbone 冻结（不训练）
- 只训练 adapter（768×8×2 = 12K 参数/层）
- 总可训练参数少（5.1M vs 86M）

---

## 6. 数据 Pipeline

### 6.1 Dataset (`av_data.py`)

```python
class AV_Dataset:
    def __getitem__(self, idx):
        # 1. 轨迹序列: 滑动窗口 [idx-7, ..., idx]
        start = max(0, idx - 7)
        traj_seq = self.traj_all[start : idx + 1]  # (≤8, 6)

        # 不足8帧时，左边补第一帧
        if traj_seq.shape[0] < 8:
            pad = traj_seq[0:1].repeat(8 - len(traj_seq), 1)
            traj_seq = torch.cat([pad, traj_seq])

        # traj_seq: (8, 6)

        # 2. 图像: 当前帧（单帧）
        img = Image.open(frame_path)
        img = transforms(img)  # (3, 224, 224)
        rgb_frames = img.unsqueeze(0)  # (1, 3, 224, 224)

        # 3. 标签
        label = int(row['分类'])

        return traj_seq, rgb_frames, label
```

**关键点**：
- 轨迹：返回 8 帧的滑动窗口序列
- 图像：返回当前单帧（`F=1`）
- 标签：当前帧的类别

### 6.2 数据标准化

```python
# 轨迹特征标准化（训练集统计）
traj_mean = train_df[TRAJ_COLS].mean()  # (6,)
traj_std  = train_df[TRAJ_COLS].std()   # (6,)
traj_norm = (traj - traj_mean) / traj_std

# 图像标准化（ImageNet 统计）
Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
```

---

## 7. 训练配置

```python
# 超参数
batch_size = 8
learning_rate = 3e-4
num_epochs = 15
optimizer = Adam
loss = CrossEntropyLoss

# 冻结策略
# - ViT backbone: 冻结（不训练）
# - BiLSTM: 训练
# - AdaptFormer adapters: 训练
# - 分类器: 训练

# 可训练参数
trajectory_only: 5,280,268
multimodal:      5,115,900
```

---

## 8. 实验对比计划

### 8.1 已完成
- ✅ Multimodal (BiLSTM + ViT): **92.66%** (epoch 2, 进行中)

### 8.2 待完成
- ⏳ Trajectory-Only (BiLSTM): 待训练

### 8.3 预期结果
- Trajectory-Only: 估计 20-30%（仅轨迹信息）
- Multimodal: 目标 ≥94%（融合视觉信息）

---

## 9. 代码文件结构

```
Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/
├── models/
│   ├── visual_model.py      # AVmodel 主模型
│   │   - forward_traj_features()    # BiLSTM 编码
│   │   - forward_rgb_features()     # ViT 编码
│   │   - forward_encoder()          # MBT 融合
│   │   - forward()                  # 主前向传播
│   │
│   └── pet_modules.py        # AdaptFormer 实现
│       - AdaptFormer                # MBT block
│       - fusion()                   # Latent bottleneck
│       - forward_*_AF()             # Adapter
│
├── dataloader/
│   └── av_data.py            # Dataset 实现
│       - 滑动窗口轨迹序列
│       - 图像加载
│       - 标准化
│
└── train_test.py             # 训练脚本
    - train_one_epoch()
    - val_one_epoch()
    - compute_detailed_metrics()
```

---

## 10. 关键设计决策

### 10.1 为什么用 BiLSTM？
✅ 时序建模：捕获轨迹的时间依赖
✅ 双向信息：前向+后向上下文
✅ 灵活长度：可处理变长序列（虽然固定8帧）

### 10.2 为什么保留 MBT？
✅ 已验证有效：94.72% 基线
✅ 参数高效：AdaptFormer adapter
✅ 跨模态融合：latent bottleneck

### 10.3 为什么冻结 ViT？
✅ 防止过拟合：数据量有限（32K）
✅ 加速训练：只训练5M参数
✅ 利用预训练：ImageNet 知识迁移

---

## 总结

当前架构 = **BiLSTM（轨迹时序编码）** + **MBT（跨模态融合）**

- **Trajectory-Only**: 纯 BiLSTM → MLP 分类（测试时序建模能力）
- **Multimodal**: BiLSTM + ViT → MBT 融合 → 分类（测试多模态融合效果）

实验目的：比较 **仅轨迹时序特征** vs **轨迹+视觉融合特征** 的性能差异
