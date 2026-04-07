# TC-AdaptFormer 架构设计文档

> **项目**: Agri-MBT — 农机11类作业模式多模态识别（视频 + GNSS 轨迹）
> **模型**: TC-AdaptFormer (Trajectory-Conditioned AdaptFormer)
> **版本**: v1.0 | 2026-03-06

---

## 一、问题形式化

**输入**:
- 视频帧序列: $V \in \mathbb{R}^{B \times T \times 3 \times H \times W}$，其中 $T=5, H=W=224$
- GNSS 轨迹特征: $G \in \mathbb{R}^{B \times 7}$（取窗口中点帧的特征）

**输出**:
- 作业类别 logits: $\hat{y} \in \mathbb{R}^{B \times 11}$，共 11 类

**GNSS 7维特征对应列**:
| 索引 | 列名 | 物理含义 |
|------|------|---------|
| 0 | 经度 | GPS 经度 |
| 1 | 纬度 | GPS 纬度 |
| 2 | 速度 | 移动速度 (m/s) |
| 3 | 深度 | 耕作深度 (cm) |
| 4 | 方向角 | 移动方向 (°) |
| 5 | 间距(米) | 与前点距离 (m) |
| 6 | 类型 | GPS点类型（数值化）|

---

## 二、整体架构

```
输入视频 (B, T, 3, 224, 224)          输入 GNSS (B, 7)
        │                                    │
        ▼                                    ▼
┌───────────────────┐            ┌────────────────────┐
│  ViT-B16 Backbone │            │    GNSS Encoder     │
│  (冻结, 每帧独立)  │            │  MLP: 7→128→768    │
│                   │            │  + LayerNorm        │
│  +AdaptFormer     │            └────────────────────┘
│   adapters(可训)  │                    │
└───────────────────┘                    │ Q_gnss (B, 768)
        │                                │
        │ patch tokens                   │
        │ (B, T, 196, 768)               │
        │                                │
        ▼                                ▼
┌─────────────────────────────────────────────────────┐
│           Cross-Attention Fusion Module              │
│  Q = Q_gnss (B, 1, 768)                             │
│  K = V = visual_tokens (B, T×196, 768)              │
│  F = Attn(Q, K, V) → (B, 768)                      │
└─────────────────────────────────────────────────────┘
        │
        │ fused_feature (B, 768)
        ▼
┌───────────────────┐
│  Classifier Head  │
│  Linear(768, 11)  │
└───────────────────┘
        │
        ▼
  logits (B, 11)
```

---

## 三、各模块详细规格

### 3.1 视觉编码器 (VisualEncoder)

**骨干网络**: ViT-B16，预训练权重来自 `timm.create_model('vit_base_patch16_224', pretrained=True)`

**帧处理方式**: T帧独立编码（2D ViT，无3D卷积）

```
单帧处理:
  输入: (B, 3, 224, 224)
  Patch Embedding: 224×224 → 14×14 patches = 196 tokens
  + CLS token → 197 tokens
  → ViT-B16 blocks (with AdaptFormer) → (B, 197, 768)
  取 patch tokens (去掉CLS): → (B, 196, 768)

T帧拼接:
  对每帧独立: (B, 3, 224, 224) → (B, 196, 768)
  重塑: (B, T, 196, 768) → flatten → (B, T×196, 768)
  即: (B, 980, 768)
```

**参数冻结策略**:
- ViT-B16 所有参数: `requires_grad = False`（~86M）
- AdaptFormer 适配器: `requires_grad = True`（~2M）

### 3.2 AdaptFormer 适配器

基于现有代码 `MBT/models/pet_modules.py` 中的设计，简化为单流版本：

**适配器结构**（插入 ViT 每个 transformer block 的 FFN 之后）:

$$x_{out} = x + \text{FFN}(x) + s \cdot \text{Adapter}(x)$$

其中 Adapter 定义为:
$$\text{Adapter}(x) = W_{up} \cdot \text{GELU}(\text{Dropout}(W_{down} \cdot x))$$

- $W_{down} \in \mathbb{R}^{768 \times 64}$（降维投影）
- $W_{up} \in \mathbb{R}^{64 \times 768}$（升维投影）
- $s \in \mathbb{R}$: 可学习缩放参数（初始化为1）

**适配器配置**:
| 参数 | 值 |
|------|-----|
| adapter_dim | 64 |
| dropout | 0.1 |
| 激活函数 | QuickGELU（$x \cdot \sigma(1.702x)$）|
| 权重初始化 | $W_{down}$: Xavier均匀；$W_{up}$: 零初始化 |
| 插入位置 | 全部12个 transformer blocks |
| 每块参数量 | $768 \times 64 \times 2 + 64 \times 2 = 98,432$ |

**12块总 Adapter 参数量**: $12 \times 98,432 \approx 1.18\text{M}$

### 3.3 GNSS 编码器 (GNSSEncoder)

**作用**: 将7维 GNSS 物理特征映射为与视觉特征同维度的 Query 向量

**结构**:
$$Q_{gnss} = \text{LayerNorm}(\text{Linear}_{768}(\text{GELU}(\text{Linear}_{128}(G))))$$

```
G (B, 7)
 → Linear(7, 128) → GELU
 → Linear(128, 768) → LayerNorm
 → Q_gnss (B, 768)
```

**参数量**: $7 \times 128 + 128 + 128 \times 768 + 768 + 768 = 99,840 \approx 0.1\text{M}$

### 3.4 Cross-Attention 融合模块 (CrossAttentionFusion)

**核心思想**: GNSS 特征作为 Query，"询问"视觉 token 中与当前农机状态最相关的区域

$$F = \text{MultiheadAttn}(Q=Q_{gnss}, K=V=X_{visual})$$

其中:
- $Q_{gnss} \in \mathbb{R}^{B \times 1 \times 768}$（扩展为序列维）
- $X_{visual} \in \mathbb{R}^{B \times (T \times 196) \times 768} = \mathbb{R}^{B \times 980 \times 768}$
- 输出: $F \in \mathbb{R}^{B \times 1 \times 768}$ → squeeze → $(B, 768)$

**多头注意力参数**:
| 参数 | 值 |
|------|-----|
| embed_dim | 768 |
| num_heads | 12 |
| head_dim | 64 |
| dropout | 0.0 |

**参数量**: $4 \times 768^2 \approx 2.36\text{M}$（Q/K/V/Out 投影）

### 3.5 分类头 (Classifier)

```
fused_feature (B, 768)
 → Dropout(0.1)
 → Linear(768, 11)
 → logits (B, 11)
```

**参数量**: $768 \times 11 + 11 = 8,459$

---

## 四、完整前向传播维度流

```
批次: B=8, T=5

输入:
  video:  (8, 5, 3, 224, 224)
  gnss:   (8, 7)

Step 1 — GNSS 编码:
  (8, 7) → Linear(7,128) → (8, 128) → GELU
         → Linear(128,768) → (8, 768) → LayerNorm
  Q_gnss: (8, 768)

Step 2 — 视觉编码 (每帧独立):
  reshape: (8, 5, 3, 224, 224) → (40, 3, 224, 224)
  patch embed: (40, 3, 224, 224) → (40, 196, 768)  [14×14 patches]
  prepend CLS: (40, 197, 768)
  12× ViT block (冻结 self-attn + FFN + 可训练 Adapter):
    → (40, 197, 768)
  取 patch tokens [1:]: (40, 196, 768)
  reshape back: (8, 5, 196, 768)
  flatten T×patches: (8, 980, 768)

Step 3 — Cross-Attention 融合:
  Q = Q_gnss.unsqueeze(1):  (8, 1, 768)
  K = V = visual_tokens:    (8, 980, 768)
  MultiheadAttn → attn_out: (8, 1, 768)
  squeeze: (8, 768)

Step 4 — 分类:
  Dropout → Linear(768, 11) → logits: (8, 11)
```

---

## 五、参数量分解

| 模块 | 参数量 | 可训练 | 说明 |
|------|--------|--------|------|
| ViT-B16 骨干 | ~86.0M | ❌ 冻结 | Patch embed + 12 blocks + norm |
| AdaptFormer (12块) | ~1.18M | ✅ | down+up proj × 12 |
| GNSS Encoder | ~0.10M | ✅ | MLP 7→128→768 |
| Cross-Attention | ~2.36M | ✅ | Q/K/V/Out 投影 |
| LayerNorm (CA) | ~3K | ✅ | 融合后归一化 |
| Classifier Head | ~8.5K | ✅ | Linear(768,11) |
| **可训练合计** | **~3.65M** | ✅ | 占总参数 4.1% |
| **总参数** | **~89.7M** | — | |

---

## 六、训练策略

### 优化器
```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=0.01
)
```

### 损失函数（处理类别不平衡）
```python
# 类别权重 = 1 / sqrt(count), 归一化
class_weights = 1 / torch.sqrt(class_counts)
class_weights = class_weights / class_weights.sum() * num_classes
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 学习率调度
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

### 训练参数
| 参数 | 值 |
|------|-----|
| batch_size | 8 |
| epochs | 50 |
| time_window T | 5 秒 |
| img_size | 224×224 |
| warmup_epochs | 3 |
| early_stopping patience | 10 |

---

## 七、创新点总结

1. **1Hz 严格对齐的极低开销跨模态融合**: GNSS 特征仅为7维单向量，通过轻量 MLP 编码为 Query，避免第二条 ViT 流的巨大开销（节省 ~86M 参数）

2. **参数高效视觉适应**: AdaptFormer 将可训练参数压缩至 ~3.65M（4.1%），在保持 ViT-B16 通用视觉表征的同时适应农业场景

3. **GNSS 条件视觉注意力**: 用当前帧的 GNSS 状态（速度/深度/方向）作为 Query，自动聚焦视频帧中与农机状态最相关的空间区域

---

*文档生成时间: 2026-03-06 | 关联代码: `src/models/`*
