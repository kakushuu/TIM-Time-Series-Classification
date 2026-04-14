# TAIF-Net 技术路线文档

## 1. 研究任务

本研究面向 `Agricultural Machinery Trajectory Time-Series Classification` 任务，目标是利用农机运行过程中的轨迹时间序列与同步视频信息，对农机作业状态进行 11 分类识别。

统一类别定义如下：

- `0` Reverse empty harvesting
- `1` Straight empty harvesting
- `2` Turning empty harvesting
- `3` Full-load harvesting
- `4` Reverse transfer
- `5` Straight transfer
- `6` Turning transfer
- `7` Engine-off waiting
- `8` Idling waiting
- `9` Unloading
- `10` Road driving

该任务覆盖田间作业、转运、等待、卸粮和道路行驶等完整作业流程，比传统 field-road 二分类或粗粒度工况识别更接近真实农业生产场景。

## 2. 研究背景与问题定义

现有农业机械轨迹分类研究主要依赖 GNSS 轨迹及其派生运动学特征，对 coarse-grained 场景有效，但在 fine-grained 作业状态识别中存在明显不足：

- 不同类别可能具有相似轨迹模式。例如 `Engine-off waiting` 与 `Idling waiting` 在位移层面接近，但物理工况不同。
- 视频与轨迹存在时间异步问题。农业场景中常见 OCR 时间误差、采样频率不一致、帧缺失和动态延迟。
- 类别分布天然不均衡，尾部类别容易在训练中被头部类别淹没。
- 多模态融合时，强模态容易主导预测，导致弱模态贡献不足。

基于上述问题，本研究拟提出一个统一解决时间对齐、轨迹建模和长尾鲁棒融合的多模态框架。

## 3. 总体技术路线

本研究设计 `TAIF-Net`，即 `Temporal Alignment and Imbalance-aware Fusion Network`。整体流程如下：

1. 对原始视频和 GNSS 轨迹进行时间同步与样本构建，得到以固定时间窗口组织的多模态训练样本。
2. 对轨迹序列提取位置、速度、方向、曲率、加速度等多维特征，构建轨迹输入张量。
3. 对视频片段进行抽帧和视觉编码，提取时空视觉 token。
4. 通过可学习时间对齐模块建立轨迹 token 与视频 token 的软对应关系。
5. 通过多尺度轨迹模式编码器提取局部运动特征和全局作业状态特征。
6. 通过长尾鲁棒 bottleneck 融合模块完成跨模态交互与最终分类。
7. 采用分类损失、对齐损失、类别原型损失和模态均衡正则进行联合训练。

整体逻辑可以概括为：

`数据对齐 -> 轨迹/视频双分支编码 -> 可学习时间匹配 -> 类别不平衡感知融合 -> 11分类输出`

## 4. 模块设计

### 4.1 数据层：多模态样本构建

输入数据包括：

- GNSS 轨迹记录：时间、经纬度、速度、航向角等
- 视频数据：与轨迹同步采集的农机运行视频

预处理过程包括：

1. 视频按固定频率抽帧，形成时间连续的图像序列或短视频片段。
2. 轨迹时间戳统一到同一时间基准，进行插值、去噪和异常值剔除。
3. 以固定滑动窗口构建样本，例如每个样本对应 `T` 个轨迹点和 `N` 个视频帧。
4. 根据作业标注生成 11 类监督标签。

输出样本形式为：

- `trajectory tensor`: `X_t in R^(T x F)`
- `video tensor`: `X_v in R^(N x C x H x W)`
- `label`: `y in {0, ..., 10}`

### 4.2 时间对齐模块 TAM

为解决轨迹与视频不同步问题，设计 `Temporal Alignment Module (TAM)`。

核心思想是：

- 不把时间对齐完全依赖于预处理阶段的硬匹配
- 在网络内部学习轨迹片段与视频片段之间的软对齐关系

实现方式：

1. 视频分支输出时序视觉 token `V = {v_1, ..., v_n}`
2. 轨迹分支输出轨迹 token `T = {t_1, ..., t_m}`
3. 以轨迹 token 为 query，以视频 token 为 key/value 计算 cross-attention：

```text
A = softmax((Q_t K_v^T) / sqrt(d))
V_aligned = A V
```

其中：

- `A` 表示轨迹时刻到视频时刻的软匹配权重
- `V_aligned` 表示与轨迹语义对齐后的视觉表示

为稳定对齐过程，引入两个约束：

- `Alignment Consistency Loss`：约束相邻轨迹点的对齐中心变化平滑
- `Timestamp Prior Loss`：利用已有粗时间戳作为弱监督先验

这样可以显著降低时间戳误差对分类性能的影响。

### 4.3 轨迹模式编码器 TPE

为增强轨迹分支的有效建模能力，设计 `Trajectory Pattern Encoder (TPE)`。

考虑到农业轨迹同时包含短时动作变化与长时作业状态变化，采用双分支结构：

#### 4.3.1 局部运动分支

局部运动分支用于捕捉短时动作模式，例如：

- 倒车
- 转弯
- 停止后再启动
- 卸粮前后的速度波动

可采用：

- `1D-CNN`
- `short-window BiLSTM`

输入为局部窗口轨迹特征，输出局部运动表示 `h_local`。

#### 4.3.2 全局状态分支

全局状态分支用于建模完整作业过程中的长依赖关系，例如：

- 连续收获
- 转运状态
- 长时间等待
- 道路行驶过程

可采用：

- `BiLSTM`
- 轻量级时序 Transformer

输出全局状态表示 `h_global`。

#### 4.3.3 轨迹特征融合

将局部与全局轨迹表示进行门控融合：

```text
h_traj = Gate([h_local, h_global])
```

同时引入可学习的 `operation-pattern tokens`，作为农业作业模式原型，引导模型学习更稳定的作业状态结构。

建议输入特征包括：

- 位置特征：`x, y`
- 运动学特征：`speed, acceleration, heading, yaw rate`
- 几何特征：`curvature, turning density`
- 统计特征：窗口均值、方差、中位数、极值

### 4.4 视频编码器 VE

视频分支负责提取场景上下文、作业对象状态和视觉动作线索。

建议路线如下：

- 轻量基线：`ViT + frame aggregation`
- 提升方案：`VideoMAE / Video Swin`

视频分支输出时序视觉 token `h_video`，重点表达：

- 农机周围场景变化
- 卸粮、等待、道路行驶等外观线索
- 转弯、直行、倒车等动作上下文

视频编码器既可以直接用于分类，也作为 TAM 和融合模块的输入。

### 4.5 长尾鲁棒融合模块 IBF

为避免视频模态完全压制轨迹模态，同时提升尾部类别识别效果，设计 `Imbalance-aware Bottleneck Fusion (IBF)` 模块。

该模块在 MBT 的 bottleneck token 机制基础上加入两类改进。

#### 4.5.1 模态可信度门控

对每个样本预测两种模态的可信度权重：

- `alpha_v`: 视频可信度
- `alpha_t`: 轨迹可信度

可信度可由以下因素驱动：

- 视频模糊、遮挡、光照变化
- 轨迹丢点、抖动、速度异常

融合表示定义为：

```text
z = alpha_v * z_v + alpha_t * z_t + z_b
```

其中 `z_b` 为 bottleneck 交互后的共享表示。

#### 4.5.2 类别不平衡感知校准

针对头部类和尾部类判别边界不均衡的问题，在融合阶段加入类别先验约束。

可采用以下策略：

- `class-balanced focal loss`
- `logit adjustment`
- `class prototype bank`

其中 `class prototype bank` 用于维护每一类的融合原型表示，增强少数类特征聚合能力。

## 5. 训练目标

总损失函数设计为：

```text
L = L_cls + lambda1 * L_align + lambda2 * L_proto + lambda3 * L_balance
```

各部分定义如下：

- `L_cls`：主分类损失，建议使用 class-balanced focal loss 或 weighted cross-entropy
- `L_align`：时间对齐损失，约束轨迹与视频软对齐的一致性
- `L_proto`：类别原型损失，增强类内紧凑性和类间可分性
- `L_balance`：模态均衡正则，防止融合时出现单模态完全主导

训练策略建议：

1. 先单独训练轨迹分支和视频分支，建立稳定基线。
2. 再进行 TAIF-Net 联合训练。
3. 对尾部类别使用重加权采样或 class-balanced loss。
4. 对时间对齐模块采用较小学习率，避免早期对齐崩溃。

## 6. 创新点归纳

本研究拟形成以下三项核心创新：

### 创新点 1：弱监督可学习时间对齐

针对农业场景中视频与轨迹时间异步问题，提出 TAM 模块，在粗时间戳先验基础上学习轨迹片段与视频片段之间的软时序对应关系，提升跨模态配准鲁棒性。

### 创新点 2：面向农业作业模式的多尺度轨迹建模

针对细粒度作业状态在轨迹层面差异细微的问题，提出 TPE 模块，联合建模局部运动模式与全局作业状态，并引入 operation-pattern tokens 以增强农业作业先验表征能力。

### 创新点 3：类别不平衡感知的多模态 bottleneck 融合

针对长尾类别和强模态主导问题，提出 IBF 模块，通过模态可信度建模与类别原型约束提升尾部类别的分类表现和多模态协同效果。

## 7. 实验设计路线

### 7.1 基线模型

为验证 TAIF-Net 的有效性，设置如下基线：

- `Trajectory only`
- `Video only`
- `Early Fusion`
- `Late Fusion`
- `MBT baseline`

### 7.2 消融实验

建议进行以下消融：

- `MBT baseline`
- `MBT + TAM`
- `MBT + TPE`
- `MBT + IBF`
- `TAIF-Net full`

进一步可细分：

- 去除 `Alignment Consistency Loss`
- 去除 `Timestamp Prior Loss`
- 去除 `operation-pattern tokens`
- 去除 `class prototype bank`
- 去除 `modality reliability gate`

### 7.3 评价指标

由于类别不平衡显著，建议同时报告：

- Accuracy
- Macro-F1
- Weighted-F1
- mAP
- Per-class Recall
- Confusion Matrix

其中 `Macro-F1` 和 `Per-class Recall` 应作为核心指标，因为它们更能反映尾部类别的识别能力。

## 8. 预期科研产出

基于本技术路线，预期形成以下科研产出：

1. 一个面向农业场景细粒度作业识别的 11 类统一任务定义。
2. 一个兼顾时间对齐、轨迹表征和长尾融合的多模态框架 TAIF-Net。
3. 一组系统的单模态、多模态与消融实验结果。
4. 一套可扩展到音频、视频、轨迹三模态融合的农业机械行为识别研究范式。

## 9. 后续实施建议

在工程实现上，建议分三个阶段推进：

### 第一阶段：构建可靠基线

- 固定 11 类任务定义
- 完成数据对齐与样本清洗
- 建立 trajectory-only 和 video-only 基线

### 第二阶段：实现核心模块

- 实现 TAM
- 实现 TPE
- 在现有 MBT 上实现 IBF

### 第三阶段：实验与论文写作

- 补充消融实验和可视化分析
- 绘制时间对齐注意力图和类别混淆矩阵
- 将技术路线整理为论文方法章节与实验章节

## 10. 总结

TAIF-Net 的技术路线不是简单堆叠多模态网络，而是围绕农业机械轨迹时间序列分类中的三个关键痛点展开：

- 时间异步导致跨模态配准不稳定
- 轨迹分支难以稳定表达细粒度作业模式
- 类别不平衡和强模态主导削弱多模态融合效果

因此，本研究通过 `TAM + TPE + IBF` 的统一设计，将时间对齐、轨迹建模和长尾鲁棒融合纳入同一框架中，形成一个具有明确问题导向、较强农业场景针对性和较好论文表达性的技术路线。
