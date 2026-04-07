# GAN-BiLSTM 风格改造 - 27 维特征 + 512 步序列

## 改造概述

将 Agri-MBT 项目的轨迹处理方式改为仿照 GAN-BiLSTM 论文：

### 改造前（6 维特征，8 步序列）
```
TRAJ_DIM = 6   # [经度, 纬度, 间距, 深度, 速度, 方向角]
TRAJ_SEQ = 8   # 8 帧滑动窗口
```

### 改造后（27 维特征，512 步序列）
```
TRAJ_DIM = 27  # 2 经纬度 + 5 运动特征 × 5 统计量
TRAJ_SEQ = 512 # 512 帧滑动窗口（仿照 GAN-BiLSTM）
```

---

## 27 维特征组成

### 1. 经纬度（2 维）
- 经度 (lon)
- 纬度 (lat)

### 2. 运动特征（5 维）

| 特征 | 公式 | 说明 |
|------|------|------|
| **speed** | 直接使用 | 原始速度 |
| **acceleration** | $\frac{speed_{i+1} - speed_i}{time_{i+1} - time_i}$ | 加速度 |
| **angle_diff** | $direction_{i+1} - direction_i$ | 角度差 |
| **angular_speed** | $\frac{angle\_diff_i}{\Delta time}$ | 角速度 |
| **angular_acceleration** | $\frac{\Delta angular\_speed}{\Delta time}$ | 角加速度 |

### 3. 时间窗口统计量（每个运动特征 × 5）

对 5 个运动特征分别计算：
1. 原始值
2. Median(window=5)
3. Median(window=50)
4. SD(window=5) - 标准差
5. SD(window=50)

**总计**: 5 特征 × 5 统计量 = **25 维**

### 4. 最终特征向量（27 维）

```python
[
  '经度', '纬度',  # 2 维

  # 速度特征（5 维）
  '速度', 'speed_med_5', 'speed_med_50', 'speed_SD_5', 'speed_SD_50',

  # 加速度特征（5 维）
  'acceleration', 'acceleration_med_5', 'acceleration_med_50',
  'acceleration_SD_5', 'acceleration_SD_50',

  # 角速度特征（5 维）
  'angular_speed', 'angular_speed_med_5', 'angular_speed_med_50',
  'angular_speed_SD_5', 'angular_speed_SD_50',

  # 角加速度特征（5 维）
  'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_50',
  'angular_acceleration_SD_5', 'angular_acceleration_SD_50',

  # 角度差特征（5 维）
  'angle_diff', 'angle_diff_med_5', 'angle_diff_med_50',
  'angle_diff_SD_5', 'angle_diff_SD_50'
]
```

---

## 代码修改

### 1. 特征计算脚本

**文件**: `scripts/compute_27_features.py`

**功能**:
- 从原始 5 列数据计算 27 维特征
- 运动特征计算（公式 1-4）
- 时间窗口统计（公式 5-6）

**运行**:
```bash
python scripts/compute_27_features.py \
  --input data/aligned_output/aligned_data.csv \
  --output data/aligned_output/aligned_data_27features.csv
```

**输出**: `aligned_data_27features.csv` (32249 行 × 29 列)

---

### 2. Dataloader 修改

**文件**: `Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/dataloader/av_data.py`

**修改**:
```python
# 修改前
TRAJ_COLS = ['经度', '纬度', '间距(米)', '深度', '速度', '方向角']
TRAJ_SEQ  = 8

# 修改后
TRAJ_COLS = [
    '经度', '纬度',
    '速度', 'speed_med_5', 'speed_med_50', 'speed_SD_5', 'speed_SD_50',
    'acceleration', 'acceleration_med_5', 'acceleration_med_50', 'acceleration_SD_5', 'acceleration_SD_50',
    'angular_speed', 'angular_speed_med_5', 'angular_speed_med_50', 'angular_speed_SD_5', 'angular_speed_SD_50',
    'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_50', 'angular_acceleration_SD_5', 'angular_acceleration_SD_50',
    'angle_diff', 'angle_diff_med_5', 'angle_diff_med_50', 'angle_diff_SD_5', 'angle_diff_SD_50'
]
TRAJ_SEQ  = 512
```

---

### 3. 模型修改

**文件**: `Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/models/visual_model.py`

**修改**:
```python
# 修改前
TRAJ_DIM = 6
TRAJ_SEQ = 8

# 修改后
TRAJ_DIM = 27
TRAJ_SEQ = 512
```

**BiLSTM 输入维度变化**:
- 输入: `(bs, 512, 27)` （原来是 `(bs, 8, 6)`）
- 输出: `(bs, 513, 768)` （cls token + 512 hidden states）

---

### 4. 训练脚本修改

**文件**: `Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/train_test.py`

**修改**:
```python
# 默认数据文件路径
parser.add_argument('--csv_file', default='../../data/aligned_output/aligned_data_27features.csv')
```

---

## 实验计划

### 实验 1: Trajectory-Only (27D, 512 seq)

**命令**:
```bash
cd Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT
python -u train_test.py --mode trajectory_only --num_epochs 15 \
  2>&1 | tee ../../experiments/train_trajectory_only_27d_512seq.log
```

**模型**: BiLSTM → MLP (3 层) → 11 类分类

**预期**:
- 训练时间: 较长（512 步序列 + BiLSTM 计算量大）
- 性能: 应优于之前的 trajectory-only (38.95%)，因为特征更丰富

---

### 实验 2: Multimodal (27D, 512 seq + ViT)

**命令**:
```bash
cd Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT
python -u train_test.py --mode multimodal --num_epochs 15 \
  2>&1 | tee ../../experiments/train_multimodal_27d_512seq.log
```

**模型**: BiLSTM + ViT → MBT 融合 → 11 类分类

**预期**:
- 性能: 应接近或超过之前的 multimodal (94.18%)
- 训练时间: 更长（需要加载图像 + 512 步轨迹）

---

## 参数量对比

### Trajectory-Only

| 组件 | 改造前 (6D, 8 seq) | 改造后 (27D, 512 seq) |
|------|-------------------|---------------------|
| BiLSTM 输入层 | 6 × 384 × 2 = 4,608 | 27 × 384 × 2 = 20,736 |
| BiLSTM hidden | ~4.7M | ~4.7M |
| MLP | 768→512→256→11 | 768→512→256→11 |
| **总参数** | **5,280,268** | **5,296,396** |

**变化**: +16K 参数（BiLSTM 输入层）

---

### Multimodal

| 组件 | 改造前 | 改造后 |
|------|--------|--------|
| BiLSTM | 同上 | 同上 |
| ViT (冻结) | 86M | 86M |
| AdaptFormer | ~5.1M | ~5.1M |
| **总参数** | **5,115,900** | **5,132,228** |

**变化**: +16K 参数（BiLSTM 输入层）

---

## 时间窗口统计量说明

### Median（中位数）

```python
def med(num, lst):
    for i in range(len(lst)):
        if i <= num:
            med[i] = median(lst[:i+1])
        else:
            med[i] = median(lst[i-num:i+1])
```

**作用**: 捕获局部趋势，对噪声鲁棒

### SD（标准差）

```python
def SD(num, lst):
    for i in range(len(lst)):
        if i <= num:
            SD[i] = std(lst[:i+1])
        else:
            SD[i] = std(lst[i-num:i+1])
```

**作用**: 捕获运动特征的变异性

---

## 与 GAN-BiLSTM 论文的差异

| 方面 | GAN-BiLSTM 论文 | Agri-MBT 改造后 |
|------|----------------|----------------|
| **任务** | 二分类（田间/道路） | 11 分类（农业活动） |
| **序列长度** | 512 | 512 |
| **特征维度** | 27 | 27 |
| **数据增强** | ✅ CTGAN | ❌ 无 |
| **损失函数** | Focal Loss | CrossEntropy |
| **视觉模态** | ❌ 无 | ✅ ViT + MBT |
| **时间窗口** | 5, 50 | 5, 50 |

---

## 内存和计算开销估算

### 单个样本内存（512 步序列）

- 轨迹数据: `512 × 27 × 4 bytes = 55 KB`
- BiLSTM 隐藏状态: `512 × 768 × 4 bytes = 1.5 MB`
- 梯度: 约 3 MB
- **总计**: ~5 MB/sample

### Batch=8 内存

- 前向: `8 × 5 MB = 40 MB`
- 反向: `~120 MB`
- **总计**: ~160 MB（仅轨迹）

### Multimodal 额外开销

- ViT 特征: `8 × 1 × 197 × 768 × 4 = 4.8 MB`
- MBT 融合: ~10 MB
- **总计**: ~175 MB

---

## 预期训练时间

### Trajectory-Only

- **单 epoch**: 约 5-10 分钟（BiLSTM 512 步较慢）
- **15 epochs**: 约 1.5-2.5 小时

### Multimodal

- **单 epoch**: 约 10-15 分钟（额外加载图像）
- **15 epochs**: 约 2.5-4 小时

---

## 当前状态

✅ **已完成**:
1. 创建特征计算脚本 `scripts/compute_27_features.py`
2. 生成 27 维特征数据 `aligned_data_27features.csv`
3. 修改 dataloader 支持新特征
4. 修改模型支持新输入维度
5. 启动 trajectory-only 训练

🔄 **进行中**:
- Trajectory-Only 训练（15 epochs）

⏳ **待完成**:
- Multimodal 训练（15 epochs）
- 性能对比分析

---

## 监控命令

```bash
# 检查训练进度
tail -f experiments/train_trajectory_only_27d_512seq.log

# 检查 GPU 使用
watch -n 1 nvidia-smi

# 检查进程
ps aux | grep train_test.py
```
