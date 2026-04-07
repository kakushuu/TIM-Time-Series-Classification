# GAN-BiLSTM 论文数据处理方法总结

**论文标题**: GAN-BiLSTM network for field-road classification on imbalanced GNSS recordings

**任务**: 农业机械轨迹点的田间/道路二分类

---

## 1. 论文数据处理方法概述

### 1.1 数据源
- **原始数据**: 160 条轨迹，1,142,149 个 GNSS 轨迹点
- **采集间隔**: 5 秒
- **原始字段**: `[longitude, latitude, speed, direction, time, tag]`
- **类别分布**: 不平衡（田间点多于道路点）

### 1.2 数据处理流程

```
原始数据
  ↓
(1) 数据预处理: 去除重复点、漂移点
  ↓
(2) CTGAN 数据增强: 生成少数类样本平衡数据集
  ↓
(3) 运动特征计算: 5 个运动特征
  ↓
(4) 时间窗口特征提取: 20 个窗口特征 (5 特征 × 2 窗口 × 2 统计量)
  ↓
(5) 最终特征向量: 27 维 (25 运动特征 + 2 经纬度)
  ↓
(6) 序列化: 滑动窗口 (step=512) 构建序列
  ↓
Att-BiLSTM 模型
```

---

## 2. 详细数据处理步骤

### 2.1 数据预处理（论文 2.2.1 节）

**目标**: 去除噪声和冗余数据

**方法**:
1. **去除重复点**: 删除经度、纬度、速度完全相同的点
2. **去除漂移点**: 删除时间戳重复的点（GNSS 信号漂移导致）

**代码对应**: `cal_25_feature.py` 第 290-336 行

```python
# 去除经纬度速度重复的
waitdelete = []
allpoint = []
for j in range(len(speed)):
    point = []
    point.append([x[j], y[j], speed[j]])
    if point in allpoint:
        waitdelete.append(j)
    else:
        allpoint.append(point)
data2 = data2.drop(waitdelete)

# 去除时间重复的
timedelete = []
all_time_point = []
for j in range(len(time)):
    if time[j] in all_time_point:
        timedelete.append(j)
    else:
        all_time_point.append(time[j])
data2 = data2.drop(timedelete)
```

---

### 2.2 CTGAN 数据增强（论文 2.2.1 节）

**目标**: 解决田间/道路类别不平衡问题

**方法**:
- 使用 **CTGAN (Conditional Tabular GAN)** 生成少数类样本
- 条件输入: 类别标签的 one-hot 编码
- 生成目标: 少数类（通常是道路点）
- 生成数量: `abs(count_field - count_road)`

**代码对应**: `loader.py` 第 62-93 行

```python
# 训练 CTGAN
if opt.useGAN == True:
    data_train = np.c_[X_train, y_train]
    data_train = pd.DataFrame(data_train, columns=[...])

    if opt.useGAN_weights and os.path.exists(weightsPath):
        model = CTGANSynthesizer.load(weightsPath)
    else:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data_train)
        model = CTGANSynthesizer(metadata, epochs=opt.GAN_epoch, verbose=True)
        model.fit(data_train)
        model.save(weightsPath)

    # 生成少数类样本
    ser = data_train['tag'].value_counts()
    num_rows = abs(ser[0] - ser[1])
    cnd = 0
    if ser[0] > ser[1]:
        cnd = 1  # 生成道路点

    condition = Condition({'tag': cnd}, num_rows=num_rows)
    synthetic_train = model.sample_from_conditions(conditions=[condition])
    data_train = pd.concat([data_train, synthetic_train], axis=0)
```

**效果**（论文表 5-6）:
- 使用 CTGAN: 准确率 92.3%, 道路召回率 87.6%
- 不使用 CTGAN: 准确率 88.6%, 道路召回率 78.5%
- **准确率提升 3.7%, 道路召回率提升 9.1%**

---

### 2.3 运动特征计算（论文 2.2.2 节，公式 1-4）

**目标**: 从原始 GNSS 数据中提取 5 个运动特征

**公式映射**:

| 特征 | 论文公式 | 代码实现 | 说明 |
|------|---------|---------|------|
| **速度 (speed)** | 直接使用 | `speed_list = data['speed']` | 原始速度 |
| **加速度 (acceleration)** | $a_i = \frac{speed_{i+1} - speed_i}{time_{i+1} - time_i}$ | `acclec[j] = float(speed_diff[j]) / time_diff[j]` | 速度变化率 |
| **角度差 (angle_diff)** | $d_i = direction_{i+1} - direction_i$ | `angular_diff[l] = angular_list[l] - angular_list[l - 1]` | 方向变化 |
| **角速度 (angular_speed)** | $h_i = \frac{d_i}{time_{i+1} - time_i}$ | `angular_speed[temp] = angular_list[temp] / time_diff[temp]` | 角度变化率 |
| **角加速度 (angular_acceleration)** | $k_i = \frac{h_{i+1} - h_i}{time_{i+1} - time_i}$ | `angular_acclec[k] = angular_speed_diff[k] / time_diff[k]` | 角速度变化率 |

**代码对应**: `cal_25_feature.py` 第 128-195 行

```python
# 加速度 (公式 1)
for j in range(len(speed_list)):
    if j == 0:
        acclec[j] = 0
    else:
        speed_diff[j] = speed_list[j] - speed_list[j - 1]
        time_diff[j] = (d2 - d1).seconds
        acclec[j] = float(speed_diff[j]) / time_diff[j]

# 角度差 (公式 2)
for l in range(len(speed_list)):
    if l == 0:
        angular_diff[l] = 0
    else:
        angular_diff[l] = angular_list[l] - angular_list[l - 1]

# 角速度 (公式 3)
for temp in range(len(speed_list)):
    if temp == 0:
        angular_speed[temp] = 0
    else:
        angular_speed[temp] = angular_list[temp] / time_diff[temp]

# 角加速度 (公式 4)
for k in range(len(speed_list)):
    if k == 0:
        angular_acclec[k] = 0
    else:
        angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
        angular_acclec[k] = angular_speed_diff[k] / time_diff[k]
```

---

### 2.4 时间窗口特征提取（论文 2.2.2 节，公式 5-6）

**目标**: 捕获轨迹点的时空相关性

**方法**:
- 对 5 个运动特征分别应用时间窗口
- 每个特征使用 **2 个窗口大小**: 5 和 50
- 每个窗口计算 **2 个统计量**: 中位数 (Median) 和标准差 (SD)
- **总共**: 5 特征 × 2 窗口 × 2 统计量 = **20 个窗口特征**

**公式映射**:

| 统计量 | 论文公式 | 代码实现 |
|--------|---------|---------|
| **中位数 (Median)** | $MD_i = \text{Median}(x_{i-size}, ..., x_i)$ | `med[i] = get_median(lst[i-num:i+1])` |
| **标准差 (SD)** | $SD_i = \sqrt{\frac{\sum_{j=i-size}^{i}(x_j - \mu)^2}{size}}$ | `SD[i] = np.std(lst[i-num:i+1])` |

**代码对应**: `cal_25_feature.py` 第 240-270 行

```python
def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = data[size // 2]
    else:
        median = data[(size - 1) // 2]
    return median

def med(num, lst):
    med = [0 for x in range(len(lst))]
    med[0] = lst[0]
    for i in range(1, len(lst)):
        if i <= num:
            med[i] = get_median(lst[:i+1])
        else:
            med[i] = get_median(lst[i-num:i+1])
    return med

def SD(num, lst):
    SD = [0 for x in range(len(lst))]
    SD[0] = np.std([lst[0]])
    for i in range(1, len(lst)):
        if i <= num:
            SD[i] = np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD
```

**特征提取调用**: `cal_25_feature.py` 第 350-381 行

```python
cit1 = 5   # 小窗口
cit2 = 50  # 大窗口

# 对每个运动特征计算窗口统计量
speed_med_5 = med(cit1, speed_list)
speed_med_20 = med(cit2, speed_list)
speed_SD_5 = SD(cit1, speed_list)
speed_SD_20 = SD(cit2, speed_list)

acc_med_5 = med(cit1, acc_list)
acc_med_20 = med(cit2, acc_list)
acc_SD_5 = SD(cit1, acc_list)
acc_SD_20 = SD(cit2, acc_list)

# ... 对 angular_speed, angular_acceleration, angle_diff 同样处理
```

---

### 2.5 最终特征向量（论文 2.2.2 节）

**组成**: 27 维特征向量

```
[lon, lat,                                                    # 2 维经纬度
 speed, speed_med_5, speed_med_50, speed_SD_5, speed_SD_50,  # 5 维速度特征
 acceleration, acc_med_5, acc_med_50, acc_SD_5, acc_SD_50,    # 5 维加速度特征
 angular_speed, ang_med_5, ang_med_50, ang_SD_5, ang_SD_50,   # 5 维角速度特征
 angular_acc, angacc_med_5, angacc_med_50, angacc_SD_5, angacc_SD_50,  # 5 维角加速度特征
 angle_diff, diff_med_5, diff_med_50, diff_SD_5, diff_SD_50]  # 5 维角度差特征
```

**代码对应**: `cal_25_feature.py` 第 385-430 行

```python
columns = ['speed', 'speed_med_5', 'speed_med_20', 'speed_SD_5', 'speed_SD_20',
           'acceleration', 'acceleration_med_5', 'acceleration_med_20', 'acceleration_SD_5', 'acceleration_SD_20',
           'angular_speed', 'angular_speed_med_5', 'angular_speed_med_20', 'angular_speed_SD_5', 'angular_speed_SD_20',
           'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_20', 'angular_acceleration_SD_5', 'angular_acceleration_SD_20',
           'angle_diff', 'angle_diff_med_5', 'angle_diff_med_20', 'angle_diff_SD_5', 'angle_diff_SD_20',
           'tag', 'lon', 'lat']
```

---

### 2.6 序列化处理（代码特有，论文未详述）

**目标**: 将独立轨迹点转换为序列输入

**方法**: 滑动窗口 (step=512)

**代码对应**: `loader.py` 第 19-28 行

```python
def transform_dataset(x_data, y_data, n_input, n_output):
    data_size = x_data.shape[0]
    X = np.empty((data_size - n_input + 1, n_input, x_data.shape[1]))
    Y = np.empty((data_size - n_input + 1, y_data.shape[1]))
    for i in range(data_size - n_input + 1):
        X[i] = x_data[i:i + n_input, :]  # 窗口内 512 个点
        Y[i] = y_data[i + n_input - 1, :]  # 最后一个点的标签
    return X, Y
```

**输出形状**:
- `X`: `(batch_size, 512, 27)` — 512 个连续轨迹点，每个点 27 维特征
- `Y`: `(batch_size, 1)` — 窗口最后一个点的标签

---

## 3. Att-BiLSTM 模型架构（论文 2.2.3 节）

### 3.1 模型结构

```
输入: (batch_size, 512, 27)
  ↓
BatchNorm1d (特征标准化)
  ↓
BiLSTM (2 层, hidden_size=256)
  → 前向: 256 维
  → 后向: 256 维
  → 拼接: 512 维
  ↓
LayerNorm
  ↓
Attention (权重计算)
  → α = tanh(W · H)
  → 加权求和
  ↓
3 层 MLP (512 → 512 → n_classes)
  ↓
输出: (batch_size, 2)  [田间, 道路]
```

**代码对应**: `models/lstm.py` 第 28-91 行

```python
class AttBiLSTM(nn.Module):
    def __init__(self, n_classes, emb_size, rnn_size, rnn_layers, dropout):
        super(AttBiLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.attention = Attention(rnn_size)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
        self.batchnorm = nn.BatchNorm1d(emb_size)
        self.layernorm = nn.LayerNorm(rnn_size)

    def forward(self, x):
        x = self.batchnorm(x)
        rnn_out, _ = self.BiLSTM(x)
        H = rnn_out[:, :, :self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        H = self.layernorm(H)
        r, alphas = self.attention(H)
        h = self.tanh(r)
        scores = self.classifier(h)
        return scores
```

### 3.2 Attention 机制（论文公式 9-10）

**目标**: 自动聚焦关键轨迹点，降低噪声影响

**公式**:
- 注意力权重: $\alpha_t = \tanh(W_t^\alpha \cdot h_t)$
- 加权输出: $\text{pred} = \text{softmax}(\sum_{t=1}^{T} \alpha_t \cdot h_t)$

**代码对应**: `models/lstm.py` 第 9-25 行

```python
class Attention(nn.Module):
    def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)               # (batch, 512, 256)
        alpha = self.w(M).squeeze(2)   # (batch, 512)
        alpha = self.softmax(alpha)     # (batch, 512)
        r = H * alpha.unsqueeze(2)     # (batch, 512, 256)
        r = r.sum(dim=1)                # (batch, 256)
        return r, alpha
```

---

## 4. Focal Loss（论文 2.2.4 节）

**目标**: 聚焦难分类样本，提升少数类性能

**公式** (论文公式 11):
$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

**代码对应**: `models/lossFun.py`（未直接提供，但使用 PyTorch 实现）

**效果**（论文表 5-6）:
- 使用 Focal Loss: 准确率 92.3%, 道路召回率 87.6%
- 使用交叉熵: 准确率 90.2%, 道路召回率 84.2%
- **准确率提升 2.1%, 道路召回率提升 3.4%**

---

## 5. 关键设计决策

### 5.1 为什么用时间窗口特征？

**论文解释**:
- 农业机械轨迹具有**时空相关性**
- 田间作业: 低速均匀运动 → 运动特征近似
- 道路行驶: 变速运动 → 运动特征变化大
- 时间窗口捕获这种**局部上下文**

**窗口大小选择**:
- 小窗口 (5): 捕获短期模式
- 大窗口 (50): 捕获长期趋势

### 5.2 为什么用 BiLSTM + Attention？

**BiLSTM 优势**:
- 双向信息流：前向 + 后向上下文
- 时序建模：捕获轨迹点依赖关系

**Attention 优势**:
- 自动权重分配：聚焦关键点
- 噪声抑制：低权重处理异常点

### 5.3 为什么用 CTGAN 而非下采样？

**对比实验**（论文表 5-6）:

| 方法 | 准确率 | F1 分数 | 道路召回率 |
|------|--------|---------|-----------|
| **CTGAN** | 92.3% | 92.1% | 87.6% |
| **下采样** | 89.3% | 89.3% | 88.6% |
| **不处理** | 88.6% | 88.0% | 78.5% |

**CTGAN 优势**:
- 保留所有原始数据（下采样丢弃数据）
- 学习数据分布，生成高质量样本
- 准确率提升 3.0%

---

## 6. 完整数据处理流程图

```
原始 GNSS 数据 (lon, lat, speed, direction, time, tag)
  ↓
【预处理】去除重复点 + 漂移点
  ↓
【CTGAN】生成少数类样本 → 平衡数据集
  ↓
【运动特征计算】
  - speed (原始)
  - acceleration = Δspeed / Δtime        (公式 1)
  - angle_diff = Δdirection              (公式 2)
  - angular_speed = angle_diff / Δtime   (公式 3)
  - angular_acceleration = Δangular_speed / Δtime  (公式 4)
  ↓
【时间窗口特征】
  对 5 个运动特征分别:
    - Median(window=5)  (公式 5)
    - Median(window=50)
    - SD(window=5)      (公式 6)
    - SD(window=50)
  → 20 个窗口特征
  ↓
【最终特征】27 维向量
  [lon, lat, 5 运动 × 5 统计量]
  ↓
【序列化】滑动窗口 (step=512)
  → (batch_size, 512, 27)
  ↓
【Att-BiLSTM】
  BatchNorm → BiLSTM → LayerNorm → Attention → MLP
  ↓
【Focal Loss 训练】
  ↓
输出: 田间/道路二分类
```

---

## 7. 与 Agri-MBT 项目的差异

| 方面 | GAN-BiLSTM 论文 | Agri-MBT 项目 |
|------|----------------|---------------|
| **任务** | 二分类（田间/道路） | 11 分类（农业活动） |
| **数据增强** | ✅ CTGAN | ❌ 无（数据充足） |
| **轨迹特征** | 27 维（时间窗口） | 6 维（经纬度、速度、方向、深度、间距） |
| **时序建模** | BiLSTM (512 步) | BiLSTM (8 步) |
| **视觉模态** | ❌ 无 | ✅ ViT + MBT 融合 |
| **损失函数** | Focal Loss | CrossEntropy |

---

## 8. 可借鉴的设计

### 8.1 适用于 Agri-MBT 的部分

1. **时间窗口特征**: 可在轨迹预处理阶段添加
2. **Focal Loss**: 用于处理 11 类不平衡问题
3. **Attention 机制**: 可集成到 BiLSTM 编码器

### 8.2 不适用的部分

1. **CTGAN**: Agri-MBT 数据已充足（32K 样本）
2. **27 维特征**: 与当前 6 维特征体系不同
3. **二分类**: Agri-MBT 是 11 分类任务

---

## 9. 代码文件对应关系

| 论文章节 | 功能 | 代码文件 | 关键函数 |
|---------|------|---------|---------|
| 2.2.1 数据预处理 | 去重去噪 | `cal_25_feature.py` | 第 290-336 行 |
| 2.2.1 CTGAN | 数据增强 | `utils/loader.py` | `get_data()` 第 62-93 行 |
| 2.2.2 运动特征 | 公式 1-4 | `cal_25_feature.py` | 第 128-195 行 |
| 2.2.2 时间窗口 | 公式 5-6 | `cal_25_feature.py` | `med()`, `SD()` 第 240-270 行 |
| 2.2.3 Att-BiLSTM | 模型架构 | `models/lstm.py` | `AttBiLSTM` 类 |
| 2.2.3 Attention | 注意力机制 | `models/lstm.py` | `Attention` 类 |
| 2.2.4 Focal Loss | 损失函数 | `models/lossFun.py` | （未提供代码） |
| - | 序列化 | `utils/loader.py` | `transform_dataset()` 第 19-28 行 |

---

## 10. 总结

GAN-BiLSTM 论文的数据处理方法核心在于：

1. **时空特征工程**: 运动特征 + 时间窗口统计量
2. **数据平衡**: CTGAN 生成少数类样本
3. **序列建模**: BiLSTM 捕获双向时序依赖
4. **噪声抑制**: Attention 自动加权关键点
5. **难样本聚焦**: Focal Loss 提升少数类性能

这套方法在田间/道路二分类任务上达到了 **92.3% 准确率**，相比基线方法提升 2.5-8.2%。

对于 Agri-MBT 项目，可借鉴其**时序特征工程**和**Attention 机制**，但 CTGAN 数据增强不适用（数据充足）。
