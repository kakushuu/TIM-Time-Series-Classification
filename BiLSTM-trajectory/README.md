# BiLSTM Trajectory Classification

基于BiLSTM的轨迹分类实验，用于11类农业活动识别。

## 项目说明

本项目修改自 GAN-BiLSTM，主要改动：
- ✅ 从2分类改为11分类（农业活动类别）
- ✅ 从25维特征改为36维轨迹特征
- ✅ 去掉GAN数据增强模块（数据充足）
- ✅ 适配Agri-MBT数据集格式

## 实验设置

### 实验1：BiLSTM Trajectory Only
- 输入：36维轨迹特征（8帧序列）
- 模型：BiLSTM + Attention
- 目标：仅使用轨迹特征进行分类

### 实验2：BiLSTM Multimodal
- 输入：36维轨迹特征 + 768维图像特征
- 模型：BiLSTM + Image Feature Fusion
- 目标：轨迹+图像多模态融合

## 项目结构

```
BiLSTM-trajectory/
├── opt.py                   # 配置文件
├── train.py                 # 训练主脚本
├── trainer.py               # 训练器
├── models/
│   ├── lstm.py              # BiLSTM模型定义
│   └── lossFun.py           # 损失函数
├── utils/
│   ├── loader.py            # 数据加载器
│   └── metrics.py           # 评估指标
└── run_experiments.sh       # 实验运行脚本
```

## 数据格式

### 输入数据：`aligned_data.csv`

**轨迹特征（36维）：**
- GPS特征：经度、纬度、间距、深度、速度、方向角
- 惯性传感器：加速度(x,y,z)、角速度(x,y,z)、线性加速度(x,y,z)
- 方向传感器：重力(x,y,z)、磁力(x,y,z)、旋转矢量(x,y,z)
- 陀螺仪：陀螺仪(x,y,z)
- 环境传感器：气压、温度、湿度、光照
- 统计特征：距离_5s、速度_5s、加速度_5s、方向变化_5s

**标签：**
- 分类（0-10）：11个农业活动类别

## 使用方法

### 1. 安装依赖

```bash
pip install torch torchvision numpy pandas scikit-learn tqdm pytorch-warmup
```

### 2. 运行单个实验

```bash
# 仅轨迹实验
python train.py --mode trajectory_only --epochs 50

# 多模态实验
python train.py --mode multimodal --epochs 50
```

### 3. 运行所有实验

```bash
bash run_experiments.sh
```

### 4. 自定义参数

```bash
python train.py \
  --mode trajectory_only \
  --data /path/to/aligned_data.csv \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cuda:0
```

## 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_classes` | 11 | 类别数 |
| `emb_size` | 36 | 轨迹特征维度 |
| `rnn_size` | 256 | LSTM隐藏层大小 |
| `rnn_layers` | 2 | LSTM层数 |
| `dropout` | 0.3 | Dropout率 |
| `time_tri` | 8 | 序列长度（帧数） |

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LEARNING_RATE` | 3e-4 | 学习率 |
| `batch_size` | 64 | 批次大小 |
| `epochs` | 50 | 训练轮数 |
| `optimizer` | adamw | 优化器 |
| `loss` | Focal | 损失函数（Focal/CE） |
| `testRatio` | 0.2 | 测试集比例 |
| `valRatio` | 0.2 | 验证集比例 |

## 输出结果

训练完成后，结果保存在 `experiments/results/` 目录：

```
experiments/results/
├── results_trajectory_only.json    # 轨迹实验结果
├── results_multimodal.json         # 多模态实验结果
└── weights/
    ├── BiLSTM_Trajectory_best.pth  # 最佳模型
    └── BiLSTM_Trajectory_last.pth  # 最终模型
```

### 结果格式

```json
{
  "mode": "trajectory_only",
  "best_val_acc": 75.5,
  "final_train_acc": 85.2,
  "final_val_acc": 74.8,
  "test_acc": 73.5,
  "test_metrics": {
    "accuracy": 73.5,
    "f1_macro": 68.2,
    "f1_weighted": 72.1,
    ...
  },
  "history": {
    "train_loss": [...],
    "train_acc": [...],
    "val_loss": [...],
    "val_acc": [...]
  }
}
```

## 预期结果

根据 MBT 多模态模型的实验结果：

| 模型 | 准确率 | Macro F1 |
|------|--------|----------|
| Trajectory Only | ~42% | ~7% |
| BiLSTM Trajectory | ~65-75% | ~55-65% |
| BiLSTM Multimodal | ~70-80% | ~60-70% |
| MBT Multimodal (对比) | 94.72% | 80.00% |

**说明：**
- 纯轨迹特征的表现受限于GPS噪声和类别不平衡
- BiLSTM能够捕获时序依赖，提升性能
- 多模态融合能进一步提升准确率

## 注意事项

1. **数据路径**：确保 `data_dir` 指向正确的 `aligned_data.csv` 文件
2. **GPU内存**：如果GPU内存不足，减小 `batch_size` 或 `rnn_size`
3. **类别不平衡**：使用 Focal Loss 处理类别不平衡问题
4. **序列长度**：`time_tri=8` 匹配 MBT 模型的8帧设置

## 与 MBT 模型对比

| 特性 | BiLSTM | MBT |
|------|--------|-----|
| 轨迹编码 | BiLSTM | MLP → 6×6 Feature Map |
| 图像编码 | - | ViT-B16 + AdaptFormer |
| 融合方式 | Concat | Multimodal Bottleneck |
| 参数效率 | 全量训练 | AdaptFormer微调 |
| 性能 | 中等 | 最优 |

## 参考文献

本项目基于以下工作修改：

```
@article{ZHAI2024108457,
  title = {GAN-BiLSTM network for field-road classification on imbalanced GNSS recordings},
  author = {Weixin Zhai and others},
  journal = {Computers and Electronics in Agriculture},
  volume = {216},
  year = {2024},
}
```

## 作者

- 原始 GAN-BiLSTM: Weixin Zhai et al.
- 修改适配: Agri-MBT Project

## 许可证

本项目仅用于学术研究。
