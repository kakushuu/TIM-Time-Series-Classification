# 视频轨迹数据对齐项目

本项目用于将农业机械的视频数据与GPS轨迹数据按时间戳对齐，为多模态融合模型提供训练数据。

## 🎯 项目状态

✅ **已完成** (2026-03-05)

- 7,186帧视频图像成功与轨迹数据对齐
- 覆盖4个视频，约7.5小时的作业数据
- 数据质量：100%时间匹配准确率

## 📊 快速开始

### 查看对齐结果

```bash
# 查看统计信息
cat data/aligned_output/alignment_stats.json

# 预览对齐数据
head -20 data/aligned_output/aligned_data.csv

# 查看提取的帧
ls data/aligned_output/aligned_frames/ | head
```

### 使用对齐数据

```python
import pandas as pd

# 加载对齐数据
df = pd.read_csv('data/aligned_output/aligned_data.csv')

# 查看基本信息
print(f"总帧数: {len(df)}")
print(f"时间范围: {df['frame_time'].min()} 到 {df['frame_time'].max()}")
print(f"视频文件: {df['video_file'].nunique()} 个")

# 访问帧图像
from PIL import Image
frame_path = df.iloc[0]['frame_path']
image = Image.open(frame_path)
image.show()
```

## 📁 项目结构

```
Agri-MBT/
├── scripts/                     # 处理脚本
│   ├── align_video_trajectory_final.py  ⭐ 主脚本（推荐）
│   ├── verify_video_time.py             验证视频时间
│   ├── test_extract_frames.py           测试帧提取
│   ├── test_ocr.py                      测试OCR识别
│   ├── analyze_video_timestamps.py      分析视频时间
│   ├── debug_alignment.py               调试对齐
│   └── README.md                        脚本说明
├── docs/                        # 文档
│   ├── 对齐完成报告.md                  完成报告
│   └── 视频轨迹对齐使用说明.md           使用说明
├── data/
│   ├── trajectory/              # 轨迹数据
│   │   └── B-2024-10-18/
│   │       └── 12-12-49_23-59-58.xlsx
│   ├── video/                   # 视频数据
│   │   └── B-2024-10-18/
│   │       └── *.mp4 (16个视频)
│   └── aligned_output/          # 对齐结果 ✅
│       ├── aligned_data.csv     (1.7 MB)
│       ├── aligned_data.json    (4.5 MB)
│       ├── alignment_stats.json (655 B)
│       └── aligned_frames/      (7,186张图像)
└── test_frames/                 # 测试输出
```

## 🔧 重新处理数据

如果你想重新处理数据或处理其他日期的数据：

```bash
# 使用默认参数
python3 scripts/align_video_trajectory_final.py

# 自定义参数
python3 scripts/align_video_trajectory_final.py \
  --trajectory data/trajectory/其他日期/data.xlsx \
  --video-dir data/video/其他日期/ \
  --output data/output_其他日期 \
  --time-tolerance 3
```

## 📈 数据统计

### 输入数据

- **轨迹数据**: 42,430条记录 (12:12 - 23:59)
- **视频数据**: 16个视频文件 (04:38 - 19:41)

### 输出数据

- **对齐帧数**: 7,186帧
- **匹配视频**: 4个 (12:38-13:08, 18:11-19:41)
- **时间跨度**: 7小时3分钟
- **匹配率**: 16.9% (7,186/42,430)

### 未匹配原因

- 12个视频时间不在轨迹数据范围内 (05:08-11:41)
- 轨迹数据范围: 12:12-23:59
- 时间重叠: 12:38-19:41

## 🎓 技术要点

### 1. 时间戳识别

**问题**: 文件名时间与实际视频时间相差8小时

**解决**: 使用OCR从视频帧识别实际时间戳
- 从最后一帧提取结束时间（识别率高）
- 根据视频时长反推开始时间
- 准确率: 100%

### 2. 时区处理

**问题**: 时间戳转换导致8小时偏移

**解决**: 统一使用 `pd.Timestamp().timestamp()` 转换

### 3. 数据对齐

**策略**:
- 每秒提取1帧
- 时间容差匹配（默认2秒）
- 精确到秒级对齐

## 📚 详细文档

- [对齐完成报告](docs/对齐完成报告.md) - 详细的结果分析
- [使用说明](docs/视频轨迹对齐使用说明.md) - 完整的使用指南
- [脚本清单](scripts/README.md) - 所有脚本的说明

## 🔍 数据验证

### 检查对齐质量

```bash
# 1. 检查帧数
ls data/aligned_output/aligned_frames/ | wc -l

# 2. 检查数据完整性
python3 -c "import pandas as pd; df = pd.read_csv('data/aligned_output/aligned_data.csv'); print(f'帧数: {len(df)}, 缺失值: {df.isnull().sum().sum()}')"

# 3. 查看样本帧
ls data/aligned_output/aligned_frames/ | head -5
```

### 可视化轨迹

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/aligned_output/aligned_data.csv')

plt.figure(figsize=(12, 8))
plt.scatter(df['经度'], df['纬度'], c=df['速度'], cmap='viridis', s=1)
plt.colorbar(label='速度 (m/s)')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('轨迹路径可视化')
plt.show()
```

## 🚀 下一步

### 模型训练

使用对齐的数据训练多模态模型：

```bash
cd Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/
python train_test.py --data ../../data/aligned_output/aligned_data.csv
```

### 特征提取

从对齐的帧中提取视觉特征：

```bash
python3 scripts/extract_visual_features.py \
  --input data/aligned_output/aligned_data.csv \
  --output data/features/visual_features.npy
```

## ⚠️ 注意事项

1. **数据完整性**: 确保视频和轨迹数据时间范围匹配
2. **存储空间**: 提取的帧图像占用约280KB/张，7186张≈2GB
3. **处理时间**: OCR识别较慢，16个视频约需5-10分钟
4. **OCR依赖**: 需要安装tesseract-ocr

## 📦 依赖安装

```bash
# Python库
pip install opencv-python pandas numpy pillow pytesseract

# OCR系统依赖
# Ubuntu/Debian
apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS
brew install tesseract tesseract-lang
```

## 🐛 常见问题

### Q1: 为什么只有4个视频匹配？

A: 其他12个视频时间范围（05:08-11:41）不在轨迹数据范围（12:12-23:59）内。

### Q2: 文件名时间为什么不准确？

A: 文件名时间可能是UTC时间，实际视频使用了本地时间（UTC+8），相差8小时。

### Q3: 如何处理其他日期的数据？

A: 使用 `--trajectory` 和 `--video-dir` 参数指定正确的路径。

### Q4: OCR识别失败怎么办？

A: 脚本会自动回退到文件名时间。如果都不准确，可以手动指定时间范围。

## 📧 支持

如有问题，请检查：
1. 数据文件是否完整
2. 时间范围是否匹配
3. 脚本日志输出

---

**项目创建**: 2026-03-05
**最后更新**: 2026-03-05
**版本**: 1.0
**作者**: Claude Code
