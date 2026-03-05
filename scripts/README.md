# 脚本清单

## 📂 目录结构

```
Agri-MBT/
├── scripts/
│   ├── align_video_trajectory.py          # 初版对齐脚本（使用OCR）
│   ├── align_video_trajectory_v2.py       # 改进版对齐脚本（推荐）
│   ├── test_extract_frames.py             # 测试：提取视频首尾帧
│   ├── test_ocr.py                        # 测试：OCR时间戳识别
│   └── analyze_video_timestamps.py        # 分析：所有视频时间范围
├── docs/
│   └── 视频轨迹对齐使用说明.md              # 详细使用文档
├── test_frames/                            # 测试输出目录
│   ├── 20241018043810_first.jpg
│   └── 20241018043810_last.jpg
└── data/
    ├── trajectory/B-2024-10-18/
    │   └── 12-12-49_23-59-58.xlsx
    ├── video/B-2024-10-18/
    │   └── *.mp4 (16个视频文件)
    └── aligned_output/                     # 对齐输出目录（待生成）
```

## 🔧 脚本说明

### 1. align_video_trajectory_v2.py ⭐（主要脚本）

**用途**：视频与轨迹数据对齐（改进版）

**特点**：
- ✅ 自动检测时间范围匹配
- ✅ 使用文件名时间戳（更可靠）
- ✅ 支持时间容差匹配
- ✅ 详细的进度和统计信息
- ✅ 自动处理多个视频文件

**使用**：
```bash
python3 scripts/align_video_trajectory_v2.py \
  --trajectory data/trajectory/<日期>/<文件>.xlsx \
  --video-dir data/video/<日期> \
  --output data/output
```

**输出**：
- `aligned_data.csv` - 对齐数据（CSV）
- `aligned_data.json` - 对齐数据（JSON）
- `alignment_stats.json` - 统计信息
- `aligned_frames/*.jpg` - 提取的视频帧

---

### 2. align_video_trajectory.py（初版）

**用途**：视频与轨迹数据对齐（使用OCR）

**特点**：
- 使用OCR从视频帧识别时间戳
- 适用于文件名无时间戳的情况
- 识别准确率较低（70-80%）

**使用**：
```bash
python3 scripts/align_video_trajectory.py
```

**注意**：推荐使用 v2 版本

---

### 3. test_extract_frames.py（测试工具）

**用途**：提取视频的第一帧和最后一帧

**使用**：
```bash
# 提取指定视频
python3 scripts/test_extract_frames.py data/video/B-2024-10-18/20241018043810.mp4

# 提取第一个视频（默认）
python3 scripts/test_extract_frames.py
```

**输出**：
- `test_frames/<视频名>_first.jpg` - 第一帧
- `test_frames/<视频名>_last.jpg` - 最后一帧

**用途**：
- 检查视频时间戳格式
- 调试OCR识别
- 验证视频质量

---

### 4. test_ocr.py（测试工具）

**用途**：测试OCR时间戳识别效果

**使用**：
```bash
python3 scripts/test_ocr.py
```

**功能**：
- 在已提取的帧上测试OCR
- 尝试不同的图像区域和参数
- 保存调试图像

**输出**：
- 控制台输出OCR识别结果
- `test_frames/debug_*.jpg` - 调试图像

---

### 5. analyze_video_timestamps.py（分析工具）

**用途**：分析所有视频的时间范围

**使用**：
```bash
python3 scripts/analyze_video_timestamps.py
```

**输出**：
- 每个视频的详细信息：
  - 文件名
  - 时长
  - 开始和结束时间
  - OCR识别结果
- 汇总信息：
  - 最早开始时间
  - 最晚结束时间

**用途**：
- 快速了解视频时间分布
- 检查视频完整性
- 调试时间对齐问题

---

## 📊 工作流程

### 标准流程

```
1. 分析数据
   ↓
   python3 scripts/analyze_video_timestamps.py

2. 提取测试帧（可选）
   ↓
   python3 scripts/test_extract_frames.py

3. 测试OCR（可选）
   ↓
   python3 scripts/test_ocr.py

4. 运行对齐
   ↓
   python3 scripts/align_video_trajectory_v2.py

5. 检查结果
   ↓
   查看 data/aligned_output/
```

### 快速开始

如果确定数据正确，直接运行：

```bash
python3 scripts/align_video_trajectory_v2.py
```

脚本会自动：
1. ✅ 加载轨迹数据
2. ✅ 分析视频文件
3. ✅ 检查时间重叠
4. ✅ 提取和对齐帧
5. ✅ 保存结果

---

## ⚙️ 依赖库

```bash
# Python库
pip install opencv-python pandas pillow pytesseract numpy

# 系统依赖（OCR）
apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

---

## 📝 参数说明

### align_video_trajectory_v2.py 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--trajectory` | `-t` | data/trajectory/.../xlsx | 轨迹数据文件路径 |
| `--video-dir` | `-v` | data/video/B-2024-10-18 | 视频文件夹路径 |
| `--output` | `-o` | data/aligned_output | 输出目录 |
| `--use-ocr` | | False | 使用OCR验证（实验性） |
| `--time-tolerance` | | 2 | 时间容差（秒） |

---

## 🐛 故障排除

### 问题1：找不到模块

```bash
ModuleNotFoundError: No module named 'cv2'
```

**解决**：
```bash
pip install opencv-python
```

### 问题2：OCR识别失败

```bash
TesseractNotFoundError: tesseract is not installed
```

**解决**：
```bash
# Ubuntu/Debian
apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### 问题3：时间不重叠

```
警告：视频和轨迹数据的时间范围不重叠！
```

**解决**：
1. 检查数据文件是否正确
2. 确认是同一天的数据
3. 检查是否有缺失文件

---

## 📈 性能优化建议

1. **SSD存储**：视频处理使用SSD可显著提升速度
2. **内存**：建议8GB以上内存
3. **并行处理**：脚本自动处理多个视频
4. **批量模式**：一次性处理所有数据

---

**最后更新**：2026-03-05
