# OCR时间矫正工具使用指南

## 📋 概述

本工具对18:41:34之后的每一帧进行OCR识别，自动检测并矫正时间不一致的问题。

## 🎯 功能

1. **逐帧OCR识别**: 对每一帧进行OCR时间戳识别
2. **自动对比**: 比较OCR时间与CSV时间
3. **智能矫正**: 自动矫正时间不一致的帧
4. **详细报告**: 生成完整的矫正报告

## 🚀 快速开始

### 步骤1: 检查模式（预览）

首先运行检查模式，查看有多少帧需要矫正：

```bash
python3 scripts/ocr_time_correction.py --check-only
```

这将：
- 识别18:41:34之后的所有帧
- 显示需要矫正的帧数量和差异分布
- **不会修改任何文件**

### 步骤2: 应用矫正

确认无误后，应用矫正：

```bash
python3 scripts/ocr_time_correction.py --apply
```

这将：
- ✅ 创建备份文件
- ✅ 矫正所有时间不一致的帧
- ✅ 重命名帧文件
- ✅ 更新CSV和JSON文件
- ✅ 生成矫正报告

---

## 📖 详细参数

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv` | `data/aligned_output/aligned_data.csv` | 输入CSV文件 |
| `--frames-dir` | `data/aligned_output/aligned_frames` | 帧图像目录 |
| `--output-dir` | `data/aligned_output` | 输出目录 |
| `--timestamp` | `2024-10-18 18:41:34` | 起始时间戳 |
| `--threshold` | `0.5` | 时间差异阈值（秒） |

### 操作模式

| 参数 | 说明 |
|------|------|
| `--check-only` | 仅检查，不矫正 |
| `--apply` | 自动应用矫正 |

---

## 📊 输出文件

### 1. aligned_data.csv (更新)
矫正后的对齐数据

### 2. aligned_data.json (更新)
矫正后的JSON数据

### 3. aligned_data_backup.csv
原始数据的备份

### 4. ocr_correction_report.json
详细的矫正报告（JSON格式）

### 5. ocr_correction_report.txt
可读的矫正报告（文本格式）

包含：
- 总矫正数
- OCR失败数
- 每个帧的矫正详情

---

## 🔧 使用示例

### 示例1: 检查并预览

```bash
# 检查18:41:34之后的帧
python3 scripts/ocr_time_correction.py --check-only

# 输出示例:
# 目标帧数: 3198
# OCR成功: 3180
# OCR失败: 18
# 需要矫正: 245
#
# 差异分布:
#   +1秒: 120 帧
#   -2秒: 80 帧
#   +2秒: 45 帧
```

### 示例2: 应用矫正

```bash
python3 scripts/ocr_time_correction.py --apply

# 输出示例:
# ✓ 已创建备份: data/aligned_output/aligned_data_backup.csv
# 应用矫正: 100%|████████| 245/245 [00:10<00:00]
# ✓ 已保存矫正后的数据
# ✓ 已生成矫正报告
```

### 示例3: 自定义参数

```bash
# 从19:00:00开始矫正，阈值1秒
python3 scripts/ocr_time_correction.py \
  --timestamp "2024-10-18 19:00:00" \
  --threshold 1.0 \
  --apply
```

---

## 🔄 增强版对齐脚本

### 使用逐帧OCR验证模式

以后运行对齐时，可以使用增强版脚本，启用逐帧OCR验证：

```bash
# 使用逐帧OCR验证（更准确但更慢）
python3 scripts/align_video_trajectory_enhanced.py --frame-by-frame-ocr
```

这将：
1. 从视频的最后一帧识别实际时间
2. 对每一帧进行OCR验证
3. 自动矫正时间不一致的帧

### 参数说明

| 参数 | 说明 |
|------|------|
| `--frame-by-frame-ocr` | 启用逐帧OCR验证 |
| `--time-tolerance 3` | 时间容差（秒） |
| `--quiet` | 减少输出 |

---

## 📝 工作流程

### 方案A: 矫正现有数据（推荐）

适用于已有对齐数据的情况：

```bash
# 1. 检查
python3 scripts/ocr_time_correction.py --check-only

# 2. 应用矫正
python3 scripts/ocr_time_correction.py --apply

# 3. 验证结果
python3 scripts/verify_time_offset.py
```

### 方案B: 重新生成数据

适用于需要完全重新处理的情况：

```bash
# 使用增强版脚本，启用逐帧OCR
rm -rf data/aligned_output
python3 scripts/align_video_trajectory_enhanced.py --frame-by-frame-ocr
```

---

## ⚠️ 注意事项

### 1. OCR成功率

- **影响因素**: 图像质量、光照、分辨率
- **典型成功率**: 90-95%
- **失败处理**: 保留原始时间，记录在报告中

### 2. 处理时间

- **检查模式**: 约1-2分钟
- **应用矫正**: 约2-3分钟
- **逐帧OCR模式**: 约10-20分钟（取决于帧数）

### 3. 数据备份

- ✅ 自动创建备份: `aligned_data_backup.csv`
- ✅ 原始帧文件会被重命名（不删除）
- ⚠️ 建议手动备份整个 `data/aligned_output/` 目录

### 4. 时间阈值

- **默认**: 0.5秒
- **含义**: OCR时间与CSV时间差异超过0.5秒才矫正
- **调整**: 使用 `--threshold` 参数

---

## 🐛 常见问题

### Q1: OCR识别失败怎么办？

A: OCR失败是正常的（约5-10%），脚本会：
- 保留原始时间
- 记录在报告中
- 继续处理其他帧

### Q2: 矫正后发现还有问题？

A: 可以：
1. 调整 `--threshold` 参数
2. 手动编辑矫正报告并重新应用
3. 使用增强版脚本重新生成

### Q3: 如何查看矫正了哪些帧？

A: 查看报告文件：
```bash
cat data/aligned_output/ocr_correction_report.txt
```

### Q4: 如何恢复原始数据？

A: 使用备份：
```bash
cp data/aligned_output/aligned_data_backup.csv data/aligned_output/aligned_data.csv
```

---

## 📚 相关脚本

1. **ocr_time_correction.py** - OCR时间矫正工具
2. **align_video_trajectory_enhanced.py** - 增强版对齐脚本（支持逐帧OCR）
3. **verify_time_offset.py** - 时间验证工具
4. **manual_time_adjustment.py** - 手动时间调整工具

---

## 🎯 推荐工作流

### 快速验证和矫正

```bash
# 1. 快速检查
python3 scripts/ocr_time_correction.py --check-only

# 2. 如果需要矫正的数量合理，应用
python3 scripts/ocr_time_correction.py --apply

# 3. 验证结果
head -20 data/aligned_output/aligned_data.csv
```

### 完全重新处理

```bash
# 使用增强版脚本，逐帧OCR验证
python3 scripts/align_video_trajectory_enhanced.py --frame-by-frame-ocr
```

---

**创建日期**: 2026-03-05
**版本**: v1.0
**作者**: Claude Code
