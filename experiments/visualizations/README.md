# 可视化成果总结

## 生成时间
2026-03-12

## 1. 模型架构图
**文件**: `model_architectures.png`
**内容**: Multimodal Bottleneck Transformer (MBT) 架构
- 视觉分支：ViT-B16 backbone
- 轨迹分支：36特征 → 6×6特征图
- 跨模态融合：4个latent tokens
- 输出：11个农业活动类别

## 2. GNSS轨迹预测可视化

### 2.1 轨迹空间分布对比
**文件**: `gnss_trajectory/gnss_trajectory_predictions.png`
**内容**: 四个子图展示
- 左上：真实GNSS轨迹分布（按真实标签着色，11个类别）
- 右上：多模态模型预测（准确率94.72%）
- 左下：仅图像模型预测（准确率92.91%）
- 右下：仅轨迹模型预测（准确率42.47%）

### 2.2 准确率空间分布
**文件**: `gnss_trajectory/gnss_trajectory_accuracy.png`
**内容**:
- 绿色点：预测正确的GNSS位置
- 红色X：预测错误的位置
- 显示每个模型的准确率

### 2.3 错误预测热力图
**文件**: `gnss_trajectory/gnss_error_spatial_distribution.png`
**内容**:
- 红色区域：预测错误密集的区域
- 浅绿色：预测正确的区域
- 六边形分箱显示错误密度

### 2.4 类别详细分布
**文件**: `gnss_trajectory/gnss_per_class_distribution.png`
**内容**: 样本数最多的4个类别的详细空间分布对比

## 3. 模型性能对比

### 3.1 轨迹预测性能对比
**文件**: `trajectory_predictions/model_predictions_comprehensive.png`
**内容**:
- 真实标签分布
- 三个模型的平均F1-Score对比
- 三个模型的验证准确率对比
- 每个类别的F1-Score热力图

### 3.2 其他性能图
- `overall_comparison.png`: 整体性能对比（准确率、Macro F1、Weighted F1）
- `per_class_f1_comparison.png`: 每类F1-Score对比
- `metrics_heatmap.png`: Precision/Recall/F1热力图
- `training_curves.png`: 训练曲线
- `sample_count_vs_f1.png`: 样本数与F1关系
- `radar_chart.png`: 雷达图综合对比

## 4. 数据处理流程

### 4.1 完整流程图
**文件**: `data_processing_pipeline.png`
**内容**: 12步完整流程

**Step 1-3**: 视频帧预处理
- 原始RGB帧 (1920×1080)
- 灰度化处理
- 提取时间戳ROI（左上角5%高度，30%宽度）

**Step 4-6**: OCR时间戳提取
- 二值化（Threshold=200）
- OCR文本识别（Tesseract）
- 正则表达式解析（YYYY-MM-DD HH:MM:SS）

**Step 7-9**: 轨迹数据匹配
- 加载轨迹数据CSV（定位时间、经度、纬度、速度等）
- 时间戳匹配（±2s容差）
- 数据对齐

**Step 10-12**: 质量控制
- OCR状态检查（ok/ocr_error/interpolated/excluded）
- 时间戳验证（局部成对速率检查，范围0.5-1.5 s/s）
- 最终输出（aligned_data.csv等）

### 4.2 OCR提取详细流程
**文件**: `ocr_extraction_detail.png`
**内容**: 8步详细展示

**Step 1-2**: 帧预处理
- 原始RGB帧（标注时间戳区域）
- 灰度化

**Step 3-4**: ROI提取和二值化
- 提取时间戳区域
- 不同阈值对比（150, 180, 200, 220）
- 最佳阈值：200

**Step 5-6**: OCR识别
- 预处理步骤（灰度、ROI、二值化、OCR配置）
- OCR识别结果展示

**Step 7-8**: 正则解析和输出
- 正则表达式匹配：`(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})`
- 最终datetime对象

## 5. 关键统计

### 数据规模
- 总样本数：32,249（B-2024-10-18: 6,259; B-2024-10-19: 25,990）
- 测试集：1,252样本
- 11个农业活动类别
- 空间覆盖：经度108.4927-108.4929°，纬度37.5813-37.5815°

### 模型性能
| 模型 | 准确率 | Macro F1 | Weighted F1 |
|------|--------|----------|-------------|
| Multimodal | 94.72% | 80.00% | 94.49% |
| Image Only | 92.91% | 74.77% | 92.71% |
| Trajectory Only | 42.47% | 7.09% | 20.65% |

### OCR配置
- OCR引擎：Tesseract
- 配置：`--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:- `
- 最佳阈值：200
- ROI区域：左上角(0,0)到(576,54)像素

## 6. 文件路径

所有可视化文件位于：
```
/home/research/Agri-MBT/experiments/visualizations/
├── model_architectures.png
├── data_processing_pipeline.png
├── ocr_extraction_detail.png
├── gnss_trajectory/
│   ├── gnss_trajectory_predictions.png
│   ├── gnss_trajectory_accuracy.png
│   ├── gnss_error_spatial_distribution.png
│   └── gnss_per_class_distribution.png
├── trajectory_predictions/
│   ├── model_predictions_comprehensive.png
│   ├── model_predictions_heatmap.png
│   ├── model_predictions_barplot.png
│   └── model_predictions_per_class_detailed.png
└── [其他性能对比图]
```

## 7. 使用说明

### 运行脚本
```bash
# 生成数据处理流程图
python experiments/visualize_data_pipeline.py

# 生成OCR详细流程图
python experiments/visualize_ocr_detail.py

# 生成GNSS轨迹可视化
python experiments/visualize_gnss_trajectory_simple.py

# 生成模型性能对比
python experiments/visualize_results.py
```

### 依赖
- matplotlib, seaborn, numpy, pandas
- opencv-python (cv2)
- pytesseract
- PIL (Pillow)
- torch, torchvision

## 8. 注意事项

1. **中文字体警告**: 可视化中的中文可能无法正确显示，建议使用英文或安装中文字体
2. **数据路径**: 脚本中的路径需要根据实际数据位置调整
3. **模型检查点**: 如果没有训练好的模型，会自动训练（需要较长时间）
4. **图片缺失**: B-2024-10-19批次的部分图片可能不存在，脚本会自动过滤

## 9. 主要发现

1. **多模态优势**: 多模态模型显著优于单模态（94.72% vs 92.91% vs 42.47%）
2. **空间聚集性**: 不同农业活动在GNSS空间上有明显聚集分布
3. **类别不平衡**: Class 7有11,069个样本，Class 1仅295个样本
4. **OCR稳定性**: 阈值200对大多数视频效果最佳
5. **轨迹特征局限**: 仅使用GNSS特征无法有效区分农业活动

---

**生成日期**: 2026-03-12
**项目**: Agri-MBT (Agricultural Multimodal Bottleneck Transformer)
**作者**: Claude Code
