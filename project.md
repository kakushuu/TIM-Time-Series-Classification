这是一个为你量身定制的、可直接执行的《Agri-MBT 实验操作完全手册》。

这份文档将复杂的科研任务拆解为 **4 个阶段** 的具体执行清单。请按照顺序执行，每完成一步打一个勾。

---

### 阶段一：环境准备与资源下载 (Environment & Assets)

在开始写代码前，你需要先把“积木”找齐。我们不从零造轮子，而是利用现有的开源组件。

#### 1. 核心代码库下载 (GitHub)

你需要参考以下代码库来实现 MBT 架构。

* **MBT (Multimodal Bottleneck Transformer) 参考实现：**
* **推荐下载：** [NMS05/Multimodal-Fusion-with-Attention-Bottlenecks](https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks) (PyTorch 版本)
* *用途：* 这是一个最干净的 PyTorch 实现。下载后，重点阅读 `models/mbt.py`，把里面的 `FusionTransformer` 类复制到你的项目中，作为融合模块的核心。
* *备选（官方）：* [google-research/scenic](https://github.com/google-research/scenic) (JAX 版本，仅作逻辑参考，不建议直接跑)。



#### 2. 预训练模型准备

你需要下载视觉部分的预训练权重，不要从头训练视觉模型。

* **库安装：** `pip install timm` (PyTorch Image Models)
* **模型权重：** 在代码中自动下载 `vit_base_patch16_224` (ImageNet-21k 预训练)。
* *用途：* 作为视频流的特征提取器。



#### 3. Python 环境依赖清单

创建一个新的 Conda 环境（建议命名为 `agri_mbt`），并安装：

```bash
pip install torch torchvision torchaudio  # 深度学习框架
pip install pandas numpy scikit-learn     # 数据处理
pip install opencv-python                 # 视频处理
pip install timm                          # 视觉模型库
pip install einops                        # 维度变换神器（MBT代码通常需要）

```

---

### 阶段二：数据工程 (Data Engineering) - 最繁琐但最重要

你需要将手中的 Excel 和 视频 处理成模型能吃的格式。

#### ✅ 步骤 1：视频抽帧 (Video Preprocessing)

* **输入：** 原始 `.mp4` 视频文件。
* **操作：**
1. 编写脚本，使用 OpenCV 读取视频。
2. 按 **1 fps** 的频率保存帧（即每秒保存一张图）。
3. **关键点：** 图片命名必须包含时间戳（例如 `20241021_161339.jpg`），以便和 Excel 对应。
4. 统一 Resize 图片大小为 **224x224** 像素。



#### ✅ 步骤 2：轨迹特征增强 (Trajectory Feature Engineering)

* **参考：** 复现 Zhai et al. (2024) 论文的 2.3 节 。


* **输入：** Excel 文件（经度, 纬度, 速度, 方向, 时间）。
* **操作：** 使用 Pandas 计算以下 **36 个特征**：
1. **基础特征 (4个):** 经度(转为UTM x), 纬度(转为UTM y), 速度, 方向。
2. 
**运动学衍生 (6个):** 速度差(), 加速度(), 方向差(), 角速度(), 角速度差(), 角加速度() 。


3. 
**统计学衍生 (26个):** 对上述特征开 **滑动窗口 (Window=5 和 20)**，计算 `mean`, `std`, `median` 。


* *注意：* 确保总列数凑齐 36 列。如果不够，可以补充 `max` 或 `min` 特征。


4. **归一化：** 使用 `MinMaxScaler` 将所有列缩放到 0-1 之间。
5. **保存：** 保存为 `trajectory_features.csv`。



#### ✅ 步骤 3：多模态对齐 (Alignment)

* **操作：**
1. 遍历 `trajectory_features.csv` 的每一行。
2. 根据 `时间` 字段，去图片文件夹里找对应的 `.jpg` 文件。
3. **生成索引文件：** 创建一个 `train_list.txt`，每一行格式为：
`path/to/image.jpg, row_index_in_csv, label_class_id`
4. 丢弃找不到对应图片的轨迹点（保证一一对应）。



---

### 阶段三：模型搭建 (Model Building) - 核心代码逻辑

你需要新建一个 `model.py` 文件，把积木搭起来。

#### ✅ 步骤 4：定义 Dataset 类

* **代码逻辑：**
```python
class AgriDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 读取图片 -> Tensor [3, 224, 224]
        img = load_image(self.img_paths[idx])

        # 2. 读取轨迹 -> Tensor [36] -> Reshape [1, 6, 6]
        traj = self.csv_data.iloc[idx] 
        traj_map = torch.tensor(traj).reshape(1, 6, 6) 

        return img, traj_map, label

```



#### ✅ 步骤 5：搭建 Agri-MBT 模型

这是你的创新部分，结合 ViT 和 Zhai 的轨迹图。

* **代码逻辑：**
1. **视频分支：** `self.vis_model = timm.create_model('vit_base_patch16_224', pretrained=True)`。去掉它的分类头，只拿特征。
2. **轨迹分支：**
* 输入是 `[Batch, 1, 6, 6]`。
* 执行 `Flatten` -> `[Batch, 36, 1]`。
* 执行 `Linear(1, 768)` -> `[Batch, 36, 768]`。
* 加上位置编码 `PositionalEncoding`。


3. **融合模块 (MBT)：**
* 定义 4 个 `Bottleneck Tokens` (维度 768)。
* 在 Transformer 的深层（如第 9-12 层），把 **[视频Tokens, 轨迹Tokens, 瓶颈Tokens]** 拼在一起做 Self-Attention。


4. **分类头：** 取出 `Bottleneck Tokens` 的均值，过一个全连接层输出 11 个分类。



---

### 阶段四：实验执行 (Execution)

#### ✅ 步骤 6：基准对比 (Baseline Training)

为了发论文，你不能只跑最好的模型，必须先跑差的作为对比。

1. **Run 1 (Video Only):** 把轨迹输入全设为 0，训练模型。记录准确率（例如 85%）。
2. **Run 2 (Traj Only):** 类似于 Zhai 的论文，只用轨迹图。记录准确率（例如 92%）。

#### ✅ 步骤 7：融合训练 (Fusion Training)

1. **Run 3 (Agri-MBT):** 全量数据训练。
* **超参数建议：**
* `Batch Size`: 32
* `Learning Rate`: 1e-4 (视频部分), 1e-3 (轨迹部分，因为它是从头学的)
* `Epochs`: 50
* `Loss`: CrossEntropyLoss




2. **预期结果：** 准确率应该比 Run 1 和 Run 2 都高（例如 95%+）。

#### ✅ 步骤 8：结果可视化

1. **混淆矩阵 (Confusion Matrix):** 重点看“停车”和“堵塞”是否分得清。
2. **Attention Map 可视化 (高级):** 抽取 Bottleneck Token 对 Video Token 的注意力权重，画在图片上。如果模型关注到了倒伏的玉米，截图保存，这将是论文里的神图。

---

### 下载清单总结 (Checklist for Download)

1. [ ] **代码：** `NMS05/Multimodal-Fusion-with-Attention-Bottlenecks` (Github) - 用于抄写 Fusion 逻辑。
2. [ ] **模型：** `vit_base_patch16_224` (通过 timm 自动下载) - 用于视觉骨干。
3. [ ] **数据：** 你自己的 Excel 和 视频文件。
4. [ ] **工具：** Python 环境 (`torch`, `pandas`, `timm`, `opencv-python`).

按照这个清单一步步来，你不仅能跑通实验，还能直接把这些步骤写进论文的 **Methodology** 和 **Experiments** 章节。