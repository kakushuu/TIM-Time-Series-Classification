# evil-read-arxiv 部署完成指南

## 🎉 部署状态

✅ **所有组件已成功部署！**

## 📦 已安装的组件

### 1. Python 依赖
- PyYAML >= 6.0
- requests >= 2.28.0
- PyMuPDF >= 1.23.0

所有依赖已安装在 `agri-mbt` conda 环境中。

### 2. Claude Code Skills
已安装 4 个技能到 `~/.claude/skills/`:
- **start-my-day**: 每日论文推荐工作流
- **paper-analyze**: 论文深度分析
- **extract-paper-images**: 论文图片提取
- **paper-search**: 论文笔记搜索

### 3. Obsidian Vault 结构
已在 `/home/research/Agri-MBT/obsidian-vault/` 创建:
```
obsidian-vault/
├── 10_Daily/                    # 每日推荐笔记
├── 20_Research/
│   └── Papers/                  # 论文详细笔记
└── 99_System/
    └── Config/
        └── research_interests.yaml  # 研究兴趣配置
```

### 4. 环境变量
`OBSIDIAN_VAULT_PATH` 已设置并添加到 `~/.bashrc`

## 🚀 使用方法

### 每日论文推荐

在 Claude Code 中输入:
```
start my day
```

或者指定日期:
```
start my day 2026-03-05
```

这将自动:
1. 搜索 arXiv 和 Semantic Scholar 的最新论文
2. 根据你的研究兴趣评分和筛选
3. 生成今日推荐笔记（保存到 `10_Daily/` 目录）
4. 对前三篇高分论文自动生成详细分析
5. 提取论文图片并插入笔记
6. 自动链接关键词到已有笔记

### 深度分析单篇论文

```
paper-analyze 2602.12345
```

或使用论文标题:
```
paper-analyze "论文标题"
```

### 提取论文图片

```
extract-paper-images 2602.12345
```

### 搜索已有论文笔记

```
paper-search 多模态融合
```

## ⚙️ 配置说明

### 研究兴趣配置文件

位置: `/home/research/Agri-MBT/obsidian-vault/99_System/Config/research_interests.yaml`

当前配置的研究领域:
1. **多模态融合** (优先级: 10)
   - 关键词: multimodal fusion, attention bottleneck, video trajectory fusion
   - 分类: cs.CV, cs.MM, cs.LG

2. **农业人工智能** (优先级: 9)
   - 关键词: agricultural activity recognition, smart farming
   - 分类: cs.CV, cs.RO, cs.AI

3. **视觉 Transformer** (优先级: 8)
   - 关键词: Vision Transformer, ViT, video understanding
   - 分类: cs.CV, cs.LG

4. **轨迹分析** (优先级: 7)
   - 关键词: trajectory prediction, motion pattern
   - 分类: cs.CV, cs.RO, cs.LG

5. **深度学习** (优先级: 6)
   - 关键词: deep learning, transfer learning, adapter
   - 分类: cs.LG, cs.AI

### 修改配置

编辑配置文件:
```bash
nano /home/research/Agri-MBT/obsidian-vault/99_System/Config/research_interests.yaml
```

修改后无需重启，下次运行时自动生效。

## 📊 评分机制

论文推荐基于四个维度:

| 维度 | 权重 | 说明 |
|------|------|------|
| 相关性 | 40% | 与研究兴趣的匹配程度 |
| 新近性 | 20% | 论文发布时间 |
| 热门度 | 30% | 引用数/影响力 |
| 质量 | 10% | 从摘要推断的方法质量 |

## 🔧 高级配置

### 修改搜索的 arXiv 分类

在调用脚本时指定:
```bash
python scripts/search_arxiv.py --categories "cs.AI,cs.LG,cs.CL,cs.CV"
```

### 修改每天推荐的论文数量

```bash
python scripts/search_arxiv.py --top-n 15
```

### 修改评分权重

编辑 `start-my-day/scripts/search_arxiv.py` 中的 `calculate_recommendation_score` 函数。

## 📁 目录结构

```
/root/.claude/skills/
├── start-my-day/
│   ├── skill.md
│   └── scripts/
│       ├── search_arxiv.py          # arXiv 搜索脚本
│       ├── scan_existing_notes.py   # 扫描现有笔记
│       └── link_keywords.py         # 关键词自动链接
├── paper-analyze/
│   ├── skill.md
│   └── scripts/
│       ├── generate_note.py         # 生成笔记模板
│       └── update_graph.py          # 更新知识图谱
├── extract-paper-images/
│   ├── skill.md
│   └── scripts/
│       └── extract_images.py        # 图片提取脚本
└── paper-search/
    └── skill.md

/home/research/Agri-MBT/obsidian-vault/
├── 10_Daily/                         # 每日推荐笔记
│   └── YYYY-MM-DD论文推荐.md
├── 20_Research/
│   └── Papers/                       # 论文详细笔记
│       ├── 多模态融合/
│       ├── 农业人工智能/
│       └── ...
└── 99_System/
    └── Config/
        └── research_interests.yaml   # 研究兴趣配置
```

## 🐛 常见问题

### Q: 搜索没有结果？

A: 检查以下几点:
1. 确认网络连接正常
2. 检查配置文件中的关键词是否正确
3. 尝试扩大搜索的 arXiv 分类范围

### Q: 图片提取失败？

A:
1. 确保安装了 PyMuPDF: `conda activate agri-mbt && pip install PyMuPDF`
2. 检查 arXiv ID 格式是否正确（如 2602.12345）

### Q: "Papers directory not found" 错误？

A:
1. 检查 `OBSIDIAN_VAULT_PATH` 环境变量: `echo $OBSIDIAN_VAULT_PATH`
2. 确认目录存在: `ls -la /home/research/Agri-MBT/obsidian-vault/20_Research/Papers/`

### Q: "未指定 vault 路径" 错误？

A: 设置环境变量:
```bash
export OBSIDIAN_VAULT_PATH="/home/research/Agri-MBT/obsidian-vault"
```

## 💡 使用技巧

1. **定期更新配置**: 根据研究进展调整 `research_interests.yaml` 中的关键词
2. **建立知识图谱**: 让系统自动链接相关论文，构建你的研究知识库
3. **批量处理**: 可以一次分析多篇论文，系统会自动管理依赖关系
4. **结合 Obsidian**: 在 Obsidian 中打开 vault，享受可视化的知识图谱和笔记链接

## 📚 下一步

1. **运行第一次推荐**:
   ```
   start my day
   ```

2. **查看生成的笔记**:
   ```bash
   ls /home/research/Agri-MBT/obsidian-vault/10_Daily/
   ```

3. **在 Obsidian 中打开 vault**:
   - 打开 Obsidian
   - 选择 "Open folder as vault"
   - 选择 `/home/research/Agri-MBT/obsidian-vault/`

4. **开始阅读和分析论文**！

## 🎯 与 Agri-MBT 项目的集成

这个工具特别适合你的 Agri-MBT 项目:

- **多模态融合**: 自动发现最新的多模态融合方法
- **农业 AI**: 跟踪农业人工智能的最新进展
- **Vision Transformer**: 了解最新的视觉 Transformer 架构
- **轨迹分析**: 发现轨迹预测和分析的新方法

## 📞 支持

- GitHub 仓库: https://github.com/juliye2025/evil-read-arxiv
- 问题反馈: 在 GitHub 上提交 Issue

---

**部署日期**: 2026-03-05
**部署环境**: agri-mbt conda 环境
**Python 版本**: 3.10.19
