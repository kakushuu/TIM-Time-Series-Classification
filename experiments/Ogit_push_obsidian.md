# Manual push to Obsidian repository

## 实验已提交到本地 Git 仓库

要 想推送到 obsidian 笔记仓库，请手动执行以下命令：

```bash
cd /tmp
git clone git@github.com:kakushuu/obsidian.git obsidian-lab
cd obsidian-lab
mkdir -p lab/2026-03-17-trajectory-innovation-experiments
# 复制实验文件
cp /home/research/Agri-MBT/experiments/EXPERIMENT_SUMMARY_2026-03-17.md lab/2026-03-17-trajectory-innovation-experiments/
cp /home/research/Agri-MBT/experiments/*.log lab/2026-03-17-trajectory-innovation-experiments/logs/
cp /home/research/Agri-MBT/experiments/*.json lab/2026-03-17-trajectory-innovation-experiments/results/ 2>/dev/null

# 提交
git add .
git commit -m "Add trajectory innovation experiments (all failed)"
git push origin main
```

如果 SSH 也不行
可以尝试：
1. 使用 GitHub token: `git clone https://YOUR_TOKEN@github.com/kakushuu/obsidian.git`
2. 或者在 VS Code 中打开 obsidian 文件夹
让 VS Code 的 Git 扩展处理认证
