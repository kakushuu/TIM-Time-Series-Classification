#!/bin/bash
# Agri-MBT Agent Status Dashboard
# 显示5个子智能体的当前状态

PROJECT="/home/research/Agri-MBT"
STORIES_DIR="$PROJECT/docs/user-stories"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
RESET='\033[0m'
BOLD='\033[1m'

print_status() {
    clear
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║       Agri-MBT TC-AdaptFormer — 多智能体实验状态面板         ║${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
    echo -e "${DIM}  更新时间: $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
    echo ""

    # 解析各 story 文件的 passes 状态
    declare -A AGENT_STATUS

    for f in "$STORIES_DIR"/*.json; do
        name=$(basename "$f")
        passes=$(python3 -c "
import json
data = json.load(open('$f'))
total = len(data)
done = sum(1 for s in data if s.get('passes', False))
print(f'{done}/{total}')
" 2>/dev/null)
        AGENT_STATUS["$name"]="$passes"
    done

    # ── Agent 状态卡片 ──────────────────────────────────────────────
    echo -e "${BOLD}  子智能体状态${RESET}"
    echo -e "  ──────────────────────────────────────────────────────────"

    agents=(
        "01-architecture-design.json|Subagent_Architect|架构设计文档"
        "02-dataset-loader.json|Subagent_Dev [Dataset]|数据集加载器"
        "03-model-implementation.json|Subagent_Dev [Model]|模型实现"
        "04-mock-testing.json|Subagent_Eval|Mock测试与剖析"
        "05-paper-draft.json|Subagent_Doc|论文草稿生成"
    )

    total_stories=0
    done_stories=0

    for entry in "${agents[@]}"; do
        IFS='|' read -r file agent desc <<< "$entry"
        status="${AGENT_STATUS[$file]:-0/0}"
        done_count="${status%/*}"
        total_count="${status#*/}"
        total_stories=$((total_stories + total_count))
        done_stories=$((done_stories + done_count))

        if [ "$done_count" = "$total_count" ] && [ "$total_count" != "0" ]; then
            icon="✅"
            color="${GREEN}"
            state="完成"
        elif [ "$done_count" = "0" ]; then
            icon="⏳"
            color="${YELLOW}"
            state="待开始"
        else
            icon="🔄"
            color="${BLUE}"
            state="进行中"
        fi

        # 进度条
        bar_filled=$((done_count * 20 / total_count))
        bar=""
        for ((i=0; i<20; i++)); do
            if [ $i -lt $bar_filled ]; then bar+="█"; else bar+="░"; fi
        done

        printf "  ${icon} ${BOLD}%-24s${RESET}  ${color}%-8s${RESET}  [%s]  ${DIM}%s/%s steps${RESET}\n" \
            "$agent" "$state" "$bar" "$done_count" "$total_count"
        printf "     ${DIM}%s${RESET}\n" "$desc"
        echo ""
    done

    echo -e "  ──────────────────────────────────────────────────────────"

    # 总进度
    if [ "$total_stories" -gt 0 ]; then
        pct=$((done_stories * 100 / total_stories))
        bar_filled=$((pct / 5))
        bar=""
        for ((i=0; i<20; i++)); do
            if [ $i -lt $bar_filled ]; then bar+="█"; else bar+="░"; fi
        done
        echo -e "  ${BOLD}总进度: [${GREEN}${bar}${RESET}${BOLD}] ${pct}%${RESET}  (${done_stories}/${total_stories} stories)"
    fi

    echo ""

    # ── Ralph 运行状态 ─────────────────────────────────────────────
    echo -e "${BOLD}  Ralph 运行器状态${RESET}"
    echo -e "  ──────────────────────────────────────────────────────────"

    # 检查是否有 ralph 进程在运行
    ralph_pid=$(pgrep -f "ralph/runner.ts" 2>/dev/null)
    if [ -n "$ralph_pid" ]; then
        echo -e "  🟢 ${GREEN}RUNNING${RESET}  PID: $ralph_pid"
    else
        echo -e "  🔴 ${RED}STOPPED${RESET}  ralph 循环未运行"
        echo -e "  ${DIM}  启动方式: cd /home/research/Agri-MBT && bun run ralph${RESET}"
    fi
    echo ""

    # ── 已知问题 ──────────────────────────────────────────────────
    echo -e "${BOLD}  ⚠️  注意事项${RESET}"
    echo -e "  ──────────────────────────────────────────────────────────"
    echo -e "  ${RED}CLAUDECODE 嵌套限制${RESET}: ralph runner 需从独立终端运行"
    echo -e "  ${DIM}  解决方案: 在新 terminal 标签中执行 bun run ralph${RESET}"
    echo -e "  ${DIM}  或直接由 Claude Code 作为 Orchestrator 实现所有 Stories${RESET}"
    echo ""
}

# 循环刷新
while true; do
    print_status
    sleep 3
done
