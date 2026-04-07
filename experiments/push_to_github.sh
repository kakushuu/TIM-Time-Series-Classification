#!/bin/bash
# Push Agri-MBT visualizations to GitHub
#
# Usage:
#   1. Edit GITHUB_REPO variable below with your repository URL
#   2. Run: bash experiments/push_to_github.sh

# ==================== CONFIGURATION ====================
# Replace with your GitHub repository URL
# Options:
#   HTTPS: https://github.com/USERNAME/Agri-MBT.git
#   SSH:   git@github.com:USERNAME/Agri-MBT.git
#   Token: https://YOUR_TOKEN@github.com/USERNAME/Agri-MBT.git

GITHUB_REPO="https://github.com/YOUR_USERNAME/Agri-MBT.git"

# ==================== PUSH SCRIPT ====================

set -e  # Exit on error

echo "========================================"
echo "Pushing Agri-MBT Visualizations"
echo "========================================"
echo ""

# Check if GITHUB_REPO is configured
if [[ "$GITHUB_REPO" == *"YOUR_USERNAME"* ]]; then
    echo "❌ Error: Please edit this script and set GITHUB_REPO variable"
    echo ""
    echo "Example:"
    echo "  GITHUB_REPO=\"https://github.com/yourname/Agri-MBT.git\""
    echo ""
    echo "Or use SSH:"
    echo "  GITHUB_REPO=\"git@github.com:yourname/Agri-MBT.git\""
    exit 1
fi

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    echo "ℹ️  Remote 'origin' already exists"
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "   Current: $CURRENT_REMOTE"

    if [ "$CURRENT_REMOTE" != "$GITHUB_REPO" ]; then
        echo "   Updating to: $GITHUB_REPO"
        git remote set-url origin "$GITHUB_REPO"
    fi
else
    echo "➕ Adding remote 'origin': $GITHUB_REPO"
    git remote add origin "$GITHUB_REPO"
fi

# Show what will be pushed
echo ""
echo "📦 Ready to push:"
echo "   Commit: $(git log -1 --oneline)"
echo "   Files:  $(git diff-tree --no-commit-id --name-only -r HEAD | wc -l) new files"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin master

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed!"
    echo ""
    echo "📊 Pushed content:"
    echo "   • 18 visualization images (English)"
    echo "   • 6 Python scripts"
    echo "   • 1 README documentation"
    echo "   • 3 experiment result JSON files"
    echo ""
    echo "🔗 View at: ${GITHUB_REPO%.git}"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "   • Authentication: Use personal access token or SSH"
    echo "   • Repository not found: Create repo on GitHub first"
    echo "   • Permission denied: Check your access rights"
    exit 1
fi
