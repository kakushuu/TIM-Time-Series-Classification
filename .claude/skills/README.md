# Project Skills

This directory contains project-specific Claude Code skills for the Agri-MBT project.

## Available Skills

### 1. `align-agri-data`

**Purpose**: Align agricultural video frames with GPS trajectory data using OCR-based timestamp extraction.

**When to use**: Processing new video+trajectory data batches for training.

**Quick start**:
```bash
python scripts/align_agri_data.py \
    --trajectory data/trajectory/B-YYYY-MM-DD/file.csv \
    --video-dir data/video/B-YYYY-MM-DD \
    --output data/aligned_output/B-YYYY-MM-DD \
    --traj-format csv
```

**Key features**:
- OCR-based timestamp extraction (threshold 200)
- Local pairwise rate validation (0.5-1.5 s/s)
- Interpolates gaps ≤5s, excludes larger gaps
- Matches frames to trajectory within ±2s tolerance

**Full documentation**: See `.claude/skills/align-agri-data/SKILL.md`

---

### 2. `deep-learning-python`

**Purpose**: Guidelines for deep learning development with PyTorch, Transformers, Diffusers, and Gradio.

**When to use**: When writing or modifying model code, training scripts, or inference pipelines.

**Key principles**:
- Use PyTorch as primary framework
- Implement proper GPU utilization and mixed precision training
- Follow PEP 8 style guidelines
- Use modular code structure (models, data loading, training, evaluation)

**Full documentation**: See `.claude/skills/deep-learning-python/SKILL.md`

---

### 3. `agri-image-autoresearch`

**Purpose**: Run a disciplined autonomous experiment loop for improving Agri-MBT image-only training before trajectory-only retuning and multimodal fusion.

**When to use**: Tuning `src/train_ablation.py --mode image_only`, comparing visual encoder changes, running repeated image-only experiments, or preparing a stable image branch for multimodal experiments.

**Quick start**:
```bash
python .agents/skills/agri-image-autoresearch/scripts/create_session_files.py \
    --gpu-ids 1,2,5,6 \
    --epochs 30 \
    --batch-size 8 \
    --suite-dir experiments/agri_image_autoresearch
```

**Key features**:
- Uses `best_val_macro_f1` as the primary keep/discard metric
- Keeps test metrics audit-only to reduce validation overfitting
- Creates `autoresearch.md`, `autoresearch.sh`, and `autoresearch.checks.sh`
- Converts image-only `summary.json` into parseable `METRIC` lines
- Defines project-specific scope, baseline commands, and high-value hypotheses

**Full documentation**: See `.claude/skills/agri-image-autoresearch/SKILL.md`

---

## Skill Management

### Installed Skills

Skills are tracked in `skills-lock.json` at the project root. Current skills:

```json
{
  "version": 1,
  "skills": {
    "deep-learning-python": {
      "source": "mindrally/skills",
      "sourceType": "github"
    },
    "align-agri-data": {
      "source": "local",
      "sourceType": "local"
    },
    "agri-image-autoresearch": {
      "source": "local",
      "sourceType": "local"
    }
  }
}
```

### Adding New Skills

To add a new skill to the project:

1. Create skill directory: `.claude/skills/<skill-name>/`
2. Add `SKILL.md` with frontmatter and documentation
3. Update `skills-lock.json` to include the new skill

### Using Skills

Skills are automatically available when working in this project with Claude Code. Reference them in your prompts:

- "Use the align-agri-data skill to process the new batch"
- "Follow deep-learning-python guidelines for model implementation"

---

## Related Documentation

- **Main project guide**: `CLAUDE.md`
- **Memory system**: `.claude/memory/MEMORY.md`
- **Script documentation**: `scripts/README.md`
- **Data pipeline**: `SCRIPTS_README.md`
