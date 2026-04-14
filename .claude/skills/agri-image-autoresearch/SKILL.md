---
name: agri-image-autoresearch
description: Project-specific autonomous experiment loop for improving Agri-MBT image-only training. Use when tuning src/train_ablation.py --mode image_only, comparing visual encoder changes, running repeated image-only experiments, or preparing the image branch before trajectory-only and multimodal ablations.
---

# Agri Image Autoresearch

Use this skill to run disciplined, repeatable image-only experiments for this repository.
It adapts the generic `autoresearch` pattern to the Agri-MBT training stack:

- Training entrypoint: `src/train_ablation.py --mode image_only`
- Current image branch: `ImageEncoder` in `src/train_ablation.py` and `VisualEncoder` in `src/models/visual_encoder.py`
- Standard split: `data/taif_20241018_split/{train,val,test}.csv`
- Standard outputs: `summary.json`, `metrics.csv`, `per_class_metrics.csv`, `predictions.csv`, `confusion_matrix.json`
- Primary optimization metric: `best_val_macro_f1`, higher is better

Do not optimize directly against test metrics. The script writes test metrics today, but autoresearch keep/discard decisions should use validation metrics. Test metrics are for milestone audits after a stable validation improvement.

## Required Protocol

1. **Start with data sanity before model edits**
   - Check CSV row counts and class distribution for train/val/test.
   - Verify `frame_path` files exist and a sample can be opened by PIL.
   - Check missing-frame rate by split and class.
   - Check whether weak classes in prior image-only runs correlate with missing frames or low support.

2. **Establish a baseline**
   - Run one smoke experiment first.
   - Run one full baseline with the same command that candidate experiments will use.
   - Record the baseline in `.autoresearch/results.jsonl` before changing code.

3. **One hypothesis per experiment**
   - State the hypothesis before edits.
   - Change one coherent idea at a time, for example augmentation policy, image sampling, loss, classifier head, backbone pooling, optimizer schedule, or class rebalancing.
   - Avoid mixing unrelated changes such as "new backbone plus new loss plus new sampler" unless the previous single-factor runs justify the bundle.

4. **Use validation macro F1 as the keep rule**
   - Primary metric: `best_val_macro_f1`.
   - Direction: higher.
   - Default minimum meaningful improvement: `+0.01 absolute macro F1`.
   - Also inspect per-class F1 and recall. A run with a tiny macro F1 gain but major collapse on rare classes should not be kept without a clear reason.

5. **Keep artifacts local**
   - Use `experiments/agri_image_autoresearch/<run_id>/` for model outputs.
   - Use `.autoresearch/` for ledgers and reports.
   - Keep `.autoresearch/` untracked.

## Scope

In scope for image-only tuning:

- `src/train_ablation.py`
- `src/models/visual_encoder.py`
- new helper scripts under `scripts/`
- project skill helper scripts under `.agents/skills/agri-image-autoresearch/scripts/`
- experiment runner files `autoresearch.md`, `autoresearch.sh`, and `autoresearch.checks.sh`

Use caution with:

- `src/dataset.py`, because it also affects other training entrypoints.
- train/val/test split creation scripts, because changing splits invalidates earlier comparisons.
- `scripts/run_ablation_suite.sh`, because it affects trajectory-only and multimodal comparisons.

Off limits unless explicitly requested:

- raw data under `data/`
- historical experiment outputs under `experiments/new_adaptive_mbt_20241018_full/`
- trajectory-only or multimodal architecture changes during image-only tuning

## Bootstrap

From the repository root:

```bash
python .agents/skills/agri-image-autoresearch/scripts/create_session_files.py \
  --gpu-ids 1,2,5,6 \
  --epochs 30 \
  --batch-size 8 \
  --suite-dir experiments/agri_image_autoresearch
```

This creates:

- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.checks.sh`
- `.autoresearch/` exclusion in `.git/info/exclude`

Then initialize the generic autoresearch ledger with the installed `autoresearch` scripts:

```bash
python /home/guozhou/.codex/skills/autoresearch/scripts/init_experiment.py \
  --goal "Improve Agri-MBT image-only validation macro F1" \
  --metric-name best_val_macro_f1 \
  --unit f1 \
  --direction higher \
  --command ./autoresearch.sh \
  --checks-command ./autoresearch.checks.sh \
  --scope src/train_ablation.py \
  --scope src/models/visual_encoder.py
```

## Baseline Commands

Smoke run for wiring and GPU sanity:

```bash
AGRI_IMAGE_RUN_ID=smoke AGRI_IMAGE_EPOCHS=1 AGRI_IMAGE_MAX_TRAIN_BATCHES=2 AGRI_IMAGE_MAX_EVAL_BATCHES=2 ./autoresearch.sh
```

Record the full baseline through the generic runner:

```bash
python /home/guozhou/.codex/skills/autoresearch/scripts/run_experiment.py \
  --id baseline \
  --hypothesis "Control image-only run using the current training code and standard split." \
  --change-summary "No code changes." \
  --baseline \
  --output .autoresearch/baseline.json
python /home/guozhou/.codex/skills/autoresearch/scripts/log_experiment.py --input .autoresearch/baseline.json
```

If you need a manually named baseline directory outside the generic runner, use:

```bash
AGRI_IMAGE_RUN_ID=baseline ./autoresearch.sh
```

## Candidate Loop

For every candidate:

1. Write a short hypothesis in `autoresearch.md`.
2. Make the smallest code or runner change needed.
3. Run a smoke check:

   ```bash
   AGRI_IMAGE_RUN_ID=exp-001-smoke AGRI_IMAGE_EPOCHS=1 AGRI_IMAGE_MAX_TRAIN_BATCHES=2 AGRI_IMAGE_MAX_EVAL_BATCHES=2 ./autoresearch.sh
   ```

4. If smoke passes, run the measured experiment:

   ```bash
   AGRI_IMAGE_RUN_ID=exp-001 ./autoresearch.sh
   ```

5. Log the experiment:

   ```bash
   python /home/guozhou/.codex/skills/autoresearch/scripts/run_experiment.py \
     --id exp-001 \
     --hypothesis "<specific hypothesis>" \
     --change-summary "<files and behavior changed>" \
     --output .autoresearch/exp-001.json
   python /home/guozhou/.codex/skills/autoresearch/scripts/log_experiment.py --input .autoresearch/exp-001.json
   ```

6. Keep only if validation macro F1 improves enough and per-class metrics do not show unacceptable regressions.

## High-Value Image-Only Hypotheses

Try these in this order unless the data audit points elsewhere:

1. **Missing frame handling**
   - Current code substitutes a black image when `frame_path` is missing.
   - Test whether logging missing-frame counts, filtering missing-heavy windows, or adding a missing-frame indicator improves validation macro F1.

2. **Train-time augmentation**
   - Current transform is resize, tensor, normalize only.
   - Try mild color jitter, random resized crop, horizontal flip only if camera geometry makes it valid, random grayscale, blur, or RandAugment.
   - Keep validation/test transforms deterministic.

3. **Class imbalance strategy**
   - Current loss uses inverse-square-root class weights.
   - Compare weighted sampler, focal loss, label smoothing, or different weight exponents.

4. **Image temporal sampling**
   - Current defaults often use `--image-window-size 9 --image-sampling center --image-radius 8`.
   - Compare window sizes 1, 5, 9, 13 and center vs uniform sampling.

5. **Visual pooling**
   - Current image encoder attention-pools patch tokens per frame and mean-pools frames.
   - Test attention over frames, CLS token usage if available, or gated frame pooling.

6. **Optimizer and schedule**
   - Add cosine schedule with warmup, lower LR for pretrained visual backbone, or differential LR for classifier vs encoder.

7. **Backbone strategy**
   - First stabilize the current pretrained visual encoder.
   - Only then compare alternate timm or torchvision backbones, keeping feature dimension and checkpoint logic clean.

## Diagnostic Readout

After each full run, inspect:

- `summary.json`: `best_val_macro_f1`, final train/val gap, test macro F1 for audit only
- `metrics.csv`: overfitting, underfitting, unstable validation
- `per_class_metrics.csv`: rare class collapse and weak classes
- `confusion_matrix.json`: dominant confusions
- `predictions.csv`: spatial or video-specific failure clusters

Use this helper to convert a run into generic autoresearch metrics:

```bash
python .agents/skills/agri-image-autoresearch/scripts/summarize_image_run.py \
  --summary experiments/agri_image_autoresearch/baseline/summary.json
```

The helper emits parseable lines such as:

```text
METRIC best_val_macro_f1=0.4324548139
METRIC test_macro_f1=0.3645020147
```

## Stop Conditions

Stop or ask for review when:

- validation macro F1 has not improved after 5 coherent experiments
- the best validation run has worse rare-class behavior than baseline
- improvements require changing data splits
- image-only appears capped and errors are mostly trajectory-dependent
- the next step would touch multimodal fusion before image-only has a stable baseline
