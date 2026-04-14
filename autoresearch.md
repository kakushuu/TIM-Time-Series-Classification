# Autoresearch: Agri-MBT Image Only

## Objective

Improve image-only validation macro F1 before trajectory-only retuning and multimodal fusion experiments.

## Up-Front Answers

- Primary metric: `best_val_macro_f1`
- Unit: F1
- Direction: higher
- Minimum meaningful improvement: +0.01 absolute macro F1
- Workload command: `scripts/run_agri_image_gpu.sh --run-id <id>`
- Correctness gates: `./autoresearch.checks.sh`
- Budget / stop criteria: stop after 5 coherent non-improving experiments, when `best_val_macro_f1 >= 0.50`, or when image-only errors are clearly trajectory-dependent

## Standard Workload

- GPUs: physical `0,1,2,5`
- Epochs: `30`
- Batch size: `8`
- Train split: `data/taif_20241018_split/train.csv`
- Validation split: `data/taif_20241018_split/val.csv`
- Test split: `data/taif_20241018_split/test.csv`
- Output root: `experiments/agri_image_autoresearch`

## Scope

- In scope: `src/train_ablation.py`, `src/models/visual_encoder.py`, focused helper scripts
- Off limits: raw data, split changes, trajectory-only and multimodal changes

## Decision Rule

Keep a candidate if `best_val_macro_f1` improves by at least +0.01 over the current best and rare-class F1/recall do not collapse. Test metrics are audit-only.

## Experiment Ledger

`.autoresearch/results.jsonl`

## Report Outputs

- `.autoresearch/report.html`
- `.autoresearch/results.csv`

## Current Best Result

Current kept best:

- Run directory: `experiments/agri_image_autoresearch/exp-003-eval`
- Model checkpoint: `experiments/agri_image_autoresearch/exp-003/best.pt`
- Change: image branch uses GRU temporal pooling over consecutive frame embeddings
- `best_val_macro_f1`: `0.49693114096928687`
- `test_macro_f1`: `0.39803123554430936` (audit only)
- Prior baseline: `experiments/new_adaptive_mbt_20241018_full/image_only`, `best_val_macro_f1=0.4324548139009304`, `test_macro_f1=0.3645020147107385`

## What We've Learned

- Historical image-only run under `experiments/new_adaptive_mbt_20241018_full/image_only` had validation macro F1 around 0.43 and test macro F1 around 0.36.
- Data sanity on 2026-04-13 found zero missing `frame_path` files in train/val/test; initial tuning should focus on imbalance, augmentation, sampling, and optimization rather than missing-frame handling.
- `exp-001` class weight power 0.75 underperformed the baseline by epoch 4 and was discarded.
- `exp-002` temporal Transformer reached validation macro F1 0.5371, but test macro F1 regressed to 0.3405 and class 9 collapsed to zero recall/F1, so it was discarded despite the validation spike.
- `exp-003` GRU temporal pooling reached validation macro F1 0.4969 and improved test macro F1 to 0.3980 without zero-recall class collapse, so it is the current kept best.
- New weak-class target from 2026-04-13: classes `1,2,4,5,6,8,9` should reach at least 0.30 on per-class metrics. Current `exp-003` reaches F1 >= 0.30 for only class 2 among those targets; class 4 is borderline at 0.2946.
- `exp-004` duration-based image radius for weak classes underperformed: test macro F1 0.2603 and target-class F1 >= 0.30 for only 2/7 classes.
- `exp-005` adjacent-frame delta features improved target recall count to 4/7, but target-class F1 >= 0.30 remained 2/7 and overall test macro F1 dropped to 0.3200.
- `exp-006` delta plus class-weight power 0.65 and `exp-007` class-balanced sampling both overfit and failed the target-class floor.
- `exp-008` was rerun with training data changed to original 2024-10-18 train split plus all available 2024-10-19 and 2024-10-20 rows, while validation/test stayed on the original 2024-10-18 split. The split lives in `data/b_ocr_dataset/split_18train_plus_19_20_keep18_eval` and has zero train/val/test overlap by `frame_path` and `frame_time`.
- `exp-008` used GRU temporal pooling plus one-vs-rest auxiliary heads for classes `1,2,4,5,6,8,9` on physical RTX 4090 GPUs `0,1,2,5`. `best_val_macro_f1=0.4692`, `test_macro_f1=0.4016`, and target-class F1 >= 0.30 for 4/7 classes. Weak remaining targets are classes 5, 6, and 9.
- Fixed `autoresearch.sh` to set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; without it, `CUDA_VISIBLE_DEVICES=0,1,2,5` can map to the wrong physical cards and accidentally include a 3090.
- Added `scripts/plot_b_deep_spatial_by_date.py`; generated `experiments/agri_image_autoresearch/b_deep_spatial_by_date.png` from `/private/data/B_deep`, excluding 2024-10-18.
- Prioritize data sanity, augmentation, imbalance handling, image sampling, frame pooling, and optimizer schedule before multimodal work.
