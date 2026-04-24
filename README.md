# TIM: Time-Series Classification

Reference implementation and lightweight release package for the TIM paper:

**TIM: Trajectory-ViT-Audio Multimodal Learning for Eleven-Class Agricultural Machinery Trajectory Time-Series Classification**

This branch is a curated paper-release version of the original research repo. It keeps the core training code, the paper result summaries, two lightweight trajectory checkpoints, and a tiny masked demo dataset. Large internal experiment folders, private data paths, logs, and machine-specific launcher scripts have been removed.

## What is included

- `src/`: core dataset and training code for trajectory-only, image-only, audio-only, multimodal, and trimodal ablations
- `scripts/`: public-facing data preparation and figure-generation scripts
- `sample_data/masked_demo/`: tiny synthetic masked demo dataset for smoke tests and command examples
- `artifacts/paper_results/`: released paper metrics (`summary.json`, `per_class_metrics.csv`)
- `checkpoints/`: two small trajectory checkpoints that fit standard GitHub limits
- `assets/`: ready-to-use paper figures and regenerated outputs

## What is not included

- Raw field data, full aligned datasets, and private trajectory spreadsheets
- Large image/audio/trimodal checkpoints that exceed GitHub single-file limits
- Internal notes, agent configs, logs, and workstation-specific 3090/4090 scripts

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Smoke-test the dataset loader with the masked demo data:

```bash
pytest tests/test_release_smoke.py
```

Run a tiny trajectory-only example on the demo data:

```bash
python src/train_ablation.py \
  --mode trajectory_only \
  --train-csv sample_data/masked_demo/train.csv \
  --val-csv sample_data/masked_demo/val.csv \
  --test-csv sample_data/masked_demo/test.csv \
  --save-dir experiments/demo_traj \
  --seq-len 4 \
  --stride 1 \
  --eval-stride 1 \
  --feature-mode raw \
  --traj-encoder lstm \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --device cpu \
  --max-train-batches 2 \
  --max-eval-batches 1
```

Run a tiny trimodal example on the same demo data:

Note: the first trimodal run will download the AST backbone unless it is already cached locally.

```bash
python src/train_ablation.py \
  --mode trimodal \
  --train-csv sample_data/masked_demo/train.csv \
  --val-csv sample_data/masked_demo/val.csv \
  --test-csv sample_data/masked_demo/test.csv \
  --save-dir experiments/demo_trimodal \
  --seq-len 4 \
  --image-window-size 3 \
  --stride 1 \
  --eval-stride 1 \
  --feature-mode engineered \
  --traj-encoder trnet_seq \
  --traj-feature-map-size 6 \
  --batch-size 1 \
  --epochs 1 \
  --num-workers 0 \
  --device cpu \
  --no-pretrained \
  --max-train-batches 1 \
  --max-eval-batches 1
```

## Released paper results

The repository includes the final released summaries used for the paper-scale comparison:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| AST | 74.24 | 46.57 | 72.85 |
| ViT | 69.49 | 51.96 | 69.53 |
| BiLSTM / TRNet-seq | 73.37 | 54.63 | 73.47 |
| TIM concat | 79.16 | 61.19 | 78.44 |
| TIM class-gate | 81.36 | 63.64 | 81.01 |

The raw released summaries are under [artifacts/paper_results](artifacts/paper_results).

## Released checkpoints

The `checkpoints/` folder intentionally contains only small trajectory checkpoints that fit normal GitHub hosting:

- `trnet_seed44_best.pt`: paper trajectory baseline
- `trajectory_only_legacy_best.pt`: earlier lightweight trajectory-only checkpoint

Image, audio, and trimodal weights are not included in this branch because their single-file sizes exceed GitHub’s standard limits.

## Dataset format

For `src/train_ablation.py`, each CSV must contain at least:

- `frame_path`
- `frame_time`
- `video_file`
- `second_in_video`
- `分类`
- `经度`
- `纬度`
- `速度`
- `深度`
- `方向角`

Audio-enabled modes additionally require:

- `audio_path`

See [docs/dataset_format.md](docs/dataset_format.md) for details.

## Public scripts

- `scripts/prepare_b_deep_part_splits.py`: filter aligned OCR/video rows with selected trajectory slices
- `scripts/build_b_deep_part_audio_dataset.py`: extract aligned 1-second WAV clips
- `scripts/build_b_deep_part_multimodal_clean_dataset.py`: build a cleaned self-contained multimodal dataset
- `scripts/analyze_behavior_durations.py`: derive adaptive sampling statistics
- `scripts/build_paper_method_figures.py`: regenerate the method figure
- `scripts/build_paper_fusion_ablation_figures.py`: regenerate paper comparison plots from released summaries

More detail: [scripts/README.md](scripts/README.md)

## Citation

If you use this repository, cite the paper metadata in [CITATION.cff](CITATION.cff).

## License

This release is provided under the MIT License. See [LICENSE](LICENSE).
