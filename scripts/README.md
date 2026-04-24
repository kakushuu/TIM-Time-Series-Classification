# Public Scripts

## Data preparation

- `prepare_b_deep_part_splits.py`
  Filters aligned OCR/video rows using selected trajectory spreadsheets and writes `train.csv`, `val.csv`, `test.csv`, and `all.csv`.

- `build_b_deep_part_audio_dataset.py`
  Extracts aligned 1-second mono WAV clips and appends `audio_path` metadata columns.

- `build_b_deep_part_multimodal_clean_dataset.py`
  Materializes a cleaned dataset directory with frames, audio clips, split CSVs, and copied trajectory provenance files.

- `analyze_behavior_durations.py`
  Summarizes contiguous behavior durations and emits `duration_sampling_config.json` for adaptive sampling.

## Paper figures

- `build_paper_method_figures.py`
  Regenerates the method schematic into `assets/fig_method_data_processing.png` by default.

- `build_paper_fusion_ablation_figures.py`
  Regenerates the main comparison figures from the released summaries in `artifacts/paper_results/`.
