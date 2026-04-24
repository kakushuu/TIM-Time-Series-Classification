# Dataset Format

`src/train_ablation.py` expects one CSV per split (`train.csv`, `val.csv`, `test.csv`).

## Required columns

| Column | Meaning |
|---|---|
| `frame_path` | Relative or absolute path to the frame image |
| `frame_time` | Timestamp for the aligned row |
| `video_file` | Source video identifier |
| `second_in_video` | Integer second index inside the source video |
| `分类` | Integer class label in `[0, 10]` |
| `经度` | Longitude |
| `纬度` | Latitude |
| `速度` | Speed |
| `深度` | Depth |
| `方向角` | Heading angle in degrees |

## Additional columns for audio-enabled modes

| Column | Meaning |
|---|---|
| `audio_path` | Relative or absolute path to a 1-second WAV clip |

## Notes

- Paths can be relative to the repository root.
- Rows are grouped by `video_file` and ordered by `second_in_video` plus `frame_time`.
- The training code splits long contiguous runs when the adjacent timestamp gap exceeds `--max-time-gap`.
- `sample_data/masked_demo/` is the minimal example layout shipped in this branch.
