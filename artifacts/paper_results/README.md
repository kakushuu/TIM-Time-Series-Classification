# Released Paper Results

This directory contains the released evaluation summaries used in the public code package.

## Included models

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| AST | 74.24 | 46.57 | 72.85 |
| ViT | 69.49 | 51.96 | 69.53 |
| BiLSTM / TRNet-seq | 73.37 | 54.63 | 73.47 |
| TIM concat | 79.16 | 61.19 | 78.44 |
| TIM class-gate | 81.36 | 63.64 | 81.01 |

Each model subdirectory contains:

- `summary.json`
- `per_class_metrics.csv`
