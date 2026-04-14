#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m py_compile src/train_ablation.py src/models/visual_encoder.py

if [[ ! -f data/taif_20241018_split/train.csv ]]; then
  echo "missing train split" >&2
  exit 1
fi
