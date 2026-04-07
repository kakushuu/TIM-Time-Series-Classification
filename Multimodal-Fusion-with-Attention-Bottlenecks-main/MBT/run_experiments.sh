#!/bin/bash
# Run 3 experiments: multimodal, trajectory_only, image_only
# Usage: ./run_experiments.sh

set -e

cd /home/research/Agri-MBT/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT

EPOCHS=15
BATCH_SIZE=8
LR=3e-4
CSV_FILE="../../data/aligned_output/B-2024-10-19/aligned_data.csv"
OUTPUT_DIR="../../experiments"

echo "========================================"
echo "Running 3 experiments"
echo "========================================"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Data: $CSV_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Experiment 1: Multimodal (trajectory + image)
echo "========================================"
echo "Experiment 1/3: MULTIMODAL (trajectory + image)"
echo "========================================"
python train_test.py \
    --mode multimodal \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --csv_file $CSV_FILE \
    --output_dir $OUTPUT_DIR

echo ""
echo "Multimodal experiment completed."
echo ""

# Experiment 2: Trajectory only
echo "========================================"
echo "Experiment 2/3: TRAJECTORY ONLY"
echo "========================================"
python train_test.py \
    --mode trajectory_only \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --csv_file $CSV_FILE \
    --output_dir $OUTPUT_DIR

echo ""
echo "Trajectory-only experiment completed."
echo ""

# Experiment 3: Image only
echo "========================================"
echo "Experiment 3/3: IMAGE ONLY"
echo "========================================"
python train_test.py \
    --mode image_only \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --csv_file $CSV_FILE \
    --output_dir $OUTPUT_DIR

echo ""
echo "Image-only experiment completed."
echo ""

# Summary
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""

# Print summary
python - <<'EOF'
import json
import os

results_dir = "../../experiments"
modes = ['multimodal', 'trajectory_only', 'image_only']

print("\n" + "="*60)
print("EXPERIMENT RESULTS SUMMARY")
print("="*60)

for mode in modes:
    result_file = os.path.join(results_dir, f'results_{mode}.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
        print(f"\n{mode:20s}: Best Val Acc = {data['best_val_acc']:.2f}%")
    else:
        print(f"\n{mode:20s}: Results not found")

print("\n" + "="*60)
EOF
