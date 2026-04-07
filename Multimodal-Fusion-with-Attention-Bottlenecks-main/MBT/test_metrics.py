#!/usr/bin/env python3
"""
Quick test: Run 1 epoch to verify metrics calculation works
"""
import subprocess
import json
import os

BASE_DIR = "/home/research/Agri-MBT"
MBT_DIR = f"{BASE_DIR}/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT"
CSV_FILE = f"{BASE_DIR}/data/aligned_output/B-2024-10-19/aligned_data.csv"
OUTPUT_DIR = f"{BASE_DIR}/experiments/metrics_test"

cmd = [
    "python", "train_test.py",
    "--mode", "multimodal",
    "--num_epochs", "1",
    "--batch_size", "8",
    "--lr", "3e-4",
    "--csv_file", CSV_FILE,
    "--output_dir", OUTPUT_DIR
]

print("Running quick test with 1 epoch...")
result = subprocess.run(cmd, cwd=MBT_DIR)

if result.returncode == 0:
    print("\nTest successful! Checking results...")
    result_file = os.path.join(OUTPUT_DIR, "results_multimodal.json")
    with open(result_file) as f:
        data = json.load(f)

    print(f"\nBest Val Acc: {data['best_val_acc']:.2f}%")
    print(f"\nMetrics keys: {list(data['metrics'].keys())}")
    print(f"\nMacro avg precision: {data['metrics']['macro_avg']['precision']:.2f}%")
    print(f"\nPer-class metrics available for {len(data['metrics']['per_class'])} classes")
else:
    print(f"\nTest failed with return code {result.returncode}")
