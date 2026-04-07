#!/usr/bin/env python3
"""
Run all 3 experiments: multimodal, trajectory_only, image_only
"""
import subprocess
import json
import os
import sys

BASE_DIR = "/home/research/Agri-MBT"
MBT_DIR = f"{BASE_DIR}/Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT"
CSV_FILE = f"{BASE_DIR}/data/aligned_output/B-2024-10-19/aligned_data.csv"
OUTPUT_DIR = f"{BASE_DIR}/experiments"

# Common arguments
COMMON_ARGS = [
    "--num_epochs", "15",
    "--batch_size", "8",
    "--lr", "3e-4",
    "--csv_file", CSV_FILE,
    "--output_dir", OUTPUT_DIR,
]

MODES = ["multimodal", "trajectory_only", "image_only"]

def run_experiment(mode):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"Experiment: {mode}")
    print(f"{'='*70}")

    cmd = ["python", "train_test.py", "--mode", mode] + COMMON_ARGS
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=MBT_DIR, capture_output=False)
    return result.returncode == 0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for mode in MODES:
        success = run_experiment(mode)
        results[mode] = "completed" if success else "failed"

    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for mode in MODES:
        result_file = os.path.join(OUTPUT_DIR, f"results_{mode}.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            print(f"\n{mode:20s}: Best Val Acc = {data['best_val_acc']:.2f}%")
        else:
            print(f"\n{mode:20s}: {results[mode]}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
