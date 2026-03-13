#!/bin/bash
# Run BiLSTM trajectory classification experiments

echo "=========================================="
echo "BiLSTM Trajectory Classification Experiments"
echo "=========================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p experiments/results/weights

# Test code first
echo ""
echo "Step 1: Testing code..."
echo "=========================================="
python test_code.py

if [ $? -ne 0 ]; then
    echo "❌ Code test failed! Please fix errors before training."
    exit 1
fi

echo ""
echo "✅ Code test passed!"
echo ""

# Ask user which experiments to run
read -p "Run all experiments? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Run all experiments
    echo ""
    echo "Step 2: Running all experiments..."
    echo "=========================================="

    python train.py --mode all --epochs 50 --batch-size 64

else
    # Ask which experiment to run
    echo ""
    echo "Which experiment to run?"
    echo "  1) Trajectory only"
    echo "  2) Multimodal (Trajectory + Image)"
    echo "  3) Both experiments"
    read -p "Enter choice (1/2/3): " -n 1 -r
    echo ""

    case $REPLY in
        1)
            echo ""
            echo "Running Experiment 1: BiLSTM Trajectory Only"
            echo "=========================================="
            python train.py --mode trajectory_only --epochs 50 --batch-size 64
            ;;
        2)
            echo ""
            echo "Running Experiment 2: BiLSTM Multimodal"
            echo "=========================================="
            python train.py --mode multimodal --epochs 50 --batch-size 64
            ;;
        3)
            echo ""
            echo "Running Both Experiments"
            echo "=========================================="
            python train.py --mode all --epochs 50 --batch-size 64
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Check results
echo ""
echo "=========================================="
echo "Experiment Results:"
echo "=========================================="

if [ -f "experiments/results/results_trajectory_only.json" ]; then
    echo ""
    echo "Trajectory Only Results:"
    python -c "
import json
with open('experiments/results/results_trajectory_only.json', 'r') as f:
    r = json.load(f)
    print(f'  Test Accuracy: {r[\"test_acc\"]:.2f}%')
    print(f'  F1 Macro: {r[\"test_metrics\"][\"f1_macro\"]:.2f}%')
    print(f'  F1 Weighted: {r[\"test_metrics\"][\"f1_weighted\"]:.2f}%')
"
fi

if [ -f "experiments/results/results_multimodal.json" ]; then
    echo ""
    echo "Multimodal Results:"
    python -c "
import json
with open('experiments/results/results_multimodal.json', 'r') as f:
    r = json.load(f)
    print(f'  Test Accuracy: {r[\"test_acc\"]:.2f}%')
    print(f'  F1 Macro: {r[\"test_metrics\"][\"f1_macro\"]:.2f}%')
    print(f'  F1 Weighted: {r[\"test_metrics\"][\"f1_weighted\"]:.2f}%')
"
fi

echo ""
echo "=========================================="
echo "✅ All experiments completed!"
echo "Results saved to: experiments/results/"
echo "=========================================="
