#!/bin/bash
# Agri-MBT Environment Setup Script

set -e  # Exit on error

echo "=================================="
echo "Agri-MBT Environment Setup"
echo "=================================="
echo ""

# Environment name
ENV_NAME="agri-mbt"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Step 1: Creating base conda environment with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo ""
echo "Step 3: Installing PyTorch with CUDA 11.8..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 4: Installing core dependencies..."
pip install numpy scipy pandas scikit-learn
pip install matplotlib seaborn tqdm
pip install opencv-python pillow pytesseract
pip install openpyxl

echo ""
echo "Step 5: Installing deep learning libraries..."
pip install timm transformers einops

echo ""
echo "Step 6: Installing Jupyter (optional)..."
pip install jupyter ipywidgets

echo ""
echo "=================================="
echo "✓ Environment setup complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To verify installation, run:"
echo "    python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
