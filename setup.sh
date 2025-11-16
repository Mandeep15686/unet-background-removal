#!/bin/bash
# U-Net Background Removal - Linux/Mac Setup Script
# Author: Mandeep Singh
# Date: November 2025

echo "============================================================"
echo "U-Net Background Removal - Setup (Linux/Mac)"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[1/5] Python found"
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv .venv
    echo "Virtual environment created successfully"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source .venv/bin/activate
echo ""

# Upgrade pip
echo "[4/5] Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install requirements
echo "[5/5] Installing dependencies..."
echo "This may take 5-10 minutes depending on your internet speed..."
echo ""

# Detect if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/images
mkdir -p data/masks
mkdir -p checkpoints
mkdir -p outputs
echo "Directories created successfully"
echo ""

echo "============================================================"
echo "Next Steps:"
echo "============================================================"
echo "1. Place your training images in: data/images/"
echo "2. Place corresponding masks in: data/masks/"
echo "3. Train the model: python train.py --epochs 50 --batch_size 16"
echo "4. Remove backgrounds: python inference.py --input photo.jpg --output result.png"
echo ""
echo "For more information, see README.md and TRAINING_GUIDE.txt"
echo "============================================================"
echo ""
