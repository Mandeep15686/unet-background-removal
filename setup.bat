@echo off
echo 🚀 U-Net Background Removal Setup
echo ================================

echo 📁 Creating directories...
mkdir data\images 2>nul
mkdir data\masks 2>nul
mkdir checkpoints 2>nul
mkdir outputs 2>nul
mkdir runs 2>nul

echo 📦 Installing Python packages...
pip install torch torchvision Pillow numpy opencv-python

echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Place training images in data/images/
echo 2. Place training masks in data/masks/
echo 3. Run quick test: python train.py --mode test
echo 4. Remove backgrounds: python inference.py --input photo.jpg
echo.
pause
