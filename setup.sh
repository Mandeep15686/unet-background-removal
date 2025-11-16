 #!/bin/bash
echo "🚀 U-Net Background Removal Setup"
echo "================================"

echo "📁 Creating directories..."
mkdir -p data/images data/masks checkpoints outputs runs

echo "📦 Installing Python packages..."
pip install torch torchvision Pillow numpy opencv-python

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Place training images in data/images/"
echo "2. Place training masks in data/masks/"
echo "3. Run quick test: python train.py --mode test"
echo "4. Remove backgrounds: python inference.py --input photo.jpg"
