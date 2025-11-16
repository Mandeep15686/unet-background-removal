```markdown
# U-Net Background Removal 🎨

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Professional U-Net implementation for automatic background removal from images using deep learning.**

Transform any image into a clean cutout with transparent background in seconds. Perfect for e-commerce, photo editing, computer vision projects, and more.

---

## ✨ Features

- 🚀 **Fast & Efficient**: GPU-accelerated inference with batch processing
- 🎯 **High Quality**: Professional-grade edge detection and segmentation
- 🔧 **Easy to Use**: Simple CLI interface for training and inference
- 📦 **Production Ready**: Optimized for real-world applications
- 🎨 **Flexible Output**: Transparent PNG, white/black backgrounds, custom colors
- 💾 **Checkpoint System**: Automatic saving of best models during training
- 📊 **Training Monitoring**: Real-time loss tracking and validation

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU
- 4GB+ RAM (8GB+ recommended)

### Step 1: Clone the Repository

```
git clone https://github.com/Mandeep15686/unet-background-removal.git
cd unet-background-removal
```

### Step 2: Create Virtual Environment

```
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

**For GPU support (CUDA 11.8):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU support (CUDA 12.1+):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Training from Scratch

```
# Prepare your dataset (see Dataset Preparation section)
# Train for 50 epochs (recommended, ~1.5-2.5 hours on RTX 4050)
python train.py --epochs 50 --batch_size 16 --num_workers 0
```

### Remove Background from Image

```
python inference.py --input path/to/image.jpg --output result.png
```

---

## 🎓 Training Your Model

### Basic Training

```
python train.py --epochs 50 --batch_size 16
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 4 | Batch size (increase for faster training) |
| `--learning_rate` | 1e-4 | Learning rate |
| `--image_size` | 256 | Input image size (256 or 512) |
| `--device` | cuda | Device to use (cuda/cpu) |
| `--num_workers` | 4 | Number of data loading workers |
| `--accumulation_steps` | 2 | Gradient accumulation steps |
| `--resume` | None | Path to checkpoint to resume training |

### Recommended Configurations

**For Quick Testing (5-10 minutes):**
```
python train.py --epochs 5 --batch_size 2
```

**For High Quality (1.5-2.5 hours on RTX 4050):**
```
python train.py --epochs 50 --batch_size 16 --num_workers 0
```

**For Maximum Quality (4-6 hours):**
```
python train.py --epochs 100 --batch_size 16 --num_workers 0 --learning_rate 5e-5
```

---

## 🎨 Inference (Background Removal)

### Single Image

```
python inference.py --input photo.jpg --output result.png
```

### Batch Processing

```
# Process entire folder
python inference.py --input photos/ --output results/ --batch
```

### Inference Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Input image or directory |
| `--output` | Required | Output path or directory |
| `--model` | checkpoints/best_model.pth | Path to model checkpoint |
| `--batch` | False | Enable batch processing |
| `--threshold` | 0.5 | Threshold for mask (0.0-1.0) |
| `--background` | transparent | Background type (transparent/white/black) |
| `--device` | cuda | Device to use (cuda/cpu) |

### Examples

```
# Adjust threshold for cleaner edges
python inference.py --input photo.jpg --output result.png --threshold 0.7

# White background instead of transparent
python inference.py --input photo.jpg --output result.png --background white

# Process multiple images
python inference.py --input input_folder/ --output output_folder/ --batch
```

---

## 📂 Dataset Preparation

### Required Structure

```
unet-background-removal/
├── data/
│   ├── images/          # Training images
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/           # Corresponding masks
│       ├── img001.png   # MUST match image filename
│       ├── img002.png
│       └── ...
```

### Important Rules

1. **Filenames Must Match**: `img001.jpg` → `img001.png`
2. **Masks Must Be Grayscale**: Single channel (1-channel)
3. **Mask Values**: 
   - White (255) = Foreground (keep)
   - Black (0) = Background (remove)
4. **Supported Formats**: JPG, JPEG, PNG, BMP

### Dataset Requirements

| Purpose | Minimum | Recommended | Best |
|---------|---------|-------------|------|
| Testing | 10-20 pairs | - | - |
| Basic Training | 50-100 pairs | 500-1000 pairs | 2000+ pairs |
| Production | 1000+ pairs | 5000+ pairs | 10,000+ pairs |

### Public Datasets

1. [COCO Dataset](http://cocodataset.org) - 330K images with segmentation
2. [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - Classic segmentation
3. [Supervisely Person Dataset](https://www.kaggle.com) - 5,711 images with masks
4. [Human Segmentation Dataset (Kaggle)](https://www.kaggle.com)

---

## 📊 Performance

### Speed Benchmarks (RTX 4050, 15,572 images)

| Batch Size | Speed (it/s) | Time per Epoch | 50 Epochs | GPU Memory |
|------------|--------------|----------------|-----------|------------|
| 2 | 12 it/s | 8 min | ~7 hours | 1.9 GB |
| 8 | 35 it/s | 3 min | ~2.5 hours | 3.5 GB |
| 16 | 60 it/s | 1.5 min | ~1.5 hours | 5.0 GB |
| 24 | 75 it/s | 1 min | ~1 hour | 5.8 GB |

### Model Specifications

- **Architecture**: U-Net with skip connections
- **Parameters**: ~31 million
- **Input Size**: 256×256 or 512×512
- **Output**: Single-channel mask (0-1)
- **Inference Speed**: 10-100 images/sec (GPU dependent)

---

## 🐛 Troubleshooting

### Common Issues

**1. "No images found"**
```
# Check directory exists and has images
ls data/images/
```

**2. "Mask not found"**
```
# Ensure filenames match: photo1.jpg → photo1.png
```

**3. "CUDA out of memory"**
```
# Reduce batch size
python train.py --batch_size 8

# Or reduce image size
python train.py --image_size 256 --batch_size 4
```

**4. "Loss not decreasing"**
```
# Train longer
python train.py --epochs 100

# Or adjust learning rate
python train.py --learning_rate 5e-5
```

**5. "Torch not compiled with CUDA"**
```
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Project Structure

```
unet-background-removal/
├── data/
│   ├── images/              # Training images
│   └── masks/               # Training masks
├── models/
│   ├── __init__.py
│   └── u2net_model.py       # U-Net architecture
├── checkpoints/             # Saved models (auto-created)
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Mandeep Singh

---

## 📞 Contact

- **Author**: Mandeep Singh
- **GitHub**: [@Mandeep15686](https://github.com/Mandeep15686)
- **Repository**: [unet-background-removal](https://github.com/Mandeep15686/unet-background-removal)
- **Issues**: [Report a bug](https://github.com/Mandeep15686/unet-background-removal/issues)

---

**Made with ❤️ by Mandeep Singh**

**Happy segmenting! 🎨🚀**
```
