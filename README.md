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

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Mandeep15686/unet-background-removal.git
cd unet-background-removal

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Create virtual environment (Linux/Mac)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Train Your Model

```bash
# Prepare dataset in data/images/ and data/masks/
# Train for 50 epochs (~1.5-2.5 hours on RTX 4050)
python train.py --epochs 50 --batch_size 16 --num_workers 0
```

### Remove Background

```bash
# Single image
python inference.py --input photo.jpg --output result.png

# Batch processing
python inference.py --input photos/ --output results/ --batch
```

---

## 🎓 Training

### Basic Command

```bash
python train.py --epochs 50 --batch_size 16
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 4 | Batch size (higher = faster) |
| `--learning_rate` | 1e-4 | Learning rate |
| `--image_size` | 256 | Input image size |
| `--device` | cuda | Device (cuda/cpu) |
| `--num_workers` | 4 | Data loading workers |
| `--resume` | None | Resume from checkpoint |

### Example Configurations

**Quick Test (5-10 min):**
```bash
python train.py --epochs 5 --batch_size 2
```

**High Quality (1.5-2.5 hours):**
```bash
python train.py --epochs 50 --batch_size 16 --num_workers 0
```

**Maximum Quality (4-6 hours):**
```bash
python train.py --epochs 100 --batch_size 16 --learning_rate 5e-5
```

---

## 🎨 Inference

### Basic Usage

```bash
python inference.py --input photo.jpg --output result.png
```

### Advanced Options

```bash
# Adjust threshold for cleaner edges
python inference.py --input photo.jpg --output result.png --threshold 0.7

# White background
python inference.py --input photo.jpg --output result.png --background white

# Batch processing
python inference.py --input photos/ --output results/ --batch

# Custom model
python inference.py --input photo.jpg --output result.png --model checkpoints/checkpoint_epoch_50.pth
```

---

## 📂 Dataset Preparation

### Directory Structure

```
unet-background-removal/
├── data/
│   ├── images/          # Your training images
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/           # Corresponding masks
│       ├── img001.png   # ⚠️ Must match image filename
│       ├── img002.png
│       └── ...
```

### Requirements

1. **Filenames must match**: `img001.jpg` → `img001.png`
2. **Masks must be grayscale** (single channel)
3. **Mask values**: White (255) = keep, Black (0) = remove
4. **Formats**: JPG, JPEG, PNG, BMP

### Recommended Dataset Sizes

| Purpose | Images |
|---------|--------|
| Testing | 10-20 |
| Basic | 50-100 |
| Good Quality | 500-1000 |
| Production | 2000+ |

### Public Datasets

- [COCO Dataset](http://cocodataset.org) - 330K images
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - Classic benchmark
- [Supervisely Person](https://www.kaggle.com) - 5,711 images
- [Human Segmentation (Kaggle)](https://www.kaggle.com)

---

## 📊 Performance

### Benchmarks (RTX 4050, 15,572 images)

| Batch Size | Speed | Time/Epoch | 50 Epochs | Memory |
|------------|-------|------------|-----------|--------|
| 2 | 12 it/s | 8 min | 7 hours | 1.9 GB |
| 8 | 35 it/s | 3 min | 2.5 hours | 3.5 GB |
| **16** | **60 it/s** | **1.5 min** | **1.5 hours** | **5.0 GB** |
| 24 | 75 it/s | 1 min | 1 hour | 5.8 GB |

### Model Info

- **Architecture**: U-Net with skip connections
- **Parameters**: ~31 million
- **Input**: 256×256 or 512×512
- **Output**: Single-channel mask
- **Inference**: 10-100 images/sec

---

## 🐛 Troubleshooting

### Common Issues

**"No images found"**
```bash
# Check directory
ls data/images/
```

**"Mask not found"**
```bash
# Ensure filenames match exactly
# photo1.jpg → photo1.png
```

**"CUDA out of memory"**
```bash
# Reduce batch size
python train.py --batch_size 8
```

**"Loss not decreasing"**
```bash
# Train longer or adjust learning rate
python train.py --epochs 100 --learning_rate 5e-5
```

**"Torch not compiled with CUDA"**
```bash
# Install CUDA PyTorch
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
│   └── u2net_model.py       # U-Net architecture
├── checkpoints/             # Saved models
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/new-feature`
5. Open Pull Request

---

## 📄 License

MIT License - Copyright (c) 2025 Mandeep Singh

See [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **GitHub**: [@Mandeep15686](https://github.com/Mandeep15686)
- **Repository**: [unet-background-removal](https://github.com/Mandeep15686/unet-background-removal)
- **Issues**: [Report bugs](https://github.com/Mandeep15686/unet-background-removal/issues)

---

**Happy segmenting! 🎨🚀**
