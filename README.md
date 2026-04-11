U-Net Background Removal 🎨






Professional U-Net implementation for automatic background removal from images using deep learning.

Transform any image into a clean cutout with transparent background in seconds. Perfect for e-commerce, photo editing, computer vision projects, and more.

✨ Features
🚀 Fast & Efficient: GPU-accelerated inference with batch processing
🎯 High Quality: Accurate segmentation with sharp edge detection
🔧 Easy to Use: CLI-based training and inference
📦 Production Ready: Designed for real-world applications
🎨 Flexible Output: Transparent PNG, white/black/custom backgrounds
💾 Checkpoint System: Saves best models automatically
📊 Training Monitoring: Loss tracking and validation support
👥 Team & Contribution Breakdown

This project was collaboratively developed with clearly defined ownership across core components to ensure modularity, scalability, and production readiness.

🔹 Mandeep Singh — Project Lead & Model Architect
Designed and implemented the U-Net architecture
Led model design, optimization, and experimentation
Implemented checkpointing and model saving mechanisms
Managed overall project integration and architecture decisions
🔹 Rahul Dewangan — ML Engineer & Training Pipeline Developer
Developed the training pipeline (train.py)
Implemented data preprocessing and augmentation
Integrated loss tracking, validation metrics, and monitoring
Optimized training using GPU acceleration and batch processing
Conducted experiments to improve model performance and convergence
🔹 Shashi Kant Kumar — Inference & Deployment Engineer
Built the inference pipeline (inference.py)
Implemented CLI interface for single and batch processing
Added support for multiple output formats (transparent, white, custom)
Optimized inference for speed and real-world usability
🧠 Key Highlights
Modular and scalable deep learning pipeline (data → training → inference)
Efficient implementation using PyTorch
Designed for real-world deployment and usability
Clean and maintainable code structure following best practices
🔧 Installation
Prerequisites
Python 3.8+
NVIDIA GPU with CUDA (recommended) or CPU
4GB+ RAM (8GB+ recommended)
Setup
# Clone repository
git clone https://github.com/Mandeep15686/unet-background-removal.git
cd unet-background-removal

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Verify Installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
🚀 Quick Start
Train Model
python train.py --epochs 50 --batch_size 16 --num_workers 0
Run Inference
# Single image
python inference.py --input photo.jpg --output result.png

# Batch processing
python inference.py --input photos/ --output results/ --batch
🎓 Training
Parameters
Parameter	Default	Description
epochs	20	Number of epochs
batch_size	4	Batch size
learning_rate	1e-4	Learning rate
image_size	256	Input size
device	cuda	cuda/cpu
num_workers	4	Data loaders
Example Configs

Quick Test

python train.py --epochs 5 --batch_size 2

High Quality

python train.py --epochs 50 --batch_size 16

Maximum Performance

python train.py --epochs 100 --learning_rate 5e-5
🎨 Inference Options
# Adjust threshold
python inference.py --input img.jpg --output out.png --threshold 0.7

# White background
python inference.py --input img.jpg --output out.png --background white

# Custom model
python inference.py --model checkpoints/model.pth
📂 Dataset Structure
data/
├── images/
│   ├── img1.jpg
│   └── ...
└── masks/
    ├── img1.png
    └── ...
Rules
Filenames must match
Masks must be grayscale
White = foreground, Black = background
📊 Performance
Architecture: U-Net
Parameters: ~31M
Input: 256×256 / 512×512
Inference Speed: 10–100 images/sec
Optimized for GPU acceleration
🐛 Troubleshooting

CUDA error

Reduce batch size

No images found

Check data path

Loss not decreasing

Increase epochs or tune learning rate
📁 Project Structure
├── data/
├── models/
├── checkpoints/
├── train.py
├── inference.py
├── requirements.txt
└── README.md
🤝 Contributing
Fork repo
Create branch
Commit changes
Open PR
📄 License

MIT License © 2025

📞 Contact
GitHub: https://github.com/Mandeep15686
Issues: https://github.com/Mandeep15686/unet-background-removal/issues
