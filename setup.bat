@echo off
REM U-Net Background Removal - Windows Setup Script
REM Author: Mandeep Singh
REM Date: November 2025

echo ============================================================
echo U-Net Background Removal - Setup (Windows)
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python found
python --version
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo [5/5] Installing dependencies...
echo This may take 5-10 minutes depending on your internet speed...
echo.

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.

REM Create necessary directories
echo Creating data directories...
if not exist "data\images" mkdir "data\images"
if not exist "data\masks" mkdir "data\masks"
if not exist "checkpoints" mkdir "checkpoints"
if not exist "outputs" mkdir "outputs"
echo Directories created successfully
echo.

echo ============================================================
echo Next Steps:
echo ============================================================
echo 1. Place your training images in: data\images\
echo 2. Place corresponding masks in: data\masks\
echo 3. Train the model: python train.py --epochs 50 --batch_size 16
echo 4. Remove backgrounds: python inference.py --input photo.jpg --output result.png
echo.
echo For more information, see README.md and TRAINING_GUIDE.txt
echo ============================================================
echo.

pause
