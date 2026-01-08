#!/bin/bash
# Install PP-OCRv4 Models for PaddleOCR

set -e

echo "=========================================="
echo "Installing PP-OCRv4 Models"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ ! -d "venv_wsl" ]; then
    echo "❌ ERROR: venv_wsl not found!"
    echo "   Please run install_dependencies.sh first"
    exit 1
fi

echo "Activating virtual environment..."
source venv_wsl/bin/activate

# Verify we're in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Failed to activate virtual environment!"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"
echo "Python version: $(python3 --version)"
echo ""

# Check if PaddleOCR is installed
echo "Checking PaddleOCR installation..."
if python3 -c "from paddleocr import PaddleOCR" 2>/dev/null; then
    echo "✅ PaddleOCR is installed"
else
    echo "❌ PaddleOCR not found. Installing PaddleOCR..."
    python3 -m pip install --upgrade pip
    python3 -m pip install paddlepaddle paddleocr
    echo "✅ PaddleOCR installed"
fi

echo ""
echo "=========================================="
echo "Downloading and Installing PP-OCRv4 Models"
echo "=========================================="
echo ""
echo "This will download ~200MB of model files."
echo "Models will be installed to: ~/.paddleocr/"
echo ""

# Run the installation script
python3 install_ppocrv4.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ PP-OCRv4 Installation Complete!"
    echo "=========================================="
    echo ""
    echo "The models are now ready to use."
    echo "PaddleOCR will automatically use PP-OCRv4 when initialized with the upgraded configuration."
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ PP-OCRv4 Installation Failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    exit 1
fi







