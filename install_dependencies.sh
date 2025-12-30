#!/bin/bash

# Install AIME Backend Dependencies

echo "=========================================="
echo "Installing AIME Backend Dependencies"
echo "=========================================="
echo ""

# Navigate to project directory
cd /mnt/d/AIME/Back_end

# Use existing venv if available, otherwise create with available Python
if [ -d "venv_wsl" ]; then
    echo "Activating existing virtual environment..."
    source venv_wsl/bin/activate
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ ERROR: Failed to activate virtual environment!"
        exit 1
    fi
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo "Python version: $(python3 --version)"
    echo "Python path: $(which python3)"
else
    # Check for Python 3.10 (preferred) or use available Python
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo "✅ Found Python 3.10 (preferred)"
    else
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Using available Python: $PYTHON_VERSION"
    fi
    
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv_wsl
    
    if [ ! -d "venv_wsl" ]; then
        echo "❌ ERROR: Failed to create virtual environment!"
        exit 1
    fi
    
    source venv_wsl/bin/activate
    
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ ERROR: Failed to activate virtual environment!"
        exit 1
    fi
    
    echo "✅ Virtual environment created and activated: $VIRTUAL_ENV"
    echo "Python version: $(python3 --version)"
    echo "Python path: $(which python3)"
fi

# Verify we're in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Virtual environment not activated!"
    echo "   Please activate venv_wsl first:"
    echo "   source venv_wsl/bin/activate"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

# Upgrade pip and build tools first
echo ""
echo "Upgrading pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel

# Install core dependencies
echo ""
echo "Installing core dependencies..."
python3 -m pip install --no-cache-dir fastapi uvicorn python-multipart pydantic pydantic-settings || {
    echo "⚠️  Some packages failed, trying with cache clear..."
    python3 -m pip cache purge
    python3 -m pip install fastapi uvicorn python-multipart pydantic pydantic-settings
}

# Install preprocessing dependencies
echo ""
echo "Installing preprocessing dependencies..."
python3 -m pip install --no-cache-dir opencv-python opencv-contrib-python Pillow numpy scikit-image pdf2image || {
    echo "⚠️  Retrying with cache clear..."
    python3 -m pip cache purge
    python3 -m pip install opencv-python opencv-contrib-python Pillow numpy scikit-image pdf2image
}

# Install logging and utilities
echo ""
echo "Installing utility dependencies..."
python3 -m pip install loguru python-dotenv

# Install SERVICE 1 dependencies (if available)
echo ""
echo "Installing SERVICE 1 dependencies..."
python3 -m pip install fvcore iopath || echo "⚠️  fvcore/iopath installation skipped (optional)"

# Check for Detectron2 (optional)
if python3 -c "import detectron2" 2>/dev/null; then
    echo "✅ Detectron2 already installed"
else
    echo "⚠️  Detectron2 not installed (optional, will use PaddleOCR only)"
fi

# Check for PaddleOCR (optional)
if python3 -c "from paddleocr import PaddleOCR" 2>/dev/null; then
    echo "✅ PaddleOCR already installed"
else
    echo "⚠️  PaddleOCR not installed (will install now...)"
    python3 -m pip install paddlepaddle paddleocr || echo "⚠️  PaddleOCR installation failed (optional)"
fi

# Install YOLOv8 (for layout detection)
echo ""
echo "Installing YOLOv8..."
python3 -m pip install --no-cache-dir ultralytics || {
    echo "⚠️  YOLOv8 installation failed, trying alternative method..."
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install --no-cache-dir ultralytics || echo "⚠️  YOLOv8 installation skipped (optional)"
}

# Install OCR Pipeline dependencies (SERVICE 3)
echo ""
echo "Installing OCR Pipeline dependencies (TrOCR)..."
python3 -m pip install --no-cache-dir transformers>=4.30.0 || {
    echo "⚠️  Transformers installation failed, trying alternative method..."
    python3 -m pip cache purge
    python3 -m pip install --no-cache-dir transformers>=4.30.0 || echo "⚠️  Transformers installation failed (TrOCR will be unavailable)"
}

# Verify OCR dependencies
echo ""
echo "Verifying OCR dependencies..."
if python3 -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel" 2>/dev/null; then
    echo "✅ Transformers (TrOCR) installed"
else
    echo "⚠️  Transformers not installed (TrOCR will be unavailable)"
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "import fastapi; print('✅ FastAPI installed')" || echo "❌ FastAPI not found"
python3 -c "import uvicorn; print('✅ Uvicorn installed')" || echo "❌ Uvicorn not found"
python3 -c "import cv2; print('✅ OpenCV installed')" || echo "❌ OpenCV not found"
python3 -c "from PIL import Image; print('✅ Pillow installed')" || echo "❌ Pillow not found"
python3 -c "from paddleocr import PaddleOCR; print('✅ PaddleOCR installed')" || echo "❌ PaddleOCR not found"
python3 -c "from transformers import TrOCRProcessor; print('✅ Transformers (TrOCR) installed')" || echo "⚠️  Transformers not found (TrOCR will be unavailable)"
python3 -c "import torch; from ultralytics import YOLO; print('✅ PyTorch & Ultralytics (GPDS verification) installed')" || echo "⚠️  PyTorch/Ultralytics not found (GPDS verification will be unavailable)"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "You can now start the server with:"
echo "  ./start_server.sh"
echo "  or"
echo "  python3 main.py"
echo ""






