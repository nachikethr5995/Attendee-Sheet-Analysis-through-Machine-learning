#!/bin/bash

# Complete installation script for AIME Backend
# Installs all dependencies including Detectron2

set -e

echo "=========================================="
echo "AIME Backend - Complete Installation"
echo "=========================================="
echo ""

cd /mnt/d/AIME/Back_end

# Use available Python (3.10 preferred but not required)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python: $PYTHON_VERSION"
echo "   (Python 3.10 is preferred but $PYTHON_VERSION should work)"
echo ""

# Check if venv exists and is valid
if [ ! -d "venv_wsl" ] || [ ! -f "venv_wsl/bin/activate" ]; then
    if [ -d "venv_wsl" ]; then
        echo "⚠️  Virtual environment directory exists but is incomplete/corrupted"
        echo "Removing corrupted venv..."
        rm -rf venv_wsl
    fi
    
    echo "Creating virtual environment..."
    if command -v python3.10 &> /dev/null; then
        echo "Using Python 3.10 to create venv..."
        python3.10 -m venv venv_wsl
    else
        echo "Using available Python to create venv..."
        python3 -m venv venv_wsl
    fi
    
    # Verify venv was created
    if [ ! -d "venv_wsl" ] || [ ! -f "venv_wsl/bin/activate" ]; then
        echo "❌ ERROR: Failed to create virtual environment!"
        echo "   Directory exists: $([ -d "venv_wsl" ] && echo 'yes' || echo 'no')"
        echo "   Activate script exists: $([ -f "venv_wsl/bin/activate" ] && echo 'yes' || echo 'no')"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists and is valid"
fi

# Activate venv for this script
echo "Activating virtual environment..."
source venv_wsl/bin/activate

# Verify venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Failed to activate virtual environment!"
    echo "   VIRTUAL_ENV is not set"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"
echo "   Python: $(which python3)"
echo ""

# Step 1: Install core dependencies (this script will also activate venv)
echo ""
echo "=========================================="
echo "Step 1: Installing Core Dependencies"
echo "=========================================="
bash install_dependencies.sh

# Step 2: Install OCR Pipeline dependencies
echo ""
echo "=========================================="
echo "Step 4: Installing OCR Pipeline Dependencies"
echo "=========================================="
bash INSTALL_OCR.sh

# Final verification
echo ""
echo "=========================================="
echo "Final Verification"
echo "=========================================="

# Ensure venv is still active
if [ -z "$VIRTUAL_ENV" ]; then
    source venv_wsl/bin/activate
fi

echo "Checking OCR Pipeline requirements..."
python3 -c "
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.trocr_recognizer import TrOCRRecognizer
from ocr.pipeline import OCRPipeline

print('✅ OCR Pipeline modules import successfully')
print('✅ PaddleOCR recognizer available:', PaddleOCRRecognizer(lang='en').is_available())
print('✅ TrOCR recognizer available:', TrOCRRecognizer().is_available())
" 2>&1 || echo "⚠️  OCR Pipeline verification failed (check logs above)"

echo ""
echo "Checking Signature Verification requirements..."
python3 check_signature_verification.py 2>&1 | tail -20

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Start the server with:"
echo "  ./start_server.sh"
echo ""





