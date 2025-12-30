#!/bin/bash
# Install OCR Pipeline Dependencies (SERVICE 3)

set -e

echo "=========================================="
echo "Installing OCR Pipeline Dependencies"
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

# Check if PyTorch is installed (required for TrOCR)
echo "Checking PyTorch installation..."
if python3 -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch is installed"
else
    echo "⚠️  PyTorch not found. Installing PyTorch..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo "✅ PyTorch installed"
fi

# Install Transformers (required for TrOCR)
echo ""
echo "Installing Transformers (TrOCR dependency)..."
python3 -m pip install --no-cache-dir transformers>=4.30.0 || {
    echo "⚠️  Transformers installation failed, trying alternative method..."
    python3 -m pip cache purge
    python3 -m pip install --no-cache-dir transformers>=4.30.0 || {
        echo "❌ Transformers installation failed!"
        echo "   TrOCR will be unavailable"
        exit 1
    }
}

# Verify installation
echo ""
echo "=========================================="
echo "Verifying OCR Pipeline Installation"
echo "=========================================="
echo ""

# Check PaddleOCR
if python3 -c "from paddleocr import PaddleOCR" 2>/dev/null; then
    echo "✅ PaddleOCR: installed"
else
    echo "❌ PaddleOCR: NOT installed"
    echo "   Install with: pip install paddlepaddle paddleocr"
fi

# Check TrOCR dependencies
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch: $TORCH_VERSION"
else
    echo "❌ PyTorch: NOT installed"
fi

if python3 -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel" 2>/dev/null; then
    TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
    echo "✅ Transformers: $TRANSFORMERS_VERSION"
else
    echo "❌ Transformers: NOT installed"
fi

# Test OCR pipeline imports
echo ""
echo "Testing OCR pipeline imports..."
if python3 -c "from ocr.paddle_recognizer import PaddleOCRRecognizer; print('✅ PaddleOCR recognizer import OK')" 2>/dev/null; then
    echo "✅ PaddleOCR recognizer: import OK"
else
    echo "⚠️  PaddleOCR recognizer: import failed"
fi

if python3 -c "from ocr.trocr_recognizer import TrOCRRecognizer; print('✅ TrOCR recognizer import OK')" 2>/dev/null; then
    echo "✅ TrOCR recognizer: import OK"
else
    echo "⚠️  TrOCR recognizer: import failed"
fi

if python3 -c "from ocr.pipeline import OCRPipeline; print('✅ OCR pipeline import OK')" 2>/dev/null; then
    echo "✅ OCR pipeline: import OK"
else
    echo "⚠️  OCR pipeline: import failed"
fi

echo ""
echo "=========================================="
echo "✅ OCR Pipeline Installation Complete!"
echo "=========================================="
echo ""
echo "Note: TrOCR models will be downloaded automatically on first use."
echo "Model size: ~500MB (microsoft/trocr-base-handwritten)"
echo ""




