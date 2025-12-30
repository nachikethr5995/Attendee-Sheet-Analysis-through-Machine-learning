#!/bin/bash

# Start AIME Backend Server

echo "=========================================="
echo "Starting AIME Backend Server"
echo "=========================================="
echo ""

# Navigate to project directory
cd /mnt/d/AIME/Back_end

# Activate virtual environment
if [ -d "venv_wsl" ]; then
    echo "Activating virtual environment..."
    source venv_wsl/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found!"
    echo "Please run the installation script first."
    exit 1
fi

# Check if Detectron2 is installed (optional check)
echo ""
echo "Checking dependencies..."
if python3 -c "import detectron2" 2>/dev/null; then
    echo "✅ Detectron2 installed"
else
    echo "⚠️  Detectron2 not found (optional, SERVICE 1 will use PaddleOCR only)"
fi

# Check if PaddleOCR is installed
if python3 -c "from paddleocr import PaddleOCR" 2>/dev/null; then
    echo "✅ PaddleOCR installed"
else
    echo "⚠️  PaddleOCR not found"
fi

# Start server
echo ""
echo "=========================================="
echo "Starting server on http://localhost:8000"
echo "=========================================="
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py






