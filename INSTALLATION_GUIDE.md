# AIME Backend Installation Guide

> **⚠️ Important:** All installation must be done in **WSL (Windows Subsystem for Linux)**.  
> Open WSL terminal and run the commands there.

## Quick Start

### 1. Open WSL Terminal

Open WSL (Windows Subsystem for Linux) terminal:
- Press `Win + R`, type `wsl`, press Enter
- Or open Ubuntu from Start Menu

### 2. Install All Dependencies (Recommended)

```bash
cd /mnt/d/AIME/Back_end
bash INSTALL_ALL.sh
```

This installs everything including OCR Pipeline dependencies automatically.

**Or install step by step:**

### 3. Install OCR Pipeline Dependencies (If not using INSTALL_ALL.sh)

```bash
./INSTALL_OCR.sh
```

This installs:
- Transformers library (required for TrOCR)
- Verifies PaddleOCR installation
- Tests OCR pipeline imports

**Or use the complete installer (installs everything including OCR):**
```bash
./INSTALL_ALL.sh
```

### 4. Start the Server

```bash
./start_server.sh
```

Or manually:
```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate
python3 main.py
```

## Environment Setup

### Virtual Environment

The project uses `venv_wsl` virtual environment. Always activate it:

```bash
source venv_wsl/bin/activate
```

### Python Version

- **Python 3.10+** (recommended)
- Python 3.11+ should work
- Python 3.12 works

**Note:** The installation script will use Python 3.10 if available, otherwise uses the default python3.

## Troubleshooting

### PaddleOCR Issues

PaddleOCR auto-downloads models on first use. If it fails:

1. Check internet connection
2. Verify disk space (models are large)
3. Check logs for specific errors

### Installing PP-OCRv4 Models (Recommended)

For better OCR accuracy, install PP-OCRv4 models:

**Option 1: Using the installation script (Recommended)**
```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate
bash INSTALL_PPOCRV4.sh
```

**Option 2: Using Python script directly**
```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate
python3 install_ppocrv4.py
```

This will:
- Download PP-OCRv4 detection model (~50MB)
- Download PP-OCRv4 recognition model (~10MB)
- Download angle classifier model (~2MB)
- Install models to `~/.paddleocr/` directory

**Note:** The installation script will automatically use PP-OCRv4 models when available. If models are not found, it will fall back to default PaddleOCR models.

### OCR Pipeline Issues

If TrOCR (Transformers) fails to install:

1. **Verify PyTorch is installed:**
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```

2. **Install Transformers manually:**
   ```bash
   source venv_wsl/bin/activate
   pip install transformers>=4.30.0
   ```

3. **Verify OCR pipeline:**
   ```bash
   python3 -c "from ocr.pipeline import OCRPipeline; print('✅ OCR Pipeline OK')"
   ```

4. **Note:** TrOCR models (~500MB) download automatically on first use

## Model Status

### Required Models
- ✅ **PaddleOCR** - Auto-downloads on first use (for detection and recognition)
- ✅ **YOLOv8s** - Layout detection model (trained)
- ✅ **Transformers** - Required for TrOCR (fallback recognition)

### Optional Models (Recommended)
- ⚠️ **PP-OCRv4 Models** - Enhanced OCR accuracy (install with `INSTALL_PPOCRV4.sh`)
  - Detection model: `PP-OCRv4_det` (~50MB)
  - Recognition model: `PP-OCRv4_rec` (~10MB)
  - Angle classifier: `ch_ppocr_mobile_v2.0_cls` (~2MB)
- ⚠️ **TrOCR Models** - Auto-downloads on first use (~500MB, handwriting recognition)
- ⚠️ **GPDS Signature** - For signature detection (not included)
- ⚠️ **Checkbox Model** - For checkbox detection (requires training)

## Verification

### Check YOLOv8s Layout Detection
```bash
python3 -c "from layout.yolov8_layout_detector import YOLOv8LayoutDetector; detector = YOLOv8LayoutDetector(); print('✅ YOLOv8s available:', detector.is_available())"
```

### Check OCR Pipeline
```bash
python3 -c "
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.trocr_recognizer import TrOCRRecognizer
print('✅ PaddleOCR recognizer:', PaddleOCRRecognizer().is_available())
print('✅ TrOCR recognizer:', TrOCRRecognizer().is_available())
"
```

### Check Server
```bash
curl http://localhost:8000/api/health
```

## File Structure

```
Back_end/
├── models/
│   ├── yolo_layout/         # YOLOv8s layout detection model
│   ├── signature_model/     # GPDS signature model (optional)
│   └── checkbox_model/      # Checkbox model (optional)
├── storage/                 # Processed files
├── venv_wsl/               # Virtual environment
└── logs/                    # Application logs
```

## Notes

- All models are stored in `models/` directory
- Processed images are in `storage/`
- Logs are in `logs/app.log`
- Server runs on `http://0.0.0.0:8000` by default




