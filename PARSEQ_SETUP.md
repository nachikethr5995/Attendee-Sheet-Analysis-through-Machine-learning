# PARSeq Setup Guide

## Overview

PARSeq (Permutation Autoregressive Sequence) is used for handwritten text recognition. This implementation uses the **official PARSeq codebase** from https://github.com/baudm/parseq, not HuggingFace.

## Setup Steps

### 1. Clone PARSeq Repository

```bash
cd Back_end/ocr/handwritten
git clone https://github.com/baudm/parseq.git
```

Or download and extract the repository into `Back_end/ocr/handwritten/parseq/`

### 2. Required Directory Structure

After cloning, you should have:

```
Back_end/ocr/handwritten/parseq/
├── models/
│   └── parseq.py          # PARSeq model architecture
├── configs/
│   └── parseq.yaml        # Model configuration
├── utils/
│   └── (various utilities)
├── weights/
│   └── parseq_base.pth    # Model weights (download separately)
└── __init__.py
```

### 3. Download Model Weights (Optional)

**Option A: Auto-download (Recommended)**
- Use `pretrained=parseq` in config (default)
- Weights will download automatically on first use
- No manual download needed

**Option B: Manual Download**
Download from: https://github.com/baudm/parseq/releases/tag/v1.0.0

Available models:
- `parseq-bb5792a6.pt` (full PARSeq model - best accuracy)
- `parseq_tiny-e7a21b54.pt` (PARSeq-Tiny - faster, smaller)
- `parseq_small_patch16_224-fcf06f5a.pt` (PARSeq-Small)

Direct download links:
- Full: https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt
- Tiny: https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt

Place in: `Back_end/ocr/handwritten/parseq/weights/`

**Note:** File extension is `.pt` (PyTorch), not `.pth`

### 4. Install Dependencies

```bash
pip install torch torchvision timm einops pillow pyyaml
```

**Note:** No transformers or HuggingFace dependencies needed.

### 5. Verify Setup

Run the diagnostic script:

```bash
python tools/diagnose_parseq.py
```

You should see:
- `[OK] PARSeqRecognizer initialized successfully`

## Configuration

The default configuration uses auto-download:

```python
PARSEQ_CHECKPOINT_PATH: str = "pretrained=parseq"  # Auto-downloads weights
```

**Options:**
1. **Auto-download (default):** `"pretrained=parseq"` or `"pretrained=parseq-tiny"`
2. **Local file:** `"./ocr/handwritten/parseq/weights/parseq-bb5792a6.pt"`

**Available pretrained models:**
- `pretrained=parseq` - Full PARSeq model (best accuracy)
- `pretrained=parseq-tiny` - PARSeq-Tiny (faster, smaller)
- `pretrained=parseq-patch16-224` - PARSeq-Small variant

## Troubleshooting

### PARSeq not found
- Ensure the repository is cloned into the correct location
- Check that `models/parseq.py` exists

### Model weights not found
- **Auto-download:** Use `pretrained=parseq` in config (default) - will download automatically
- **Manual download:** Download `.pt` files from https://github.com/baudm/parseq/releases/tag/v1.0.0
- Update `PARSEQ_CHECKPOINT_PATH` in config to point to downloaded file
- **Note:** Files are `.pt` (PyTorch), not `.pth`

### Import errors
- Ensure all dependencies are installed: `pip install torch torchvision timm einops pillow pyyaml`
- Check Python path includes the parseq directory

