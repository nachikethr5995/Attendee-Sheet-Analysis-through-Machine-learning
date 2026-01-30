# 📘 Installation & Setup Guide
## OCR Pipeline (YOLOv8 + PaddleOCR + PARSeq)

**Aligned to Existing Project Structure & Configurations**

---

## 1. Scope of This Document

This document explains how to install, configure, and run the existing OCR system by:

- Using library versions already defined in the project
- Using the existing project folder structure
- Respecting all architectural rules, constraints, and logic
- Providing explicit steps for PARSeq installation and weight placement

⚠️ **This guide does not introduce new logic, models, or versions.**

---

## 2. System & Library Versions (SOURCE OF TRUTH)

All versions must be taken directly from the project, specifically:

- `requirements.txt`
- Any existing lock files (if present)
- Code comments or config files referencing model expectations

❌ **Do NOT install "latest" versions manually**  
❌ **Do NOT upgrade/downgrade libraries arbitrarily**

**Before installation, always review:**

```bash
cat requirements.txt
```

---

## 3. Existing Project Structure (REFERENCE)

The project structure must be used as-is.

```
Back_end/
│
├── api/
│   ├── analyze.py
│   ├── upload.py
│   ├── health.py
│   └── debug.py
│
├── core/
│   ├── config.py
│   ├── logging.py
│   └── utils.py
│
├── layout/
│   ├── yolov8_layout_detector.py
│   ├── layout_service.py
│   └── ...
│
├── ocr/
│   ├── class_based_router.py
│   ├── paddle_recognizer.py
│   ├── handwritten/
│   │   ├── parseq_recognizer.py
│   │   └── parseq/
│   │       ├── strhub/
│   │       ├── configs/
│   │       └── weights/
│   │           └── parseq-bb5792a6.pt
│   └── ...
│
├── postprocessing/
│   ├── unified_pipeline.py
│   ├── api_output_formatter.py
│   └── ...
│
├── models/
│   ├── yolo_layout/
│   │   └── best.pt (or yolov8s_layout.pt)
│   └── ...
│
├── requirements.txt
├── main.py
└── README.md
```

⚠️ **Do not rename, move, or restructure any folders or files.**

---

## 4. Python Environment Setup

### 4.1 Python Version

Use the same Python version already used by the project  
(check CI config, Dockerfile, or README if unsure).

**Verify:**

```bash
python --version
```

**Recommended:** Python 3.10+ (Python 3.11+ should work, Python 3.12 works)

### 4.2 Virtual Environment

From the project root:

```bash
cd Back_end
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

**Note:** The project may also use `venv_wsl` for WSL environments.

---

## 5. Dependency Installation

Install dependencies exactly as specified:

```bash
pip install -r requirements.txt
```

**If installation fails:**

- Do NOT change versions
- Resolve system-level dependencies instead

**Key dependencies from requirements.txt:**
- `ultralytics>=8.0.196` (YOLOv8)
- `paddleocr>=2.7.0.3` (PaddleOCR)
- `paddlepaddle>=2.5.2` (PaddlePaddle backend)
- `torch>=2.1.0` (PyTorch for PARSeq)
- `transformers>=4.30.0` (For TrOCR - legacy, PARSeq doesn't use it)
- `pytorch-lightning` (For PARSeq - check version compatibility)
- `lmdb` (For PARSeq datasets)
- `nltk` (For PARSeq edit distance)

---

## 6. Model Weights Placement (CRITICAL)

### 6.1 YOLOv8 Layout Model

Place YOLO weights at:

```
models/yolo_layout/yolov8s_layout.pt
```

**Or if using trained model:**

```
models/yolo_layout/best.pt
```

**Rules:**

- YOLO is the **ONLY detector**
- Detects only layout classes:
  - `Text_box`
  - `Handwritten`
  - `Signature`
  - `Checkbox`
  - `Table`
- **No OCR model is allowed to detect regions.**

### 6.2 PARSeq Handwritten Recognition Model

Place PARSeq assets at:

```
ocr/handwritten/parseq/weights/parseq-bb5792a6.pt
```

**Rules:**

- Files must be **local**
- **No auto-download**
- **No HuggingFace dependency**
- Filenames must match exactly
- Checkpoint file: `parseq-bb5792a6.pt` (not `.pth` or `.ckpt`)

**Download from:**

- GitHub Releases: https://github.com/baudm/parseq/releases
- Direct link: https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt

**Alternative models (if needed):**
- `parseq_tiny-e7a21b54.pt` (faster, smaller)
- `parseq_small_patch16_224-fcf06f5a.pt` (small variant)

---

## 7. OCR Configuration Rules (MANDATORY)

### PaddleOCR (Printed Text)

- **Detection disabled** (`det=False`)
- **Recognition only**
- Receives only YOLO `Text_box` crops
- **Must never see full image**

⚠️ **If PaddleOCR runs detection → installation is invalid.**

### PARSeq (Handwritten Text)

- Receives only YOLO `Handwritten` crops
- **One YOLO box → one recognition output**
- **Handwritten boxes must NOT be merged**
- **No fallback to TrOCR or any other model**

---

## 8. Core Pipeline Behavior (EXPECTED)

```
Image
 ↓
YOLOv8 Layout Detection (ONLY detector)
 ↓
Class-based routing
 ├─ Text_box     → PaddleOCR
 ├─ Handwritten  → PARSeq
 ├─ Signature    → Presence only
 ├─ Checkbox     → Presence + checked state
 ├─ Table        → Row grouping anchor
 ↓
Row & column grouping
 ↓
Handwritten sanitization
 ↓
Final API output shaping
```

**Any deviation indicates misinstallation or unintended changes.**

---

## 9. Mandatory Safety Assertion

Every OCR invocation validates:

```python
assert source_bbox_origin == "YOLO"
```

**If violated → pipeline must fail loudly.**

---

## 10. Handwritten Sanitization Logic (Phase-1.5)

Applied only to handwritten OCR output.

**Purpose:**

- Remove OCR-introduced symbols
- Preserve recognition intent
- Avoid guessing

**Constraints:**

- ❌ No spell correction
- ❌ No dictionary
- ❌ No inference

**Sanitization steps:**

1. Trim obvious edge junk: `text.strip(" |-_.,;:")`
2. Remove repeated decoder artifacts: `re.sub(r"[|_]{2,}", "", text)`
3. Remove trailing non-alphanumeric characters: `re.sub(r"[^A-Za-z0-9]+$", "", text)`
4. Column-aware allowed charset filtering
5. Final whitespace normalization: `re.sub(r"\s{2,}", " ", text).strip()`

---

## 11. Row & Column Grouping Rules

- Table detections define row anchors
- Rows grouped via vertical overlap
- Columns assigned by x-center alignment
- Each OCR output remains independent
- **Handwritten boxes must not be merged**

---

## 12. Metrics Logic

### `layout.handwritten`

- Equals **number of YOLO Handwritten detections**
- Counted at **YOLO stage only**
- **Not derived from rows or OCR output**
- **YOLO is the source of truth.**

---

## 13. API Output (POST /api/analyze)

### Output Schema

```json
{
  "rows": [
    {
      "last_name": "...",
      "first_name": "...",
      "attendee_type": "...",
      "credential": "...",
      "state_of_license": "...",
      "license_number": "...",
      "signature": true,
      "checkbox": false
    }
  ]
}
```

### "NO NULL IF DATA EXISTS" Rule

For each field in a row:

1. Use `PrintedText` if present
2. Else use `HandwrittenText` if present
3. Emit `null` only if both are empty

**No cross-row borrowing.**  
**No inference.**

---

## 14. Running the Application

```bash
cd Back_end
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

**Or using uvicorn directly:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Swagger UI:**

```
http://localhost:8000/docs
```

---

## 15. Installation Verification Checklist

Installation is correct if:

- ✅ YOLO detects all layout regions
- ✅ PaddleOCR runs with detection disabled
- ✅ PARSeq runs only on handwritten crops
- ✅ OCR never sees full image
- ✅ Symbols are removed from handwritten output
- ✅ No nulls appear where OCR data exists
- ✅ `layout.handwritten` equals YOLO handwritten detections
- ✅ API output matches schema exactly

---

## 📌 PARSeq Installation & Setup Notes (Detailed)

### 16. Purpose of PARSeq in This Project

- Used only for handwritten recognition
- Does NOT detect regions
- One YOLO handwritten box → one PARSeq output
- No fallback OCR engine

### 17. Official PARSeq References (READ-ONLY)

**GitHub:**

https://github.com/baudm/parseq

**Paper:**

https://arxiv.org/abs/2207.06966

⚠️ **These links are for reference only.**  
**Do NOT change the implementation based on them.**

### 18. PARSeq Prerequisites

**Ensure PyTorch is working:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**If this fails → fix PyTorch before proceeding.**

### 19. PARSeq Dependency Installation

All required dependencies are already installed via:

```bash
pip install -r requirements.txt
```

**Additional PARSeq-specific dependencies:**

```bash
pip install pytorch-lightning lmdb nltk
```

**Note:** Check `pytorch-lightning` version compatibility:
- PARSeq checkpoint may require `pytorch-lightning==2.2.0.post0`
- If loading fails, downgrade: `pip install pytorch-lightning==2.2.0.post0`

❌ **Do NOT install HuggingFace transformers for PARSeq**  
❌ **Do NOT install extra OCR libraries**

### 20. Downloading PARSeq Weights (One-Time)

**From:**

https://github.com/baudm/parseq/releases

**Download:**

- Pretrained checkpoint: `parseq-bb5792a6.pt`
- (No separate vocabulary file needed - vocab is embedded)

**Direct download:**

```bash
cd Back_end/ocr/handwritten/parseq/weights
wget https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt
```

**Or on Windows (PowerShell):**

```powershell
cd Back_end\ocr\handwritten\parseq\weights
Invoke-WebRequest -Uri "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt" -OutFile "parseq-bb5792a6.pt"
```

### 21. Placing PARSeq Weights (CRITICAL)

Files must exist at:

```
Back_end/ocr/handwritten/parseq/weights/parseq-bb5792a6.pt
```

**Rules:**

- No renaming
- No nested folders
- No symlinks
- Exact path: `./ocr/handwritten/parseq/weights/parseq-bb5792a6.pt`

**Verify path matches config:**

```python
# In core/config.py
PARSEQ_CHECKPOINT_PATH: str = "./ocr/handwritten/parseq/weights/parseq-bb5792a6.pt"
```

### 22. PARSeq Runtime Rules

At runtime, PARSeq must:

- Load weights locally
- Run in inference mode only
- Keep encoder frozen
- Skip handwritten OCR if weights are missing (log error, no fallback)

### 23. What PARSeq Must NOT Do

- ❌ Detect boxes
- ❌ Merge boxes
- ❌ Guess text
- ❌ Correct spelling
- ❌ Use dictionaries
- ❌ Switch OCR engines

### 24. Verifying PARSeq Installation

**Run the app and check logs:**

```bash
python main.py
```

**Expected log output:**

```
✅ PARSeq recognizer initialized successfully
PARSeq weights loaded from: ./ocr/handwritten/parseq/weights/parseq-bb5792a6.pt
```

**If errors occur:**

- Check file path matches config
- Verify PyTorch Lightning version
- Check CUDA availability (if using GPU)
- Verify `lmdb` and `nltk` are installed

### 25. Common PARSeq Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| PARSeq not loading | Wrong path | Verify `ocr/handwritten/parseq/weights/` |
| CUDA error | GPU mismatch | Run CPU or fix CUDA |
| Empty output | Bad YOLO crop | Check handwritten boxes |
| Symbols remain | Sanitizer skipped | Verify sanitizer execution |
| `KeyError: 'pytorch-lightning_version'` | Version mismatch | Downgrade: `pip install pytorch-lightning==2.2.0.post0` |
| `ModuleNotFoundError: No module named 'lmdb'` | Missing dependency | `pip install lmdb` |
| `ModuleNotFoundError: No module named 'nltk'` | Missing dependency | `pip install nltk` |

### 26. Phase-2 Note (Future)

PARSeq training / fine-tuning is **NOT part of installation**.

Phase-2 includes:

- Dataset creation
- Decoder fine-tuning
- Training scripts

**Do NOT attempt training during setup.**

---

## Final Note

This installation guide is intentionally strict and complete.  
**If system behavior differs from what is described here, it indicates misinstallation or unintended changes.**

---

## Quick Reference Commands

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PARSeq-specific dependencies
pip install pytorch-lightning lmdb nltk

# 4. Download PARSeq weights (if not already present)
cd ocr/handwritten/parseq/weights
wget https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt

# 5. Verify installation
python -c "from ocr.handwritten.parseq_recognizer import PARSeqRecognizer; r = PARSeqRecognizer(); print('✅ PARSeq OK' if r.is_available() else '❌ PARSeq failed')"

# 6. Run application
python main.py
```

---

## Troubleshooting

### PaddleOCR Issues

PaddleOCR auto-downloads models on first use. If it fails:

1. Check internet connection
2. Verify disk space (models are large)
3. Check logs for specific errors

**Important:** PaddleOCR must run with `det=False` (detection disabled).

### PARSeq Issues

**Check initialization:**

```python
from ocr.handwritten.parseq_recognizer import PARSeqRecognizer
recognizer = PARSeqRecognizer()
print(f"Available: {recognizer.is_available()}")
```

**Check weights path:**

```python
from core.config import settings
import os
print(f"Expected path: {settings.PARSEQ_CHECKPOINT_PATH}")
print(f"Exists: {os.path.exists(settings.PARSEQ_CHECKPOINT_PATH)}")
```

### YOLO Issues

**Verify YOLO model:**

```python
from layout.yolov8_layout_detector import YOLOv8LayoutDetector
detector = YOLOv8LayoutDetector()
print(f"Available: {detector.is_available()}")
```

---

## File Structure Summary

```
Back_end/
├── models/
│   └── yolo_layout/
│       └── best.pt (or yolov8s_layout.pt)
├── ocr/
│   └── handwritten/
│       └── parseq/
│           └── weights/
│               └── parseq-bb5792a6.pt
├── storage/                 # Processed files
├── venv/                    # Virtual environment
└── logs/                    # Application logs
```

---

## Notes

- All models are stored in `models/` and `ocr/handwritten/parseq/weights/` directories
- Processed images are in `storage/`
- Logs are in `logs/app.log`
- Server runs on `http://0.0.0.0:8000` by default
- YOLO is the single source of truth for detection counts
- PARSeq replaces TrOCR for handwritten recognition
- No OCR model performs detection (only YOLO detects regions)
