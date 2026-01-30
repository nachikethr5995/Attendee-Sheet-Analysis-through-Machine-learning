# **AIME Backend — Implementation Plan**

> **⚠️ LEGACY DOCUMENT**: This is a historical planning document from the initial project design phase. The actual implementation has evolved significantly:
> - **Current workflow**: Uses Unified Pipeline (not separate SERVICE 2/4)
> - **OCR**: PARSeq replaces TrOCR for handwritten recognition
> - **Detection**: YOLOv8s is the only detector (not Detectron2/PubLayNet)
> - **API**: POST /api/analyze returns structured rows (not separate service outputs)
> 
> **For current architecture, see**: `README.md`, `ARCHITECTURE_RULES.md`, `INSTALLATION_GUIDE.md`

## **📋 Overview**

This document provides a **strict, sequential implementation plan** for building the AIME Backend OCR & Computer Vision Pipeline. The plan follows a **mandatory linear development approach** where each service must be fully completed, tested, and approved before proceeding to the next.

**Note**: This plan reflects the original design. The actual implementation uses a unified pipeline architecture.

**Two-Stage Preprocessing Strategy:**

1. **SERVICE 0 — Basic Preprocessing (Always Runs)**
   - Produces `pre_0_id`
   - Lightweight, fast corrections
   - Always applied to every uploaded image

2. **SERVICE 0.1 — Advanced Preprocessing (Conditional)**
   - Produces `pre_01_id`
   - Heavy CV enhancements
   - Only triggered if Services 1-4 fail or produce low confidence

**Critical ID Usage Rule:**

* **Canonical ID Logic:**
  - If SERVICE 0.1 is NOT triggered → canonical_id = `pre_0_id`
  - If SERVICE 0.1 IS triggered → canonical_id = `pre_01_id`
* **Services 1-4 always use the canonical ID** (never raw `file_id`)
* All intermediate files (layout, tables, OCR) are stored using the canonical ID
* **Endpoint ID Resolution:**
  1. Prefer `pre_01_id` if exists
  2. Else prefer `pre_0_id`
  3. Else use `file_id` (fallback only)

---

## **🔐 Development Rules**

### **Critical Rule: Sequential Development**

**NO SERVICE CAN BE STARTED UNTIL THE PREVIOUS SERVICE IS FULLY COMPLETE AND APPROVED.**

1. **SERVICE 0** → Build → Test → Validate → **APPROVE** → Proceed
2. **SERVICE 1** → Build → Test → Validate → **APPROVE** → Proceed
3. **SERVICE 2** → Build → Test → Validate → **APPROVE** → Proceed
4. **SERVICE 3** → Build → Test → Validate → **APPROVE** → Proceed
5. **SERVICE 4** → Build → Test → Validate → **APPROVE** → Complete

**Developer must explicitly approve each service before moving forward.**

---

## **📁 Phase 0: Project Setup & Infrastructure**

### **0.1 Directory Structure Setup**

**Task:** Create complete folder structure

```
Backend/
├── api/
│   ├── __init__.py
│   ├── upload.py
│   ├── analyze.py
│   └── health.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   └── utils.py
├── ingestion/
│   ├── __init__.py
│   └── file_handler.py
├── preprocessing/
│   ├── __init__.py
│   ├── basic.py         # SERVICE 0
│   ├── advanced.py      # SERVICE 0.1
│   ├── converter.py
│   ├── enhancer.py
│   ├── quality.py
│   ├── deskew.py
│   ├── perspective.py
│   └── text_regions.py
├── layout/
│   ├── __init__.py
│   ├── line_detection.py
│   ├── layout_classifier.py
│   ├── table_detector.py
│   ├── yolo_detector.py
│   └── east_text_detector.py
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py
│   └── contour_finder.py
├── ocr/
│   ├── __init__.py
│   ├── paddle_engine.py
│   ├── trocr_engine.py
│   └── router.py
├── postprocessing/
│   ├── __init__.py
│   ├── table_reconstruct.py
│   ├── field_mapper.py
│   └── scoring.py
├── models/
│   ├── signature_model/
│   ├── yolo_weights/
│   └── classifiers/
├── storage/
│   ├── raw/
│   ├── processed/
│   ├── intermediate/
│   └── output/
├── tests/
│   ├── __init__.py
│   ├── test_service_0.py
│   ├── test_service_1.py
│   ├── test_service_2.py
│   ├── test_service_3.py
│   └── test_service_4.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

**Deliverable:** Complete directory structure with `__init__.py` files

---

### **0.2 Storage Structure Setup**

**Task:** Create storage directories with proper permissions

```
storage/
├── raw/                    # Original uploaded files
├── processed_basic/        # SERVICE 0 output (pre_0_id)
├── processed_advanced/     # SERVICE 0.1 output (pre_01_id)
├── intermediate/          # Service outputs (JSON) - uses canonical_id
└── output/                # Final analysis results
```

**Deliverable:** Storage directories created and documented

---

### **0.3 Python Environment & Dependencies**

**Task:** Set up virtual environment and install dependencies

**Dependencies to install:**

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
Pillow==10.1.0
pillow-heif==0.13.0
pillow-avif-plugin==1.4.3
pdf2image==1.16.3
numpy==1.24.3
scikit-image==0.22.0
ultralytics==8.0.196
torch==2.1.0
torchvision==0.16.0
efficientnet-pytorch==0.7.1
scikit-learn==1.3.2
paddleocr==2.7.0.3
paddlepaddle==2.5.2
loguru==0.7.2
python-dotenv==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

**Deliverable:** `requirements.txt` and virtual environment configured

---

### **0.4 Core Configuration**

**Task:** Implement core configuration and utilities

**Files to create:**

1. **`core/config.py`**
   - Environment variables
   - Storage paths
   - Model paths
   - API settings
   - GPU/CPU configuration

2. **`core/logging.py`**
   - Loguru configuration
   - Log levels
   - File rotation
   - Structured logging

3. **`core/utils.py`**
   - ID generation (file_id, pre_0_id, pre_01_id, analysis_id)
   - Canonical ID resolution logic
   - File path utilities
   - Image utilities
   - Validation helpers

**Deliverable:** Core modules functional and tested

---

### **0.5 FastAPI Application Skeleton**

**Task:** Create main FastAPI app with health check

**Files to create:**

1. **`main.py`**
   - FastAPI app initialization
   - CORS configuration
   - Route registration
   - Error handlers

2. **`api/health.py`**
   - Health check endpoint
   - System status
   - Dependency checks

**Deliverable:** FastAPI app runs and responds to health checks

---

### **✅ Phase 0 Approval Checklist**

- [ ] Directory structure complete
- [ ] Storage directories created
- [ ] Virtual environment active
- [ ] All dependencies installed
- [ ] Core modules implemented
- [ ] FastAPI app runs successfully
- [ ] Health check endpoint works
- [ ] Logging configured and tested

**Status:** ⏳ **PENDING APPROVAL**

---

## **🔧 SERVICE 0: Basic Preprocessing (Always Runs)**

### **Goal**

Transform raw uploaded files (JPG, PNG, HEIC, AVIF, PDF) into normalized, standardized PNG images (2048px long edge) ready for layout detection and OCR.

**SERVICE 0 ALWAYS RUNS** - it provides fast, minimal, non-destructive enhancement to every uploaded image.

### **Key Characteristics**

- **Fast** - Lightweight operations for performance
- **Non-destructive** - Minimal changes to preserve original quality
- **Standardized** - Consistent output format for downstream services
- **Always applied** - No conditional logic, runs on every upload

### **Processing Steps**

1. Format conversion (HEIC/AVIF/PDF → PNG)
2. Light denoising
3. Image resize to 2048px long edge
4. Soft deskew (minor angle corrections only)
5. Light contrast normalization
6. Minimal artifact removal

### **Output**

- Generates `pre_0_id` (format: `pre_0_{timestamp}_{random}`)
- Stores processed image in `/storage/processed_basic/{pre_0_id}.png`
- This becomes the canonical input for Services 1-4 (unless SERVICE 0.1 is triggered)

---

### **S0.1 File Upload Endpoint**

**Task:** Implement `/api/upload` endpoint

**File:** `api/upload.py`

**Requirements:**
- Accept multipart file uploads
- Support: JPG, PNG, HEIC, HEIF, AVIF, PDF
- Generate `file_id` (format: `file_{timestamp}_{random}`)
- Store in `/storage/raw/{file_id}.{ext}`
- Return JSON: `{"file_id": "..."}`

**Validation:**
- File size limits (e.g., 50MB max)
- File type validation
- Error handling for invalid files

**Deliverable:** Upload endpoint functional with file_id generation

---

### **S0.2 Basic Preprocessing Pipeline**

**Task:** Implement SERVICE 0 basic preprocessing

**File:** `preprocessing/basic.py` (new file)

**Requirements:**
- Load image from `file_id`
- Apply format conversion (HEIC/AVIF/PDF → PNG)
- Apply light denoising (minimal strength)
- Resize to 2048px long edge (maintain aspect ratio)
- Apply soft deskew (only for angles > 0.5°)
- Apply light contrast normalization
- Remove minimal artifacts
- Generate `pre_0_id` (format: `pre_0_{timestamp}_{random}`)
- Store in `/storage/processed_basic/{pre_0_id}.png`
- Return response with `pre_0_id`

**Deliverable:** Basic preprocessing pipeline functional

---

### **S0.5 Input Loader & Format Detection**

**Task:** Implement file format detection and loading

**File:** `ingestion/file_handler.py`

**Requirements:**
- Detect file format (MIME type + extension)
- Load images using Pillow
- Load HEIC using pillow-heif
- Load AVIF using pillow-avif-plugin
- Load PDF using pdf2image (convert first page)
- Convert all to RGB format
- Handle errors gracefully

**Deliverable:** All supported formats load correctly

---

### **S0.6 Basic Preprocessing Components**

**Task:** Implement lightweight preprocessing components

**Files:** `preprocessing/basic.py`, `preprocessing/converter.py`

**Requirements:**
- **Format Conversion:** HEIC/AVIF/PDF → PNG
- **Light Denoising:** Minimal strength (preserve detail)
- **Resize:** 2048px long edge (maintain aspect ratio)
- **Soft Deskew:** Only for angles > 0.5° (minimal correction)
- **Light Contrast Normalization:** Subtle adjustments
- **Minimal Artifact Removal:** Basic cleanup only

**Deliverable:** All basic preprocessing components functional

---

### **S0.3 Basic Preprocessing API Endpoint**

**Task:** Implement `/api/analyze/preprocess` endpoint (SERVICE 0)

**File:** `api/analyze.py`

**Requirements:**
- Accept `file_id` in request body
- Load file from `/storage/raw/`
- Run SERVICE 0 basic preprocessing pipeline
- Generate `pre_0_id`
- Save processed image in `/storage/processed_basic/{pre_0_id}.png`
- Return response:
  ```json
  {
    "pre_0_id": "pre_0_20250111_99e12d",
    "clean_image_path": "/storage/processed_basic/pre_0_20250111_99e12d.png",
    "metadata": {
      "width": 2048,
      "height": 1536,
      "quality_score": 0.92,
      "processing_type": "basic"
    }
  }
  ```

**Deliverable:** Basic preprocessing endpoint functional

---

### **S0.4 Main Analysis Endpoint with Fallback**

**Task:** Implement main analysis endpoint with automatic SERVICE 0.1 fallback

**File:** `api/analyze.py` (main analysis endpoint)

**Requirements:**
- Accept `file_id` in request body
- **Step 1:** Run SERVICE 0 (basic preprocessing) → get `pre_0_id`
- **Step 2:** Run Services 1-4 using `pre_0_id` as canonical ID
- **Step 3:** Monitor for failure conditions:
  - Layout detection returns empty/insufficient results
  - OCR confidence below threshold (e.g., < 0.5)
  - Table isolation fails
  - Any exception during processing
- **Step 4:** If failure detected:
  - Automatically trigger SERVICE 0.1 (advanced preprocessing) → get `pre_01_id`
  - Rerun Services 1-4 using `pre_01_id` as canonical ID
- **Step 5:** Return final result

**Canonical ID Logic:**
- If SERVICE 0.1 is NOT triggered → canonical_id = `pre_0_id`
- If SERVICE 0.1 IS triggered → canonical_id = `pre_01_id`
- Services 1-4 always use canonical_id (never raw `file_id`)

**Fallback Conditions:**
```python
# Example fallback triggers
if layout_results["tables"] == [] and layout_results["text_blocks"] == []:
    # Trigger SERVICE 0.1
if ocr_confidence < 0.5:
    # Trigger SERVICE 0.1
if table_isolation_failed:
    # Trigger SERVICE 0.1
```

**Deliverable:** Main analysis endpoint with automatic fallback to advanced preprocessing

---

### **S0.7 Unit Tests**

**Task:** Write comprehensive unit tests for SERVICE 0

**File:** `tests/test_service_0.py`

**Test Cases:**
- File upload with various formats
- Format conversion (HEIC → PNG, AVIF → PNG, PDF → PNG)
- Light denoising functionality
- Image resize to 2048px
- Soft deskew (minor angles only)
- Light contrast normalization
- Minimal artifact removal
- `pre_0_id` generation
- End-to-end basic preprocessing pipeline

**Deliverable:** Test suite with >80% coverage

---

### **S0.8 Integration Tests**

**Task:** Test full upload → basic preprocessing flow

**Test Scenarios:**
1. Upload JPG → Run SERVICE 0 → Verify `pre_0_id` and PNG output
2. Upload HEIC → Run SERVICE 0 → Verify conversion and output
3. Upload PDF → Run SERVICE 0 → Verify conversion and output
4. Upload rotated image → Verify soft deskew
5. Upload noisy image → Verify light denoising
6. Verify `pre_0_id` is generated correctly
7. Verify processed image stored in `processed_basic/` directory

**Deliverable:** Integration tests passing

---

### **✅ SERVICE 0 Approval Checklist**

- [ ] File upload endpoint functional
- [ ] All input formats supported (JPG, PNG, HEIC, AVIF, PDF)
- [ ] Basic preprocessing pipeline implemented (`preprocessing/basic.py`)
- [ ] Format conversion working (HEIC/AVIF/PDF → PNG)
- [ ] Light denoising functional
- [ ] Image resize to 2048px working
- [ ] Soft deskew working correctly
- [ ] Light contrast normalization implemented
- [ ] Minimal artifact removal functional
- [ ] `pre_0_id` generation working
- [ ] Basic preprocessing endpoint returns correct response
- [ ] Storage structure created (`processed_basic/` directory)
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Code reviewed and documented
- [ ] Performance acceptable (<3s for typical image)

**Status:** ⏳ **PENDING APPROVAL**

**⚠️ DO NOT PROCEED TO SERVICE 1 UNTIL SERVICE 0 IS APPROVED**

**Note:** SERVICE 0 (basic preprocessing) always runs and must be fully implemented and stable before proceeding. SERVICE 0.1 (advanced preprocessing) will be built later, after Services 1-4 are complete.

---

## **🔍 SERVICE 1: Hybrid Layout Detection**

### **Goal**

Perform **document layout detection** using a hybrid ensemble of specialized, pretrained models. Output unified JSON containing tables, text blocks, signature regions, checkbox regions, spatial metadata, and confidence scores.

**Important:** SERVICE 1 runs on **`pre_0_id`** (from SERVICE 0 basic preprocessing). If Services 1-4 fail, SERVICE 0.1 (advanced preprocessing) is triggered, and Services 1-4 rerun using `pre_01_id`.

### **High-Level Architecture**

```
PubLayNet (Detectron2) → Tables, Text Blocks
GPDS Signature Detector → Signatures
Checkbox Detector (Tiny YOLOv8) → Checkboxes
PaddleOCR Text Detector → Refined Text Regions
                    ↓
            Fusion Engine
                    ↓
        Unified Layout JSON
```

---

### **S1.1 PubLayNet Layout Detector (Detectron2)**

**Task:** Integrate PubLayNet pretrained model for table and text block detection

**File:** `layout/publaynet_detector.py`

**Requirements:**
- Install Detectron2
- Load pretrained PubLayNet weights (R50-FPN or Swin Transformer)
- Run inference on canonical image
- Extract bounding boxes for:
  - `table`
  - `text` (text blocks)
  - `title` (optional)
- Normalize coordinates to 0-1 range
- Return structured detection results

**Output Format:**
```json
[
  { "type": "table", "bbox": [x1, y1, x2, y2], "confidence": 0.95 },
  { "type": "text_block", "bbox": [x1, y1, x2, y2], "confidence": 0.92 }
]
```

**Deliverable:** PubLayNet detector functional, no training required

---

### **S1.2 GPDS Signature Detector**

**Task:** Integrate GPDS pretrained signature detection model

**File:** `layout/signature_detector.py`

**Requirements:**
- Download YOLOv5 or Faster R-CNN GPDS signature model
- Load into PyTorch
- Run inference on canonical image
- Normalize coordinates
- Convert to Service 1 schema

**Output Format:**
```json
[
  { "type": "signature", "bbox": [x1, y1, x2, y2], "confidence": 0.88 }
]
```

**Deliverable:** GPDS signature detector functional, no training required

---

### **S1.3 Checkbox Detector (Tiny Custom Model)**

**Task:** Train and integrate tiny YOLOv8 model for checkbox detection

**File:** `layout/checkbox_detector.py`

**Requirements:**

**Step 1: Data Preparation**
- Annotate 20-50 checkbox images (using LabelImg or Roboflow)
- Create YOLOv8 dataset format
- Split into train/val (80/20)

**Step 2: Model Training**
- Train YOLOv8n (nano) model:
  ```bash
  yolo detect train data=checkbox.yaml model=yolov8n.pt epochs=25 imgsz=640
  ```
- Target: 10-20 minutes training time
- Save best model weights

**Step 3: Inference Integration**
- Load trained model
- Run inference on canonical image
- Filter noise boxes via confidence threshold > 0.4
- Normalize coordinates

**Output Format:**
```json
[
  { "type": "checkbox", "bbox": [x1, y1, x2, y2], "confidence": 0.75 }
]
```

**Deliverable:** Custom checkbox detector trained and functional

---

### **S1.4 PaddleOCR Text Detector (DBNet/SAST)**

**Task:** Integrate PaddleOCR text detection for refined text regions

**File:** `layout/paddleocr_detector.py`

**Requirements:**
- Install PaddleOCR
- Load text detection model (`ch_PP-OCRv4_det`)
- Run DBNet to get text polygons
- Merge small polygons into larger text_block regions
- Align with PubLayNet text blocks
- Convert polygons to bounding boxes

**Output Format:**
```json
[
  { "type": "text_block_refined", "bbox": [x1, y1, x2, y2], "confidence": 0.90 }
]
```

**Deliverable:** PaddleOCR text detector functional, no training required

---

### **S1.5 Fusion Engine**

**Task:** Implement fusion engine to merge all detector outputs

**File:** `layout/fusion_engine.py`

**Requirements:**

**Core Functions:**

1. **Coordinate Unification**
   - Convert mixed outputs (boxes + polygons) to standard format: `{x_min, y_min, x_max, y_max}`
   - Normalize all coordinates to 0-1 range

2. **Overlap Handling (IoU Rules)**
   - If text_block overlaps with table → assign to table
   - If signature overlaps with any text → signature wins
   - If checkbox exists inside table → keep but mark `"in_table": true`
   - Discard duplicate regions with IoU > 0.85

3. **Confidence Normalization**
   - Normalize scores from different models to unified scale [0, 1]
   - Formula: `confidence_norm = (score - min_score) / (max_score - min_score)`

4. **Region Ranking**
   - Sort by priority: signature > table > checkbox > text_block
   - Then by confidence
   - Then by area if needed

5. **Final JSON Assembly**
   - Output unified structure:
   ```json
   {
     "canonical_id": "xxx",
     "tables": [...],
     "text_blocks": [...],
     "signatures": [...],
     "checkboxes": [...],
     "metadata": {
       "image_dims": [w, h],
       "models_used": ["PubLayNet", "GPDS", "checkbox_v1", "PaddleOCR"],
       "confidence_summary": {...}
     }
   }
   ```

**Deliverable:** Fusion engine functional with all merge rules

---

### **S1.6 Layout Service Integration**

**Task:** Integrate all detectors into unified layout service

**File:** `layout/layout_service.py`

**Requirements:**
- Load image using canonical ID (pre_0_id or pre_01_id)
- Run all four detectors in parallel (if possible) or sequentially
- Pass results to fusion engine
- Handle errors gracefully (fallback if one detector fails)
- Detect failure conditions for SERVICE 0.1 trigger
- Store layout JSON using canonical ID: `/storage/intermediate/{canonical_id}_layout.json`

**Error Handling:**
- If PubLayNet fails → Trigger SERVICE 0.1, then retry
- If GPDS fails → Return empty signature array, proceed normally
- If checkbox model fails → Return empty checkbox array
- If PaddleOCR fails → Fallback to PubLayNet text blocks only

**Deliverable:** Unified layout service functional

---

### **S1.7 Layout Output Storage**

**Task:** Save layout detection results

**File:** `layout/layout_service.py` (extend)

**Requirements:**
- Store layout JSON using canonical ID:
  - `/storage/intermediate/{canonical_id}_layout.json`
  - Where canonical_id = pre_01_id (if SERVICE 0.1 triggered) or pre_0_id (otherwise)
- Structure:
  ```json
  {
    "canonical_id": "pre_0_20251212_...",
    "tables": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "source": "PubLayNet"
      }
    ],
    "text_blocks": [...],
    "signatures": [...],
    "checkboxes": [...],
    "dimensions": {
      "width": 2048,
      "height": 1536
    },
    "metadata": {
      "models_used": ["PubLayNet", "GPDS", "checkbox_v1", "PaddleOCR"],
      "detection_times": {...},
      "confidence_summary": {...}
    }
  }
  ```

**Deliverable:** Layout JSON stored correctly

---

### **S1.8 Layout Detection API Endpoint**

**Task:** Implement `/api/analyze/layout` endpoint

**File:** `api/analyze.py` (extend)

**Requirements:**
- Accept `file_id`, `pre_0_id`, or `pre_01_id` in request body
- Determine canonical ID using resolution logic:
  1. Prefer `pre_01_id` if exists
  2. Else prefer `pre_0_id`
  3. Else use `file_id` (fallback only)
- Load image using canonical ID
- Run hybrid layout detection (all four detectors + fusion)
- **Detect failure conditions** (empty results, low confidence) for SERVICE 0.1 trigger
- Store layout JSON using canonical ID: `/storage/intermediate/{canonical_id}_layout.json`
- Return layout results

**Deliverable:** Layout detection endpoint functional

---

### **S1.9 Performance Targets**

| Model             | Est. Inference Time (GPU) |
| ----------------- | ------------------------- |
| PubLayNet         | 60-90 ms                  |
| GPDS Signature    | 10-20 ms                  |
| Checkbox Detector | 3-7 ms                    |
| PaddleOCR Det     | 40-80 ms                  |
| Fusion Engine     | 3-5 ms                    |

**Total per page:** 150-200 ms

**Deliverable:** Performance targets met or documented deviations

---

### **S1.10 Unit Tests**

**Task:** Write unit tests for layout detection

**File:** `tests/test_service_1.py`

**Test Cases:**
- PubLayNet model loading
- PubLayNet detection on sample images
- GPDS signature model loading
- GPDS signature detection
- Checkbox model loading
- Checkbox detection
- PaddleOCR text detection
- Fusion engine coordinate unification
- Fusion engine overlap handling
- Fusion engine confidence normalization
- JSON output format

**Deliverable:** Test suite with >80% coverage

---

### **S1.11 Integration Tests**

**Task:** Test preprocessing → layout detection flow

**Test Scenarios:**
1. Run layout detection on `pre_0_id` (from SERVICE 0) → Verify JSON output
2. Test failure detection → Trigger SERVICE 0.1 → Rerun layout on `pre_01_id` → Verify JSON output
3. Test with images containing tables
4. Test with images containing signatures
5. Test with images containing checkboxes
6. Test with images containing all element types
7. Verify coordinate accuracy
8. Verify detection confidence scores
9. Verify canonical ID resolution logic (pre_01_id > pre_0_id > file_id)
10. Test error handling (one detector fails)

**Deliverable:** Integration tests passing

---

### **✅ SERVICE 1 Approval Checklist**

- [ ] PubLayNet model loaded and functional
- [ ] PubLayNet detection working (tables, text blocks)
- [ ] GPDS signature model loaded and functional
- [ ] GPDS signature detection working
- [ ] Checkbox detector trained and functional
- [ ] Checkbox detection working
- [ ] PaddleOCR text detector loaded and functional
- [ ] PaddleOCR text detection working
- [ ] Fusion engine implemented correctly
- [ ] **Layout detection works on `pre_0_id` (from SERVICE 0)**
- [ ] **Failure detection logic implemented**
- [ ] **SERVICE 0.1 trigger works correctly (when Services 1-4 fail)**
- [ ] **Canonical ID resolution logic implemented (pre_01_id > pre_0_id > file_id)**
- [ ] Layout JSON stored in correct format
- [ ] Layout detection endpoint functional
- [ ] Performance targets met (<200ms per page on GPU)
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Detection accuracy validated on sample images
- [ ] Code reviewed and documented

**Status:** ⏳ **PENDING APPROVAL**

**⚠️ DO NOT PROCEED TO SERVICE 2 UNTIL SERVICE 1 IS APPROVED**

---

## **📊 SERVICE 2: Table Isolation**

### **Goal**

Isolate and extract table structures from detected table regions using line detection, gap clustering, and row/column segmentation.

---

### **S2.1 Line Detection**

**Task:** Implement vertical and horizontal line detection

**File:** `layout/line_detection.py`

**Requirements:**
- Hough line transform for table lines
- Vertical line detection
- Horizontal line detection
- Line filtering and clustering
- Return line coordinates

**Deliverable:** Line detection functional

---

### **S2.2 Gap Clustering**

**Task:** Implement gap detection for cell boundaries

**File:** `layout/table_detector.py`

**Requirements:**
- Detect gaps between text regions
- Cluster gaps into rows/columns
- Identify cell boundaries
- Handle irregular tables

**Deliverable:** Gap clustering functional

---

### **S2.3 Row/Column Segmentation**

**Task:** Segment tables into rows and columns

**File:** `layout/table_detector.py` (extend)

**Requirements:**
- Identify table rows
- Identify table columns
- Extract cell regions
- Handle merged cells
- Return structured table data

**Deliverable:** Table segmentation functional

---

### **S2.4 Table Region Cropping**

**Task:** Crop table regions from full image

**File:** `layout/table_detector.py` (extend)

**Requirements:**
- Use layout detection results (table bboxes)
- Crop table regions from processed image
- Save cropped table images (optional)
- Return table region coordinates

**Deliverable:** Table cropping functional

---

### **S2.5 Table Structure Output**

**Task:** Generate table structure JSON

**File:** `layout/table_detector.py` (extend)

**Requirements:**
- Store table JSON using canonical ID:
  - `/storage/intermediate/{canonical_id}_tables.json`
  - Where canonical_id = pre_01_id (if SERVICE 0.1 triggered) or pre_0_id (otherwise)
- Structure:
  ```json
  {
    "tables": [
      {
        "table_id": "table_0",
        "bbox": [x1, y1, x2, y2],
        "rows": [
          {
            "row_id": "row_0",
            "cells": [
              {
                "cell_id": "cell_0_0",
                "bbox": [x1, y1, x2, y2],
                "row": 0,
                "col": 0
              }
            ]
          }
        ]
      }
    ]
  }
  ```

**Deliverable:** Table structure JSON stored correctly

---

### **S2.6 Table Isolation API Endpoint**

**Task:** Implement `/api/analyze/tables` endpoint

**File:** `api/analyze.py` (extend)

**Requirements:**
- Accept `file_id`, `pre_0_id`, or `pre_01_id` in request body
- Determine canonical ID using resolution logic (same as SERVICE 1)
- Load image and layout JSON using canonical ID
- Run table isolation
- Store table JSON using canonical ID: `/storage/intermediate/{canonical_id}_tables.json`
- Return table structure

**Deliverable:** Table isolation endpoint functional

---

### **S2.7 Unit Tests**

**Task:** Write unit tests for table isolation

**File:** `tests/test_service_2.py`

**Test Cases:**
- Line detection accuracy
- Gap clustering logic
- Row/column segmentation
- Table cropping
- JSON output format
- Edge cases (irregular tables, merged cells)

**Deliverable:** Test suite with >80% coverage

---

### **S2.8 Integration Tests**

**Task:** Test layout detection → table isolation flow

**Test Scenarios:**
1. Layout detection → Table isolation → Verify table structure
2. Test with regular tables
3. Test with irregular tables
4. Test with merged cells
5. Verify cell coordinate accuracy

**Deliverable:** Integration tests passing

---

### **✅ SERVICE 2 Approval Checklist**

- [ ] Line detection working (vertical + horizontal)
- [ ] Gap clustering functional
- [ ] Row/column segmentation accurate
- [ ] Table region cropping working
- [ ] Table structure JSON stored correctly
- [ ] Table isolation endpoint functional
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Table extraction validated on sample images
- [ ] Code reviewed and documented
- [ ] Performance acceptable (<8s for typical table)

**Status:** ⏳ **PENDING APPROVAL**

**⚠️ DO NOT PROCEED TO SERVICE 3 UNTIL SERVICE 2 IS APPROVED**

---

## **📝 SERVICE 3: OCR Routing**

### **Goal**

Route detected regions to appropriate OCR engines (PaddleOCR for printed text, TrOCR for handwriting, SignatureNet for signatures, checkbox detector for checkboxes).

---

### **S3.1 PaddleOCR Engine Setup**

**Task:** Configure PaddleOCR engine

**File:** `ocr/paddle_engine.py`

**Requirements:**
- Initialize PaddleOCR
- Configure language (English)
- Support GPU/CPU inference
- Configure confidence thresholds
- Handle text region extraction

**Deliverable:** PaddleOCR engine ready

---

### **S3.2 PaddleOCR Text Extraction**

**Task:** Implement PaddleOCR text extraction

**File:** `ocr/paddle_engine.py`

**Requirements:**
- Extract text from image regions
- Return text + confidence scores
- Handle multiple text lines
- Preserve text order

**Deliverable:** PaddleOCR extraction functional

---

### **S3.3 TrOCR Engine Setup**

**Task:** Configure TrOCR engine for handwriting

**File:** `ocr/trocr_engine.py`

**Requirements:**
- Load TrOCR model (handwriting variant)
- Configure tokenizer
- Support GPU/CPU inference
- Handle image preprocessing for TrOCR

**Deliverable:** TrOCR engine ready

---

### **S3.4 TrOCR Handwriting Extraction**

**Task:** Implement TrOCR handwriting recognition

**File:** `ocr/trocr_engine.py`

**Requirements:**
- Extract handwritten text from regions
- Return text + confidence scores
- Handle cursive and printed handwriting
- Preserve text order

**Deliverable:** TrOCR extraction functional

---

### **S3.5 Signature Detection**

**Task:** Implement signature presence detection

**File:** `detection/yolo_detector.py` (extend) or new file

**Requirements:**
- Use YOLO signature detections
- Validate signature regions
- Return signature presence boolean
- Optional: SignatureNet model for verification

**Deliverable:** Signature detection functional

---

### **S3.6 Checkbox Detection**

**Task:** Implement checkbox state detection

**File:** `detection/contour_finder.py` or new file

**Requirements:**
- Detect checkbox regions (from layout detection)
- Classify checkbox state (checked/unchecked)
- Use contour analysis or ML model
- Return checkbox states

**Deliverable:** Checkbox detection functional

---

### **S3.7 OCR Router**

**Task:** Implement routing logic

**File:** `ocr/router.py`

**Requirements:**
- Route text blocks to PaddleOCR or TrOCR
- Classify text as printed vs handwritten
- Route signatures to signature detector
- Route checkboxes to checkbox detector
- Aggregate all OCR results

**Deliverable:** OCR routing functional

---

### **S3.8 OCR Output Storage**

**Task:** Save OCR results

**File:** `ocr/router.py` (extend)

**Requirements:**
- Store OCR JSON using canonical ID:
  - `/storage/intermediate/{canonical_id}_ocr.json`
  - Where canonical_id = pre_01_id (if SERVICE 0.1 triggered) or pre_0_id (otherwise)
- Structure:
  ```json
  {
    "text_regions": [
      {
        "region_id": "text_0",
        "text": "Sample text",
        "confidence": 0.95,
        "engine": "paddleocr",
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "signatures": [
      {
        "region_id": "sig_0",
        "present": true,
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "checkboxes": [
      {
        "region_id": "cb_0",
        "checked": true,
        "confidence": 0.88,
        "bbox": [x1, y1, x2, y2]
      }
    ]
  }
  ```

**Deliverable:** OCR JSON stored correctly

---

### **S3.9 OCR API Endpoint**

**Task:** Implement `/api/analyze/ocr` endpoint

**File:** `api/analyze.py` (extend)

**Requirements:**
- Accept `file_id`, `pre_0_id`, or `pre_01_id` in request body
- Determine canonical ID using resolution logic (same as previous services)
- Load image and layout/table JSONs using canonical ID
- Run OCR routing
- Store OCR JSON using canonical ID: `/storage/intermediate/{canonical_id}_ocr.json`
- Return OCR results

**Deliverable:** OCR endpoint functional

---

### **S3.10 Unit Tests**

**Task:** Write unit tests for OCR routing

**File:** `tests/test_service_3.py`

**Test Cases:**
- PaddleOCR initialization
- PaddleOCR text extraction
- TrOCR initialization
- TrOCR handwriting extraction
- Signature detection
- Checkbox detection
- OCR routing logic
- JSON output format

**Deliverable:** Test suite with >80% coverage

---

### **S3.11 Integration Tests**

**Task:** Test table isolation → OCR routing flow

**Test Scenarios:**
1. Table isolation → OCR routing → Verify OCR results
2. Test with printed text
3. Test with handwritten text
4. Test with signatures
5. Test with checkboxes
6. Verify confidence scores
7. Verify text accuracy

**Deliverable:** Integration tests passing

---

### **✅ SERVICE 3 Approval Checklist**

- [ ] PaddleOCR engine configured and functional
- [ ] PaddleOCR text extraction working
- [ ] TrOCR engine configured and functional
- [ ] TrOCR handwriting extraction working
- [ ] Signature detection functional
- [ ] Checkbox detection functional
- [ ] OCR routing logic implemented
- [ ] OCR JSON stored correctly
- [ ] OCR endpoint functional
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] OCR accuracy validated on sample images
- [ ] Code reviewed and documented
- [ ] Performance acceptable (<15s for typical document)

**Status:** ⏳ **PENDING APPROVAL**

**⚠️ DO NOT PROCEED TO SERVICE 4 UNTIL SERVICE 3 IS APPROVED**

---

## **📈 SERVICE 4: Scoring & Output**

### **Goal**

Aggregate OCR results, calculate confidence scores, reconstruct tables, map fields, and generate final structured JSON output for the frontend.

---

### **S4.1 Confidence Aggregation**

**Task:** Implement confidence scoring system

**File:** `postprocessing/scoring.py`

**Requirements:**
- Aggregate field-level confidence scores
- Calculate row-level confidence
- Calculate overall document confidence
- Handle missing fields
- Return confidence metrics

**Deliverable:** Confidence scoring functional

---

### **S4.2 Table Reconstruction**

**Task:** Reconstruct tables from OCR results

**File:** `postprocessing/table_reconstruct.py`

**Requirements:**
- Map OCR text to table cells
- Reconstruct table structure
- Handle empty cells
- Preserve row/column relationships
- Return structured table data

**Deliverable:** Table reconstruction functional

---

### **S4.3 Field Mapping**

**Task:** Map extracted text to structured fields

**File:** `postprocessing/field_mapper.py`

**Requirements:**
- Map text regions to field names
- Extract key-value pairs
- Handle attendee sheet fields:
  - First Name
  - Last Name
  - Attendee Type
  - Credential
  - State of License
  - License Number
  - Signature presence
  - Meal opt-out
- Return mapped fields

**Deliverable:** Field mapping functional

---

### **S4.4 Final Output Generation**

**Task:** Generate final structured JSON

**File:** `postprocessing/scoring.py` (extend)

**Requirements:**
- Combine all results (tables, OCR, fields)
- Generate `analysis_id` (format: `analysis_{timestamp}_{random}`)
- Store in `/storage/output/{analysis_id}.json`
- Structure:
  ```json
  {
    "analysis_id": "analysis_20250111_abc123",
    "file_id": "file_20250111_abcd123",
    "pre_0_id": "pre_0_20250111_99e12d",
    "pre_01_id": "pre_01_20250111_88f34e",
    "canonical_id": "pre_01_20250111_88f34e"
    "overall_confidence": 0.88,
    "rows": [
      {
        "firstName": {
          "value": "John",
          "confidence": 0.93,
          "status": "matched"
        },
        "lastName": {
          "value": "Doe",
          "confidence": 0.91,
          "status": "matched"
        },
        "attendeeType": {
          "value": "Business Guest",
          "confidence": 0.76,
          "status": "matched"
        },
        "credential": {
          "value": "MD",
          "confidence": 0.62,
          "status": "needs_review"
        },
        "state": {
          "value": "MA",
          "confidence": 0.96,
          "status": "matched"
        },
        "licenseNumber": {
          "value": "70346",
          "confidence": 0.71,
          "status": "matched"
        },
        "signaturePresent": {
          "value": true,
          "status": "matched"
        },
        "mealOptOut": {
          "value": false,
          "status": "matched"
        },
        "rowConfidence": 0.83
      }
    ],
    "metadata": {
      "processing_time": 12.5,
      "tables_detected": 1,
      "signatures_detected": 5
    }
  }
  ```

**Deliverable:** Final output JSON generated correctly

---

### **S4.5 Complete Analysis API Endpoint**

**Task:** Implement `/api/analyze/complete` endpoint

**File:** `api/analyze.py` (extend)

**Requirements:**
- Accept `file_id`, `pre_0_id`, or `pre_01_id` in request body
- Determine canonical ID using resolution logic (same as previous services)
- Run all services in sequence (if not already run):
  - SERVICE 0 (basic preprocessing) if not done
  - Layout detection
  - Table isolation
  - OCR routing
  - Scoring & output
- Load intermediate JSONs using canonical ID
- Include all IDs in output: `file_id`, `pre_0_id`, `pre_01_id` (if exists), `canonical_id`
- Return final analysis JSON
- Handle errors gracefully

**Deliverable:** Complete analysis endpoint functional

---

### **S4.6 Unit Tests**

**Task:** Write unit tests for scoring & output

**File:** `tests/test_service_4.py`

**Test Cases:**
- Confidence aggregation logic
- Table reconstruction accuracy
- Field mapping correctness
- Output JSON format
- Edge cases (missing fields, low confidence)

**Deliverable:** Test suite with >80% coverage

---

### **S4.7 Integration Tests**

**Task:** Test complete pipeline end-to-end

**Test Scenarios:**
1. Upload → SERVICE 0 → Services 1-4 → Output (no SERVICE 0.1)
2. Upload → SERVICE 0 → Services 1-4 fail → SERVICE 0.1 → Rerun Services 1-4 → Output
3. Verify complete pipeline works
4. Verify output JSON format matches frontend expectations (includes all IDs)
5. Test with various document types
6. Verify confidence scores are accurate
7. Test error handling
8. Verify canonical ID resolution in output

**Deliverable:** End-to-end integration tests passing

---

### **S4.8 Performance Optimization**

**Task:** Optimize pipeline performance

**Requirements:**
- Profile each service
- Optimize slow operations
- Add caching where appropriate
- Optimize model loading
- Target: <30s for complete pipeline

**Deliverable:** Performance optimized

---

### **✅ SERVICE 4 Approval Checklist**

- [ ] Confidence aggregation functional
- [ ] Table reconstruction accurate
- [ ] Field mapping correct
- [ ] Final output JSON generated correctly
- [ ] Complete analysis endpoint functional
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] End-to-end pipeline validated
- [ ] Output format matches frontend expectations
- [ ] Code reviewed and documented
- [ ] Performance acceptable (<30s for complete pipeline)

**Status:** ⏳ **PENDING APPROVAL**

---

## **🔧 SERVICE 0.1: Advanced Preprocessing (Conditional / Fallback Only)**

### **Goal**

Provide heavy CV enhancements as a fallback when Services 1-4 fail or produce low confidence.

**SERVICE 0.1 ONLY RUNS** when triggered by failure conditions in Services 1-4.

### **Key Characteristics**

- **Heavy processing** - Advanced corrections for difficult images
- **Conditional** - Only runs on failure/low confidence
- **Comprehensive** - Multiple enhancement techniques
- **Performance trade-off** - Slower but more effective for challenging inputs

### **Processing Steps**

1. Strong denoising (higher strength than basic)
2. Shadow removal
3. Full perspective correction
4. Orientation correction (EfficientNet classifier)
5. CLAHE++ tuning (enhanced contrast)
6. Handwriting enhancement
7. Adaptive binarization
8. Strong structural cleanup
9. SSIM-optimized compression

### **Output**

- Generates `pre_01_id` (format: `pre_01_{timestamp}_{random}`)
- Stores processed image in `/storage/processed_advanced/{pre_01_id}.png`
- This becomes the new canonical ID for rerunning Services 1-4

### **Trigger Conditions**

SERVICE 0.1 is automatically triggered if:
- Layout detection returns empty/insufficient results
- OCR confidence is below threshold (e.g., < 0.5)
- Table isolation fails
- Any exception occurs during Services 1-4
- Output is empty, invalid, or error-prone

### **Implementation Notes**

- SERVICE 0.1 will be built **after Services 1-4 are complete**
- Requires explicit developer approval before implementation
- Must integrate seamlessly with existing fallback logic
- Should reuse components from SERVICE 0 where possible

---

## **🎯 Final Integration & Testing**

### **F1. End-to-End Testing**

**Task:** Comprehensive end-to-end testing

**Test Scenarios:**
1. Upload JPG → SERVICE 0 → Services 1-4 → Verify output (no SERVICE 0.1)
2. Upload difficult image → SERVICE 0 → Services 1-4 fail → SERVICE 0.1 → Rerun Services 1-4 → Verify output
3. Upload PDF → Complete pipeline → Verify output
4. Upload HEIC → Complete pipeline → Verify output
5. Test canonical ID resolution logic
6. Test with various document qualities
7. Test error handling (invalid files, corrupted images)
8. Test performance under load
9. Verify ID traceability (file_id → pre_0_id → pre_01_id → analysis_id)

**Deliverable:** All end-to-end tests passing

---

### **F2. API Documentation**

**Task:** Generate API documentation

**Requirements:**
- FastAPI automatic docs (Swagger/OpenAPI)
- Document all endpoints
- Include request/response examples
- Document error codes

**Deliverable:** API documentation complete

---

### **F3. Deployment Preparation**

**Task:** Prepare for deployment

**Requirements:**
- Environment configuration
- Docker setup (optional)
- Production settings
- Logging configuration
- Error monitoring

**Deliverable:** Deployment-ready codebase

---

### **✅ Final Approval Checklist**

- [ ] All services (0, 0.1, 1-4) completed and approved
- [ ] Two-stage preprocessing working correctly
- [ ] Canonical ID resolution logic tested
- [ ] End-to-end tests passing (with and without SERVICE 0.1)
- [ ] API documentation complete
- [ ] Code reviewed and documented
- [ ] Performance benchmarks met
- [ ] Error handling robust
- [ ] Deployment configuration ready

**Status:** ⏳ **PENDING FINAL APPROVAL**

---

## **📅 Timeline Summary**

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0: Setup | 2-3 days | ⏳ Pending |
| SERVICE 0: Basic Preprocessing | 1-2 weeks | ⏳ Pending |
| SERVICE 1: Layout Detection | 1 week | ⏳ Pending |
| SERVICE 2: Table Isolation | 1 week | ⏳ Pending |
| SERVICE 3: OCR Routing | 1 week | ⏳ Pending |
| SERVICE 4: Scoring & Output | 1 week | ⏳ Pending |
| SERVICE 0.1: Advanced Preprocessing | 1 week | ⏳ Pending (after Services 1-4) |
| Final Integration | 3-5 days | ⏳ Pending |

**Total Estimated Time:** 7-9 weeks

---

## **📝 Notes**

1. **Strict Sequential Development:** Each service must be fully approved before proceeding.
2. **Testing Requirements:** Each service requires unit tests (>80% coverage) and integration tests.
3. **Documentation:** All code must be documented with docstrings.
4. **Performance Targets:** Each service has specific performance requirements.
5. **Error Handling:** Robust error handling required at all levels.
6. **ID Workflow:** Maintain file_id → pre_0_id → pre_01_id → analysis_id traceability.
7. **Two-Stage Preprocessing:** SERVICE 0 always runs; SERVICE 0.1 only on failure.
8. **Canonical ID:** Services 1-4 always use canonical ID (pre_0_id or pre_01_id), never raw file_id.

---

## **🚨 Critical Reminders**

- ⚠️ **DO NOT** start Service N+1 until Service N is approved
- ⚠️ **DO NOT** skip testing phases
- ⚠️ **DO NOT** proceed without explicit developer approval
- ✅ **DO** maintain code quality and documentation
- ✅ **DO** follow the ID workflow strictly
- ✅ **DO** test with real-world sample images

---

**Last Updated:** 2025-01-11
**Version:** 1.0.0
