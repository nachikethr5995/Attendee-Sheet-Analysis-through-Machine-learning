# **AIME Backend — OCR & Computer Vision Pipeline**

## **README.md**

> **Note:** For information on setting up optional model files (GPDS signature, checkbox models), see [MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)

# **📌 Overview**

AIME Backend is a linear, GPU-accelerated FastAPI pipeline for high-accuracy document parsing in life-sciences workflows. It performs:

* OCR for handwritten + printed text
* Table detection & reconstruction
* Signature & checkbox detection
* Global confidence scoring
* Full auditability using file-level IDs

AIME uses a **two-stage preprocessing strategy** to maximize both performance and robustness:

### **Preprocessing Strategy**

1. **SERVICE 0 — Basic Preprocessing (Always Runs)**
   Lightweight corrections applied to every uploaded image.

2. **SERVICE 0.1 — Advanced Preprocessing (Only if Services 1–4 fail)**
   Heavy CV enhancements used as a fallback for difficult images.

### **Final Pipeline Flow**

```
UPLOAD → SERVICE 0 (basic preprocessing - optional)
       → SERVICE 1 (YOLOv8s layout detection)
       → Unified Pipeline:
          ├─ Class-based OCR routing (Text_box → PaddleOCR, Handwritten → PARSeq)
          ├─ Signature handling (presence + crop, no OCR)
          ├─ Checkbox handling (presence + checked/unchecked)
          ├─ Table-aware row grouping
          ├─ Column grouping
          └─ Structured output (row-wise & column-wise)
             │
             └── failure / low confidence → SERVICE 0.1 (advanced preprocessing)
                                               → rerun pipeline
```

This ensures:

* High performance for clean inputs
* High reliability for challenging inputs

---

# **📦 Project Goals**

* Robust OCR for mixed handwritten/printed content
* Improved fallback processing for difficult inputs
* Full traceability using file_id, pre_0_id, pre_01_id, and analysis_id
* Maintainable, modular backend
* Production-ready design with future extensibility
* Strict service-by-service build approval workflow

---

# **🚀 Core Technologies**

* FastAPI
* OpenCV
* scikit-image
* Pillow (HEIF/AVIF support)
* YOLOv8s (layout detection - Text_box, Handwritten, Signature, Checkbox, Table)
* GPDS Signature Detector
* PaddleOCR (printed text recognition only, detection disabled)
* PARSeq (handwritten text recognition)
* PyTorch & PyTorch Lightning
* pdf2image

---

# **📂 Folder Structure**

```
backend/
├── api/
│   ├── upload.py
│   ├── analyze.py
│   └── health.py
├── core/
│   ├── config.py
│   ├── logging.py
│   └── utils.py
├── ingestion/
│   └── file_handler.py
├── preprocessing/
│   ├── basic.py         # SERVICE 0
│   ├── advanced.py      # SERVICE 0.1
│   ├── converter.py
│   ├── quality.py
│   ├── deskew.py
│   └── perspective.py
├── layout/
│   ├── yolo_detector.py
│   ├── east_text_detector.py
│   └── fusion.py
├── tables/
│   ├── table_detector.py
│   └── line_cluster.py
├── ocr/
│   ├── paddle_engine.py
│   ├── parseq_recognizer.py  # PARSeq for handwritten recognition
│   └── router.py
├── postprocessing/
│   ├── table_reconstruct.py
│   ├── field_mapper.py
│   └── scoring.py
├── models/
│   ├── yolo_weights/
│   ├── signature_model/
│   └── classifiers/
└── main.py
```

---

# **FILE-ID / PREPROCESSING ID WORKFLOW**

### **IDs Generated**

| ID            | Description                                    |
| ------------- | ---------------------------------------------- |
| `file_id`     | Original uploaded file                         |
| `pre_0_id`    | Output of SERVICE 0 (basic preprocessing)      |
| `pre_01_id`   | Output of SERVICE 0.1 (advanced preprocessing) |
| `analysis_id` | Final result identifier                        |

### **Canonical ID Logic**

A canonical ID is the image used for Services 1–4:

* If **SERVICE 0.1 is NOT triggered** → canonical_id = `pre_0_id`
* If **SERVICE 0.1 IS triggered** → canonical_id = `pre_01_id`

### **Storage Layout**

```
storage/
├── raw/{file_id}.{ext}
├── processed_basic/{pre_0_id}.png
├── processed_advanced/{pre_01_id}.png
├── intermediate/{canonical_id}_layout.json
├── intermediate/{canonical_id}_tables.json
├── intermediate/{canonical_id}_ocr.json
└── output/{analysis_id}.json
```

### **Endpoint Acceptance Rule**

All endpoints must accept:

```
file_id
pre_0_id (optional)
pre_01_id (optional)
```

Endpoints automatically determine which ID to use:

1. Prefer `pre_01_id` if exists
2. Else prefer `pre_0_id`
3. Else use `file_id`

---

# **🧠 SERVICE 0 — Basic Preprocessing (Always Runs)**

Fast, minimal, non-destructive enhancement:

* Format conversion (HEIC/AVIF/PDF → PNG)
* Light denoising
* Image resize to 2048px long edge
* Soft deskew
* Light contrast normalization
* Minimal artifact removal

**Output:**

```
pre_0_id + standardized PNG
```

---

# **🧠 SERVICE 0.1 — Advanced Preprocessing (Conditional)**

Triggered **ONLY IF** Services 1–4 fail or produce low confidence.

Advanced corrections include:

* Strong denoising
* Shadow removal
* Full perspective correction
* Orientation correction (if needed)
* CLAHE++ tuning
* Handwriting enhancement
* Adaptive binarization
* Strong structural cleanup
* SSIM-optimized compression

**Output:**

```
pre_01_id + enhanced PNG
```

This becomes the canonical input for rerunning Services 1–4.

---

# **🧠 SERVICE 1 — Hybrid Layout Detection**

Using a hybrid ensemble of specialized pretrained models:

* **YOLOv8s** — Layout detection (Text_box, Handwritten, Signature, Checkbox, Table)
* **GPDS Signature Detector** — Handwritten signatures
* **Custom Checkbox Detector (Tiny YOLOv8)** — Checkboxes
* **PaddleOCR Text Detector (DBNet)** — Refined text regions
* **Fusion Engine** — Unify all detections with overlap handling

**Features:**
* High-accuracy layout detection using specialized models
* No training required for most components (pretrained models)
* Fast inference (~150-200ms per page on GPU)
* Robust error handling with fallbacks

**Stored as:**

```
{canonical_id}_layout.json
```

---

# **🧠 Unified Pipeline — Complete Analysis**

The unified pipeline orchestrates the complete analysis flow:

1. **YOLOv8s Layout Detection** - Detects all layout elements
2. **Class-based OCR Routing**:
   * Text_box → PaddleOCR (printed text)
   * Handwritten → PARSeq (handwriting)
   * Signature → Presence flag only (no OCR)
   * Checkbox → Presence + checked/unchecked (no OCR)
3. **Table-aware Row Grouping** - Groups detections into rows using table anchors
4. **Column Grouping** - Assigns cells to columns based on header positions
5. **Structured Output**:
   * Row-wise structured JSON
   * Column-wise structured JSON
   * Final API output with field mapping

**Output:**

```
POST /api/analyze returns:
{
  "rows": [
    {
      "last_name": "...",
      "first_name": "...",
      ...
    }
  ]
}
```

---

# **Frontend Requirements (Tables)**

Frontend must:

* Render extracted rows/columns as structured tables
* Apply low-confidence highlighting
* Maintain mapping between bounding boxes and cells
* Provide scroll support for wide tables
* Support mixed handwritten/printed cell content

---

# **🔐 STRICT DEVELOPMENT WORKFLOW (MANDATORY)**

AIME uses an **approval-based sequential build**.
No service can be built until the previous service is fully tested and explicitly approved.

### **Build Order:**

1. **SERVICE 0 (Basic preprocessing)**
   * Build & test → Developer approval required.

2. **SERVICE 1 (Layout detection)**
   * Build only after SERVICE 0 approval.

3. **Unified Pipeline (OCR routing, row/column grouping, structured output)**

### **SERVICE 0.1 (Advanced Preprocessing)**

Will only be built after the unified pipeline is functional and **only when the developer explicitly requests it**.

This preserves clarity, debuggability, and pipeline stability.

---

# **Required Python Libraries**

```
fastapi
uvicorn
python-multipart
pydantic
opencv-python
opencv-contrib-python
Pillow
pillow-heif
pillow-avif-plugin
pdf2image
numpy
scikit-image
ultralytics
torch
torchvision
# efficientnet-pytorch  # Not used in current workflow
# scikit-learn  # Not used in current workflow
paddleocr
paddlepaddle
loguru
python-dotenv
```

---

# **Implementation Timeline**

| Week | Deliverable                          |
| ---- | ------------------------------------ |
| 1–2  | SERVICE 0 (basic)                    |
| 3    | SERVICE 1                            |
| 4    | Unified Pipeline (OCR + Grouping + Output) |
| 7    | SERVICE 0.1 (advanced preprocessing) |

---

# **API Endpoints**

## **File Upload**

### **POST /api/upload**
Upload a file for processing.

**Request:**
- `file` (multipart/form-data): File to upload (JPG, PNG, HEIC, HEIF, AVIF, PDF)
- Maximum file size: 50MB

**Response:**
```json
{
  "file_id": "unique_file_identifier",
  "message": "File uploaded successfully"
}
```

### **GET /api/upload/{file_id}/image**
Retrieve the uploaded raw image by file_id.

**Parameters:**
- `file_id` (path): File identifier

**Response:** Image file (FileResponse)

---

## **Health Check**

### **GET /api/health**
Check system health and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "opencv": "available",
    "pillow": "available",
    "torch": "available",
    "gpu": "available"
  }
}
```

---

## **Preprocessing (SERVICE 0 & 0.1)**

### **POST /api/analyze/preprocess**
SERVICE 0: Basic preprocessing (always runs).

**Request:**
```json
{
  "file_id": "your_file_id"
}
```

**Response:**
```json
{
  "pre_0_id": "preprocessing_identifier",
  "clean_image_path": "path/to/processed/image.png",
  "metadata": {
    "original_format": "jpg",
    "processed_format": "png",
    "dimensions": {"width": 2048, "height": 1536}
  }
}
```

**Processing Steps:**
1. Format conversion (HEIC/AVIF/PDF → PNG)
2. Light denoising
3. Image resize to 2048px long edge
4. Soft deskew
5. Light contrast normalization
6. Minimal artifact removal

### **GET /api/analyze/preprocess**
Helper endpoint providing usage information (returns 405 Method Not Allowed with instructions).

### **GET /api/analyze/preprocess/{pre_0_id}/image**
Retrieve the basic processed image by pre_0_id.

**Parameters:**
- `pre_0_id` (path): Basic preprocessing identifier

**Response:** PNG image file (FileResponse)

### **POST /api/analyze/preprocess/advanced**
SERVICE 0.1: Advanced preprocessing (conditional/fallback only).

**Request:**
```json
{
  "file_id": "your_file_id",
  "pre_0_id": "your_pre_0_id (optional)"
}
```

**Response:**
```json
{
  "pre_01_id": "advanced_preprocessing_identifier",
  "clean_image_path": "path/to/enhanced/image.png",
  "metadata": {
    "original_format": "jpg",
    "processed_format": "png",
    "dimensions": {"width": 2048, "height": 1536}
  }
}
```

**Processing Steps:**
1. Strong denoising
2. Shadow removal
3. Full perspective correction
4. Orientation correction (EfficientNet classifier)
5. CLAHE++ tuning
6. Handwriting enhancement
7. Adaptive binarization
8. Strong structural cleanup

### **GET /api/analyze/preprocess/advanced**
Helper endpoint providing usage information (returns 405 Method Not Allowed with instructions).

### **GET /api/analyze/preprocess/advanced/{pre_01_id}/image**
Retrieve the advanced processed image by pre_01_id.

**Parameters:**
- `pre_01_id` (path): Advanced preprocessing identifier

**Response:** PNG image file (FileResponse)

---

## **Analysis Services**

### **POST /api/analyze/layout**
SERVICE 1: Layout detection using YOLOv8s.

**Request:**
```json
{
  "file_id": "your_file_id (optional)",
  "pre_0_id": "your_pre_0_id (optional)",
  "pre_01_id": "your_pre_01_id (optional)"
}
```
*Note: At least one ID must be provided. Uses canonical ID resolution: pre_01_id > pre_0_id > file_id*

**Response:**
```json
{
  "canonical_id": "resolved_image_id",
  "detections": {
    "tables": [...],
    "text_blocks": [...],
    "handwritten": [...],
    "signatures": [...],
    "checkboxes": [...]
  },
  "failed": false,
  "metadata": {...}
}
```

**Detection Models:**
- YOLOv8s for tables, text blocks, signatures, and checkboxes
- GPDS signature detector for handwritten signatures
- Custom checkbox detector (tiny YOLOv8)
- PaddleOCR text detector for refined text regions
- Fusion engine to unify all detections

### **POST /api/analyze/ocr**
SERVICE 3: OCR pipeline.

**Request:**
```json
{
  "file_id": "your_file_id (optional)",
  "pre_0_id": "your_pre_0_id (optional)",
  "pre_01_id": "your_pre_01_id (optional)",
  "paddle_confidence_threshold": 0.5 (optional)
}
```
*Note: At least one ID must be provided*

**Response:**
```json
{
  "text_regions": [
    {
      "bbox": [x1, y1, x2, y2],
      "text": "extracted text",
      "confidence": 0.95,
      "source": "paddleocr" or "parseq"
    }
  ],
  "dimensions": {"width": 2048, "height": 1536},
  "failed": false,
  "metadata": {...}
}
```

**OCR Routing:**
- PaddleOCR for printed text (Text_box class only)
- PARSeq for handwriting (Handwritten class only)
- Class-based routing: Text_box → PaddleOCR, Handwritten → PARSeq
- No fallbacks - strict class-based routing

### **POST /api/analyze/signatures/verify**
Signature verification endpoint (post-detection verification).

**Request:**
```json
{
  "canonical_id": "image_id (optional)",
  "file_id": "your_file_id (optional)",
  "pre_0_id": "your_pre_0_id (optional)",
  "pre_01_id": "your_pre_01_id (optional)",
  "verification_threshold": 0.5 (optional)
}
```
*Note: At least one ID must be provided*

**Response:**
```json
{
  "signatures": [
    {
      "bbox": [x1, y1, x2, y2],
      "is_valid_signature": true,
      "verification_score": 0.92,
      "crop_path": "path/to/signature/crop.png"
    }
  ],
  "total_signatures": 3,
  "valid_signatures": 2,
  "failed": false,
  "metadata": {...}
}
```

**Note:** This endpoint verifies already-detected signatures using GPDS verification model. It does NOT detect signatures - it only validates detected regions.

### **POST /api/analyze/rowwise**
Unified pipeline endpoint returning dual row-wise and column-wise structured output.

**Request:**
```json
{
  "file_id": "your_file_id (optional)",
  "pre_0_id": "your_pre_0_id (optional)",
  "pre_01_id": "your_pre_01_id (optional)"
}
```
*Note: At least one ID must be provided*

**Response:**
```json
{
  "rowwise": {
    "rows": [
      {
        "row_index": 1,
        "columns": {
          "PrintedText": ["First Name", "Last Name"],
          "HandwrittenText": ["Drew"],
          "Signature": false,
          "Checkbox": null
        }
      }
    ],
    "total_rows": 9,
    "total_columns": 4
  },
  "columnwise": {
    "columns": [
      {
        "column_index": 1,
        "header": "First Name",
        "class": "Text",
        "rows": {
          "1": ["Drew"],
          "2": [],
          "3": ["Alexis"]
        }
      }
    ],
    "total_columns": 9,
    "total_rows": 9
  }
}
```

**Pipeline Flow:**
1. YOLOv8s layout detection
2. Class-based OCR routing with table filtering:
   - Filter text_boxes using center-point containment (center must be inside table bbox)
   - Text_box → PaddleOCR (only for filtered text_boxes inside tables)
   - Handwritten → PARSeq
3. Signature handling (presence + crop, NO OCR)
4. Checkbox handling (presence + checked/unchecked state)
5. Table-aware row grouping (Y-center clustering)
6. Column assignment (X-center distance-based)
7. Dual structured JSON output (row-wise + column-wise)

**Table Filtering Logic:**
- A text_box is considered inside a table if its center point (`x_center`, `y_center`) lies inside the table bounding box
- This matches the row/column assignment logic for end-to-end geometric consistency
- No tables detected → No PaddleOCR (strict policy)

### **POST /api/analyze**
Complete analysis endpoint running all services in sequence with automatic fallback.

**Request:**
```json
{
  "file_id": "your_file_id"
}
```

**Response:**
```json
{
  "file_id": "your_file_id",
  "preprocessing_applied": true,
  "pre_0_id": "preprocessing_id",
  "pre_01_id": null,
  "canonical_id": "resolved_image_id",
  "services_used": ["SERVICE_0", "SERVICE_1", "SERVICE_3"],
  "fallbacks_used": [],
  "layout_result": {...},
  "ocr_result": {...},
  "signature_verification_result": {...},
  "rowwise_result": {...},
  "columnwise_result": {...}
}
```

**Flow:**
1. Run SERVICE 1 (YOLOv8s) on original file_id (NO preprocessing)
2. Evaluate YOLOv8s results
3. Conditionally trigger SERVICE 0 (basic) or SERVICE 0.1 (advanced) if needed
4. If preprocessing triggered: rerun SERVICE 1 on processed image
5. Run SERVICE 3 (OCR pipeline)
6. Run signature verification
7. Run unified pipeline (row-wise + column-wise extraction)

---

## **Endpoint ID Resolution**

All analysis endpoints accept:
- `file_id` (original uploaded file)
- `pre_0_id` (optional, SERVICE 0 output)
- `pre_01_id` (optional, SERVICE 0.1 output)

**Canonical ID Resolution Logic:**
1. Prefer `pre_01_id` if exists
2. Else prefer `pre_0_id`
3. Else use `file_id`

This ensures endpoints automatically use the best available processed image.

---

# **Future Capability: Customizable Preprocessing**

Later versions will allow the developer/system to request preprocessing based on custom rules:

* "Make the image look like a scanned PDF"
* "Normalize lighting + flatten shadows"
* "Enhance pencil handwriting"
* "Remove background patterns"
* "Aggressive deskew only if angle > 5°"

These customization rules will be integrated once approved.

**Current state:** Preprocessing uses standard modes (basic always, advanced on fallback).

---

# **📋 Implementation History & Architecture Evolution**

This section documents the evolution of the AIME backend architecture, including critical fixes and improvements that shaped the current system.

## **Core Architecture Principles**

### **Golden Rule: YOLO Defines Structure, OCR Fills Content**

The system follows a strict architectural principle:

> **YOLO is the only detector. OCR only reads what YOLO explicitly gives it.**

This ensures:
- ✅ No OCR detection (PaddleOCR runs with `det=False`)
- ✅ No handwriting heuristics (YOLO class determines routing)
- ✅ Strict class-based routing (Text_box → PaddleOCR, Handwritten → PARSeq)
- ✅ No OCR outside YOLO regions
- ✅ Deterministic, reproducible results

### **Dual Row-wise + Column-wise Extraction**

The system provides two orthogonal views of the same document structure:

1. **Rowwise View** - Record-oriented (rows = records, columns = fields)
2. **Columnwise View** - Field-oriented (columns = fields, rows = values)

Both views are computed independently and consistently from the same YOLO detections.

## **Architecture Evolution**

### **Phase 1: Initial Row-wise Implementation**

**Components:**
- `RowGrouper` - Groups detections into rows using Y-axis clustering
- `RowwiseFormatter` - Formats rows into structured output
- Class-based OCR routing (Text_box → PaddleOCR, Handwritten → PARSeq)

**Output Schema:**
```json
{
  "rows": [
    {
      "row_index": 1,
      "columns": {
        "PrintedText": ["..."],
        "HandwrittenText": ["..."],
        "Signature": true,
        "Checkbox": false
      }
    }
  ],
  "total_rows": 10,
  "total_columns": 4
}
```

### **Phase 2: Dual Row/Column Pipeline**

**Problem:** Need column-wise view for analytics, validation, and database ingestion.

**Solution:** Added independent column grouping using X-axis clustering.

**New Components:**
- `ColumnGrouper` - Groups detections into columns using X-center clustering
- `ColumnwiseFormatter` - Formats columns into structured output

**Key Principle:** Rows and columns are orthogonal dimensions - computed independently.

### **Phase 3: Independent Column Grouping Fix**

**Problem:** Column grouping was derived from rowwise aggregation, causing:
- Column collapse (multiple columns merging)
- Class contamination (printed/handwritten mixing)
- Unstable headers
- Row-dependent column inference

**Solution:** Pure X-center clustering on ALL detections (not anchor-based).

**Algorithm:**
```python
# Sort all detections by X-center
# Cluster using threshold (3% of image width)
# Assign all detections to nearest column
```

### **Phase 4: Explicit Column Anchors**

**Problem:** Columns were inferred dynamically, causing:
- No persistent column anchors
- Headers not locked to X-ranges
- Classes mixed in columns
- Columnwise reconstructed from rowwise

**Solution:** Explicit column anchors with frozen X-ranges from header row.

**Components:**
- `ColumnAnchor` - Dataclass with locked X-ranges, header text, dominant class
- `ColumnAnchorBuilder` - Builds anchors from header row
- Dominant class classification per column

**Key Feature:** Columns defined once from header row, reused everywhere.

### **Phase 5: X-Axis Geometry-Based Assignment (Current)**

**Problem:** Column grouping happened AFTER OCR formatting, causing:
- Column membership inferred from text class
- Handwritten text not properly assigned
- Inconsistent column assignment

**Solution:** Column assignment using X-axis overlap BEFORE OCR formatting.

**Correct Architecture:**
```
YOLO → Row Grouper → Column Assigner (X-based) → OCR → Rowwise → Columnwise
```

**Key Principle:** Column membership decided using X-axis overlap, not text class or row order.

**Components:**
- `ColumnAssigner` - Assigns `column_index` to ALL detections using X-axis overlap
- Locks column boundaries from header row
- Expands X-ranges with tolerance (±3%) for drift handling

**Algorithm:**
1. Find header row (maximum PrintedText, no other classes)
2. Build column anchors from header row (locked X-ranges)
3. Expand X-ranges with tolerance
4. Assign every detection to a column based on X-center overlap

**Table Filtering (OCR Stage):**
- Before OCR routing, filter text_boxes using center-point containment
- A text_box is processed by PaddleOCR only if its center point (`x_center`, `y_center`) lies inside a table bounding box
- This ensures geometric consistency with row/column assignment (which also uses center points)
- No tables detected → No PaddleOCR (strict policy prevents processing non-tabular text)

## **Current System Architecture**

### **Pipeline Flow**

```
1. YOLOv8s Layout Detection
   ├── Tables
   ├── Text blocks
   ├── Handwritten regions
   ├── Signatures
   └── Checkboxes

2. Row Grouper (Y-axis clustering)
   └── Groups detections into rows

3. Column Assigner (X-axis overlap) ← CRITICAL
   ├── Find header row
   ├── Build column anchors (locked X-ranges)
   ├── Expand X-ranges with tolerance
   └── Assign column_index to ALL detections

4. OCR Routing
   ├── Text_box → PaddleOCR
   ├── Handwritten → PARSeq
   ├── Signature → Presence flag (no OCR)
   └── Checkbox → Checked state (no OCR)

5. Dual Formatting
   ├── Rowwise Formatter (uses assigned column_index)
   └── Columnwise Formatter (uses assigned column_index)
```

### **Column Assignment Details**

**Header Row Selection:**
- Priority 1: Row with maximum PrintedText count
- Priority 2: No handwritten, signatures, or checkboxes
- Fallback: Configured `HEADER_ROW_INDEX` (default: 2)

**Column Boundary Locking:**
- Extract PrintedText detections from header row
- Compute X-center for each header cell
- Sort left → right
- Assign column_index = 1, 2, 3, ...
- Store locked X-ranges: `[x_min, x_max]`

**X-Range Expansion:**
- Tolerance: 3% of image width
- Handles handwriting drift and slight misalignments
- Prevents column boundary edge cases

**Detection Assignment:**
- For EVERY detection (text / handwritten / signature / checkbox):
  - Compute X-center
  - Find column where `x_min <= x_center <= x_max`
  - Assign `column_index`
- One detection → one column
- Works for ALL classes

### **Output Schemas**

**Rowwise Output:**
```json
{
  "rowwise": {
    "rows": [
      {
        "row_index": 1,
        "columns": {
          "PrintedText": ["First Name", "Last Name"],
          "HandwrittenText": [],
          "Signature": false,
          "Checkbox": null
        }
      }
    ],
    "total_rows": 9,
    "total_columns": 36
  }
}
```

**Columnwise Output:**
```json
{
  "columnwise": {
    "columns": [
      {
        "column_index": 1,
        "header": "First Name",
        "class": "Mixed",
        "rows": {
          "1": ["Drew"],
          "2": [],
          "3": ["Alexis"],
          "7": ["Jack"],      // HandwrittenText
          "8": ["Dawn"]       // HandwrittenText
        }
      },
      {
        "column_index": 7,
        "header": "Signature",
        "class": "Mixed",
        "rows": {
          "3": true,
          "5": true,
          "6": true
        }
      }
    ],
    "total_columns": 9,
    "total_rows": 9
  }
}
```

## **Critical Fixes Applied**

### **1. OCR Architecture Fix**

**Problem:** PaddleOCR was detecting regions (`det=True`), violating YOLO authority.

**Fix:**
- Disabled PaddleOCR detection (`det=False`)
- PaddleOCR now ONLY does recognition (receives cropped regions from YOLO)
- Removed handwriting classification heuristics
- Strict class-based routing with no fallbacks

### **2. Boolean Type Normalization**

**Problem:** Checkbox/Signature values could be numpy bool types, causing validation failures.

**Fix:**
- Convert all boolean values to Python `bool` type
- Normalize during assignment and validation
- Handle numpy bool types gracefully

### **3. Column Class Isolation**

**Problem:** Classes were mixed within columns, causing semantic confusion.

**Fix:**
- Allow multiple classes in same column ("Mixed" class)
- PrintedText + HandwrittenText both allowed
- Signature/Checkbox remain boolean
- Preserve empty rows for rectangular structure

### **4. Center-Point Based Table Filtering**

**Problem:** Overlap-ratio filtering (IoU ≥ 0.3) was rejecting valid text boxes whose area slightly crossed table borders, causing missing values in leftmost columns (e.g., "First Name", "Drew").

**Solution:** Center-point containment - a text_box is considered inside a table if its center point lies inside the table bounding box.

**Implementation:**
- Added `is_center_inside_bbox()` utility function (`Back_end/core/utils.py`)
- Filters text_boxes using center-point containment before PaddleOCR routing
- Matches row/column assignment logic (uses same `x_center`/`y_center` primitives)
- Deterministic binary check (no thresholds needed)

**Key Benefits:**
- ✅ Matches `_internal.x_center` logic used in row/column assignment
- ✅ Stable for narrow columns
- ✅ Immune to partial boundary crossings
- ✅ End-to-end geometric consistency (OCR filtering → Row assignment → Column assignment)

**Architecture Consistency:**
1. **YOLO Detection** → Provides bboxes
2. **OCR Filtering** → Uses center-point containment (`x_center`, `y_center`)
3. **Row Assignment** → Uses `y_center` clustering
4. **Column Assignment** → Uses `x_center` distance matching

All stages now use the same geometric primitive (center points), ensuring consistency.

**Configuration:**
```python
OCR_TABLE_FILTER_MODE: str = "center"  # Filter mode: "center" (center-point containment) or "none" (process all)
```

**Logging:**
```
PaddleOCR routing: 24/29 text_boxes passed (center-in-table)
```

## **Configuration Settings**

```python
# Row/Column Grouping
ROW_HEIGHT_THRESHOLD: float = 0.02      # 2% of image height
COLUMN_WIDTH_THRESHOLD: float = 0.03    # 3% of image width
HEADER_ROW_INDEX: int = 2               # Row index for header (1-based)

# OCR Thresholds
OCR_PADDLE_CONFIDENCE_THRESHOLD: float = 0.5
# PARSeq settings (see core/config.py for PARSEQ_* settings)
OCR_TABLE_FILTER_MODE: str = "center"  # Filter mode: "center" (center-point containment) or "none" (process all)
```

## **Validation & Guarantees**

✅ **YOLO defines structure** - All detections come from YOLOv8s  
✅ **OCR defines content** - PaddleOCR/PARSeq only fill content within YOLO regions  
✅ **No OCR outside YOLO regions** - Strict enforcement  
✅ **Table-only OCR filtering** - PaddleOCR only processes text_boxes whose center point lies inside table bbox  
✅ **Center-point containment** - Uses same geometric primitive as row/column assignment for consistency  
✅ **No cross-contamination** - Printed/handwritten routing is separate  
✅ **Deterministic indices** - Consistent row/column indexing  
✅ **Column boundaries locked** - From header row, not inferred  
✅ **X-axis overlap** - Column membership based on geometry  
✅ **All detections assigned** - PrintedText, HandwrittenText, Signature, Checkbox  
✅ **Rectangular data** - All rows present in all columns  
✅ **API-safe serialization** - All outputs are JSON-safe  

## **Use Cases Enabled**

1. **CSV Export** - Column-wise view enables direct CSV conversion
2. **Database Ingestion** - Column structure maps to database schemas
3. **Spreadsheet Reconstruction** - Dual views enable table reconstruction
4. **Rule-based Validation** - Column-wise checks (e.g., missing signatures per column)
5. **Data Analysis** - Both row and column perspectives for analysis
6. **UI Grids** - Column-wise structure perfect for table/grid components

## **Testing Recommendations**

1. Test with various document layouts (tables, forms, sign-in sheets)
2. Verify row/column alignment across both views
3. Test edge cases (single row, single column, irregular spacing)
4. Validate OCR routing in both views
5. Check signature/checkbox detection in both views
6. Verify column boundaries remain consistent across rows
7. Test header row selection with different row positions
8. Validate empty row normalization (all rows present)
9. Test with mixed printed/handwritten content
10. Verify class isolation (no cross-contamination)

---

**Note:** This implementation history represents the evolution of the system from initial row-wise extraction to the current dual row/column architecture with geometry-based column assignment. All fixes have been applied and tested in production.
