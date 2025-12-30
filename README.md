# **AIME Backend — OCR & Computer Vision Pipeline**

## **README.md**

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
UPLOAD → SERVICE 0 (basic preprocessing)
       → SERVICE 1 (layout detection)
       → SERVICE 2 (table isolation)
       → SERVICE 3 (OCR routing)
       → SERVICE 4 (scoring & output)
             │
             └── failure / low confidence → SERVICE 0.1 (advanced preprocessing)
                                               → rerun SERVICE 1–4
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
* Detectron2 (PubLayNet)
* GPDS Signature Detector
* YOLOv8 (for checkboxes)
* PaddleOCR (DBNet text detection)
* TrOCR
* EfficientNet
* PyTorch
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
│   ├── trocr_engine.py
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
* Orientation correction (EfficientNet classifier)
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

* **PubLayNet (Detectron2)** — Tables and text blocks
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

# **🧠 SERVICE 2 — Table Isolation**

* Line detection
* Gap clustering
* Row/column segmentation
* Table bounding box refinement

**Output:**

```
{canonical_id}_tables.json
```

---

# **🧠 SERVICE 3 — OCR Routing**

* PaddleOCR for printed text
* TrOCR for handwriting
* SignatureNet for signatures
* Checkbox classifier

**Output:**

```
{canonical_id}_ocr.json
```

---

# **🧠 SERVICE 4 — Scoring & Output**

* Confidence aggregation
* Row/column reconstruction
* Field mapping
* Final JSON output for frontend

**Stored as:**

```
analysis_{timestamp}.json
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

3. **SERVICE 2 (Table isolation)**

4. **SERVICE 3 (OCR routing)**

5. **SERVICE 4 (Scoring & output)**

### **SERVICE 0.1 (Advanced Preprocessing)**

Will only be built after Services 1–4 are functional and **only when the developer explicitly requests it**.

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
efficientnet-pytorch
scikit-learn
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
| 4    | SERVICE 2                            |
| 5    | SERVICE 3                            |
| 6    | SERVICE 4                            |
| 7    | SERVICE 0.1 (advanced preprocessing) |

---

# **API Endpoints**

-   POST /api/upload
-   POST /api/analyze/preprocess (SERVICE 0 - basic)
-   POST /api/analyze/preprocess/advanced (SERVICE 0.1 - advanced, fallback only)
-   POST /api/analyze (main analysis endpoint with automatic fallback)
-   GET /api/health

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
