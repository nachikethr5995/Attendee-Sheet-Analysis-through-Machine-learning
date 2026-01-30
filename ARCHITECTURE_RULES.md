# Architecture Rules (Non-Negotiable)

## Golden Rule: YOLOv8s is the ONLY Authority for Regions

**OCR must NEVER decide where to read. YOLO is the ONLY authority for regions.**

### Component Responsibilities

| Component | Allowed to Detect Boxes? | Responsibility |
|-----------|-------------------------|----------------|
| **YOLOv8s** | ✅ **YES (ONLY ONE)** | Layout detection: Text_box, Handwritten, Signature, Checkbox, Table |
| **PaddleOCR** | ❌ **NO** | Recognition ONLY (receives cropped regions from YOLO) |
| **PARSeq** | ❌ **NO** | Recognition ONLY (receives cropped regions from YOLO) |

### Strict Routing Rules

1. **Text_box class** → PaddleOCR ONLY
   - No fallbacks
   - No handwriting heuristics
   - YOLO decides it's printed text

2. **Handwritten class** → PARSeq ONLY
   - No fallbacks
   - No confidence-based switching
   - YOLO decides it's handwriting

3. **Signature class** → NO OCR
   - Presence flag only
   - Optional crop storage

4. **Checkbox class** → NO OCR
   - Presence + checked/unchecked state
   - Pixel density analysis

5. **Table class** → NO OCR
   - Used for row grouping anchor only

### PaddleOCR Configuration (MANDATORY)

```python
PaddleOCR(
    det=False,   # ❌ CRITICAL: Disable detection
    rec=True,    # ✅ Only recognition
    lang="en"
)
```

**PaddleOCR must NEVER see the full image. It only receives cropped regions from YOLO.**

### Preprocessing Rules

- **Preprocessing is OPTIONAL** (disabled by default)
- **YOLOv8s ALWAYS runs on original image** (no preprocessing interference)
- Preprocessing only triggers if:
  - YOLOv8s returns 0 detections
  - YOLOv8s confidence < threshold
  - Explicitly enabled via config

### Output Schema (Clean)

```json
{
  "text_regions": [
    {
      "bbox": [...],
      "class": "Text_box",
      "text": "Health Care Provider",
      "confidence": 0.99,
      "source": "paddleocr"
    },
    {
      "bbox": [...],
      "class": "Handwritten",
      "text": "McNuff",
      "confidence": 0.87,
      "source": "parseq"
    }
  ]
}
```

**NO fallback fields:**
- ❌ `fallback_used`
- ❌ `fallback_reason`
- ❌ `paddle_text` (duplicate)
- ❌ `is_handwriting` (should come from YOLO class)

### Safety Assertions

All OCR routers must verify:
```python
assert source_bbox_origin == "YOLO", \
    "OCR received non-YOLO region — this is a pipeline violation"
```

### Violations to Watch For

1. ❌ PaddleOCR running `det=True` (detection enabled)
2. ❌ OCR pipeline using PaddleOCR detection instead of YOLO
3. ❌ Handwriting heuristics triggering PARSeq fallback
4. ❌ Mixed OCR sources for one detection
5. ❌ OCR seeing full image instead of cropped regions

### Correct Flow

```
Image
  ↓
YOLOv8s Layout Detection (ONLY detector)
  ↓
Class-based routing
  ├── class=Text_box      → PaddleOCR (REC ONLY, det=False)
  ├── class=Handwritten   → PARSeq
  ├── class=Signature     → Crop + presence (NO OCR)
  ├── class=Checkbox      → Presence + checked (NO OCR)
  ↓
Table-aware row grouping
  ↓
Row-wise structured output
```
















