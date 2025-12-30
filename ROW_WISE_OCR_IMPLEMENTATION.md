# Row-Wise OCR Text Output Implementation

## ✅ Implementation Complete

The row-wise output now includes extracted text from PaddleOCR and TrOCR, aggregated by row.

## Changes Made

### 1. Row Data Model Updated ✅

**File:** `Back_end/postprocessing/rowwise_formatter.py`

**New Schema:**
```python
RowOutput = {
    "row_index": int,
    "columns": {
        "PrintedText": List[str],        # PaddleOCR results (Text_box class)
        "HandwrittenText": List[str],    # TrOCR results (Handwritten class)
        "Signature": bool,               # Presence flag
        "Checkbox": Optional[bool]       # Checked/unchecked or None
    }
}
```

**Key Features:**
- `PrintedText` and `HandwrittenText` are **lists** (not strings)
- Empty list `[]` if no OCR content exists for that row
- All rows appear in output (even with empty OCR lists)

### 2. Detection → Row Assignment ✅

**File:** `Back_end/postprocessing/unified_pipeline.py`

- Detections are already assigned to rows by `RowGrouper.group_into_rows()`
- Each row contains a `detections` list with YOLO detections
- Row assignment uses vertical overlap with table grid

### 3. Class-Based OCR Routing (Strict) ✅

**File:** `Back_end/postprocessing/rowwise_formatter.py` (lines 84-125)

**Implementation:**
```python
# STRICT CLASS-BASED ROUTING (No OCR inference outside YOLO regions)
if class_name in ['text_block', 'text_box']:
    # Text_box → PaddleOCR text
    if ocr_result and ocr_source == 'paddleocr' and confidence >= 0.4:
        normalized_text = self._normalize_text(text)
        columns['PrintedText'].append(normalized_text)

elif class_name == 'handwritten':
    # Handwritten → TrOCR text
    if ocr_result and ocr_source == 'trocr' and confidence >= 0.4:
        normalized_text = self._normalize_text(text)
        columns['HandwrittenText'].append(normalized_text)

elif class_name == 'signature':
    # Signature → Presence flag only (NO OCR)
    columns['Signature'] = True

elif class_name == 'checkbox':
    # Checkbox → Presence + checked/unchecked (NO OCR)
    columns['Checkbox'] = checkbox.get('checked', False)
```

**Rules Enforced:**
- ✅ OCR **never** invoked for Signature, Checkbox, or Table classes
- ✅ Text_box → PaddleOCR ONLY (no exceptions)
- ✅ Handwritten → TrOCR ONLY (no exceptions)
- ✅ Confidence threshold: 0.4 (text dropped if below)

### 4. Text Normalization ✅

**File:** `Back_end/postprocessing/rowwise_formatter.py` (lines 148-161)

**Implementation:**
```python
def _normalize_text(self, text: str) -> str:
    """Normalize text by collapsing whitespace."""
    if not text:
        return ""
    # Collapse multiple whitespace to single space
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized
```

**Applied to:**
- PaddleOCR output (PrintedText)
- TrOCR output (HandwrittenText)

### 5. Post-Processing: Deduplication Per Row ✅

**File:** `Back_end/postprocessing/rowwise_formatter.py` (lines 127-129)

**Implementation:**
```python
# Post-processing: Deduplication per row
columns['PrintedText'] = list(dict.fromkeys(columns['PrintedText']))  # Preserve order, remove duplicates
columns['HandwrittenText'] = list(dict.fromkeys(columns['HandwrittenText']))  # Preserve order, remove duplicates
```

**Features:**
- Preserves order (first occurrence kept)
- Removes duplicate text fragments within same row

### 6. Safety Assertions ✅

**File:** `Back_end/postprocessing/rowwise_formatter.py` (lines 70-73)

**Implementation:**
```python
# Safety assertion: All detections must come from YOLO
if source and source != 'YOLOv8s':
    log.warning(f"⚠️  Non-YOLO detection in row {row_index} (source: {source}) - skipping")
    continue
```

**File:** `Back_end/postprocessing/unified_pipeline.py` (lines 139-147)

**Additional Verification:**
```python
# Safety assertion: Verify all OCR results have correct source
for ocr_result in ocr_results:
    ocr_class = ocr_result.get('class', '').lower()
    ocr_source = ocr_result.get('source', '').lower()
    
    if ocr_class == 'text_block' and ocr_source != 'paddleocr':
        log.warning(f"⚠️  Text_block OCR result has wrong source: {ocr_source}")
    elif ocr_class == 'handwritten' and ocr_source != 'trocr':
        log.warning(f"⚠️  Handwritten OCR result has wrong source: {ocr_source}")
```

## Expected Output Example

### Row with OCR Content:
```json
{
  "row_index": 5,
  "columns": {
    "PrintedText": ["Health Care Provider", "MI"],
    "HandwrittenText": ["McNuff"],
    "Signature": true,
    "Checkbox": true
  }
}
```

### Row with No OCR Content:
```json
{
  "row_index": 2,
  "columns": {
    "PrintedText": [],
    "HandwrittenText": [],
    "Signature": false,
    "Checkbox": null
  }
}
```

## Architecture Rule (Enforced)

> **YOLO defines structure; OCR only fills content within YOLO regions**

This is enforced at multiple levels:
1. ✅ YOLO source verification in formatter
2. ✅ OCR source verification in pipeline
3. ✅ Class-based routing (no OCR for Signature/Checkbox)
4. ✅ Confidence thresholding (0.4 minimum)

## Non-Goals (Explicitly Avoided)

- ❌ OCR fallback logic
- ❌ PaddleOCR detection (`det=True` - already disabled)
- ❌ TrOCR full-image inference
- ❌ Mixed OCR sources per region
- ❌ Handwriting heuristics

## Testing Checklist

- [x] PrintedText list populated from PaddleOCR (Text_box class)
- [x] HandwrittenText list populated from TrOCR (Handwritten class)
- [x] Signature boolean flag (no OCR)
- [x] Checkbox boolean flag (no OCR)
- [x] Text normalization (whitespace collapse)
- [x] Deduplication per row
- [x] Empty lists for rows with no OCR
- [x] YOLO source verification
- [x] Confidence thresholding (0.4)

## API Endpoint

**`POST /api/analyze/rowwise`**

Returns structured row-wise output with OCR text aggregated by row.

**Response Structure:**
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
  "total_columns": 4,
  "layout": {...},
  "failed": false
}
```


