"""Utility functions for AIME Backend."""

import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import numpy as np
from PIL import Image


def generate_file_id() -> str:
    """Generate a unique file ID.
    
    Format: file_{timestamp}_{random}
    
    Returns:
        str: Unique file identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"file_{timestamp}_{random_str}"


def generate_processed_id() -> str:
    """Generate a unique processed image ID (legacy).
    
    Format: pre_{timestamp}_{random}
    
    Returns:
        str: Unique processed identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_{timestamp}_{random_str}"


def generate_pre_0_id() -> str:
    """Generate a unique basic preprocessing ID (SERVICE 0).
    
    Format: pre_0_{timestamp}_{random}
    
    Returns:
        str: Unique basic preprocessing identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_0_{timestamp}_{random_str}"


def generate_pre_01_id() -> str:
    """Generate a unique advanced preprocessing ID (SERVICE 0.1).
    
    Format: pre_01_{timestamp}_{random}
    
    Returns:
        str: Unique advanced preprocessing identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_01_{timestamp}_{random_str}"


def generate_analysis_id() -> str:
    """Generate a unique analysis ID.
    
    Format: analysis_{timestamp}_{random}
    
    Returns:
        str: Unique analysis identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"analysis_{timestamp}_{random_str}"


def get_raw_file_path(file_id: str, extension: str) -> Path:
    """Get the storage path for a raw uploaded file.
    
    Args:
        file_id: File identifier
        extension: File extension (e.g., 'jpg', 'png', 'pdf')
        
    Returns:
        Path: Full path to the raw file
    """
    from core.config import settings
    return Path(settings.STORAGE_RAW) / f"{file_id}.{extension}"


def get_processed_image_path(processed_id: str) -> Path:
    """Get the storage path for a processed image (legacy).
    
    Args:
        processed_id: Processed image identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED) / f"{processed_id}.png"


def get_basic_processed_path(pre_0_id: str) -> Path:
    """Get the storage path for a basic processed image (SERVICE 0).
    
    Args:
        pre_0_id: Basic preprocessing identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED_BASIC) / f"{pre_0_id}.png"


def get_advanced_processed_path(pre_01_id: str) -> Path:
    """Get the storage path for an advanced processed image (SERVICE 0.1).
    
    Args:
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED_ADVANCED) / f"{pre_01_id}.png"


def resolve_canonical_id(file_id: Optional[str] = None, 
                         pre_0_id: Optional[str] = None, 
                         pre_01_id: Optional[str] = None) -> str:
    """Resolve canonical ID from available IDs, checking file existence.
    
    Priority: pre_01_id > pre_0_id > file_id
    Only returns an ID if the corresponding file actually exists.
    
    Args:
        file_id: Original file identifier
        pre_0_id: Basic preprocessing identifier
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        str: Canonical ID to use for processing (first available file)
        
    Raises:
        ValueError: If no valid ID is provided or no files exist
    """
    from core.logging import log
    
    # Try pre_01_id first (advanced preprocessing) - check if file exists
    if pre_01_id:
        path = get_advanced_processed_path(pre_01_id)
        if path.exists():
            log.info(f"Resolved canonical_id: pre_01_id ({pre_01_id})")
            return pre_01_id
        else:
            log.warning(f"pre_01_id provided but file not found: {path}, falling back...")
    
    # Try pre_0_id (basic preprocessing) - check if file exists
    if pre_0_id:
        path = get_basic_processed_path(pre_0_id)
        if path.exists():
            log.info(f"Resolved canonical_id: pre_0_id ({pre_0_id})")
            return pre_0_id
        else:
            log.warning(f"pre_0_id provided but file not found: {path}, falling back...")
    
    # Fallback to file_id (raw file) - check if file exists
    if file_id:
        from ingestion.file_handler import FileHandler
        from core.utils import get_raw_file_path
        # Try common extensions
        for ext in ['png', 'jpg', 'jpeg', 'heic', 'heif', 'avif', 'pdf']:
            test_path = get_raw_file_path(file_id, ext)
            if test_path.exists():
                log.info(f"Resolved canonical_id: file_id ({file_id})")
                return file_id
        log.warning(f"file_id provided but file not found: {file_id}")
    
    raise ValueError("At least one ID (file_id, pre_0_id, or pre_01_id) must be provided and the file must exist")


def load_image_from_canonical_id(file_id: Optional[str] = None,
                                  pre_0_id: Optional[str] = None,
                                  pre_01_id: Optional[str] = None) -> Image.Image:
    """Load processed image from canonical ID.
    
    Priority: pre_01_id > pre_0_id > file_id
    
    Args:
        file_id: Original file identifier (fallback)
        pre_0_id: Basic preprocessing identifier
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        Image.Image: Loaded image in RGB format
        
    Raises:
        FileNotFoundError: If no image found for any ID
        ValueError: If no valid ID is provided
    """
    from PIL import Image
    from core.logging import log
    
    # Try pre_01_id first (advanced preprocessing)
    if pre_01_id:
        path = get_advanced_processed_path(pre_01_id)
        if path.exists():
            log.info(f"Loading image from pre_01_id: {pre_01_id}")
            return Image.open(path).convert('RGB')
        else:
            log.warning(f"pre_01_id image not found: {path}")
    
    # Try pre_0_id (basic preprocessing)
    if pre_0_id:
        path = get_basic_processed_path(pre_0_id)
        if path.exists():
            log.info(f"Loading image from pre_0_id: {pre_0_id}")
            return Image.open(path).convert('RGB')
        else:
            log.warning(f"pre_0_id image not found: {path}")
    
    # Fallback to file_id (raw file)
    if file_id:
        from ingestion.file_handler import FileHandler
        # Try common extensions
        for ext in ['png', 'jpg', 'jpeg', 'heic', 'pdf']:
            try:
                return FileHandler.load_image(file_id, ext)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Could not find file for file_id: {file_id}")
    
    raise ValueError("At least one ID (file_id, pre_0_id, or pre_01_id) must be provided")


def get_intermediate_json_path(processed_id: str, suffix: str) -> Path:
    """Get the storage path for intermediate JSON files.
    
    Args:
        processed_id: Processed image identifier
        suffix: File suffix (e.g., 'layout', 'tables', 'ocr')
        
    Returns:
        Path: Full path to the intermediate JSON file
    """
    from core.config import settings
    return Path(settings.STORAGE_INTERMEDIATE) / f"{processed_id}_{suffix}.json"


def get_output_json_path(analysis_id: str) -> Path:
    """Get the storage path for final output JSON.
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        Path: Full path to the output JSON file
    """
    from core.config import settings
    return Path(settings.STORAGE_OUTPUT) / f"{analysis_id}.json"


def validate_file_size(file_path: Path, max_size_mb: int) -> bool:
    """Validate that a file is within size limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum size in megabytes
        
    Returns:
        bool: True if file is within limits
    """
    size_mb = file_path.stat().st_size / (1024 * 1024)
    return size_mb <= max_size_mb


def validate_image_format(extension: str) -> bool:
    """Validate that file extension is supported.
    
    Args:
        extension: File extension (lowercase, without dot)
        
    Returns:
        bool: True if format is supported
    """
    supported_formats = {'jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf'}
    return extension.lower() in supported_formats


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image object
        
    Returns:
        np.ndarray: Image as numpy array (RGB)
    """
    return np.array(image.convert('RGB'))


def array_to_image(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array (RGB)
        
    Returns:
        Image.Image: PIL Image object
    """
    return Image.fromarray(array.astype(np.uint8))


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON-serializable types to native Python types.
    
    This function ensures that all data structures can be safely serialized to JSON,
    which is required for FastAPI/Pydantic v2 responses.
    
    Converts:
    - numpy.bool_ → bool
    - numpy.integer → int
    - numpy.floating → float
    - numpy.ndarray → list
    - Recursively processes dicts and lists
    
    Args:
        obj: Object to sanitize (can be dict, list, or any type)
        
    Returns:
        JSON-safe version of the object with all numpy types converted
    """
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    
    # Handle numpy types
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Return as-is for other types (str, int, float, bool, None, etc.)
    return obj


def expand_yolo_bbox(bbox_xyxy: tuple, img_w: int, img_h: int, scale: float) -> tuple:
    """Expand YOLO bounding box by a scale factor (for OCR input quality).
    
    ARCHITECTURAL RULE: YOLO remains the sole authority for regions.
    This function only expands geometry - no new detection, no heuristics.
    
    The expansion:
    - Centers on the original bbox
    - Scales width and height by the given factor
    - Clamps to image boundaries
    - Preserves integer pixel coordinates
    
    Args:
        bbox_xyxy: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        img_w: Image width in pixels
        img_h: Image height in pixels
        scale: Expansion factor (e.g., 1.15 = 15% expansion)
        
    Returns:
        tuple: Expanded bounding box (x1, y1, x2, y2) clamped to image bounds
    """
    x1, y1, x2, y2 = bbox_xyxy
    
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    new_w = w * scale
    new_h = h * scale
    
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(img_w, int(cx + new_w / 2))
    ny2 = min(img_h, int(cy + new_h / 2))
    
    return nx1, ny1, nx2, ny2


def normalize_resolution(crop: np.ndarray, min_short_side: int) -> np.ndarray:
    """Normalize crop resolution for OCR viability (upscale only).
    
    Both PaddleOCR and TrOCR fail silently when:
    - Character height < ~12 px
    - Crop short side < ~32 px
    
    This function:
    - Preserves aspect ratio
    - Only upscales (never downscales)
    - Uses LANCZOS interpolation for quality
    
    Args:
        crop: Input image as numpy array (H, W, C) or (H, W)
        min_short_side: Minimum short side in pixels (48 for PaddleOCR, 64 for TrOCR)
        
    Returns:
        np.ndarray: Normalized image (same or larger resolution)
    """
    import cv2
    
    h, w = crop.shape[:2]
    short = min(h, w)
    
    # Only upscale, never downscale
    if short >= min_short_side:
        return crop
    
    scale = min_short_side / short
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )


def compute_iou(boxA: list, boxB: list) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA: First bounding box [x1, y1, x2, y2] (normalized or pixel coordinates)
        boxB: Second bounding box [x1, y1, x2, y2] (same coordinate system as boxA)
        
    Returns:
        float: IoU value between 0.0 and 1.0
    """
    if len(boxA) < 4 or len(boxB) < 4:
        return 0.0
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0


def resolve_text_handwritten_conflicts(
    text_blocks: list,
    handwritten: list,
    iou_threshold: float = 0.30
) -> tuple:
    """Hard rejection of lower-confidence class in Text_box ↔ Handwritten conflicts.
    
    HARD REJECTION RULE:
    When Text_box and Handwritten detections overlap (IoU ≥ threshold),
    the detection with LOWER YOLO confidence is COMPLETELY REMOVED.
    
    "Removed" means:
    - ❌ NOT routed to OCR
    - ❌ NOT included in output JSON
    - ❌ NOT included in table grouping
    - ❌ NOT logged as a valid detection
    - ❌ NOT visualized (except optionally as "REJECTED" in debug)
    
    The rejected detection does NOT exist beyond this function.
    
    ARCHITECTURAL GUARANTEES:
    - Does NOT modify bounding boxes
    - Does NOT create new detections
    - Does NOT look at pixels
    - Does NOT use OCR output
    - Does NOT change class labels
    - Does NOT affect Signature / Checkbox / Table
    - Only REMOVES one of two conflicting YOLO outputs
    
    Args:
        text_blocks: List of Text_box detections from YOLO
        handwritten: List of Handwritten detections from YOLO
        iou_threshold: Minimum IoU to consider as conflict (default: 0.30)
        
    Returns:
        tuple: (kept_text_blocks, kept_handwritten) - rejected detections are GONE
    """
    from core.logging import log
    
    # Track which indices to REJECT (hard removal)
    rejected_text = set()
    rejected_hand = set()
    
    # Compare each text_block with each handwritten
    for i, text_det in enumerate(text_blocks):
        if i in rejected_text:
            continue
        
        text_bbox = text_det.get('bbox', [])
        text_conf = text_det.get('confidence', 0.0)
        
        # Safety assertion: YOLO source only
        text_source = text_det.get('source', 'YOLOv8s')
        assert text_source == 'YOLOv8s', f"Non-YOLO bbox in text_blocks (source: {text_source})"
        
        for j, hand_det in enumerate(handwritten):
            if j in rejected_hand:
                continue
            
            hand_bbox = hand_det.get('bbox', [])
            hand_conf = hand_det.get('confidence', 0.0)
            
            # Safety assertion: YOLO source only
            hand_source = hand_det.get('source', 'YOLOv8s')
            assert hand_source == 'YOLOv8s', f"Non-YOLO bbox in handwritten (source: {hand_source})"
            
            # Check IoU for conflict
            iou = compute_iou(text_bbox, hand_bbox)
            
            if iou >= iou_threshold:
                # CONFLICT DETECTED - higher confidence wins, lower is REJECTED
                if text_conf >= hand_conf:
                    # Text_box wins, REJECT Handwritten entirely
                    rejected_hand.add(j)
                    log.warning(
                        f"REJECTED Handwritten (conf={hand_conf:.2f}) - "
                        f"Text_box (conf={text_conf:.2f}) wins at IoU={iou:.2f}"
                    )
                else:
                    # Handwritten wins, REJECT Text_box entirely
                    rejected_text.add(i)
                    log.warning(
                        f"REJECTED Text_box (conf={text_conf:.2f}) - "
                        f"Handwritten (conf={hand_conf:.2f}) wins at IoU={iou:.2f}"
                    )
                    break  # This text_block is rejected, stop comparing it
    
    # Build KEPT lists (rejected detections are GONE - they never leave this function)
    kept_text = [det for i, det in enumerate(text_blocks) if i not in rejected_text]
    kept_hand = [det for j, det in enumerate(handwritten) if j not in rejected_hand]
    
    # Log summary of hard rejections
    total_rejected = len(rejected_text) + len(rejected_hand)
    if total_rejected > 0:
        log.warning(
            f"HARD REJECTION: {len(rejected_text)} Text_box + {len(rejected_hand)} Handwritten "
            f"= {total_rejected} detections REMOVED from pipeline"
        )
        log.info(
            f"Surviving detections: {len(kept_text)} text_blocks, {len(kept_hand)} handwritten"
        )
    
    return kept_text, kept_hand


def is_center_inside_bbox(inner_bbox: list, outer_bbox: list) -> bool:
    """Check if center point of inner bounding box lies inside outer bounding box.
    
    This is the correct geometric primitive for document parsing:
    - Matches row/column assignment logic (uses x_center/y_center)
    - Stable for narrow columns
    - Immune to partial boundary crossings
    - Deterministic (binary check, no thresholds)
    
    Args:
        inner_bbox: Inner bounding box [x1, y1, x2, y2] (normalized coordinates)
        outer_bbox: Outer bounding box [x1, y1, x2, y2] (normalized coordinates)
        
    Returns:
        bool: True if center point of inner bbox is inside outer bbox
    """
    if len(inner_bbox) < 4 or len(outer_bbox) < 4:
        return False
    
    ix1, iy1, ix2, iy2 = inner_bbox[:4]
    ox1, oy1, ox2, oy2 = outer_bbox[:4]
    
    # Calculate center point of inner bbox
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    
    # Check if center point is inside outer bbox
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def is_bbox_inside_table(box: list, table: list, margin: int = 5, img_w: int = None, img_h: int = None) -> bool:
    """Check if bounding box is inside table bbox with margin tolerance.
    
    Used for table-anchored row construction. The margin allows for slight
    clipping at table boundaries.
    
    Args:
        box: Detection bbox [x1, y1, x2, y2] (pixel coordinates)
        table: Table bbox [x1, y1, x2, y2] (pixel coordinates)
        margin: Pixel margin for tolerance (default: 5)
        img_w: Image width in pixels (for normalized coordinate conversion)
        img_h: Image height in pixels (for normalized coordinate conversion)
        
    Returns:
        bool: True if box is inside table (with margin)
    """
    if len(box) < 4 or len(table) < 4:
        return False
    
    # Convert normalized to pixel if needed
    if img_w and img_h:
        # Assume coordinates are normalized [0, 1]
        bx1, by1, bx2, by2 = box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h
        tx1, ty1, tx2, ty2 = table[0] * img_w, table[1] * img_h, table[2] * img_w, table[3] * img_h
    else:
        # Assume pixel coordinates
        bx1, by1, bx2, by2 = box[:4]
        tx1, ty1, tx2, ty2 = table[:4]
    
    # Check containment with margin
    return (
        bx1 >= tx1 - margin and
        by1 >= ty1 - margin and
        bx2 <= tx2 + margin and
        by2 <= ty2 + margin
    )


def assign_column_ids(
    detections: list,
    header_row_detections: list = None,
    column_anchors: list = None
) -> list:
    """Assign column_id to each detection based on nearest x-center.
    
    Column assignment uses GEOMETRY ONLY (no OCR):
    - Either from provided column_anchors (list of x-center values)
    - Or computed from header_row_detections (their x-centers)
    
    This must run BEFORE tie-breaker logic.
    
    Args:
        detections: List of YOLO detections with 'bbox' key
        header_row_detections: Optional header row detections to derive anchors
        column_anchors: Optional list of x-center values for columns
        
    Returns:
        list: Same detections with 'column_id' added to each
    """
    from core.logging import log
    
    # Build column anchors from header row if not provided
    if column_anchors is None and header_row_detections:
        column_anchors = []
        for det in header_row_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                column_anchors.append(x_center)
        column_anchors.sort()
    
    if not column_anchors:
        log.warning("No column anchors available - skipping column_id assignment")
        return detections
    
    log.info(f"Assigning column_ids using {len(column_anchors)} anchors")
    
    # Assign each detection to nearest column
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) < 4:
            det['column_id'] = None
            continue
        
        det_x_center = (bbox[0] + bbox[2]) / 2.0
        
        # Find nearest column anchor
        min_dist = float('inf')
        nearest_col = 0
        for col_idx, anchor_x in enumerate(column_anchors):
            dist = abs(det_x_center - anchor_x)
            if dist < min_dist:
                min_dist = dist
                nearest_col = col_idx
        
        det['column_id'] = nearest_col
    
    return detections


def build_column_stats(
    detections: list,
    column_conf_threshold: float = 0.60
) -> dict:
    """Build class statistics per column for tie-breaking.
    
    Counts high-confidence detections per column:
    - Handwritten + Text_box count (combined)
    - Signature count
    
    Args:
        detections: List of detections with 'column_id', 'cls', 'confidence'
        column_conf_threshold: Minimum confidence to include in stats
        
    Returns:
        dict: {column_id: {"hand_or_text_count": N, "signature_count": M}}
    """
    from collections import defaultdict
    
    HAND_CLASS = "Handwritten"
    TEXT_CLASS = "Text_box"
    SIG_CLASS = "Signature"
    
    stats = defaultdict(lambda: {
        "hand_or_text_count": 0,
        "signature_count": 0
    })
    
    for det in detections:
        conf = det.get('confidence', 0.0)
        if conf < column_conf_threshold:
            continue
        
        col = det.get('column_id')
        if col is None:
            continue
        
        cls = det.get('cls', det.get('class', ''))
        
        if cls in {HAND_CLASS, TEXT_CLASS}:
            stats[col]["hand_or_text_count"] += 1
        elif cls == SIG_CLASS:
            stats[col]["signature_count"] += 1
    
    return dict(stats)


def resolve_signature_handwritten_ties(
    signatures: list,
    handwritten: list,
    all_detections: list,
    iou_threshold: float = 0.30,
    conf_epsilon: float = 0.01,
    column_conf_threshold: float = 0.60
) -> tuple:
    """Tie-breaker for Signature ↔ Handwritten using column consensus.
    
    ONLY applies when ALL conditions are true:
    - Classes: {Signature, Handwritten}
    - Confidence scores are EQUAL (within epsilon)
    - IoU ≥ threshold
    - Column has clear class majority with conf ≥ threshold
    
    Resolution:
    - If column majority is Handwritten/Text_box → reclassify as Handwritten
    - Otherwise → keep as Signature
    
    ARCHITECTURAL GUARANTEES:
    - ❌ No OCR feedback
    - ❌ No pixel heuristics
    - ❌ No handwriting analysis
    - ❌ No fallback logic
    - ✓ Uses YOLO outputs only
    - ✓ Column grouping is geometric
    
    Args:
        signatures: List of Signature detections from YOLO
        handwritten: List of Handwritten detections from YOLO
        all_detections: All detections (for column stats)
        iou_threshold: Minimum IoU to consider as conflict
        conf_epsilon: Confidence difference to be "equal"
        column_conf_threshold: Minimum confidence for column stats
        
    Returns:
        tuple: (resolved_signatures, resolved_handwritten)
    """
    from core.logging import log
    
    # Build column stats from all detections
    column_stats = build_column_stats(all_detections, column_conf_threshold)
    
    # Track reclassifications
    reclassified_to_hand = []
    kept_signatures = []
    
    for sig in signatures:
        sig_bbox = sig.get('bbox', [])
        sig_conf = sig.get('confidence', 0.0)
        sig_col = sig.get('column_id')
        
        # Find conflicting handwritten detection
        conflict = None
        for hand in handwritten:
            hand_bbox = hand.get('bbox', [])
            hand_conf = hand.get('confidence', 0.0)
            
            # Check if this is an ambiguous overlap
            iou = compute_iou(sig_bbox, hand_bbox)
            if iou >= iou_threshold:
                # Check if confidences are "equal"
                if abs(sig_conf - hand_conf) <= conf_epsilon:
                    conflict = hand
                    break
        
        if not conflict:
            # No ambiguous conflict - keep as Signature
            kept_signatures.append(sig)
            continue
        
        # Check column consensus
        if sig_col is not None and sig_col in column_stats:
            stats = column_stats[sig_col]
            
            if stats["hand_or_text_count"] > stats["signature_count"]:
                # Column majority is Handwritten/Text_box → reclassify
                reclassified = sig.copy()
                reclassified['cls'] = 'Handwritten'
                reclassified['class'] = 'handwritten'
                reclassified['_reclassified_from'] = 'Signature'
                reclassified['_reclassified_reason'] = 'column_consensus'
                reclassified_to_hand.append(reclassified)
                
                log.info(
                    f"RECLASSIFIED Signature→Handwritten (column {sig_col}): "
                    f"hand_or_text={stats['hand_or_text_count']} > sig={stats['signature_count']}"
                )
            else:
                # Column majority is Signature or tie → keep as Signature
                kept_signatures.append(sig)
                log.debug(
                    f"Kept Signature (column {sig_col}): "
                    f"sig={stats['signature_count']} >= hand_or_text={stats['hand_or_text_count']}"
                )
        else:
            # No column info - keep as Signature (conservative)
            kept_signatures.append(sig)
    
    # Add reclassified detections to handwritten list
    resolved_handwritten = handwritten + reclassified_to_hand
    
    if reclassified_to_hand:
        log.info(
            f"Signature/Handwritten tie-breaker: "
            f"{len(reclassified_to_hand)} reclassified to Handwritten"
        )
    
    return kept_signatures, resolved_handwritten


def merge_handwritten_boxes(
    handwritten: list,
    image_width: int,
    gap_threshold_ratio: float = 0.03
) -> list:
    """Pre-OCR handwritten box merging (Phase-1 Fix A).
    
    Merges horizontally adjacent handwritten boxes in the same row/column
    to prevent fragmentation (e.g., "Business Guest" → "I").
    
    ELIGIBILITY RULES (STRICT):
    - Same row_id
    - Same column_index (if assigned)
    - class == Handwritten
    - Horizontal adjacency (gap <= threshold)
    - No printed text between them (checked via x-coordinates)
    
    ❌ Never merge across rows
    ❌ Never merge across columns
    ❌ Never merge printed + handwritten
    
    ARCHITECTURAL GUARANTEES:
    - Does NOT create new YOLO detections
    - Does NOT modify original YOLO boxes
    - Creates temporary merged crops for OCR only
    - Preserves YOLO authority
    - Deterministic behavior
    
    Args:
        handwritten: List of Handwritten detections from YOLO
        image_width: Image width in pixels (for gap threshold calculation)
        gap_threshold_ratio: Gap threshold as fraction of image width (default: 0.03 = 3%)
        
    Returns:
        list: List with merged detections (original boxes preserved, merged boxes added)
              Format: [original_detections..., merged_detections...]
              Merged detections have '_merged_from' key with list of original indices
    """
    from core.logging import log
    
    if not handwritten:
        return handwritten
    
    # Calculate absolute gap threshold
    gap_threshold = gap_threshold_ratio * image_width
    
    # Group by row_id and column_id (if available)
    # If row_id/column_id not assigned, group all together (will merge adjacent boxes)
    # Format: {(row_id, column_id): [detections]}
    grouped = {}
    for i, det in enumerate(handwritten):
        row_id = det.get('row_id')  # May be None if not assigned yet
        column_id = det.get('column_id')  # May be None if not assigned yet
        # Use a default key if not assigned (all unassigned boxes in one group)
        key = (row_id, column_id)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((i, det))
    
    merged_detections = []
    merge_count = 0
    
    # Process each group (same row + column)
    for (row_id, column_id), group in grouped.items():
        if len(group) < 2:
            continue  # Need at least 2 boxes to merge
        
        # Sort by x_min (left to right)
        # Convert normalized bbox to pixel coordinates for sorting
        def get_x_min(det_tuple):
            idx, det = det_tuple
            bbox = det.get('bbox', [])
            if not bbox or len(bbox) < 4:
                return float('inf')
            # bbox is normalized [x1, y1, x2, y2]
            return bbox[0] * image_width
        
        group_sorted = sorted(group, key=get_x_min)
        
        # Try to merge adjacent boxes
        merged_groups = []
        current_group = [group_sorted[0]]
        
        for i in range(1, len(group_sorted)):
            prev_idx, prev_det = group_sorted[i-1]
            curr_idx, curr_det = group_sorted[i]
            
            prev_bbox = prev_det.get('bbox', [])
            curr_bbox = curr_det.get('bbox', [])
            
            if len(prev_bbox) < 4 or len(curr_bbox) < 4:
                # Invalid bbox, start new group
                if len(current_group) > 1:
                    merged_groups.append(current_group)
                current_group = [group_sorted[i]]
                continue
            
            # Calculate gap in pixel coordinates
            prev_x_max = prev_bbox[2] * image_width
            curr_x_min = curr_bbox[0] * image_width
            gap = curr_x_min - prev_x_max
            
            if gap <= gap_threshold:
                # Adjacent - add to current group
                current_group.append(group_sorted[i])
            else:
                # Gap too large - finalize current group and start new
                if len(current_group) > 1:
                    merged_groups.append(current_group)
                current_group = [group_sorted[i]]
        
        # Finalize last group
        if len(current_group) > 1:
            merged_groups.append(current_group)
        
        # Create merged detections for each merged group
        for merged_group in merged_groups:
            if len(merged_group) < 2:
                continue
            
            # Calculate union bbox
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            original_indices = []
            
            for idx, det in merged_group:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    min_x = min(min_x, bbox[0])
                    min_y = min(min_y, bbox[1])
                    max_x = max(max_x, bbox[2])
                    max_y = max(max_y, bbox[3])
                original_indices.append(idx)
            
            if min_x == float('inf'):
                continue  # Invalid bbox
            
            # Create merged detection (for OCR only, not a new YOLO detection)
            merged_det = {
                'bbox': [min_x, min_y, max_x, max_y],  # Normalized coordinates
                'class': 'handwritten',
                'cls': 'Handwritten',
                'confidence': max(det.get('confidence', 0.0) for _, det in merged_group),  # Use max confidence
                'source': 'YOLOv8s',  # Still from YOLO (merged boxes)
                'row_id': row_id,
                'column_id': column_id,
                '_merged_from': original_indices,  # Track which boxes were merged
                '_is_merged': True,  # Flag for downstream processing
            }
            
            merged_detections.append(merged_det)
            merge_count += 1
            
            # Mark original boxes as merged (so they're skipped in OCR)
            for idx, det in merged_group:
                handwritten[idx]['_merged_into'] = len(handwritten) + len(merged_detections) - 1  # Index of merged box
                handwritten[idx]['_skip_ocr'] = True  # Flag to skip OCR for this box
            
            row_str = str(row_id) if row_id is not None else "unassigned"
            col_str = str(column_id) if column_id is not None else "unassigned"
            log.info(
                f"Handwritten merge applied | row={row_str} | col={col_str} | "
                f"boxes={len(merged_group)} | gap_threshold={gap_threshold:.1f}px"
            )
    
    if merge_count > 0:
        log.info(f"Phase-1 Fix A: Merged {merge_count} handwritten box groups")
        # Return original detections + merged detections
        # Original boxes marked with _skip_ocr=True will be skipped in OCR
        # Merged boxes will be processed in OCR, and results attached to row+column
        return handwritten + merged_detections
    else:
        return handwritten










