"""Debug endpoints for observability (not part of production pipeline).

These endpoints are read-only and have no side effects.
They are designed for field debugging and diagnosis.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import io
import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import Response
from PIL import Image
from core.logging import log
from core.config import settings

# Import YOLO detector (same model, same weights as pipeline)
from layout.yolov8_layout_detector import YOLOv8LayoutDetector

router = APIRouter(prefix="/debug", tags=["Debug"])

# Class → Color mapping for visualization
# BGR format for OpenCV
CLASS_COLORS = {
    "text_box": (0, 255, 0),        # Green
    "text_block": (0, 255, 0),      # Green (alias)
    "Text_box": (0, 255, 0),        # Green (model name)
    "handwritten": (0, 165, 255),   # Orange (BGR)
    "Handwritten": (0, 165, 255),   # Orange (model name)
    "signature": (0, 0, 255),       # Red
    "Signature": (0, 0, 255),       # Red (model name)
    "checkbox": (255, 0, 0),        # Blue
    "Checkbox": (255, 0, 0),        # Blue (model name)
    "table": (128, 0, 128),         # Purple
    "Table": (128, 0, 128),         # Purple (model name)
}

# Lazy-loaded YOLO detector (same as pipeline)
_yolo_detector: Optional[YOLOv8LayoutDetector] = None


def get_yolo_detector() -> YOLOv8LayoutDetector:
    """Get or initialize the YOLO detector (singleton, same as pipeline)."""
    global _yolo_detector
    if _yolo_detector is None:
        log.info("Initializing YOLO detector for debug visualization...")
        _yolo_detector = YOLOv8LayoutDetector(
            confidence_threshold=0.01,  # Same as pipeline
            iou_threshold=0.7,
            use_gpu=settings.USE_GPU
        )
    return _yolo_detector


def draw_yolo_detections(
    image: np.ndarray,
    yolo_results: dict,
    conf_threshold: float = 0.0,
    show_labels: bool = True
) -> np.ndarray:
    """Draw YOLO detections on image.
    
    This is a pure visualization function - no OCR, no bbox expansion.
    
    Args:
        image: Input image as numpy array (BGR)
        yolo_results: YOLO detection results dict with 'tables', 'text_blocks', etc.
        conf_threshold: Minimum confidence to display
        show_labels: Whether to show class labels
        
    Returns:
        np.ndarray: Image with drawn bounding boxes
    """
    img = image.copy()
    height, width = img.shape[:2]
    
    # Collect all detections from all classes
    all_detections = []
    for class_key in ['tables', 'text_blocks', 'signatures', 'checkboxes', 'handwritten']:
        for det in yolo_results.get(class_key, []):
            det['_display_class'] = class_key.rstrip('s')  # Remove plural
            all_detections.append(det)
    
    # Draw each detection
    for det in all_detections:
        conf = det.get('confidence', 0.0)
        if conf < conf_threshold:
            continue
        
        bbox = det.get('bbox', [])
        if not bbox or len(bbox) < 4:
            continue
        
        # Convert normalized bbox to pixel coordinates
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # Get class name and color
        class_name = det.get('class', det.get('_display_class', 'unknown'))
        color = CLASS_COLORS.get(class_name, (255, 255, 255))  # White fallback
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            label = f"{class_name} {conf:.2f}"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            label_y = max(y1 - 5, text_height + 5)
            cv2.rectangle(
                img,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y + 5),
                color,
                -1  # Filled
            )
            
            # Draw text (white on colored background)
            cv2.putText(
                img,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
    
    return img


@router.post("/yolo-visualize", 
             summary="Visualize YOLO detections",
             description="""
Debug endpoint to visualize what YOLO actually detects on an image.

**This endpoint is read-only observability, not part of the pipeline.**

Returns the input image with:
- YOLO-detected bounding boxes
- Colored by class (green=text, orange=handwritten, red=signature, blue=checkbox, purple=table)
- Annotated with class name + confidence

**Architectural Guardrails:**
- Same YOLO model and weights as pipeline
- No OCR invoked
- No bbox expansion applied
- No preprocessing
- No side effects
""",
             responses={
                 200: {
                     "content": {"image/png": {}},
                     "description": "Annotated image with YOLO detections"
                 }
             })
async def yolo_visualize(
    file: UploadFile = File(..., description="Image file to analyze"),
    conf_threshold: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0 = show all)"
    ),
    show_labels: bool = Query(
        default=True,
        description="Show class labels and confidence scores"
    )
):
    """Visualize YOLO detections on uploaded image.
    
    This endpoint answers one question only:
    "What did YOLO actually detect on this image?"
    """
    log.info(f"Debug: YOLO visualization requested for {file.filename}")
    
    # Read uploaded image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        log.error(f"Failed to decode image: {file.filename}")
        return Response(
            content=b"Failed to decode image",
            status_code=400,
            media_type="text/plain"
        )
    
    # Convert BGR to RGB for PIL (YOLO expects PIL Image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run YOLO inference (SAME model, SAME weights as pipeline)
    detector = get_yolo_detector()
    if not detector.is_available():
        log.error("YOLO detector not available")
        return Response(
            content=b"YOLO detector not available",
            status_code=500,
            media_type="text/plain"
        )
    
    log.info(f"Running YOLO inference on {pil_image.size[0]}x{pil_image.size[1]} image...")
    yolo_results = detector.detect(pil_image)
    
    # Count detections
    total_detections = sum(
        len(yolo_results.get(k, []))
        for k in ['tables', 'text_blocks', 'signatures', 'checkboxes', 'handwritten']
    )
    log.info(f"YOLO detected {total_detections} total regions")
    
    # Draw detections on image (BGR for OpenCV)
    annotated_image = draw_yolo_detections(
        image_bgr,
        yolo_results,
        conf_threshold=conf_threshold,
        show_labels=show_labels
    )
    
    # Encode as PNG
    _, encoded = cv2.imencode(".png", annotated_image)
    
    log.info(f"Debug: Returning annotated image with {total_detections} detections")
    
    return Response(
        content=encoded.tobytes(),
        media_type="image/png"
    )


@router.post("/yolo-visualize-json",
             summary="Get YOLO detections as JSON",
             description="""
Debug endpoint to get raw YOLO detection data as JSON.

Same as /yolo-visualize but returns JSON instead of image.
Useful for programmatic analysis of YOLO output.
""")
async def yolo_visualize_json(
    file: UploadFile = File(..., description="Image file to analyze"),
    conf_threshold: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0 = show all)"
    )
):
    """Get YOLO detections as JSON (raw data, no visualization)."""
    log.info(f"Debug: YOLO JSON requested for {file.filename}")
    
    # Read uploaded image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        return {"error": "Failed to decode image", "filename": file.filename}
    
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run YOLO inference
    detector = get_yolo_detector()
    if not detector.is_available():
        return {"error": "YOLO detector not available"}
    
    yolo_results = detector.detect(pil_image)
    
    # Filter by confidence threshold
    filtered_results = {}
    for class_key in ['tables', 'text_blocks', 'signatures', 'checkboxes', 'handwritten']:
        filtered_results[class_key] = [
            det for det in yolo_results.get(class_key, [])
            if det.get('confidence', 0.0) >= conf_threshold
        ]
    
    # Count detections
    total_detections = sum(len(v) for v in filtered_results.values())
    
    return {
        "filename": file.filename,
        "image_size": {"width": pil_image.size[0], "height": pil_image.size[1]},
        "conf_threshold": conf_threshold,
        "total_detections": total_detections,
        "detections": filtered_results,
        "model_info": detector.get_model_info()
    }




