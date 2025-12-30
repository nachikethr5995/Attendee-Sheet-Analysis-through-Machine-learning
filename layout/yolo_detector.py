"""YOLOv8 layout detection for tables, text blocks, signatures, and checkboxes."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import image_to_array, resolve_canonical_id, load_image_from_canonical_id

# Optional YOLOv8 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("ultralytics not installed. YOLOv8 detection will be unavailable.")


class YOLODetector:
    """YOLOv8-based layout detection for document elements.
    
    Note: By default, uses YOLOv8n (COCO dataset) which detects general objects.
    For production, a custom-trained layout detection model should be used.
    The class mapping below is a placeholder for when a custom model is available.
    """
    
    # Detection classes (for custom layout model)
    # Note: Default YOLOv8 uses COCO classes (80 classes), not these
    CLASS_NAMES = {
        0: 'table',
        1: 'text_block',
        2: 'signature',
        3: 'checkbox'
    }
    
    # COCO class mapping (for default YOLOv8 model)
    # Map COCO classes to layout classes (heuristic mapping)
    COCO_TO_LAYOUT = {
        # COCO doesn't have direct layout classes, so we'll use general objects
        # This is a placeholder - custom model needed for proper layout detection
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLOv8 weights file (optional, will download if not provided)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if not YOLO_AVAILABLE:
            log.warning("YOLOv8 not available. Install with: pip install ultralytics")
            return
        
        try:
            if model_path and Path(model_path).exists():
                log.info(f"Loading YOLO model from: {model_path}")
                self.model = YOLO(model_path)
            else:
                # Use default YOLOv8 model (will download if needed)
                log.info("Loading default YOLOv8 model (will download if needed)...")
                self.model = YOLO('yolov8n.pt')  # nano model for speed
                log.info("YOLOv8 model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
    
    def detect(self, image: Image.Image) -> Dict[str, List[Dict[str, Any]]]:
        """Detect layout elements in image.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            dict: Detection results with keys: 'tables', 'text_blocks', 'signatures', 'checkboxes'
            Each list contains dicts with 'bbox', 'confidence', 'class'
        """
        if not self.model:
            log.warning("YOLO model not available, returning empty detections")
            return {
                'tables': [],
                'text_blocks': [],
                'signatures': [],
                'checkboxes': []
            }
        
        try:
            # Convert PIL to numpy array
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run YOLO inference
            results = self.model(img_bgr, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            detections = {
                'tables': [],
                'text_blocks': [],
                'signatures': [],
                'checkboxes': []
            }
            
            if results and len(results) > 0:
                result = results[0]
                width, height = image.size
                
                # Process each detection
                # Note: Default YOLOv8 uses COCO dataset (80 classes: person, car, etc.)
                # For proper layout detection, a custom-trained model is needed
                # For now, we'll use all detections as potential text blocks
                # and note that custom model training is required
                
                for box in result.boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Normalize coordinates to 0-1 range
                    bbox_normalized = [
                        float(x1 / width),
                        float(y1 / height),
                        float(x2 / width),
                        float(y2 / height)
                    ]
                    
                    # Get COCO class name
                    coco_class_name = result.names.get(class_id, 'unknown')
                    
                    # For now, treat all detections as text blocks
                    # TODO: Train custom YOLOv8 model for layout detection (tables, signatures, checkboxes)
                    # Custom model should output: table, text_block, signature, checkbox classes
                    class_name = 'text_block'
                    
                    # Add to appropriate list
                    detection = {
                        'bbox': bbox_normalized,
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id,
                        'coco_class': coco_class_name,  # Keep original for reference
                        'note': 'Using default YOLOv8 (COCO). Custom layout model recommended.'
                    }
                    
                    if class_name in detections:
                        detections[class_name].append(detection)
            
            log.info(f"YOLO detected: {len(detections['tables'])} tables, "
                    f"{len(detections['text_blocks'])} text blocks, "
                    f"{len(detections['signatures'])} signatures, "
                    f"{len(detections['checkboxes'])} checkboxes")
            
            return detections
            
        except Exception as e:
            log.error(f"YOLO detection failed: {str(e)}", exc_info=True)
            return {
                'tables': [],
                'text_blocks': [],
                'signatures': [],
                'checkboxes': []
            }
    
    def detect_failure(self, detections: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, str]:
        """Detect if layout detection failed (for SERVICE 0.1 trigger).
        
        Args:
            detections: Detection results from detect()
            
        Returns:
            Tuple[bool, str]: (has_failed, reason)
        """
        total_detections = sum(len(detections[key]) for key in detections)
        
        # Failure conditions:
        # 1. No detections at all
        if total_detections == 0:
            return True, "No layout elements detected"
        
        # 2. Very low confidence (all detections below threshold)
        all_confidences = []
        for key in detections:
            for det in detections[key]:
                all_confidences.append(det.get('confidence', 0.0))
        
        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            if avg_confidence < 0.3:  # Very low average confidence
                return True, f"Low average confidence: {avg_confidence:.2f}"
        
        # 3. No tables detected but image likely contains tables (heuristic)
        # This is a placeholder - in production, you might use image analysis
        # to determine if tables should be present
        
        return False, "Detection successful"









