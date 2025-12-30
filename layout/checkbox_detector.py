"""Checkbox detector using custom trained tiny YOLOv8 model.

Requires training on 20-50 checkbox images.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import image_to_array

# Optional YOLOv8 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("ultralytics not installed. Checkbox detection will be unavailable.")


class CheckboxDetector:
    """Tiny YOLOv8-based checkbox detector."""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.4):
        """Initialize checkbox detector.
        
        Args:
            model_path: Path to trained checkbox model weights
                       If None, will look for default checkbox model
            confidence_threshold: Minimum confidence for detections (default 0.4 for checkboxes)
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if not YOLO_AVAILABLE:
            log.warning("YOLOv8 not available. Checkbox detection will be unavailable.")
            return
        
        try:
            # Try to load trained checkbox model
            default_model_path = Path(settings.MODELS_ROOT) / "checkbox_model" / "checkbox_best.pt"
            
            if model_path:
                model_path = Path(model_path)
            elif default_model_path.exists():
                model_path = default_model_path
            else:
                log.warning(f"Checkbox model not found at: {default_model_path}")
                log.warning("Checkbox detection will be unavailable.")
                log.warning("Train checkbox model using: python train_checkbox_model.py")
                log.warning("Or place trained model at: models/checkbox_model/checkbox_best.pt")
                return
            
            if model_path.exists():
                log.info(f"Loading checkbox model from: {model_path}")
                self.model = YOLO(str(model_path))
                log.info("Checkbox model loaded successfully")
            else:
                log.warning(f"Checkbox model not found: {model_path}")
                
        except Exception as e:
            log.error(f"Failed to load checkbox model: {str(e)}")
            self.model = None
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect checkboxes in image.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            list: List of checkbox detections with 'bbox', 'confidence', 'type', 'source'
        """
        if not self.model:
            log.warning("Checkbox model not available, returning empty detections")
            return []
        
        try:
            # Convert PIL to numpy array
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model(img_bgr, conf=self.confidence_threshold, verbose=False)
            
            checkboxes = []
            width, height = image.size
            
            if results and len(results) > 0:
                result = results[0]
                
                # Process each detection
                for box in result.boxes:
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
                    
                    checkboxes.append({
                        'bbox': bbox_normalized,
                        'confidence': confidence,
                        'type': 'checkbox',
                        'source': 'checkbox_detector'
                    })
            
            log.info(f"Checkbox detector found {len(checkboxes)} checkboxes")
            return checkboxes
            
        except Exception as e:
            log.error(f"Checkbox detection failed: {str(e)}", exc_info=True)
            return []









