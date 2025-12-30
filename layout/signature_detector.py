"""GPDS signature detector for handwritten signatures.

Uses pretrained GPDS signature detection model.
No training required - uses pretrained weights.
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

# Optional YOLOv5 import (common for GPDS models)
try:
    import torch
    # Try to import YOLOv5 (if available)
    try:
        import yolov5
        YOLOV5_AVAILABLE = True
    except ImportError:
        YOLOV5_AVAILABLE = False
    
    # Alternative: Try ultralytics YOLOv8 (can load YOLOv5 weights)
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    ULTRALYTICS_AVAILABLE = False
    log.warning("PyTorch not available. GPDS signature detection will be unavailable.")


class SignatureDetector:
    """GPDS signature detector for handwritten signatures."""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """Initialize GPDS signature detector.
        
        Args:
            model_path: Path to GPDS signature model weights
                       If None, will look for default GPDS model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if not ULTRALYTICS_AVAILABLE:
            log.warning("PyTorch/Ultralytics not available. GPDS signature detection will be unavailable.")
            return
        
        try:
            # Try to load GPDS model
            # Note: GPDS models are typically YOLOv5 or YOLOv8 format
            # User needs to download and place model file
            default_model_path = Path(settings.MODELS_ROOT) / "signature_model" / "gpds_signature.pt"
            
            if model_path:
                model_path = Path(model_path)
            elif default_model_path.exists():
                model_path = default_model_path
            else:
                log.warning(f"GPDS signature model not found at: {default_model_path}")
                log.warning("GPDS signature detection will be unavailable.")
                log.warning("Download GPDS signature model and place at: models/signature_model/gpds_signature.pt")
                return
            
            if model_path.exists():
                log.info(f"Loading GPDS signature model from: {model_path}")
                # Use ultralytics YOLO (can load YOLOv5/YOLOv8 weights)
                self.model = YOLO(str(model_path))
                log.info("GPDS signature model loaded successfully")
            else:
                log.warning(f"GPDS signature model not found: {model_path}")
                
        except Exception as e:
            log.error(f"Failed to load GPDS signature model: {str(e)}")
            self.model = None
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect signatures in image.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            list: List of signature detections with 'bbox', 'confidence', 'type', 'source'
        """
        if not self.model:
            log.warning("GPDS signature model not available, returning empty detections")
            return []
        
        try:
            # Convert PIL to numpy array
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model(img_bgr, conf=self.confidence_threshold, verbose=False)
            
            signatures = []
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
                    
                    signatures.append({
                        'bbox': bbox_normalized,
                        'confidence': confidence,
                        'type': 'signature',
                        'source': 'GPDS'
                    })
            
            log.info(f"GPDS detected {len(signatures)} signatures")
            return signatures
            
        except Exception as e:
            log.error(f"GPDS signature detection failed: {str(e)}", exc_info=True)
            return []









