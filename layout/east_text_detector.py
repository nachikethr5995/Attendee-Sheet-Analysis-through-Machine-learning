"""EAST text detection for fine-grained text region detection."""

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

# EAST model path (OpenCV DNN)
# Default EAST model filename
EAST_MODEL_FILENAME = "frozen_east_text_detection.pb"


class EASTTextDetector:
    """EAST (Efficient and Accurate Scene Text) detector for text regions."""
    
    def __init__(self, model_path: Optional[Path] = None, confidence_threshold: float = 0.5):
        """Initialize EAST detector.
        
        Args:
            model_path: Path to EAST frozen model file (frozen_east_text_detection.pb)
                       If None, will look in settings.EAST_MODEL_PATH
            confidence_threshold: Minimum confidence for text detections
        """
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.model_loaded = False
        
        # Determine model path
        if model_path is None:
            # Try default location
            model_path = Path(settings.EAST_MODEL_PATH) / EAST_MODEL_FILENAME
        elif isinstance(model_path, str):
            model_path = Path(model_path)
        
        # If model_path is a directory, append filename
        if model_path.is_dir():
            model_path = model_path / EAST_MODEL_FILENAME
        
        if model_path.exists():
            try:
                log.info(f"Loading EAST model from: {model_path}")
                self.net = cv2.dnn.readNet(str(model_path))
                
                # Set backend (prefer GPU if available)
                if settings.USE_GPU:
                    try:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        log.info("EAST model using CUDA backend")
                    except:
                        log.warning("CUDA not available, using CPU for EAST")
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                else:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                self.model_loaded = True
                log.info("EAST model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load EAST model: {str(e)}")
                self.net = None
        else:
            log.warning(f"EAST model not found at: {model_path}")
            log.warning("EAST detection will be unavailable. Download model from:")
            log.warning("https://github.com/argman/EAST")
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect text regions in image.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            list: List of text region dicts with 'bbox', 'confidence'
        """
        if not self.model_loaded or not self.net:
            log.warning("EAST model not available, returning empty detections")
            return []
        
        try:
            # Convert PIL to numpy array
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            original_height, original_width = img_bgr.shape[:2]
            
            # EAST requires input size to be multiple of 32
            # Resize while maintaining aspect ratio
            target_width = (original_width // 32) * 32
            target_height = (original_height // 32) * 32
            
            if target_width == 0:
                target_width = 32
            if target_height == 0:
                target_height = 32
            
            # Resize image
            resized = cv2.resize(img_bgr, (target_width, target_height))
            
            # Prepare blob for EAST
            blob = cv2.dnn.blobFromImage(
                resized, 1.0, (target_width, target_height),
                (123.68, 116.78, 103.94), swapRB=True, crop=False
            )
            
            # Run inference
            self.net.setInput(blob)
            output_layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
            scores, geometry = self.net.forward(output_layers)
            
            # Decode detections
            text_regions = self._decode_predictions(scores, geometry, original_width, original_height, target_width, target_height)
            
            # Filter by confidence
            filtered_regions = [
                region for region in text_regions
                if region['confidence'] >= self.confidence_threshold
            ]
            
            log.info(f"EAST detected {len(filtered_regions)} text regions")
            return filtered_regions
            
        except Exception as e:
            log.error(f"EAST detection failed: {str(e)}", exc_info=True)
            return []
    
    def _decode_predictions(self, scores: np.ndarray, geometry: np.ndarray,
                           orig_w: int, orig_h: int, target_w: int, target_h: int) -> List[Dict[str, Any]]:
        """Decode EAST predictions to bounding boxes.
        
        Args:
            scores: Confidence scores from EAST
            geometry: Geometry predictions from EAST
            orig_w: Original image width
            orig_h: Original image height
            target_w: Resized image width
            target_h: Resized image height
            
        Returns:
            list: Text region detections
        """
        text_regions = []
        
        try:
            # Extract scores and geometry
            scores_map = scores[0, 0, :, :]
            geometry_map = geometry[0, :, :, :]
            
            # Find high-confidence regions
            y_coords, x_coords = np.where(scores_map >= self.confidence_threshold)
            
            for y, x in zip(y_coords, x_coords):
                score = float(scores_map[y, x])
                
                # Get geometry
                x_offset = geometry_map[0, y, x]
                y_offset = geometry_map[1, y, x]
                w = geometry_map[2, y, x]
                h = geometry_map[3, y, x]
                angle = geometry_map[4, y, x]
                
                # Calculate bounding box
                x1 = int((x * 4) + x_offset - (w / 2))
                y1 = int((y * 4) + y_offset - (h / 2))
                x2 = int((x * 4) + x_offset + (w / 2))
                y2 = int((y * 4) + y_offset + (h / 2))
                
                # Scale to original image size
                scale_x = orig_w / target_w
                scale_y = orig_h / target_h
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Normalize to 0-1 range
                bbox_normalized = [
                    max(0.0, min(1.0, x1 / orig_w)),
                    max(0.0, min(1.0, y1 / orig_h)),
                    max(0.0, min(1.0, x2 / orig_w)),
                    max(0.0, min(1.0, y2 / orig_h))
                ]
                
                text_regions.append({
                    'bbox': bbox_normalized,
                    'confidence': score,
                    'class': 'text_region'
                })
        
        except Exception as e:
            log.error(f"Error decoding EAST predictions: {str(e)}", exc_info=True)
        
        return text_regions









