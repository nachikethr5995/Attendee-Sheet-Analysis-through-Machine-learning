"""Signature handler - presence detection and optional cropping (NO OCR)."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import image_to_array
import cv2
import numpy as np


class SignatureHandler:
    """Handles signature detection: presence flag + optional crop (NO OCR)."""
    
    def __init__(self, save_crops: bool = False, crop_dir: Optional[Path] = None):
        """Initialize signature handler.
        
        Args:
            save_crops: Whether to save cropped signature images
            crop_dir: Directory to save crops (if save_crops=True)
        """
        self.save_crops = save_crops
        self.crop_dir = crop_dir or (Path(settings.STORAGE_ROOT) / "signatures")
        
        if self.save_crops:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Signature crops will be saved to: {self.crop_dir}")
    
    def process_signatures(self,
                          image: Image.Image,
                          signature_detections: List[Dict[str, Any]],
                          canonical_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process signature detections: presence + optional crop.
        
        Args:
            image: Full image (PIL Image)
            signature_detections: List of signature detections from YOLOv8s
            canonical_id: Optional ID for saving crops
            
        Returns:
            List of signature results with 'present', 'bbox', 'crop_path' (optional)
        """
        if not signature_detections:
            return [{'present': False}]
        
        width, height = image.size
        signature_results = []
        
        for i, detection in enumerate(signature_detections):
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            
            if not bbox or len(bbox) < 4:
                continue
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            # Ensure valid coordinates
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            result = {
                'present': True,
                'bbox': bbox,
                'confidence': float(confidence),
                'pixel_bbox': [x1, y1, x2, y2]
            }
            
            # Optional: Save crop
            if self.save_crops and canonical_id:
                try:
                    crop = image.crop((x1, y1, x2, y2))
                    crop_filename = f"{canonical_id}_signature_{i}.png"
                    crop_path = self.crop_dir / crop_filename
                    crop.save(crop_path)
                    result['crop_path'] = str(crop_path.relative_to(Path(settings.STORAGE_ROOT).parent))
                    log.debug(f"Saved signature crop: {crop_path}")
                except Exception as e:
                    log.warning(f"Failed to save signature crop {i}: {str(e)}")
            
            signature_results.append(result)
        
        log.info(f"Processed {len(signature_results)} signature(s)")
        return signature_results







