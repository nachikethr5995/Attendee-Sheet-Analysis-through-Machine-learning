"""Checkbox handler - presence detection and checked/unchecked state."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any
from PIL import Image
from core.logging import log
from core.utils import image_to_array
import cv2
import numpy as np


class CheckboxHandler:
    """Handles checkbox detection: presence + checked/unchecked state."""
    
    def __init__(self, checked_threshold: float = 0.15):
        """Initialize checkbox handler.
        
        Args:
            checked_threshold: Black pixel ratio threshold for "checked" state
                              (0.0-1.0, default 0.15 = 15% black pixels = checked)
        """
        self.checked_threshold = checked_threshold
        log.info(f"Checkbox handler initialized (checked_threshold: {checked_threshold})")
    
    def process_checkboxes(self,
                          image: Image.Image,
                          checkbox_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process checkbox detections: presence + checked/unchecked state.
        
        Args:
            image: Full image (PIL Image)
            checkbox_detections: List of checkbox detections from YOLOv8s
            
        Returns:
            List of checkbox results with 'present', 'checked', 'bbox', 'confidence'
        """
        if not checkbox_detections:
            return [{'present': False}]
        
        width, height = image.size
        checkbox_results = []
        
        for i, detection in enumerate(checkbox_detections):
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
            
            # Crop checkbox region
            try:
                crop = image.crop((x1, y1, x2, y2))
            except Exception as e:
                log.warning(f"Failed to crop checkbox {i}: {str(e)}")
                continue
            
            # Determine checked state using pixel density
            is_checked = self._is_checked(crop)
            
            result = {
                'present': True,
                'checked': is_checked,
                'bbox': bbox,
                'confidence': float(confidence),
                'pixel_bbox': [x1, y1, x2, y2]
            }
            
            checkbox_results.append(result)
        
        checked_count = sum(1 for r in checkbox_results if r.get('checked', False))
        log.info(f"Processed {len(checkbox_results)} checkbox(es) ({checked_count} checked, {len(checkbox_results) - checked_count} unchecked)")
        return checkbox_results
    
    def _is_checked(self, crop: Image.Image) -> bool:
        """Determine if checkbox is checked using pixel density.
        
        Args:
            crop: Cropped checkbox image
            
        Returns:
            True if checked, False if unchecked
        """
        # Convert to grayscale
        gray = crop.convert('L')
        img_array = np.array(gray)
        
        # Calculate black pixel ratio (pixels < 128 in 0-255 range)
        total_pixels = img_array.size
        black_pixels = np.sum(img_array < 128)
        black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Checked if black pixel ratio exceeds threshold
        is_checked = black_ratio > self.checked_threshold
        
        log.debug(f"Checkbox black pixel ratio: {black_ratio:.3f}, threshold: {self.checked_threshold}, checked: {is_checked}")
        return is_checked
















