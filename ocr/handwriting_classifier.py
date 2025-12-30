"""Handwriting classification module.

Simple heuristic-based classifier to determine if a text region contains handwriting.
This is a lightweight classifier that can be replaced with a trained model later.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.utils import image_to_array


class HandwritingClassifier:
    """Heuristic-based handwriting classifier.
    
    Uses simple image analysis heuristics to classify text regions as handwriting.
    Can be replaced with a trained ML model in the future.
    """
    
    def __init__(self):
        """Initialize handwriting classifier."""
        log.info("Initializing handwriting classifier (heuristic-based)")
    
    def is_handwriting(self, image: Image.Image, bbox: Optional[list] = None) -> bool:
        """Classify if a text region contains handwriting.
        
        Args:
            image: PIL Image (RGB) of cropped text region
            bbox: Optional bounding box [x1, y1, x2, y2] for context
            
        Returns:
            bool: True if region likely contains handwriting
        """
        try:
            # Convert to numpy array
            img_array = image_to_array(image)
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Heuristic 1: Variability in stroke width
            # Handwriting typically has more variable stroke widths than printed text
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Heuristic 2: Baseline irregularity
            # Handwriting has less regular baselines
            # Use horizontal projection to detect baseline irregularity
            h_projection = np.sum(gray < 128, axis=1)  # Sum of dark pixels per row
            if len(h_projection) > 0:
                h_variance = np.var(h_projection)
                h_mean = np.mean(h_projection)
                irregularity = h_variance / (h_mean + 1e-6)  # Normalized variance
            else:
                irregularity = 0.0
            
            # Heuristic 3: Character spacing variability
            # Handwriting has more variable spacing
            v_projection = np.sum(gray < 128, axis=0)  # Sum of dark pixels per column
            if len(v_projection) > 0:
                v_variance = np.var(v_projection)
                spacing_variability = v_variance / (np.mean(v_projection) + 1e-6)
            else:
                spacing_variability = 0.0
            
            # Combine heuristics (tunable thresholds)
            # These thresholds are based on empirical observations
            is_handwriting = (
                edge_density > 0.15 or  # High edge density
                irregularity > 0.3 or    # High baseline irregularity
                spacing_variability > 0.4  # High spacing variability
            )
            
            log.debug(f"Handwriting classification: edge_density={edge_density:.3f}, "
                     f"irregularity={irregularity:.3f}, spacing_var={spacing_variability:.3f}, "
                     f"result={is_handwriting}")
            
            return is_handwriting
            
        except Exception as e:
            log.warning(f"Handwriting classification failed: {str(e)}")
            # Default to False (assume printed) on error
            return False
    
    def classify_region(self, image: Image.Image, 
                       bbox: Optional[list] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify a text region and return detailed results.
        
        Args:
            image: PIL Image (RGB) of cropped text region
            bbox: Optional bounding box [x1, y1, x2, y2]
            metadata: Optional metadata about the region
            
        Returns:
            dict: Classification results with 'is_handwriting' boolean
        """
        is_handwriting = self.is_handwriting(image, bbox)
        
        return {
            'is_handwriting': is_handwriting,
            'confidence': 0.7 if is_handwriting else 0.3,  # Heuristic confidence
            'method': 'heuristic'
        }




