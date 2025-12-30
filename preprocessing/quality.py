"""Blur detection and quality scoring."""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image
from core.logging import log
from core.utils import image_to_array


class QualityScorer:
    """Handles image quality assessment."""
    
    @staticmethod
    def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
        """Detect blur using Laplacian variance.
        
        Args:
            image: Input image as numpy array (grayscale or BGR)
            threshold: Blur threshold (lower = more sensitive)
            
        Returns:
            Tuple[bool, float]: (is_blurred, variance_score)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        is_blurred = bool(variance < threshold)  # Convert to native Python bool
        
        log.info(f"Blur detection: variance={variance:.2f}, blurred={is_blurred}")
        
        return is_blurred, float(variance)  # Return native Python types
    
    @staticmethod
    def calculate_quality_score(image: np.ndarray) -> float:
        """Calculate overall quality score (0-1).
        
        Combines blur detection, contrast, and brightness metrics.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Blur score (normalized)
        _, blur_variance = QualityScorer.detect_blur(gray, threshold=50.0)
        blur_score = min(blur_variance / 200.0, 1.0)  # Normalize to 0-1
        
        # Contrast score (standard deviation)
        contrast_std = np.std(gray)
        contrast_score = min(contrast_std / 64.0, 1.0)  # Normalize to 0-1
        
        # Brightness score (avoid too dark or too bright)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        
        # Combined score (weighted average)
        quality_score = (
            0.5 * blur_score +
            0.3 * contrast_score +
            0.2 * brightness_score
        )
        
        log.info(f"Quality score: {quality_score:.3f} (blur={blur_score:.3f}, contrast={contrast_score:.3f}, brightness={brightness_score:.3f})")
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    @staticmethod
    def assess_quality(image: Image.Image) -> dict:
        """Assess image quality and return metrics.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            dict: Quality metrics including score, blur status, etc.
        """
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        is_blurred, blur_variance = QualityScorer.detect_blur(gray)
        quality_score = QualityScorer.calculate_quality_score(img_bgr)
        
        return {
            "quality_score": float(quality_score),
            "blur_detected": bool(is_blurred),  # Ensure native Python bool
            "blur_variance": float(blur_variance),
            "is_high_quality": bool(quality_score >= 0.7),  # Ensure native Python bool
        }










