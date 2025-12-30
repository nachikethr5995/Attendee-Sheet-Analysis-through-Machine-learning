"""Adaptive image analysis for determining preprocessing needs."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
from core.logging import log
from core.utils import image_to_array
from preprocessing.quality import QualityScorer


class AdaptiveAnalyzer:
    """Analyzes images to determine optimal preprocessing strategy."""
    
    @staticmethod
    def analyze_image(image: Image.Image) -> Dict[str, Any]:
        """Analyze image to determine preprocessing needs.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            dict: Analysis results with recommended preprocessing steps
        """
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        analysis = {
            "needs_enhancement": False,
            "needs_denoising": False,
            "needs_deskew": False,
            "needs_perspective_correction": False,
            "enhancement_level": "normal",  # light, normal, aggressive
            "denoising_strength": "medium",  # light, medium, strong
            "quality_metrics": {},
        }
        
        # 1. Quality Assessment
        quality_metrics = QualityScorer.assess_quality(image)
        analysis["quality_metrics"] = quality_metrics
        
        # 2. Determine if enhancement is needed
        quality_score = quality_metrics["quality_score"]
        contrast_std = np.std(gray)
        mean_brightness = np.mean(gray)
        
        # Balanced approach - apply to most images but skip truly excellent ones
        if quality_score < 0.8 or contrast_std < 25:
            analysis["needs_enhancement"] = True
            if quality_score < 0.5 or contrast_std < 18:
                analysis["enhancement_level"] = "aggressive"
            elif quality_score < 0.65 or contrast_std < 22:
                analysis["enhancement_level"] = "normal"
            else:
                analysis["enhancement_level"] = "light"
        
        # 3. Determine if denoising is needed
        # Calculate noise level using variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.std(laplacian)
        
        # Balanced approach - apply when noise is noticeable
        if noise_level > 40:  # Moderate threshold
            analysis["needs_denoising"] = True
            if noise_level > 100:
                analysis["denoising_strength"] = "strong"
            elif noise_level > 65:
                analysis["denoising_strength"] = "medium"
            else:
                analysis["denoising_strength"] = "light"
        
        # 4. Detect if deskew is needed
        skew_angle = AdaptiveAnalyzer._detect_skew_angle(gray)
        # Moderate threshold - correct noticeable angles
        if abs(skew_angle) > 0.3:  # More than 0.3 degrees
            analysis["needs_deskew"] = True
            analysis["skew_angle"] = float(skew_angle)
        
        # 5. Detect if perspective correction is needed
        needs_perspective, perspective_score = AdaptiveAnalyzer._needs_perspective_correction(gray)
        # Moderate threshold - correct noticeable warping
        if needs_perspective or perspective_score > 0.08:  # Moderate threshold
            analysis["needs_perspective_correction"] = True
            analysis["perspective_score"] = float(perspective_score)
        
        # 6. Detect image type/content
        image_type = AdaptiveAnalyzer._detect_image_type(gray, img_bgr)
        analysis["image_type"] = image_type
        
        log.info(f"Image analysis: enhancement={analysis['needs_enhancement']}, "
                f"denoising={analysis['needs_denoising']}, deskew={analysis['needs_deskew']}, "
                f"perspective={analysis['needs_perspective_correction']}, type={image_type}")
        
        return analysis
    
    @staticmethod
    def _detect_skew_angle(gray: np.ndarray) -> float:
        """Detect skew angle in degrees."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta)
            if angle > 45:
                angle = angle - 90
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        return float(np.median(angles))
    
    @staticmethod
    def _needs_perspective_correction(gray: np.ndarray) -> Tuple[bool, float]:
        """Determine if perspective correction is needed.
        
        Returns:
            Tuple[bool, float]: (needs_correction, confidence_score)
        """
        # Detect document edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have exactly 4 points, likely needs perspective correction
        if len(approx) == 4:
            # Check if it's roughly rectangular
            area = cv2.contourArea(approx)
            rect = cv2.minAreaRect(largest_contour)
            rect_area = rect[1][0] * rect[1][1]
            
            if rect_area > 0:
                extent = area / rect_area
            # If extent is low, shape is irregular (needs correction)
            # Moderate threshold - correct noticeable warping
            if extent < 0.87:
                return True, float(1.0 - extent)
        
        return False, 0.0
    
    @staticmethod
    def _detect_image_type(gray: np.ndarray, color: np.ndarray) -> str:
        """Detect image type (document, photo, form, etc.)."""
        # Calculate text-like features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate variance (text has high variance)
        variance = np.var(gray)
        
        # Detect if it's mostly text (high edge density, high variance)
        if edge_density > 0.15 and variance > 1000:
            return "document"
        elif edge_density > 0.10:
            return "form"
        else:
            return "photo"









