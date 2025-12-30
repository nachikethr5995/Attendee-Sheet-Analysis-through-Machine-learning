"""Auto rotation and deskew correction."""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image
from core.logging import log
from core.utils import image_to_array, array_to_image


class DeskewCorrector:
    """Handles image rotation and deskew correction."""
    
    @staticmethod
    def detect_orientation(image: np.ndarray) -> int:
        """Detect image orientation (0°, 90°, 180°, 270°).
        
        Uses Hough line detection and text orientation analysis.
        
        Args:
            image: Input image as numpy array (grayscale)
            
        Returns:
            int: Rotation angle needed (0, 90, 180, or 270)
        """
        # Use Hough lines to detect dominant direction
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            log.info("No lines detected, assuming 0° rotation")
            return 0
        
        # Calculate angles
        angles = []
        for line in lines[:20]:  # Use first 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta)
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)
        
        if not angles:
            return 0
        
        # Get median angle
        median_angle = np.median(angles)
        
        # Determine rotation needed
        if abs(median_angle) < 5:
            rotation = 0
        elif abs(median_angle - 90) < 5:
            rotation = 90
        elif abs(median_angle - 180) < 5 or abs(median_angle + 180) < 5:
            rotation = 180
        elif abs(median_angle + 90) < 5:
            rotation = 270
        else:
            rotation = 0
        
        log.info(f"Detected orientation: {median_angle:.2f}°, rotation needed: {rotation}°")
        return rotation
    
    @staticmethod
    def detect_skew_angle(image: np.ndarray) -> float:
        """Detect skew angle using Hough lines.
        
        Args:
            image: Input image as numpy array (grayscale)
            
        Returns:
            float: Skew angle in degrees
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta)
            # Normalize to -45 to 45 degrees
            if angle > 45:
                angle = angle - 90
            angles.append(angle)
        
        # Get median angle (most common)
        skew_angle = np.median(angles)
        
        # Only correct if skew is significant (> 0.5°)
        if abs(skew_angle) < 0.5:
            return 0.0
        
        log.info(f"Detected skew angle: {skew_angle:.2f}°")
        return float(skew_angle)
    
    @staticmethod
    def rotate_image(image: Image.Image, angle: int) -> Image.Image:
        """Rotate image by 90° increments.
        
        Args:
            image: Input PIL Image
            angle: Rotation angle (0, 90, 180, or 270)
            
        Returns:
            Image.Image: Rotated image
        """
        if angle == 0:
            return image
        
        # PIL rotates counter-clockwise, so we need to adjust
        rotations = {
            90: Image.ROTATE_270,   # 90° CCW = 270° CW
            180: Image.ROTATE_180,
            270: Image.ROTATE_90,   # 270° CCW = 90° CW
        }
        
        if angle in rotations:
            rotated = image.transpose(rotations[angle])
            log.info(f"Rotated image by {angle}°")
            return rotated
        
        return image
    
    @staticmethod
    def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Deskew image by small angle.
        
        Args:
            image: Input image as numpy array
            angle: Skew angle in degrees
            
        Returns:
            np.ndarray: Deskewed image
        """
        if abs(angle) < 0.5:
            return image
        
        # Get image center
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation with better interpolation and border handling
        deskewed = cv2.warpAffine(
            image, M, (new_w, new_h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
        )
        
        log.info(f"Deskewed image by {angle:.2f}°")
        return deskewed
    
    @staticmethod
    def crop_to_content(image: np.ndarray, margin: int = 10) -> np.ndarray:
        """Crop image to remove empty borders and focus on content.
        
        Uses multiple methods to find document boundaries:
        1. Edge detection with Canny
        2. Contour analysis
        3. Variance-based content detection
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            margin: Additional margin in pixels to keep around content
            
        Returns:
            np.ndarray: Cropped image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Get image dimensions
        h, w = gray.shape
        
        # Method 1: Use edge detection to find document boundaries
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: Use variance-based method
            log.info("No contours found, using variance-based cropping")
            # Calculate row and column variances
            row_vars = np.var(gray, axis=1)
            col_vars = np.var(gray, axis=0)
            
            # Find content boundaries (where variance is above threshold)
            threshold = np.max(row_vars) * 0.1
            y_indices = np.where(row_vars > threshold)[0]
            x_indices = np.where(col_vars > threshold)[0]
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min = max(0, y_indices[0] - margin)
                y_max = min(h, y_indices[-1] + margin)
                x_min = max(0, x_indices[0] - margin)
                x_max = min(w, x_indices[-1] + margin)
                
                cropped = image[y_min:y_max, x_min:x_max]
                log.info(f"Cropped using variance: {w}x{h} -> {x_max-x_min}x{y_max-y_min}")
                return cropped
            else:
                log.info("No content detected for cropping, returning original")
                return image
        
        # Find largest contour (likely the document)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        
        # Check if the contour is significant (at least 20% of image)
        if w_cont * h_cont < (w * h * 0.2):
            # Try to find all significant content
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for contour in contours:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                # Only consider significant contours (at least 1% of image)
                if w_c * h_c > (w * h * 0.01):
                    x_min = min(x_min, x_c)
                    y_min = min(y_min, y_c)
                    x_max = max(x_max, x_c + w_c)
                    y_max = max(y_max, y_c + h_c)
            
            if x_max <= x_min or y_max <= y_min:
                log.info("No significant content detected for cropping")
                return image
        else:
            # Use the largest contour
            x_min = x
            y_min = y
            x_max = x + w_cont
            y_max = y + h_cont
        
        # Add margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Log crop info
        original_area = w * h
        cropped_area = (x_max - x_min) * (y_max - y_min)
        reduction = (1 - cropped_area / original_area) * 100
        
        log.info(f"Cropped image: {w}x{h} -> {x_max-x_min}x{y_max-y_min} ({reduction:.1f}% reduction)")
        
        return cropped
    
    @classmethod
    def correct_orientation(cls, image: Image.Image, crop_after_deskew: bool = True) -> Tuple[Image.Image, bool]:
        """Correct image orientation and deskew with optional cropping.
        
        Args:
            image: Input PIL Image (RGB)
            crop_after_deskew: Whether to crop empty borders after deskewing
            
        Returns:
            Tuple[Image.Image, bool]: (corrected_image, was_corrected)
        """
        # Convert to numpy array
        img_array = image_to_array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        was_corrected = False
        
        # Detect and correct orientation
        rotation = cls.detect_orientation(gray)
        if rotation != 0:
            image = cls.rotate_image(image, rotation)
            img_array = image_to_array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            was_corrected = True
        
        # Detect and correct skew
        skew_angle = cls.detect_skew_angle(gray)
        if abs(skew_angle) >= 0.5:
            # Convert to BGR for deskewing
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_bgr = cls.deskew_image(img_bgr, skew_angle)
            
            # Crop to content if requested
            if crop_after_deskew:
                img_bgr = cls.crop_to_content(img_bgr, margin=10)
            
            # Convert back to RGB
            img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = array_to_image(img_array)
            was_corrected = True
        
        return image, was_corrected










