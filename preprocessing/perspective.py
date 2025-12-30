"""Perspective correction."""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image
from core.logging import log
from core.utils import image_to_array, array_to_image


class PerspectiveCorrector:
    """Handles perspective correction using four-point transform."""
    
    @staticmethod
    def find_document_corners(image: np.ndarray) -> np.ndarray:
        """Find document corners using contour detection.
        
        Args:
            image: Input image as numpy array (grayscale or BGR)
            
        Returns:
            np.ndarray: Four corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found, return image corners
            h, w = gray.shape
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have 4 points, use them
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            log.info("Found 4 corner points")
            return corners
        
        # Otherwise, find bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
        
        log.info("Using bounding rectangle corners")
        return corners
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            np.ndarray: Ordered points
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left: smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right: largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right: smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left: largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    @staticmethod
    def calculate_dimensions(pts: np.ndarray) -> Tuple[int, int]:
        """Calculate output dimensions from corner points.
        
        Args:
            pts: Four corner points
            
        Returns:
            Tuple[int, int]: (width, height)
        """
        (tl, tr, br, bl) = pts
        
        # Calculate width
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Calculate height
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        return max_width, max_height
    
    @classmethod
    def correct_perspective(cls, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Correct perspective distortion.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            Tuple[Image.Image, bool]: (corrected_image, was_corrected)
        """
        # Convert to numpy array
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Find corners
        corners = cls.find_document_corners(img_bgr)
        
        # Order points
        ordered_corners = cls.order_points(corners)
        
        # Calculate output dimensions
        width, height = cls.calculate_dimensions(ordered_corners)
        
        # Destination points (rectangular output)
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        
        # Apply perspective transform
        corrected = cv2.warpPerspective(img_bgr, M, (width, height))
        
        # Convert back to RGB PIL Image
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        corrected_image = array_to_image(corrected_rgb)
        
        # Check if correction was significant
        was_corrected = not np.allclose(ordered_corners, dst, atol=10)
        
        if was_corrected:
            log.info(f"Applied perspective correction: {width}x{height}")
        else:
            log.info("No perspective correction needed")
        
        return corrected_image, was_corrected










