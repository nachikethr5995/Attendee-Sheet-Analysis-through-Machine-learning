"""Color and lighting enhancement, noise reduction."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import cv2
import numpy as np
from PIL import Image
from core.logging import log
from core.utils import image_to_array, array_to_image


class ImageEnhancer:
    """Handles image enhancement and noise reduction."""
    
    def __init__(self, use_gpu: bool = True, enhancement_level: str = "normal"):
        """Initialize enhancer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            enhancement_level: Enhancement intensity ("light", "normal", "aggressive")
        """
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.enhancement_level = enhancement_level
    
    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply auto white balance correction.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            np.ndarray: White-balanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        lab = cv2.merge([l, a, b])
        balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return balanced
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = None) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image as numpy array (BGR)
            clip_limit: CLAHE clip limit (auto-determined if None)
            
        Returns:
            np.ndarray: CLAHE-enhanced image
        """
        # Determine clip limit based on enhancement level
        if clip_limit is None:
            clip_limits = {
                "light": 2.0,
                "normal": 3.0,
                "aggressive": 4.0,
            }
            clip_limit = clip_limits.get(self.enhancement_level, 3.0)
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction.
        
        Args:
            image: Input image as numpy array (BGR)
            gamma: Gamma value (default 1.2)
            
        Returns:
            np.ndarray: Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def denoise(self, image: np.ndarray, strength: str = "medium") -> np.ndarray:
        """Apply adaptive denoising.
        
        Uses GPU acceleration if available, otherwise CPU fallback.
        
        Args:
            image: Input image as numpy array (BGR)
            strength: Denoising strength ("light", "medium", "strong")
            
        Returns:
            np.ndarray: Denoised image
        """
        # Determine denoising parameters based on strength
        h_values = {
            "light": 5,
            "medium": 10,
            "strong": 15,
        }
        h = h_values.get(strength, 10)
        h_color = h
        
        if self.use_gpu:
            try:
                # GPU-accelerated denoising
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                denoised_gpu = cv2.cuda.fastNlMeansDenoisingColored(
                    gpu_img, h=h, hColor=h_color, templateWindowSize=7, searchWindowSize=21
                )
                denoised = denoised_gpu.download()
                log.info(f"Applied GPU denoising (strength: {strength})")
                return denoised
            except Exception as e:
                log.warning(f"GPU denoising failed, using CPU: {str(e)}")
        
        # CPU fallback
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=h, hColor=h_color, templateWindowSize=7, searchWindowSize=21
        )
        log.info(f"Applied CPU denoising (strength: {strength})")
        return denoised
    
    def enhance(self, image: Image.Image, apply_denoise: bool = True, denoise_strength: str = "medium") -> Image.Image:
        """Apply full enhancement pipeline.
        
        Args:
            image: Input PIL Image (RGB)
            apply_denoise: Whether to apply denoising
            denoise_strength: Denoising strength if applied
            
        Returns:
            Image.Image: Enhanced PIL Image (RGB)
        """
        # Convert to numpy array (BGR for OpenCV)
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply enhancements
        enhanced = self.auto_white_balance(img_bgr)
        enhanced = self.apply_clahe(enhanced)
        enhanced = self.gamma_correction(enhanced)
        
        # Conditionally apply denoising
        if apply_denoise:
            enhanced = self.denoise(enhanced, strength=denoise_strength)
        
        # Convert back to RGB PIL Image
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return array_to_image(enhanced_rgb)










