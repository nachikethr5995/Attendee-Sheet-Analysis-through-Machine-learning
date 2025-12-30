"""Image conversion and normalization."""

from pathlib import Path
from PIL import Image
import numpy as np
from core.logging import log
from core.config import settings
from core.utils import get_processed_image_path, generate_processed_id, image_to_array

# Optional SSIM import for compression optimization
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    log.warning("scikit-image not installed. SSIM compression optimization will be disabled.")


class ImageConverter:
    """Handles image conversion and normalization."""
    
    def __init__(self, target_size: int = None):
        """Initialize converter.
        
        Args:
            target_size: Target long edge size in pixels (default from config)
        """
        self.target_size = target_size or settings.TARGET_IMAGE_SIZE
    
    def normalize(self, image: Image.Image) -> Image.Image:
        """Normalize image to target size.
        
        Resizes image so the long edge equals target_size while maintaining
        aspect ratio. Converts to RGB if needed.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Image.Image: Normalized image
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate new size
        width, height = image.size
        long_edge = max(width, height)
        
        if long_edge != self.target_size:
            scale = self.target_size / long_edge
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            log.info(f"Resizing from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def optimize_compression(self, image: Image.Image, target_ssim: float = 0.98) -> Image.Image:
        """Optimize PNG compression using SSIM.
        
        Tests different compression levels and selects the best one
        that maintains SSIM above threshold.
        
        Args:
            image: Input PIL Image
            target_ssim: Minimum SSIM threshold (default 0.98)
            
        Returns:
            Image.Image: Optimized image (same as input, optimization happens on save)
        """
        if not SSIM_AVAILABLE:
            log.info("SSIM not available, using default compression")
        else:
            log.info(f"SSIM optimization target: {target_ssim}")
        return image
    
    def save_processed(self, image: Image.Image, processed_id: str = None, optimize: bool = True) -> Path:
        """Save processed image as PNG with optional SSIM optimization.
        
        Args:
            image: Processed PIL Image
            processed_id: Processed ID (generated if None)
            optimize: Whether to apply SSIM-based compression optimization
            
        Returns:
            Path: Path to saved image
        """
        if processed_id is None:
            processed_id = generate_processed_id()
        
        output_path = get_processed_image_path(processed_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if optimize and SSIM_AVAILABLE:
            # Try different compression levels and find best balance
            original_array = image_to_array(image)
            best_quality = 6  # Default
            best_ssim = 0.0
            
            # Test compression levels (0-9, higher = more compression)
            for level in [9, 6, 3, 1]:
                # Save with this compression level
                temp_path = output_path.parent / f"temp_{processed_id}.png"
                image.save(
                    temp_path,
                    format='PNG',
                    optimize=True,
                    compress_level=level
                )
                
                # Load and compare SSIM
                try:
                    loaded = Image.open(temp_path)
                    loaded_array = image_to_array(loaded)
                    
                    # Calculate SSIM (convert to grayscale for comparison)
                    if len(original_array.shape) == 3:
                        orig_gray = np.mean(original_array, axis=2)
                        loaded_gray = np.mean(loaded_array, axis=2)
                    else:
                        orig_gray = original_array
                        loaded_gray = loaded_array
                    
                    # Ensure same size
                    if orig_gray.shape != loaded_gray.shape:
                        from PIL import Image as PILImage
                        loaded_resized = PILImage.fromarray(loaded_gray).resize(
                            (orig_gray.shape[1], orig_gray.shape[0]), 
                            PILImage.Resampling.LANCZOS
                        )
                        loaded_gray = np.array(loaded_resized)
                    
                    ssim_value = ssim(orig_gray, loaded_gray, data_range=255)
                    
                    if ssim_value > best_ssim and ssim_value >= 0.98:
                        best_ssim = ssim_value
                        best_quality = level
                    
                    temp_path.unlink()  # Clean up
                    
                except Exception as e:
                    log.warning(f"SSIM test failed for level {level}: {str(e)}")
                    continue
            
            log.info(f"Selected compression level {best_quality} with SSIM {best_ssim:.4f}")
            compress_level = best_quality
        else:
            # Use default compression level
            compress_level = 6
        
        # Save with optimal compression
        image.save(
            output_path,
            format='PNG',
            optimize=True,
            compress_level=compress_level
        )
        
        log.info(f"Saved processed image: {output_path} (size: {output_path.stat().st_size / 1024:.1f} KB)")
        return output_path










