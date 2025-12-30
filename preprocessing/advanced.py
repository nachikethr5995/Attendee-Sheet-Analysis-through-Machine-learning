"""SERVICE 0.1: Advanced Preprocessing (Conditional / Fallback Only)

This module implements heavy CV enhancements used as a fallback when Services 1-4 fail
or produce low confidence results.

Key Characteristics:
- Heavy processing: Advanced corrections for difficult images
- Conditional: Only runs on failure/low confidence
- Comprehensive: Multiple enhancement techniques
- Performance trade-off: Slower but more effective for challenging inputs

Processing Steps:
1. Strong denoising (higher strength than basic)
2. Shadow removal
3. Full perspective correction
4. Orientation correction (EfficientNet classifier)
5. CLAHE++ tuning (enhanced contrast)
6. Handwriting enhancement
7. Adaptive binarization
8. Strong structural cleanup
9. SSIM-optimized compression
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import cv2
import numpy as np
from core.logging import log
from core.config import settings
from core.utils import (
    generate_pre_01_id,
    get_advanced_processed_path,
    get_basic_processed_path,
    image_to_array,
    array_to_image,
    resolve_canonical_id
)
from preprocessing.converter import ImageConverter
from preprocessing.enhancer import ImageEnhancer
from preprocessing.deskew import DeskewCorrector
from preprocessing.perspective import PerspectiveCorrector
from preprocessing.quality import QualityScorer


class AdvancedPreprocessor:
    """SERVICE 0.1: Advanced preprocessing pipeline (conditional/fallback only)."""
    
    def __init__(self):
        """Initialize advanced preprocessor."""
        self.converter = ImageConverter(target_size=settings.TARGET_IMAGE_SIZE)
        self.enhancer = ImageEnhancer(use_gpu=settings.USE_GPU, enhancement_level="aggressive")
        self.deskew_corrector = DeskewCorrector()
        self.perspective_corrector = PerspectiveCorrector()
        self.quality_scorer = QualityScorer()
    
    def process(self, file_id: Optional[str] = None, 
                pre_0_id: Optional[str] = None) -> Dict[str, Any]:
        """Run advanced preprocessing pipeline.
        
        This processes the image with heavy CV enhancements. It can work from either
        the original file_id or from a pre_0_id (preferred, as it's already normalized).
        
        Args:
            file_id: Original file identifier (optional if pre_0_id provided)
            pre_0_id: Basic preprocessing identifier (preferred input)
            
        Returns:
            dict: Processing results with pre_01_id and metadata
        """
        log.info("Starting SERVICE 0.1 (advanced preprocessing)")
        
        # Determine input source
        if pre_0_id:
            log.info(f"Loading from pre_0_id: {pre_0_id}")
            input_path = get_basic_processed_path(pre_0_id)
            if not input_path.exists():
                raise FileNotFoundError(f"pre_0_id image not found: {input_path}")
            image = Image.open(input_path).convert('RGB')
        elif file_id:
            log.info(f"Loading from file_id: {file_id}")
            # Need to determine extension - this is a simplified approach
            # In production, you'd track the extension with the file_id
            from ingestion.file_handler import FileHandler
            from core.utils import get_raw_file_path
            # Try common extensions
            for ext in ['png', 'jpg', 'jpeg', 'heic', 'pdf']:
                test_path = get_raw_file_path(file_id, ext)
                if test_path.exists():
                    image = FileHandler.load_image(file_id, ext)
                    break
            else:
                raise FileNotFoundError(f"Could not find file for file_id: {file_id}")
        else:
            raise ValueError("Either file_id or pre_0_id must be provided")
        
        log.info(f"Input image: {image.size[0]}x{image.size[1]}, mode: {image.mode}")
        
        # Step 1: Strong denoising (higher strength than basic)
        log.info("Step 1: Applying strong denoising...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_bgr = self.enhancer.denoise(img_bgr, strength="strong")
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Strong denoising applied")
        
        # Step 2: Gentle shadow removal and brightness enhancement
        log.info("Step 2: Removing shadows and brightening image...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to LAB color space for better brightness control
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (brightness) only - this avoids color artifacts
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Increase overall brightness (add to L channel)
        l_channel = cv2.add(l_channel, 15)  # Brighten by 15 points
        
        # Merge channels back
        lab = cv2.merge([l_channel, a, b])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Shadow removal and brightness enhancement applied")
        
        # Step 3: Full perspective correction
        log.info("Step 3: Applying full perspective correction...")
        image, perspective_corrected = self.perspective_corrector.correct_perspective(image)
        if perspective_corrected:
            log.info("Perspective correction applied")
        else:
            log.info("Perspective correction not needed")
        
        # Step 4: Orientation correction (using EfficientNet if available, else Hough lines)
        log.info("Step 4: Applying orientation correction...")
        try:
            image, orientation_corrected = self.deskew_corrector.correct_orientation(image, crop_after_deskew=True)
            if orientation_corrected:
                log.info("Orientation correction applied")
            else:
                log.info("Image orientation is correct")
        except Exception as e:
            log.warning(f"Orientation correction failed, continuing: {str(e)}")
            orientation_corrected = False
        
        # Step 5: Gentle contrast enhancement (avoid dark artifacts)
        log.info("Step 5: Applying gentle contrast enhancement...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to LAB for better control
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Gentle CLAHE on L channel only (avoid color artifacts)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Brighten further to avoid dark parts
        l_channel = cv2.add(l_channel, 10)
        
        # Merge back
        lab = cv2.merge([l_channel, a, b])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Gentle contrast enhancement applied")
        
        # Step 6: Gentle handwriting enhancement (preserve brightness)
        log.info("Step 6: Applying gentle handwriting enhancement...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Use unsharp masking instead of aggressive sharpening
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.3, blurred, -0.3, 0)
        
        # Blend sharpened grayscale with original color (preserve color and brightness)
        for i in range(3):
            img_bgr[:, :, i] = cv2.addWeighted(img_bgr[:, :, i], 0.8, sharpened, 0.2, 0)
        
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Gentle handwriting enhancement applied")
        
        # Step 7: Final brightness boost and dark artifact removal
        log.info("Step 7: Applying final brightness boost and removing dark artifacts...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to LAB for brightness control
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Brighten the image further
        l_channel = cv2.add(l_channel, 20)
        
        # Remove very dark pixels (likely artifacts) by lightening them
        # Find dark areas (below threshold) and brighten them
        dark_mask = l_channel < 50
        l_channel[dark_mask] = np.minimum(l_channel[dark_mask] + 30, 255)
        
        # Merge back
        lab = cv2.merge([l_channel, a, b])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Final brightness boost and dark artifact removal applied")
        
        # Step 8: Skip structural cleanup - it creates the topographic texture
        # The morphological operations were creating the dark artifacts
        log.info("Step 8: Skipping structural cleanup to avoid dark artifacts...")
        
        # Step 9: Ensure proper size (re-normalize if needed)
        log.info("Step 9: Re-normalizing size...")
        image = self.converter.normalize(image)
        log.info(f"Image re-normalized: {image.size[0]}x{image.size[1]}")
        
        # Step 10: Quality assessment
        log.info("Step 10: Assessing final image quality...")
        quality_metrics = self.quality_scorer.assess_quality(image)
        quality_score = quality_metrics.get("quality_score", 0.0)
        log.info(f"Final quality score: {quality_score:.3f}")
        
        # Step 11: Generate pre_01_id and save
        log.info("Step 11: Generating pre_01_id and saving processed image...")
        pre_01_id = generate_pre_01_id()
        output_path = get_advanced_processed_path(pre_01_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with SSIM optimization (using converter's save logic)
        # We'll save directly to the correct path
        image.save(
            output_path,
            format='PNG',
            optimize=True,
            compress_level=6  # Default compression
        )
        log.info(f"Saved processed image: {output_path} (size: {output_path.stat().st_size / 1024:.1f} KB)")
        
        log.info(f"SERVICE 0.1 complete. pre_01_id: {pre_01_id}, saved to: {output_path}")
        
        # Prepare response
        width, height = image.size
        metadata = {
            "width": width,
            "height": height,
            "quality_score": float(quality_score),
            "blur_detected": quality_metrics.get("blur_detected", False),
            "processing_type": "advanced",
            "perspective_corrected": perspective_corrected,
            "orientation_corrected": orientation_corrected,
            "shadow_removed": True,
            "handwriting_enhanced": True,
            "adaptive_binarization": False,  # Removed - too destructive
            "gentle_enhancement": True,
        }
        
        return {
            "pre_01_id": pre_01_id,
            "clean_image_path": f"storage/processed_advanced/{pre_01_id}.png",
            "metadata": metadata
        }









