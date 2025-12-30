"""SERVICE 0: Basic Preprocessing (Always Runs)

This module implements lightweight, fast preprocessing that runs on every uploaded image.
It provides minimal, non-destructive enhancements to standardize images for downstream services.

Key Characteristics:
- Fast: Lightweight operations for performance
- Non-destructive: Minimal changes to preserve original quality
- Standardized: Consistent output format for downstream services
- Always applied: No conditional logic, runs on every upload
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
    generate_pre_0_id,
    get_basic_processed_path,
    image_to_array,
    array_to_image
)
from ingestion.file_handler import FileHandler
from preprocessing.converter import ImageConverter
from preprocessing.enhancer import ImageEnhancer
from preprocessing.deskew import DeskewCorrector
from preprocessing.perspective import PerspectiveCorrector
from preprocessing.quality import QualityScorer


class BasicPreprocessor:
    """SERVICE 0: Basic preprocessing pipeline (always runs)."""
    
    def __init__(self):
        """Initialize basic preprocessor."""
        self.converter = ImageConverter(target_size=settings.TARGET_IMAGE_SIZE)
        self.enhancer = ImageEnhancer(use_gpu=settings.USE_GPU, enhancement_level="light")
        self.deskew_corrector = DeskewCorrector()
        self.perspective_corrector = PerspectiveCorrector()
        self.quality_scorer = QualityScorer()
    
    def process(self, file_id: str, extension: str) -> Dict[str, Any]:
        """Run basic preprocessing pipeline.
        
        Processing steps:
        1. Format conversion (HEIC/AVIF/PDF → PNG)
        2. Light denoising
        3. Image resize to 2048px long edge
        4. Soft deskew (minor angle corrections only)
        5. Light contrast normalization
        6. Minimal artifact removal
        
        Args:
            file_id: Original file identifier
            extension: File extension
            
        Returns:
            dict: Processing results with pre_0_id and metadata
        """
        log.info(f"Starting SERVICE 0 (basic preprocessing) for file_id: {file_id}")
        
        # Step 1: Load image
        log.info("Step 1: Loading image...")
        image = FileHandler.load_image(file_id, extension)
        log.info(f"Image loaded: {image.size[0]}x{image.size[1]}, mode: {image.mode}")
        
        # Step 2: Format conversion and normalization FIRST
        log.info("Step 2: Converting format and normalizing size...")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Normalize size first to standardize processing
        image = self.converter.normalize(image)
        log.info(f"Image normalized: {image.size[0]}x{image.size[1]}")
        
        # Step 3: Detect and correct skew ONLY (no perspective correction - too aggressive)
        log.info("Step 3: Checking for deskew correction...")
        gray = cv2.cvtColor(image_to_array(image), cv2.COLOR_RGB2GRAY)
        skew_angle = DeskewCorrector.detect_skew_angle(gray)
        
        if abs(skew_angle) > 0.5:  # Only correct noticeable angles
            log.info(f"Applying deskew correction (angle: {skew_angle:.2f}°)...")
            # Deskew WITHOUT cropping first - cropping is too aggressive
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_bgr = DeskewCorrector.deskew_image(img_bgr, skew_angle)
            img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = array_to_image(img_array)
            log.info("Deskew correction applied")
        else:
            log.info(f"Skew angle ({skew_angle:.2f}°) is minimal, skipping deskew")
        
        # Step 4: Conservative cropping - only if we can clearly detect document boundaries
        log.info("Step 4: Attempting conservative content cropping...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Try cropping, but check if result is reasonable
        original_size = img_bgr.shape[:2]
        cropped = DeskewCorrector.crop_to_content(img_bgr, margin=20)
        cropped_size = cropped.shape[:2]
        
        # Only use cropped version if it's reasonable (not too small, preserves most content)
        area_reduction = 1 - (cropped_size[0] * cropped_size[1]) / (original_size[0] * original_size[1])
        content_cropped = False
        if area_reduction < 0.5:  # Only if we're not removing more than 50% of image
            img_bgr = cropped
            content_cropped = True
            log.info(f"Content cropping applied (reduced by {area_reduction*100:.1f}%)")
        else:
            log.info(f"Content cropping too aggressive ({area_reduction*100:.1f}% reduction), skipping")
        
        img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = array_to_image(img_array)
        
        # Re-normalize after cropping if needed
        if content_cropped:
            image = self.converter.normalize(image)
            log.info(f"Re-normalized after cropping: {image.size[0]}x{image.size[1]}")
        
        # Step 5: Very light denoising (minimal - preserve detail)
        log.info("Step 5: Applying very light denoising...")
        img_array = image_to_array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Very light denoising - minimal strength to preserve detail
        img_bgr = self.enhancer.denoise(img_bgr, strength="light")
        image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        log.info("Light denoising applied")
        
        # Step 6: Skip CLAHE - it's washing out the image
        # Only apply if image is very dark
        log.info("Step 6: Checking if contrast enhancement is needed...")
        img_array = image_to_array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Only apply CLAHE if image is quite dark (mean < 100)
        if mean_brightness < 100:
            log.info(f"Image is dark (mean brightness: {mean_brightness:.1f}), applying light CLAHE...")
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Very light CLAHE
            img_bgr = self.enhancer.apply_clahe(img_bgr, clip_limit=1.5)  # Very light
            image = array_to_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            log.info("Light contrast enhancement applied")
        else:
            log.info(f"Image brightness is good ({mean_brightness:.1f}), skipping contrast enhancement")
        
        # Step 7: Quality assessment
        log.info("Step 7: Assessing image quality...")
        quality_metrics = self.quality_scorer.assess_quality(image)
        quality_score = quality_metrics.get("quality_score", 0.0)
        log.info(f"Quality score: {quality_score:.3f}")
        
        # Step 8: Generate pre_0_id and save
        log.info("Step 8: Generating pre_0_id and saving processed image...")
        pre_0_id = generate_pre_0_id()
        output_path = get_basic_processed_path(pre_0_id)
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
        
        log.info(f"SERVICE 0 complete. pre_0_id: {pre_0_id}, saved to: {output_path}")
        
        # Prepare response
        width, height = image.size
        metadata = {
            "width": width,
            "height": height,
            "quality_score": float(quality_score),
            "blur_detected": quality_metrics.get("blur_detected", False),
            "processing_type": "basic",
            "skew_corrected": abs(skew_angle) > 0.5,
            "skew_angle": float(skew_angle) if abs(skew_angle) > 0.5 else 0.0,
            "perspective_corrected": False,  # Disabled - too aggressive
            "content_cropped": content_cropped,
        }
        
        return {
            "pre_0_id": pre_0_id,
            "clean_image_path": f"storage/processed_basic/{pre_0_id}.png",
            "metadata": metadata
        }









