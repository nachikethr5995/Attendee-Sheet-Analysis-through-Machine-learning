"""Diagnostic script for YOLOv8s detection issues."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.yolov8_layout_detector import YOLOv8LayoutDetector
from core.config import settings
from PIL import Image
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_with_real_image():
    """Test YOLOv8s with a real document image if available."""
    log.info("=" * 60)
    log.info("YOLOv8s Diagnostic Test")
    log.info("=" * 60)
    
    # Initialize detector
    log.info("\n1. Initializing YOLOv8s detector...")
    detector = YOLOv8LayoutDetector(
        model_version="v1.0",
        confidence_threshold=0.15,
        use_gpu=False
    )
    
    if not detector.model:
        log.error("❌ Model failed to load!")
        return False
    
    log.info(f"✅ Model loaded: {detector.is_custom_model}")
    log.info(f"   Device: {detector.device}")
    log.info(f"   Confidence threshold: {detector.confidence_threshold}")
    log.info(f"   Model classes: {detector.model.names}")
    
    # Try to find a real image in storage
    storage_path = Path(settings.STORAGE_ROOT)
    test_image_path = None
    
    # Look for processed images first
    for storage_dir in [settings.STORAGE_PROCESSED_BASIC, settings.STORAGE_PROCESSED_ADVANCED, settings.STORAGE_RAW]:
        storage_dir_path = Path(storage_dir)
        if storage_dir_path.exists():
            # Find first image file
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images = list(storage_dir_path.glob(ext))
                if images:
                    test_image_path = images[0]
                    log.info(f"\n2. Found test image: {test_image_path}")
                    break
            if test_image_path:
                break
    
    if not test_image_path:
        log.warning("\n2. No test image found in storage directories")
        log.warning("   Creating a test pattern image...")
        # Create a test pattern that might trigger detections
        test_image = Image.new('RGB', (800, 1000), color='white')
        # Add some patterns that might look like text
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        # Draw some rectangles that might look like text blocks
        draw.rectangle([50, 50, 750, 200], fill='lightgray', outline='black', width=2)
        draw.rectangle([50, 250, 750, 400], fill='lightgray', outline='black', width=2)
        log.info("   Created test pattern image")
    else:
        try:
            test_image = Image.open(test_image_path)
            log.info(f"   Image size: {test_image.size}, mode: {test_image.mode}")
        except Exception as e:
            log.error(f"   Failed to load image: {e}")
            return False
    
    # Test detection
    log.info("\n3. Running detection...")
    try:
        results = detector.detect(test_image)
        
        log.info(f"\n4. Detection Results:")
        log.info(f"   Tables: {len(results.get('tables', []))}")
        log.info(f"   Text blocks: {len(results.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(results.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(results.get('checkboxes', []))}")
        
        total = sum(len(results.get(k, [])) for k in ['tables', 'text_blocks', 'signatures', 'checkboxes'])
        log.info(f"   Total: {total}")
        
        if total > 0:
            log.info("\n✅ YOLOv8s is working!")
            # Show sample detections
            for key in ['tables', 'text_blocks', 'signatures', 'checkboxes']:
                items = results.get(key, [])
                if items:
                    sample = items[0]
                    log.info(f"\n   Sample {key}:")
                    log.info(f"      Class: {sample.get('class')}")
                    log.info(f"      Confidence: {sample.get('confidence', 0):.3f}")
                    log.info(f"      BBox: {sample.get('bbox')}")
        else:
            log.warning("\n⚠️  YOLOv8s detected 0 items")
            log.warning("   Possible reasons:")
            log.warning("   1. Image doesn't contain detectable elements")
            log.warning("   2. Confidence threshold too high (try lowering)")
            log.warning("   3. Model not trained for this type of document")
            log.warning("   4. Image quality/preprocessing issues")
        
        return True
        
    except Exception as e:
        log.error(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_real_image()
    sys.exit(0 if success else 1)

















