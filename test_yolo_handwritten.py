"""Test YOLOv8s handwritten class detection and PaddleOCR exclusion."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.yolov8_layout_detector import YOLOv8LayoutDetector
from layout.layout_service import LayoutService
from core.config import settings
from PIL import Image
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_yolo_handwritten():
    """Test YOLOv8s handwritten class detection."""
    log.info("=" * 60)
    log.info("YOLOv8s Handwritten Class Test")
    log.info("=" * 60)
    
    # Initialize detector
    log.info("\n1. Initializing YOLOv8s detector...")
    detector = YOLOv8LayoutDetector(
        model_version="v1.0",
        confidence_threshold=0.01,
        iou_threshold=0.7,
        use_gpu=False
    )
    
    if not detector.model:
        log.error("❌ Model failed to load!")
        return False
    
    log.info(f"✅ Model loaded: {detector.is_custom_model}")
    log.info(f"   Model classes: {detector.model.names}")
    
    # Check if handwritten class exists in model
    model_classes = detector.model.names.values() if hasattr(detector.model, 'names') else []
    has_handwritten = 'Handwritten' in model_classes
    log.info(f"   Model has 'Handwritten' class: {has_handwritten}")
    
    # Try to find a real image
    storage_path = Path(settings.STORAGE_ROOT)
    test_image_path = None
    
    for storage_dir in [settings.STORAGE_PROCESSED_BASIC, settings.STORAGE_PROCESSED_ADVANCED, settings.STORAGE_RAW]:
        storage_dir_path = Path(storage_dir)
        if storage_dir_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images = list(storage_dir_path.glob(ext))
                if images:
                    test_image_path = images[0]
                    break
            if test_image_path:
                break
    
    if not test_image_path:
        log.warning("No test image found, creating dummy image...")
        test_image = Image.new('RGB', (800, 1000), color='white')
    else:
        log.info(f"\n2. Loading test image: {test_image_path}")
        try:
            test_image = Image.open(test_image_path)
            log.info(f"   Image size: {test_image.size}, mode: {test_image.mode}")
        except Exception as e:
            log.error(f"   Failed to load image: {e}")
            return False
    
    # Test detection
    log.info("\n3. Running YOLOv8s detection...")
    try:
        results = detector.detect(test_image)
        
        log.info(f"\n4. Detection Results:")
        log.info(f"   Tables: {len(results.get('tables', []))}")
        log.info(f"   Text blocks: {len(results.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(results.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(results.get('checkboxes', []))}")
        log.info(f"   Handwritten: {len(results.get('handwritten', []))}")
        
        total = (len(results.get('tables', [])) + 
                len(results.get('text_blocks', [])) + 
                len(results.get('signatures', [])) + 
                len(results.get('checkboxes', [])) + 
                len(results.get('handwritten', [])))
        log.info(f"   Total: {total}")
        
        # Check if handwritten key exists
        if 'handwritten' in results:
            log.info("✅ Handwritten key exists in results")
            if len(results['handwritten']) > 0:
                log.info(f"✅ Found {len(results['handwritten'])} handwritten regions")
                sample = results['handwritten'][0]
                log.info(f"   Sample handwritten: confidence={sample.get('confidence', 0):.3f}, bbox={sample.get('bbox')}")
            else:
                log.info("   No handwritten regions detected (may not exist in image)")
        else:
            log.error("❌ Handwritten key missing from results!")
            return False
        
        return True
        
    except Exception as e:
        log.error(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_service():
    """Test LayoutService with handwritten class."""
    log.info("\n" + "=" * 60)
    log.info("LayoutService Handwritten Class Test")
    log.info("=" * 60)
    
    # Initialize service
    log.info("\n1. Initializing LayoutService...")
    service = LayoutService(yolo_model_version="v1.0")
    
    # Try to find a real image ID
    storage_path = Path(settings.STORAGE_PROCESSED_BASIC)
    test_id = None
    
    if storage_path.exists():
        images = list(storage_path.glob("pre_0_*.png"))
        if images:
            # Extract ID from filename
            test_id = images[0].stem
            log.info(f"\n2. Testing with image ID: {test_id}")
    
    if not test_id:
        log.warning("No test image ID found, skipping LayoutService test")
        return True
    
    try:
        # Test layout detection
        log.info("\n3. Running layout detection...")
        result = service.detect_layout(pre_0_id=test_id)
        
        log.info(f"\n4. Layout Detection Results:")
        log.info(f"   Tables: {len(result.get('tables', []))}")
        log.info(f"   Text blocks: {len(result.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(result.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(result.get('checkboxes', []))}")
        log.info(f"   Handwritten: {len(result.get('handwritten', []))}")
        
        # Check if handwritten key exists
        if 'handwritten' in result:
            log.info("✅ Handwritten key exists in layout result")
            if len(result['handwritten']) > 0:
                log.info(f"✅ Found {len(result['handwritten'])} handwritten regions in layout")
            else:
                log.info("   No handwritten regions in layout (may not exist in image)")
        else:
            log.error("❌ Handwritten key missing from layout result!")
            return False
        
        # Check that PaddleOCR is excluded when YOLO works
        total_yolo = (len(result.get('tables', [])) + 
                     len(result.get('text_blocks', [])) + 
                     len(result.get('signatures', [])) + 
                     len(result.get('checkboxes', [])) + 
                     len(result.get('handwritten', [])))
        
        if total_yolo > 0:
            # Check for PaddleOCR text_block_refined
            text_blocks = result.get('text_blocks', [])
            paddleocr_count = sum(1 for tb in text_blocks if tb.get('source') == 'PaddleOCR' or tb.get('type') == 'text_block_refined')
            if paddleocr_count > 0:
                log.warning(f"⚠️  Found {paddleocr_count} PaddleOCR results in text_blocks (should be 0 when YOLO works)")
            else:
                log.info("✅ PaddleOCR correctly excluded from layout (YOLO is working)")
        
        return True
        
    except Exception as e:
        log.error(f"\n❌ LayoutService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_yolo_handwritten()
    success2 = test_layout_service()
    
    if success1 and success2:
        log.info("\n" + "=" * 60)
        log.info("✅ All tests passed!")
        log.info("=" * 60)
        sys.exit(0)
    else:
        log.error("\n" + "=" * 60)
        log.error("❌ Some tests failed")
        log.error("=" * 60)
        sys.exit(1)



