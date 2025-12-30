"""Quick test script to verify YOLOv8s model is working."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.yolov8_layout_detector import YOLOv8LayoutDetector
from core.config import settings
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_yolov8():
    """Test YOLOv8s model loading and detection."""
    log.info("=" * 60)
    log.info("YOLOv8s Model Test")
    log.info("=" * 60)
    
    # Check model directory
    model_dir = Path(settings.MODELS_ROOT) / "yolo_layout"
    log.info(f"Model directory: {model_dir}")
    log.info(f"Directory exists: {model_dir.exists()}")
    
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pt"))
        log.info(f"Model files found: {[f.name for f in model_files]}")
    
    # Initialize detector
    log.info("\nInitializing YOLOv8s detector...")
    detector = YOLOv8LayoutDetector(
        model_version="v1.0",
        confidence_threshold=0.15,
        use_gpu=False  # Use CPU for testing
    )
    
    if not detector.model:
        log.error("❌ Model failed to load!")
        return False
    
    log.info(f"✅ Model loaded successfully")
    log.info(f"   Is custom model: {detector.is_custom_model}")
    log.info(f"   Device: {detector.device}")
    log.info(f"   Confidence threshold: {detector.confidence_threshold}")
    
    if hasattr(detector.model, 'names'):
        log.info(f"   Model classes: {detector.model.names}")
    
    # Test with a dummy image (white image)
    log.info("\nTesting detection with dummy image...")
    test_image = Image.new('RGB', (800, 1000), color='white')
    
    try:
        results = detector.detect(test_image)
        log.info(f"\nDetection results:")
        log.info(f"   Tables: {len(results.get('tables', []))}")
        log.info(f"   Text blocks: {len(results.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(results.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(results.get('checkboxes', []))}")
        
        total = sum(len(results.get(k, [])) for k in ['tables', 'text_blocks', 'signatures', 'checkboxes'])
        log.info(f"   Total detections: {total}")
        
        if total > 0:
            log.info("✅ Model is working!")
            return True
        else:
            log.warning("⚠️  Model loaded but detected 0 items (this is expected for a white image)")
            log.info("✅ Model is working, but needs a real document image to detect items")
            return True
            
    except Exception as e:
        log.error(f"❌ Detection failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_yolov8()
    sys.exit(0 if success else 1)



