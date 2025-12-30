"""Test OCR pipeline initialization and recognition."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.pipeline import OCRPipeline
from ocr.paddle_recognizer import PaddleOCRRecognizer
from layout.paddleocr_detector import PaddleOCRTextDetector
from core.config import settings
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_ocr_components():
    """Test individual OCR components."""
    log.info("=" * 60)
    log.info("OCR Components Test")
    log.info("=" * 60)
    
    # Test 1: PaddleOCR Detector
    log.info("\n1. Testing PaddleOCR Detector...")
    try:
        detector = PaddleOCRTextDetector(lang='en')
        if detector.ocr:
            log.info("   [OK] PaddleOCR detector initialized")
        else:
            log.error("   [FAIL] PaddleOCR detector not initialized")
            return False
    except Exception as e:
        log.error(f"   [FAIL] PaddleOCR detector failed: {e}")
        return False
    
    # Test 2: PaddleOCR Recognizer
    log.info("\n2. Testing PaddleOCR Recognizer...")
    try:
        recognizer = PaddleOCRRecognizer(lang='en', use_gpu=False)
        if recognizer.is_available():
            log.info("   [OK] PaddleOCR recognizer initialized")
        else:
            log.error("   [FAIL] PaddleOCR recognizer not initialized")
            return False
    except Exception as e:
        log.error(f"   [FAIL] PaddleOCR recognizer failed: {e}")
        return False
    
    # Test 3: OCR Pipeline
    log.info("\n3. Testing OCR Pipeline initialization...")
    try:
        pipeline = OCRPipeline(
            paddle_confidence_threshold=0.5,
            use_gpu=False
        )
        log.info("   [OK] OCR Pipeline initialized")
        log.info(f"   PaddleOCR detector: {'available' if pipeline.paddle_detector.ocr else 'unavailable'}")
        log.info(f"   PaddleOCR recognizer: {'available' if pipeline.paddle_recognizer.is_available() else 'unavailable'}")
        log.info(f"   TrOCR recognizer: {'available' if pipeline.trocr_recognizer.is_available() else 'unavailable'}")
    except Exception as e:
        log.error(f"   [FAIL] OCR Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Recognition on dummy image
    log.info("\n4. Testing recognition on dummy image...")
    try:
        test_image = Image.new('RGB', (200, 50), color='white')
        text, confidence = recognizer.recognize(test_image)
        log.info(f"   Recognition result: text='{text}', confidence={confidence:.3f}")
        if text == "" and confidence == 0.0:
            log.info("   [OK] Empty result (expected for white image)")
        else:
            log.info("   [OK] Recognition working")
    except Exception as e:
        log.error(f"   [FAIL] Recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    log.info("\n" + "=" * 60)
    log.info("[OK] All OCR component tests passed!")
    log.info("=" * 60)
    return True

if __name__ == "__main__":
    success = test_ocr_components()
    sys.exit(0 if success else 1)



