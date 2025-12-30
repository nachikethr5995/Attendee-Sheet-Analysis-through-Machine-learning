"""Test signature and checkbox detection with detailed diagnostics."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.yolov8_layout_detector import YOLOv8LayoutDetector
from core.config import settings
from PIL import Image
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_signature_checkbox_detection():
    """Test signature and checkbox detection with detailed diagnostics."""
    log.info("=" * 60)
    log.info("Signature & Checkbox Detection Diagnostic Test")
    log.info("=" * 60)
    
    # Initialize detector
    log.info("\n1. Initializing YOLOv8s detector...")
    detector = YOLOv8LayoutDetector(
        model_version="v1.0",
        confidence_threshold=0.01,  # Very low to catch all detections
        iou_threshold=0.7,
        use_gpu=False
    )
    
    if not detector.model:
        log.error("❌ Model failed to load!")
        return False
    
    log.info(f"✅ Model loaded")
    log.info(f"   Model classes: {detector.model.names}")
    
    # Load the specific image
    image_path = Path(settings.STORAGE_PROCESSED_BASIC) / "pre_0_20251229_191005_185b70d3.png"
    if not image_path.exists():
        log.error(f"❌ Image not found: {image_path}")
        return False
    
    log.info(f"\n2. Loading test image: {image_path}")
    try:
        test_image = Image.open(image_path)
        log.info(f"   Image size: {test_image.size}, mode: {test_image.mode}")
    except Exception as e:
        log.error(f"   Failed to load image: {e}")
        return False
    
    # Test detection
    log.info("\n3. Running YOLOv8s detection...")
    try:
        results = detector.detect(test_image)
        
        log.info(f"\n4. Detection Results Summary:")
        log.info(f"   Tables: {len(results.get('tables', []))}")
        log.info(f"   Text blocks: {len(results.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(results.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(results.get('checkboxes', []))}")
        log.info(f"   Handwritten: {len(results.get('handwritten', []))}")
        
        # Detailed signature analysis
        signatures = results.get('signatures', [])
        if len(signatures) > 0:
            log.info(f"\n5. Signature Details ({len(signatures)} found):")
            for i, sig in enumerate(signatures, 1):
                log.info(f"   Signature {i}:")
                log.info(f"     Confidence: {sig.get('confidence', 0):.3f}")
                log.info(f"     Bbox: {sig.get('bbox')}")
                log.info(f"     Class ID: {sig.get('class_id')}")
                log.info(f"     Source: {sig.get('source')}")
        else:
            log.warning("\n5. ⚠️  NO SIGNATURES DETECTED")
            log.warning("   This could mean:")
            log.warning("   - Model is not detecting signatures in this image")
            log.warning("   - Signatures are being filtered out (check raw detections)")
            log.warning("   - Class mapping issue")
        
        # Detailed checkbox analysis
        checkboxes = results.get('checkboxes', [])
        if len(checkboxes) > 0:
            log.info(f"\n6. Checkbox Details ({len(checkboxes)} found):")
            for i, cb in enumerate(checkboxes, 1):
                log.info(f"   Checkbox {i}:")
                log.info(f"     Confidence: {cb.get('confidence', 0):.3f}")
                log.info(f"     Bbox: {cb.get('bbox')}")
                log.info(f"     Class ID: {cb.get('class_id')}")
                log.info(f"     Source: {cb.get('source')}")
        else:
            log.warning("\n6. ⚠️  NO CHECKBOXES DETECTED")
            log.warning("   This could mean:")
            log.warning("   - Model is not detecting checkboxes in this image")
            log.warning("   - Checkboxes are being filtered out (check raw detections)")
            log.warning("   - Class mapping issue")
        
        # Check raw model output
        log.info("\n7. Checking raw model output...")
        import cv2
        import numpy as np
        from core.utils import image_to_array
        
        img_array = image_to_array(test_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run inference directly
        raw_results = detector.model.predict(
            img_bgr,
            conf=0.01,  # Very low confidence
            iou=0.7,
            verbose=False
        )
        
        if raw_results and len(raw_results) > 0:
            result = raw_results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                log.info(f"   Raw detections: {len(result.boxes)} boxes")
                
                # Count by class
                class_counts = {}
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = detector.model.names.get(class_id, 'unknown')
                    confidence = float(box.conf[0])
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = []
                    class_counts[class_name].append(confidence)
                
                log.info("   Raw class distribution:")
                for class_name, confidences in class_counts.items():
                    log.info(f"     {class_name}: {len(confidences)} detections, "
                            f"conf range: {min(confidences):.3f}-{max(confidences):.3f}")
                
                # Check specifically for Signature and Checkbox
                if 'Signature' in class_counts:
                    log.info(f"\n   ✅ Signature class detected in raw output: {len(class_counts['Signature'])} boxes")
                    log.info(f"      Confidences: {[f'{c:.3f}' for c in class_counts['Signature']]}")
                else:
                    log.warning("\n   ⚠️  Signature class NOT found in raw output")
                
                if 'Checkbox' in class_counts:
                    log.info(f"\n   ✅ Checkbox class detected in raw output: {len(class_counts['Checkbox'])} boxes")
                    log.info(f"      Confidences: {[f'{c:.3f}' for c in class_counts['Checkbox']]}")
                else:
                    log.warning("\n   ⚠️  Checkbox class NOT found in raw output")
            else:
                log.warning("   No boxes in raw results")
        else:
            log.warning("   No raw results")
        
        return True
        
    except Exception as e:
        log.error(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signature_checkbox_detection()
    
    if success:
        log.info("\n" + "=" * 60)
        log.info("✅ Diagnostic test completed")
        log.info("=" * 60)
        sys.exit(0)
    else:
        log.error("\n" + "=" * 60)
        log.error("❌ Diagnostic test failed")
        log.error("=" * 60)
        sys.exit(1)



