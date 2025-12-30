"""Test full layout service to verify signatures and checkboxes are included."""

import sys
from pathlib import Path

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from layout.layout_service import LayoutService
from core.config import settings
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def test_layout_service_full():
    """Test full layout service with the specific image."""
    log.info("=" * 60)
    log.info("Full Layout Service Test")
    log.info("=" * 60)
    
    # Initialize service
    log.info("\n1. Initializing LayoutService...")
    service = LayoutService(yolo_model_version="v1.0")
    
    # Test with the specific image ID
    test_id = "pre_0_20251229_191005_185b70d3"
    log.info(f"\n2. Testing with image ID: {test_id}")
    
    try:
        # Test layout detection
        log.info("\n3. Running layout detection...")
        result = service.detect_layout(pre_0_id=test_id)
        
        log.info(f"\n4. Final Layout Detection Results:")
        log.info(f"   Tables: {len(result.get('tables', []))}")
        log.info(f"   Text blocks: {len(result.get('text_blocks', []))}")
        log.info(f"   Signatures: {len(result.get('signatures', []))}")
        log.info(f"   Checkboxes: {len(result.get('checkboxes', []))}")
        log.info(f"   Handwritten: {len(result.get('handwritten', []))}")
        
        # Check signatures
        signatures = result.get('signatures', [])
        if len(signatures) > 0:
            log.info(f"\n✅ Signatures found: {len(signatures)}")
            for i, sig in enumerate(signatures[:3], 1):  # Show first 3
                log.info(f"   Signature {i}: conf={sig.get('confidence', 0):.3f}, source={sig.get('source')}")
        else:
            log.error("\n❌ NO SIGNATURES in final result!")
        
        # Check checkboxes
        checkboxes = result.get('checkboxes', [])
        if len(checkboxes) > 0:
            log.info(f"\n✅ Checkboxes found: {len(checkboxes)}")
            for i, cb in enumerate(checkboxes[:3], 1):  # Show first 3
                log.info(f"   Checkbox {i}: conf={cb.get('confidence', 0):.3f}, source={cb.get('source')}")
        else:
            log.error("\n❌ NO CHECKBOXES in final result!")
        
        # Save result to file for inspection
        output_path = Path(settings.STORAGE_ROOT) / "test_layout_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"\n5. Full result saved to: {output_path}")
        
        return len(signatures) > 0 and len(checkboxes) > 0
        
    except Exception as e:
        log.error(f"\n❌ LayoutService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_layout_service_full()
    
    if success:
        log.info("\n" + "=" * 60)
        log.info("✅ Test passed - Signatures and checkboxes are included!")
        log.info("=" * 60)
        sys.exit(0)
    else:
        log.error("\n" + "=" * 60)
        log.error("❌ Test failed - Signatures or checkboxes missing")
        log.error("=" * 60)
        sys.exit(1)



