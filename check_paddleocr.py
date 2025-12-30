#!/usr/bin/env python3
"""Check PaddleOCR installation and initialization."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("PaddleOCR Installation Check")
print("=" * 60)
print()

# Check 1: Import
print("1. Checking PaddleOCR import...")
try:
    from paddleocr import PaddleOCR
    print("   [OK] PaddleOCR imported successfully")
except ImportError as e:
    print(f"   [FAIL] PaddleOCR import failed: {e}")
    print("   Install with: pip install paddlepaddle paddleocr")
    sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Unexpected error during import: {e}")
    sys.exit(1)

# Check 2: PaddlePaddle
print("\n2. Checking PaddlePaddle...")
try:
    import paddle
    print(f"   [OK] PaddlePaddle version: {paddle.__version__}")
except ImportError:
    print("   [FAIL] PaddlePaddle not installed")
    print("   Install with: pip install paddlepaddle")
    sys.exit(1)

# Check 3: Initialize PaddleOCR (try different methods)
print("\n3. Testing PaddleOCR initialization...")

# Method 1: Absolute minimal (most compatible)
print("   Method 1: Absolute minimal initialization (lang only)...")
ocr = None
try:
    ocr = PaddleOCR(lang='en')
    print("   [OK] Initialized with minimal parameters")
except Exception as e1:
    print(f"   [WARN] Minimal init failed: {e1}")
    # Method 2: With use_textline_orientation (newer versions)
    print("   Method 2: With use_textline_orientation parameter...")
    try:
        ocr = PaddleOCR(lang='en', use_textline_orientation=False)
        print("   [OK] Initialized with use_textline_orientation=False")
    except (TypeError, ValueError) as e2:
        print(f"   [WARN] use_textline_orientation not supported: {e2}")
        # Method 3: With show_log (older versions)
        print("   Method 3: With show_log parameter...")
        try:
            ocr = PaddleOCR(lang='en', show_log=False)
            print("   [OK] Initialized with show_log=False")
        except Exception as e3:
            print(f"   [FAIL] All initialization methods failed")
            print(f"   Last error: {e3}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Test recognition
print("\n4. Testing PaddleOCR recognition...")
if ocr is None:
    print("   [SKIP] OCR not initialized, skipping recognition test")
else:
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_img = Image.new('RGB', (100, 30), color='white')
        test_array = np.array(test_img)
        
        # Test OCR
        import cv2
        img_bgr = cv2.cvtColor(test_array, cv2.COLOR_RGB2BGR)
        
        # Try new API first (predict), fallback to old API (ocr)
        try:
            result = ocr.predict(img_bgr)
            print("   [OK] PaddleOCR recognition test passed (using predict() method)")
            print(f"   Result type: {type(result)}")
        except AttributeError:
            try:
                result = ocr.ocr(img_bgr)
                print("   [OK] PaddleOCR recognition test passed (using ocr() method)")
                print(f"   Result type: {type(result)}")
            except Exception as e:
                print(f"   [WARN] Recognition test failed: {e}")
        except Exception as e:
            print(f"   [WARN] Recognition test failed: {e}")
        
    except Exception as e:
        print(f"   [WARN] Recognition test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("[OK] PaddleOCR check complete!")
print("=" * 60)

