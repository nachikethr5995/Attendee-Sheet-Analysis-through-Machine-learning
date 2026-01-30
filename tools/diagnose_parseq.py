"""Quick diagnostic script to check PARSeq setup.

Run this to verify:
1. Dependencies are installed
2. PARSeq can be imported/loaded
3. Device compatibility
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("PARSeq Diagnostic Check")
print("=" * 60)

# Check 1: Transformers
print("\n[1] Checking transformers...")
try:
    import transformers
    print(f"[OK] transformers: {transformers.__version__}")
except ImportError as e:
    print(f"[FAIL] transformers not available: {e}")
    sys.exit(1)

# Check 2: PyTorch
print("\n[2] Checking PyTorch...")
try:
    import torch
    print(f"[OK] torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"[FAIL] torch not available: {e}")
    sys.exit(1)

# Check 3: Other dependencies
print("\n[3] Checking other dependencies...")
deps = ['timm', 'PIL', 'cv2', 'einops']
for dep in deps:
    try:
        if dep == 'PIL':
            import PIL
            print(f"[OK] {dep}: {PIL.__version__}")
        elif dep == 'cv2':
            import cv2
            print(f"[OK] {dep}: {cv2.__version__}")
        else:
            mod = __import__(dep)
            print(f"[OK] {dep}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"[FAIL] {dep} not available")

# Check 4: Try to import PARSeq components
print("\n[4] Checking PARSeq imports...")
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("[OK] AutoProcessor and AutoModelForVision2Seq available")
except ImportError as e:
    print(f"[FAIL] PARSeq components not available: {e}")
    sys.exit(1)

# Check 5: Try to load a test model (this will fail if model doesn't exist)
print("\n[5] Testing model loading...")
test_model = "baudm/parseq"
print(f"  Attempting to load: {test_model}")
try:
    # Just try to get processor info, don't fully load
    from huggingface_hub import model_info
    info = model_info(test_model)
    print(f"[OK] Model '{test_model}' exists on HuggingFace")
    print(f"  Model ID: {info.id}")
except Exception as e:
    print(f"[WARN] Model '{test_model}' may not exist: {e}")
    print("  This is expected if PARSeq is not available as a HuggingFace model")
    print("  You may need to:")
    print("    - Use a different model name")
    print("    - Load PARSeq from local checkpoint")
    print("    - Use TrOCR as an alternative")

# Check 6: Try to initialize PARSeqRecognizer
print("\n[6] Testing PARSeqRecognizer initialization...")
try:
    from ocr.parseq_recognizer import PARSeqRecognizer
    recognizer = PARSeqRecognizer(use_gpu=False)  # Force CPU for testing
    if recognizer.is_available():
        print("[OK] PARSeqRecognizer initialized successfully")
    else:
        print("[FAIL] PARSeqRecognizer initialization failed (check logs above)")
except Exception as e:
    print(f"[FAIL] PARSeqRecognizer initialization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)

