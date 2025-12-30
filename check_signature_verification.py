#!/usr/bin/env python3
"""Check GPDS Signature Verification setup."""

import sys
from pathlib import Path

print("=" * 60)
print("GPDS Signature Verification Requirements Check")
print("=" * 60)
print()

# Check Python version
print("1. Python Version:")
try:
    version = sys.version_info
    print(f"   Python: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version OK (3.8+)")
    else:
        print("   ⚠️  Python 3.8+ recommended")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check PyTorch
print("2. PyTorch:")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("   ❌ PyTorch NOT installed")
    print("   Install with: pip install torch torchvision")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check Ultralytics (YOLO)
print("3. Ultralytics (YOLO):")
try:
    from ultralytics import YOLO
    print("   ✅ Ultralytics installed")
    # Try to create a YOLO instance (without loading model)
    print("   ✅ YOLO import OK")
except ImportError:
    print("   ❌ Ultralytics NOT installed")
    print("   Install with: pip install ultralytics")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check OpenCV
print("4. OpenCV:")
try:
    import cv2
    print(f"   ✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("   ❌ OpenCV NOT installed")
    print("   Install with: pip install opencv-python")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check Pillow
print("5. Pillow:")
try:
    from PIL import Image
    print("   ✅ Pillow installed")
except ImportError:
    print("   ❌ Pillow NOT installed")
    print("   Install with: pip install Pillow")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check NumPy
print("6. NumPy:")
try:
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
except ImportError:
    print("   ❌ NumPy NOT installed")
    print("   Install with: pip install numpy")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Check GPDS model files
print("7. GPDS Model Files:")
from core.config import settings

model_paths = [
    Path(settings.MODELS_ROOT) / "signature_model" / "gpds_verification.pt",
    Path(settings.MODELS_ROOT) / "signature_model" / "gpds_signature.pt"
]

model_found = False
for model_path in model_paths:
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model found: {model_path}")
        print(f"      Size: {size_mb:.1f} MB")
        model_found = True
        break

if not model_found:
    print("   ⚠️  GPDS model NOT found")
    print("   Expected locations:")
    for model_path in model_paths:
        print(f"      - {model_path}")
    print("   Download GPDS verification model and place in one of the above locations")
print()

# Check signature verification module
print("8. Signature Verification Module:")
try:
    from postprocessing.signature_verifier import GPDSSignatureVerifier
    from postprocessing.signature_verification_service import SignatureVerificationService
    print("   ✅ Signature verifier: import OK")
    print("   ✅ Verification service: import OK")
    
    # Try to initialize (without model)
    verifier = GPDSSignatureVerifier()
    print(f"   ✅ Verifier initialized: {'available' if verifier.is_available() else 'model not found'}")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")
print()

print("=" * 60)
print("Summary")
print("=" * 60)

# Final check
missing = []
try:
    import torch
except ImportError:
    missing.append("PyTorch")

try:
    from ultralytics import YOLO
except ImportError:
    missing.append("Ultralytics")

try:
    import cv2
except ImportError:
    missing.append("OpenCV")

try:
    from PIL import Image
except ImportError:
    missing.append("Pillow")

if missing:
    print(f"❌ Missing dependencies: {', '.join(missing)}")
    print()
    print("Install missing dependencies:")
    if "PyTorch" in missing:
        print("  pip install torch torchvision")
    if "Ultralytics" in missing:
        print("  pip install ultralytics")
    if "OpenCV" in missing:
        print("  pip install opencv-python")
    if "Pillow" in missing:
        print("  pip install Pillow")
else:
    print("✅ All dependencies installed!")
    if not model_found:
        print("⚠️  GPDS model not found (verification will be unavailable)")
        print("   Download model and place at: models/signature_model/gpds_verification.pt")
    else:
        print("✅ GPDS Signature Verification ready to use!")

print()




