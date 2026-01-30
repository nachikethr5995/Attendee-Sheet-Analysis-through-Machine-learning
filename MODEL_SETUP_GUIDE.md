# Model Setup Guide - GPDS Signature & Checkbox Models

## Overview

Your pipeline requires two optional model files that are **not included** in the repository:

1. **GPDS Signature Model** (`gpds_signature.pt`) - For signature detection
2. **Checkbox Model** (`checkbox_best.pt`) - For checkbox detection

**Important:** These are **NOT** files that can be copied/renamed. They are trained model weights that must be obtained separately.

---

## 1. GPDS Signature Model (`gpds_signature.pt`)

### What It Is
A pretrained YOLOv5/YOLOv8 model specifically trained for signature detection using the GPDS (Graphic Pattern Description System) dataset.

### Where to Get It

**Option A: Download Pretrained Model (Recommended)**
1. Search for "GPDS signature detection YOLOv5" or "GPDS signature YOLOv8" on:
   - GitHub (search for GPDS signature detection repositories)
   - Hugging Face Model Hub
   - Roboflow Universe
   - Papers with Code

2. Common sources:
   - GPDS-960 dataset repositories often include pretrained models
   - YOLOv5 signature detection models on GitHub
   - Community-trained models on Hugging Face

3. Once downloaded, place the `.pt` file at:
   ```
   Back_end/models/signature_model/gpds_signature.pt
   ```

**Option B: Train Your Own (Advanced)**
If you have access to GPDS dataset:
```bash
# Train YOLOv8 on GPDS signature dataset
yolo detect train data=gpds_signature.yaml model=yolov8n.pt epochs=100 imgsz=640
# Save the best.pt as gpds_signature.pt
```

**Option C: Use Alternative Signature Detection**
You can modify the code to use YOLOv8s layout detection for signatures (class 2) instead of GPDS:
- The layout detector already detects signatures as part of its 5-class model
- GPDS is optional and provides additional signature detection capability

### File Location
```
Back_end/models/signature_model/gpds_signature.pt
```

### Verification
After placing the file, restart your application. You should see:
```
INFO | Loading GPDS signature model from: models/signature_model/gpds_signature.pt
INFO | GPDS signature model loaded successfully
```

---

## 2. Checkbox Model (`checkbox_best.pt`)

### What It Is
A custom-trained YOLOv8 model specifically for checkbox detection. This **must be trained** on your checkbox images.

### How to Get It

**Step 1: Prepare Dataset**
1. Collect 20-50 images containing checkboxes
2. Annotate checkboxes using:
   - **LabelImg**: https://github.com/HumanSignal/labelImg
   - **Roboflow**: https://roboflow.com
   - **CVAT**: https://cvat.org

3. Export annotations in YOLO format:
   ```
   class_id center_x center_y width height
   ```
   Where `class_id = 0` for checkbox (single class)

**Step 2: Create Dataset Structure**
```bash
mkdir -p dataset/checkbox/images/train
mkdir -p dataset/checkbox/images/val
mkdir -p dataset/checkbox/labels/train
mkdir -p dataset/checkbox/labels/val
```

**Step 3: Create dataset.yaml**
```yaml
path: ./dataset/checkbox
train: images/train
val: images/val
names:
  0: checkbox
nc: 1
```

**Step 4: Train the Model**
```bash
cd Back_end
# Activate your virtual environment
source venv_wsl/bin/activate  # or venv\Scripts\activate on Windows

# Train YOLOv8n (nano) - fastest, good for checkboxes
yolo detect train data=dataset/checkbox/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16

# Or use the training script
python train_yolo_layout.py
```

**Step 5: Copy Best Model**
After training, copy the best model:
```bash
# Training saves to: runs/detect/train/weights/best.pt
mkdir -p models/checkbox_model
cp runs/detect/train/weights/best.pt models/checkbox_model/checkbox_best.pt
```

### File Location
```
Back_end/models/checkbox_model/checkbox_best.pt
```

### Quick Training Command
```bash
# Minimum viable training (20-30 images, ~10 minutes)
yolo detect train \
  data=dataset/checkbox/dataset.yaml \
  model=yolov8n.pt \
  epochs=25 \
  imgsz=640 \
  batch=8 \
  project=runs/checkbox \
  name=train

# Copy best model
cp runs/checkbox/train/weights/best.pt models/checkbox_model/checkbox_best.pt
```

### Verification
After placing the file, restart your application. You should see:
```
INFO | Loading checkbox model from: models/checkbox_model/checkbox_best.pt
INFO | Checkbox model loaded successfully
```

---

## 3. Alternative: Use Layout Detector for Both

**Good News:** Your YOLOv8s layout detector already detects both signatures and checkboxes!

The layout model (`models/yolo_layout/best.pt`) includes:
- Class 0: text_block
- Class 1: table
- Class 2: signature ✅
- Class 3: checkbox ✅

**So you can:**
1. **Skip GPDS** - Use layout detector for signatures (class 2)
2. **Skip checkbox training** - Use layout detector for checkboxes (class 3)

The GPDS and checkbox-specific models are **optional enhancements** that provide:
- Better accuracy for signatures/checkboxes specifically
- Additional detection capability (secondary detector)

---

## 4. Current Status Check

### Check What Models You Have
```bash
# Check signature model
ls -la Back_end/models/signature_model/

# Check checkbox model
ls -la Back_end/models/checkbox_model/

# Check layout model (should exist)
ls -la Back_end/models/yolo_layout/
```

### What Happens Without These Models

**Without GPDS Signature Model:**
- ✅ Layout detector still detects signatures (class 2)
- ⚠️ No secondary GPDS signature detection
- ✅ Pipeline continues normally

**Without Checkbox Model:**
- ✅ Layout detector still detects checkboxes (class 3)
- ⚠️ No secondary checkbox-specific detection
- ✅ Pipeline continues normally

**Both are optional** - your pipeline will work with just the layout detector!

---

## 5. Quick Setup (If You Have Existing Models)

If you have `.pt` files elsewhere that might be these models:

### For GPDS Signature:
```bash
# If you have a signature detection model with a different name
cp /path/to/your/signature_model.pt Back_end/models/signature_model/gpds_signature.pt
```

### For Checkbox:
```bash
# If you have a trained checkbox model
mkdir -p Back_end/models/checkbox_model
cp /path/to/your/checkbox_model.pt Back_end/models/checkbox_model/checkbox_best.pt
```

**Note:** The models must be in YOLOv5/YOLOv8 format (`.pt` files compatible with Ultralytics YOLO).

---

## 6. Summary

| Model | Required? | How to Get | Location |
|-------|-----------|------------|----------|
| **GPDS Signature** | ❌ Optional | Download pretrained or train | `models/signature_model/gpds_signature.pt` |
| **Checkbox** | ❌ Optional | Train on your data | `models/checkbox_model/checkbox_best.pt` |
| **Layout (YOLOv8s)** | ✅ Required | Already included | `models/yolo_layout/best.pt` |

**Bottom Line:**
- These are **NOT** files you can just copy/rename
- GPDS needs to be **downloaded** (pretrained model)
- Checkbox needs to be **trained** (custom model)
- **Both are optional** - your layout detector handles signatures and checkboxes already!

---

## 7. Need Help?

If you need help finding or training these models:
1. **GPDS Signature**: Search GitHub for "GPDS signature YOLOv5" or "signature detection YOLOv8"
2. **Checkbox**: Use the training guide in `TRAINING_GUIDE.md` or train with 20-50 annotated images
3. **Alternative**: Just use the layout detector - it already works for both!






