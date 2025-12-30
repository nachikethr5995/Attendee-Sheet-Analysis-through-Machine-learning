# YOLOv8s Layout Detection Training Guide

## Overview

This comprehensive guide covers training a custom YOLOv8s model for document layout detection with fixed class mappings and training-ready architecture.

**Key Features:**
- ✅ One-command training
- ✅ Model versioning support
- ✅ Incremental dataset growth
- ✅ Resume from checkpoints
- ✅ Production-ready implementation

---

## Quick Start

### One-Command Training

```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate

# Train with default settings
python3 training/train_yolov8_layout.py --data dataset/dataset.yaml --epochs 80
```

### Minimum Requirements

- **Training images**: 80-100
- **Validation images**: 20-30
- **Format**: YOLO (images + labels)
- **Classes**: text_block, table, signature, checkbox

---

## Fixed Class Mappings

**⚠️ CRITICAL: These class indices must NEVER change across training runs**

```
0: text_block
1: table
2: signature
3: checkbox
```

**Important**: Always use these exact mappings when annotating. Changing them will break existing models.

---

## Dataset Structure

### Directory Structure

```
dataset/
├── images/
│   ├── train/          # Training images (80-100 minimum)
│   ├── val/            # Validation images (20-30 minimum)
│   └── test/           # Test images (optional)
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   ├── val/            # Validation labels
│   └── test/           # Test labels (optional)
└── dataset.yaml        # Dataset configuration (auto-generated)
```

### Annotation Format (YOLO)

Each image must have a corresponding `.txt` label file with the same name.

**Format:** `class_id center_x center_y width height` (all normalized 0-1)

**Example (`image001.txt`):**
```
0 0.5 0.3 0.4 0.2
1 0.2 0.6 0.3 0.25
2 0.8 0.9 0.15 0.1
```

Where:
- `0` = text_block
- `1` = table
- `2` = signature
- `3` = checkbox

### Annotation Rules

1. **Signature**: Annotate the full visible signature ink region
2. **Checkbox**: Annotate the full checkbox boundary (including border)
3. **Text blocks**: Group logical text regions (paragraphs, sections), not individual words
4. **Tables**: Full table boundary only (not individual cells)

---

## Dataset Preparation

### Step 1: Organize Images

```bash
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}
```

### Step 2: Annotate Images

Use annotation tools:
- **LabelImg**: https://github.com/HumanSignal/labelImg
- **Roboflow**: https://roboflow.com
- **CVAT**: https://cvat.org

**Important**: Export in YOLO format with class mappings:
- `text_block` → class 0
- `table` → class 1
- `signature` → class 2
- `checkbox` → class 3

### Step 3: Create dataset.yaml

```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate
python3 training/prepare_dataset.py dataset/ --create-yaml --validate --stats
```

This will:
- Validate dataset structure
- Show statistics
- Create `dataset.yaml`

---

## Training

### Basic Training

```bash
cd /mnt/d/AIME/Back_end
source venv_wsl/bin/activate
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --epochs 80 \
  --imgsz 1024 \
  --batch 8 \
  --version v1.0
```

### Training with Custom Model

```bash
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --version v2.0
```

### Resume Training

```bash
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --resume runs/layout/yolov8s_layout/weights/last.pt \
  --epochs 80
```

### CPU Training (for testing)

```bash
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --device cpu \
  --batch 4
```

---

## Training Parameters

### Recommended Settings (Initial Phase)

- **Images**: ~100 (minimum)
- **Epochs**: 60-80
- **Image size**: 1024
- **Batch size**: 4-8 (adjust based on GPU memory)
- **Learning rate**: Auto (0.01 initial, 0.1 final)

### Augmentation (Automatic)

- Scaling: ±50%
- Rotation: ±3° (light, document-appropriate)
- Brightness/Contrast: Light adjustments
- Noise: Light
- **No perspective/shear** (documents are flat)

---

## Model Output

After training, models are saved to:

```
runs/layout/yolov8s_layout/
├── weights/
│   ├── best.pt        # Best model (use this)
│   └── last.pt        # Last checkpoint
```

**Auto-copied to:**
```
models/yolo_layout/
├── yolov8s_layout_v1.0.pt    # Versioned model
└── yolov8s_layout.pt         # Default model (updated)
```

---

## Model Versioning

### Versioning Strategy

- **v1.0**: Initial training (100 images)
- **v1.1**: Fine-tuned on expanded dataset (500 images)
- **v2.0**: Retrained with larger model (yolov8m)
- **v2.1**: Fine-tuned v2.0

### Swapping Models

To use a different model version:

1. **Update default model:**
   ```bash
   cp models/yolo_layout/yolov8s_layout_v2.0.pt models/yolo_layout/yolov8s_layout.pt
   ```

2. **Or specify in code:**
   ```python
   detector = YOLOv8LayoutDetector(
       model_path="models/yolo_layout/yolov8s_layout_v2.0.pt",
       model_version="v2.0"
   )
   ```

---

## Acceptance Rules

YOLOv8s output is **ACCEPTED** if ANY condition holds:

- ≥1 `text_block` with confidence ≥ 0.45
- ≥1 `signature` with confidence ≥ 0.40
- ≥1 `checkbox` with confidence ≥ 0.40

If not accepted, the system will log a warning but continue with available detections.

---

## Incremental Dataset Growth

### Phase 1: Initial (100 images)

```bash
# Train initial model
python3 training/train_yolov8_layout.py --data dataset/dataset.yaml --epochs 80 --version v1.0
```

### Phase 2: Expansion (500 images)

1. Add new images to `dataset/images/train/`
2. Add corresponding labels to `dataset/labels/train/`
3. Update validation set
4. Retrain:

```bash
# Fine-tune from v1.0
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --resume models/yolo_layout/yolov8s_layout_v1.0.pt \
  --epochs 60 \
  --version v1.1
```

### Phase 3: Large Dataset (2000+ images)

```bash
# Full retrain with larger model
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --version v2.0
```

---

## Model Upgrade Path

| Component | v1 | v2+ |
|-----------|----|-----|
| Layout | YOLOv8s | YOLOv8m/l |
| Signature | YOLOv8s | YOLOv8s (dedicated) |
| Checkbox | YOLOv8s | YOLOv8s + Classifier |

**Upgrade steps:**
1. Train new model
2. Version it (e.g., v2.0)
3. Test on validation set
4. Update default model if performance improved
5. No code changes required!

---

## Architecture

### Primary Pipeline: YOLOv8s

- **Model**: YOLOv8s (trainable, fine-tunable)
- **Classes**: Fixed mappings (never change)
  - `0: text_block`
  - `1: table`
  - `2: signature`
  - `3: checkbox`
- **Role**: Primary layout detection (all elements)
- **Training**: Fully supported, one-command execution

### File Structure

```
Back_end/
├── layout/
│   ├── yolov8_layout_detector.py    # Training-ready YOLOv8s detector
│   ├── layout_service.py            # YOLOv8s layout service
│   └── ...
├── training/
│   ├── train_yolov8_layout.py       # Training script
│   ├── prepare_dataset.py           # Dataset preparation
│   └── __init__.py
├── dataset.yaml.example             # Dataset template (fixed class mappings)
└── models/yolo_layout/              # Trained models (versioned)
    ├── yolov8s_layout.pt            # Default model
    ├── yolov8s_layout_v1.0.pt       # Versioned models
    └── yolov8s_layout_v2.0.pt
```

---

## Training Workflow

### Phase 1: Initial Training (100 images)

```bash
# 1. Prepare dataset
python3 training/prepare_dataset.py dataset/ --create-yaml --validate --stats

# 2. Train model
python3 training/train_yolov8_layout.py --data dataset/dataset.yaml --epochs 80 --version v1.0

# 3. Model auto-saved to models/yolo_layout/yolov8s_layout.pt
```

### Phase 2: Dataset Expansion (500 images)

```bash
# 1. Add new images to dataset/images/train/
# 2. Add labels to dataset/labels/train/

# 3. Fine-tune from v1.0
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --resume models/yolo_layout/yolov8s_layout_v1.0.pt \
  --epochs 60 \
  --version v1.1
```

### Phase 3: Model Upgrade (yolov8m)

```bash
# Train with larger model
python3 training/train_yolov8_layout.py \
  --data dataset/dataset.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --version v2.0
```

---

## Inference Flow

```
Input Image
    ↓
YOLOv8s Detection (PRIMARY)
    ↓
Acceptance Check
    ├─ Accepted → Use YOLOv8s results
    └─ Rejected → Log warning, continue with available detections
    ↓
Fusion & Output
```

---

## Output Format

```json
{
  "type": "text_block | table | signature | checkbox",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.85,
  "source": "YOLOv8s",
  "model_version": "v1.0",
  "is_custom_model": true
}
```

---

## Testing Trained Model

```python
from layout.yolov8_layout_detector import YOLOv8LayoutDetector

detector = YOLOv8LayoutDetector(model_version="v1.0")
print("Available:", detector.is_available())
print("Info:", detector.get_model_info())
```

Or via command line:

```bash
python3 -c "
from layout.yolov8_layout_detector import YOLOv8LayoutDetector
detector = YOLOv8LayoutDetector()
print('Model available:', detector.is_available())
print('Model info:', detector.get_model_info())
"
```

---

## Troubleshooting

### Low Detection Accuracy

1. **Check dataset quality:**
   ```bash
   python3 training/prepare_dataset.py dataset/ --stats
   ```

2. **Verify class balance:**
   - Ensure all classes have sufficient examples
   - Minimum: 20-30 examples per class

3. **Check annotations:**
   - Verify bounding boxes are correct
   - Ensure class IDs match fixed mappings

### Training Fails

1. **GPU Memory Error:**
   - Reduce batch size: `--batch 4`
   - Reduce image size: `--imgsz 640`

2. **CUDA Out of Memory:**
   - Use CPU: `--device cpu`
   - Or reduce batch size

### Model Not Loading

1. **Check model path:**
   ```bash
   ls -lh models/yolo_layout/
   ```

2. **Verify model file:**
   ```python
   from ultralytics import YOLO
   model = YOLO("models/yolo_layout/yolov8s_layout.pt")
   print(model.names)  # Should show: {0: 'text_block', 1: 'table', ...}
   ```

---

## Best Practices

1. **Always version models** - Never overwrite trained models
2. **Keep training logs** - `runs/layout/` contains training history
3. **Validate before deployment** - Test on validation set
4. **Document dataset changes** - Track dataset versions
5. **Monitor acceptance rates** - Track detection quality
6. **Iterate incrementally** - Start small, expand gradually

---

## Key Design Principles

1. **Training-Ready from Day One**: No architectural changes needed for training
2. **Fixed Class Mappings**: Never change, enforced in code
3. **Model Versioning**: Automatic, supports incremental improvements
4. **Easy Model Swapping**: No code changes required
5. **Production-Grade**: Error handling, logging, versioning

---

## Next Steps

After training:

1. **Test model on validation images**
2. **Monitor detection quality in production**
3. **Collect failure cases for dataset expansion**
4. **Iterate**: expand dataset → retrain → deploy

### Complete Workflow

1. **Annotate 100 images** (use LabelImg/Roboflow)
2. **Prepare dataset**: `python3 training/prepare_dataset.py dataset/ --create-yaml`
3. **Train**: `python3 training/train_yolov8_layout.py --data dataset/dataset.yaml`
4. **Deploy**: Model auto-loaded on next server start
5. **Monitor**: Track detection quality and acceptance rates
6. **Iterate**: Expand dataset → retrain → deploy

---

## Success Criteria

✅ System trains successfully with 100 images  
✅ New images can be added and retrained without refactoring  
✅ Model replacement does not break inference  
✅ Clear upgrade path to stronger models (yolov8m/l)  
✅ Fixed class mappings enforced  
✅ Acceptance rules implemented  
✅ Production-ready implementation  

**The system is ready for immediate training and iterative improvement!**
