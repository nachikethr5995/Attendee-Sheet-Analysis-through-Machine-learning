"""Train custom YOLOv8 model for document layout detection.

This script trains a YOLOv8 model to detect:
- table (class 0)
- text_block (class 1)
- signature (class 2)
- checkbox (class 3)

Usage:
    python train_yolo_layout.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import sys

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Check if dataset.yaml exists
dataset_yaml = Path('dataset.yaml')
if not dataset_yaml.exists():
    print("ERROR: dataset.yaml not found!")
    print("Please create dataset.yaml with your dataset configuration.")
    print("\nExample dataset.yaml:")
    print("""
path: ./dataset
train: images/train
val: images/val
names:
  0: table
  1: text_block
  2: signature
  3: checkbox
nc: 4
    """)
    sys.exit(1)

# Model selection
print("\nAvailable models:")
print("1. yolov8n.pt - Nano (fastest, smallest)")
print("2. yolov8s.pt - Small (balanced)")
print("3. yolov8m.pt - Medium (better accuracy)")
print("4. yolov8l.pt - Large (high accuracy)")
print("5. yolov8x.pt - XLarge (best accuracy, slowest)")

model_choice = input("\nSelect model (1-5, default=1): ").strip() or "1"
model_map = {
    "1": "yolov8n.pt",
    "2": "yolov8s.pt",
    "3": "yolov8m.pt",
    "4": "yolov8l.pt",
    "5": "yolov8x.pt"
}
model_name = model_map.get(model_choice, "yolov8n.pt")

print(f"\nInitializing model: {model_name}")
model = YOLO(model_name)

# Training parameters
print("\nTraining Configuration:")
epochs = int(input("Epochs (default=100): ").strip() or "100")
imgsz = int(input("Image size (default=640): ").strip() or "640")
batch = int(input("Batch size (default=16): ").strip() or "16")
project_name = input("Project name (default=layout_detector): ").strip() or "layout_detector"

print(f"\nStarting training...")
print(f"  Epochs: {epochs}")
print(f"  Image size: {imgsz}")
print(f"  Batch size: {batch}")
print(f"  Project: {project_name}")

# Train the model
try:
    results = model.train(
        data=str(dataset_yaml),        # Dataset configuration
        epochs=epochs,                 # Number of training epochs
        imgsz=imgsz,                   # Image size
        batch=batch,                   # Batch size
        name=project_name,             # Project name
        project='runs/detect',         # Project directory
        exist_ok=True,                 # Overwrite existing project
        pretrained=True,               # Use pretrained weights
        optimizer='AdamW',             # Optimizer
        verbose=True,                  # Verbose output
        seed=42,                       # Random seed
        deterministic=True,            # Deterministic training
        single_cls=False,              # Multi-class detection
        rect=False,                    # Rectangular training
        cos_lr=False,                  # Cosine learning rate scheduler
        close_mosaic=10,               # Disable mosaic in last N epochs
        resume=False,                  # Resume from checkpoint
        amp=True,                      # Automatic Mixed Precision
        fraction=1.0,                  # Dataset fraction
        profile=False,                 # Profile speeds
        freeze=None,                   # Freeze layers
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate
        momentum=0.937,                # SGD momentum
        weight_decay=0.0005,           # Weight decay
        warmup_epochs=3.0,             # Warmup epochs
        warmup_momentum=0.8,           # Warmup momentum
        warmup_bias_lr=0.1,            # Warmup bias lr
        box=7.5,                       # Box loss gain
        cls=0.5,                       # Class loss gain
        dfl=1.5,                       # DFL loss gain
        val=True,                      # Validate during training
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Best model: runs/detect/{project_name}/weights/best.pt")
    print(f"Last model: runs/detect/{project_name}/weights/last.pt")
    print(f"\nTo use this model, update layout/yolo_detector.py:")
    print(f"  model_path = 'runs/detect/{project_name}/weights/best.pt'")
    print(f"\nTo view training metrics:")
    print(f"  tensorboard --logdir runs/detect/{project_name}")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    print("You can resume training later with:")
    print(f"  yolo detect train resume project=runs/detect/{project_name}")
except Exception as e:
    print(f"\nERROR during training: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)









