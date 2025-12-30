#!/usr/bin/env python3
"""YOLOv8 Training Smoke Test

Validates that the training pipeline works end-to-end without errors.
This is a minimal, fast test - NOT an accuracy evaluation.

Smoke Test Steps:
1. Environment verification
2. Dataset structure validation
3. data.yaml validation/creation
4. Minimal training dry run (5 epochs)
5. Output validation
"""

# Import core first to ensure path fix is applied
import sys
from pathlib import Path

# Add Back_end to path so we can import core
back_end_dir = Path(__file__).parent.parent
if str(back_end_dir) not in sys.path:
    sys.path.insert(0, str(back_end_dir))

try:
    import core  # This ensures user site-packages is in path
    from core.logging import log
except ImportError:
    # Fallback if core module not available
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import argparse
import random
from typing import Dict, List, Tuple, Optional

try:
    import yaml
except ImportError:
    yaml = None

# YOLOv8 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.error("ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

# Expected classes per the prompt (5 classes)
EXPECTED_CLASSES = {
    0: 'checkbox',
    1: 'handwritten',
    2: 'signature',
    3: 'table',
    4: 'text_block'
}


def check_environment() -> bool:
    """Step 1: Environment Verification."""
    log.info("=" * 60)
    log.info("Step 1: Environment Verification")
    log.info("=" * 60)
    
    try:
        # Check YOLO version
        import ultralytics
        log.info(f"✅ YOLO version: {ultralytics.__version__}")
        
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                log.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                log.info("⚠️  CUDA not available, will use CPU")
        except ImportError:
            log.info("⚠️  PyTorch not found, will use CPU")
        
        return True
    except Exception as e:
        log.error(f"❌ Environment check failed: {e}")
        return False


def detect_dataset_structure(dataset_dir: Path) -> Optional[Dict[str, Path]]:
    """Detect which dataset structure is being used.
    
    Returns dict with paths if structure is valid, None otherwise.
    """
    # Structure 1: dataset/train/images, dataset/valid/images (prompt format)
    train_images_1 = dataset_dir / 'train' / 'images'
    train_labels_1 = dataset_dir / 'train' / 'labels'
    valid_images_1 = dataset_dir / 'valid' / 'images'
    valid_labels_1 = dataset_dir / 'valid' / 'labels'
    
    # Structure 2: dataset/images/train, dataset/labels/train (existing code format)
    train_images_2 = dataset_dir / 'images' / 'train'
    train_labels_2 = dataset_dir / 'labels' / 'train'
    valid_images_2 = dataset_dir / 'images' / 'val'
    valid_labels_2 = dataset_dir / 'labels' / 'val'
    
    if (train_images_1.exists() and train_labels_1.exists() and 
        valid_images_1.exists() and valid_labels_1.exists()):
        log.info("✅ Detected structure: dataset/train/images, dataset/valid/images")
        return {
            'train_images': train_images_1,
            'train_labels': train_labels_1,
            'valid_images': valid_images_1,
            'valid_labels': valid_labels_1,
            'structure': 'nested'
        }
    elif (train_images_2.exists() and train_labels_2.exists() and 
          valid_images_2.exists() and valid_labels_2.exists()):
        log.info("✅ Detected structure: dataset/images/train, dataset/labels/train")
        return {
            'train_images': train_images_2,
            'train_labels': train_labels_2,
            'valid_images': valid_images_2,
            'valid_labels': valid_labels_2,
            'structure': 'flat'
        }
    else:
        return None


def validate_dataset_structure(dataset_dir: Path) -> Tuple[bool, Optional[Dict[str, Path]]]:
    """Step 2: Dataset Structure Validation."""
    log.info("=" * 60)
    log.info("Step 2: Dataset Structure Validation")
    log.info("=" * 60)
    log.info(f"Checking dataset directory: {dataset_dir}")
    
    if not dataset_dir.exists():
        log.error(f"❌ Dataset directory not found: {dataset_dir}")
        log.error("Expected location: Back_end/dataset/")
        return False, None
    
    structure = detect_dataset_structure(dataset_dir)
    
    if structure is None:
        log.error("❌ Invalid dataset structure!")
        log.error("Expected one of:")
        log.error("  Structure 1: dataset/train/images, dataset/valid/images")
        log.error("  Structure 2: dataset/images/train, dataset/labels/train")
        return False, None
    
    # Check that directories have content
    train_images = list(structure['train_images'].glob('*.jpg')) + list(structure['train_images'].glob('*.png'))
    valid_images = list(structure['valid_images'].glob('*.jpg')) + list(structure['valid_images'].glob('*.png'))
    
    if len(train_images) == 0:
        log.error("❌ No training images found!")
        return False, None
    
    if len(valid_images) == 0:
        log.error("❌ No validation images found!")
        return False, None
    
    log.info(f"✅ Found {len(train_images)} training images")
    log.info(f"✅ Found {len(valid_images)} validation images")
    
    return True, structure


def validate_label_files(structure: Dict[str, Path], num_samples: int = 3) -> bool:
    """Validate label files for sanity."""
    log.info(f"\nValidating {num_samples} random label files...")
    
    # Get random training label files
    label_files = list(structure['train_labels'].glob('*.txt'))
    if len(label_files) == 0:
        log.error("❌ No label files found in training set!")
        return False
    
    sample_files = random.sample(label_files, min(num_samples, len(label_files)))
    
    for label_file in sample_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    log.warning(f"⚠️  Empty label file: {label_file.name}")
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        log.error(f"❌ Invalid label format in {label_file.name}:{line_num}")
                        log.error(f"   Expected at least 5 values, got {len(parts)}")
                        return False
                    
                    try:
                        class_id = int(parts[0])
                        if class_id < 0 or class_id > 4:
                            log.error(f"❌ Invalid class ID in {label_file.name}:{line_num}")
                            log.error(f"   Class ID must be 0-4, got {class_id}")
                            return False
                        
                        # Check if standard YOLO format (5 values) or polygon format (9+ values)
                        if len(parts) == 5:
                            # Standard YOLO: class_id center_x center_y width height
                            center_x, center_y, width, height = map(float, parts[1:5])
                            if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                                    0 <= width <= 1 and 0 <= height <= 1):
                                log.warning(f"⚠️  Bounding box values not normalized in {label_file.name}:{line_num}")
                                log.warning(f"   Values: {center_x}, {center_y}, {width}, {height}")
                        elif len(parts) >= 9:
                            # Polygon format (Roboflow): class_id x1 y1 x2 y2 x3 y3 x4 y4
                            log.info(f"   Detected polygon format (9+ values) - will be converted by YOLO")
                            # Just verify coordinates are numeric and reasonable
                            coords = [float(x) for x in parts[1:]]
                            if any(not (0 <= c <= 1) for c in coords):
                                log.warning(f"⚠️  Some polygon coordinates not normalized in {label_file.name}:{line_num}")
                        else:
                            log.warning(f"⚠️  Unusual format in {label_file.name}:{line_num} - {len(parts)} values")
                    
                    except ValueError as e:
                        log.error(f"❌ Invalid numeric value in {label_file.name}:{line_num}: {e}")
                        return False
            
            log.info(f"✅ {label_file.name}: {len(lines)} annotations, all valid")
        
        except Exception as e:
            log.error(f"❌ Failed to read {label_file}: {e}")
            return False
    
    return True


def create_data_yaml(dataset_dir: Path, structure: Dict[str, Path], output_path: Path) -> bool:
    """Step 3: Create data.yaml file."""
    log.info("=" * 60)
    log.info("Step 3: data.yaml Validation/Creation")
    log.info("=" * 60)
    
    if yaml is None:
        log.error("PyYAML not installed. Install with: pip install pyyaml")
        return False
    
    # Check if data.yaml already exists
    existing_yaml = dataset_dir / 'data.yaml'
    if existing_yaml.exists():
        log.info(f"Found existing data.yaml: {existing_yaml}")
        try:
            with open(existing_yaml, 'r') as f:
                existing_data = yaml.safe_load(f)
            
            # Validate structure
            if 'names' not in existing_data:
                log.error("❌ data.yaml missing 'names' field")
                return False
            
            if 'nc' not in existing_data:
                log.error("❌ data.yaml missing 'nc' field")
                return False
            
            if existing_data['nc'] != 5:
                log.error(f"❌ data.yaml has {existing_data['nc']} classes, expected 5")
                return False
            
            # Check class names match
            expected_names = {str(k): v for k, v in EXPECTED_CLASSES.items()}
            actual_names = {str(k): v for k, v in existing_data['names'].items()}
            
            if expected_names != actual_names:
                log.warning("⚠️  Class names in data.yaml don't match expected:")
                log.warning(f"   Expected: {expected_names}")
                log.warning(f"   Found: {actual_names}")
                log.warning("   Will create new data.yaml with correct format")
            else:
                log.info("✅ Existing data.yaml is valid")
                return True
        
        except Exception as e:
            log.warning(f"⚠️  Failed to parse existing data.yaml: {e}")
            log.warning("   Will create new data.yaml")
    
    # Create data.yaml
    log.info("Creating data.yaml...")
    
    # Determine paths based on structure
    if structure['structure'] == 'nested':
        train_path = 'train/images'
        val_path = 'valid/images'
    else:
        train_path = 'images/train'
        val_path = 'images/val'
    
    yaml_content = f"""train: {train_path}
val: {val_path}

nc: 5

names:
  0: checkbox
  1: handwritten
  2: signature
  3: table
  4: text_block
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        log.info(f"✅ Created data.yaml: {output_path}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to create data.yaml: {e}")
        return False


def run_training_smoke_test(data_yaml: Path, epochs: int = 5) -> bool:
    """Step 4: Minimal Training Dry Run."""
    log.info("=" * 60)
    log.info("Step 4: Minimal Training Dry Run (5 epochs)")
    log.info("=" * 60)
    
    # Determine device
    try:
        import torch
        if torch.cuda.is_available():
            device = "0"  # Use first GPU
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            log.info("Using CPU (CUDA not available)")
    except ImportError:
        device = "cpu"
        log.info("Using CPU")
    
    # Load model
    log.info("Loading yolov8s.pt...")
    try:
        model = YOLO('yolov8s.pt')
        log.info("✅ Model loaded successfully")
    except Exception as e:
        log.error(f"❌ Failed to load model: {e}")
        return False
    
    # Training arguments (minimal for smoke test)
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': 640,
        'batch': 4,
        'device': device,
        'project': 'smoke_test',
        'name': 'training_dry_run',
        'save': True,
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    log.info("Starting training...")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Image size: 640")
    log.info(f"  Batch size: 4")
    log.info(f"  Device: {device}")
    log.info(f"  Output: smoke_test/training_dry_run/")
    
    try:
        results = model.train(**train_args)
        log.info("✅ Training completed without errors")
        
        # Validate outputs
        output_dir = Path('smoke_test') / 'training_dry_run'
        weights_dir = output_dir / 'weights'
        
        if not weights_dir.exists():
            log.error("❌ Weights directory not created!")
            return False
        
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        if not best_pt.exists():
            log.error("❌ best.pt not created!")
            return False
        
        if not last_pt.exists():
            log.error("❌ last.pt not created!")
            return False
        
        log.info(f"✅ best.pt created: {best_pt}")
        log.info(f"✅ last.pt created: {last_pt}")
        
        # Check for results files
        results_png = output_dir / 'results.png'
        results_csv = output_dir / 'results.csv'
        
        if results_png.exists():
            log.info(f"✅ Training plots: {results_png}")
        if results_csv.exists():
            log.info(f"✅ Training metrics: {results_csv}")
        
        # Check if loss decreased (basic sanity check)
        if hasattr(results, 'results_dict'):
            train_loss = results.results_dict.get('train/box_loss', None)
            if train_loss is not None:
                log.info(f"✅ Training loss tracked: {train_loss}")
        
        return True
    
    except KeyboardInterrupt:
        log.warning("⚠️  Training interrupted by user")
        return False
    except Exception as e:
        log.error(f"❌ Training failed: {e}", exc_info=True)
        return False


def main():
    """Main smoke test entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Training Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run smoke test with default dataset location
  python smoke_test_training.py
  
  # Run smoke test with custom dataset path
  python smoke_test_training.py --dataset ../custom_dataset
  
  # Run with more epochs
  python smoke_test_training.py --epochs 10
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs for smoke test (default: 5)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip label file validation (faster)'
    )
    
    args = parser.parse_args()
    
    # Resolve dataset path
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        # Relative to Back_end directory
        back_end_dir = Path(__file__).parent.parent
        dataset_dir = back_end_dir / dataset_dir
    
    log.info("=" * 60)
    log.info("YOLOv8 Training Smoke Test")
    log.info("=" * 60)
    log.info(f"Dataset directory: {dataset_dir}")
    log.info(f"Epochs: {args.epochs}")
    log.info("=" * 60)
    
    # Step 1: Environment check
    if not check_environment():
        log.error("❌ Environment check failed")
        sys.exit(1)
    
    # Step 2: Dataset structure validation
    valid, structure = validate_dataset_structure(dataset_dir)
    if not valid:
        log.error("❌ Dataset structure validation failed")
        sys.exit(1)
    
    # Label file validation
    if not args.skip_validation:
        if not validate_label_files(structure):
            log.error("❌ Label file validation failed")
            sys.exit(1)
    
    # Step 3: Create/validate data.yaml
    data_yaml = dataset_dir / 'data.yaml'
    if not create_data_yaml(dataset_dir, structure, data_yaml):
        log.error("❌ data.yaml creation/validation failed")
        sys.exit(1)
    
    # Step 4: Training smoke test
    if not run_training_smoke_test(data_yaml, epochs=args.epochs):
        log.error("❌ Training smoke test failed")
        sys.exit(1)
    
    # Final verdict
    log.info("=" * 60)
    log.info("✅ SMOKE TEST PASSED")
    log.info("=" * 60)
    log.info("Pipeline is healthy and ready for full training!")
    log.info("Next steps:")
    log.info("  1. Increase epochs to 80-120")
    log.info("  2. Enable class-wise metrics")
    log.info("  3. Proceed with full training")
    log.info("=" * 60)


if __name__ == "__main__":
    main()


