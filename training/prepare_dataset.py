#!/usr/bin/env python3
"""Dataset preparation script for YOLOv8 layout detection.

This script helps prepare and validate datasets for training.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from core.logging import log

# FIXED CLASS MAPPINGS
CLASS_MAPPINGS = {
    0: 'text_block',
    1: 'table',
    2: 'signature',
    3: 'checkbox'
}


def validate_dataset_structure(dataset_dir: Path) -> bool:
    """Validate dataset directory structure.
    
    Expected structure:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/ (optional)
    └── labels/
        ├── train/
        ├── val/
        └── test/ (optional)
    
    Args:
        dataset_dir: Path to dataset root directory
        
    Returns:
        bool: True if structure is valid
    """
    log.info(f"Validating dataset structure: {dataset_dir}")
    
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = dataset_dir / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    if missing:
        log.error("Missing required directories:")
        for dir_path in missing:
            log.error(f"  - {dir_path}")
        return False
    
    log.info("✅ Dataset structure is valid")
    return True


def count_annotations(dataset_dir: Path) -> Dict[str, Any]:
    """Count annotations in dataset.
    
    Args:
        dataset_dir: Path to dataset root directory
        
    Returns:
        dict: Statistics about the dataset
    """
    stats = {
        'train': {'images': 0, 'labels': 0, 'classes': {}},
        'val': {'images': 0, 'labels': 0, 'classes': {}},
        'test': {'images': 0, 'labels': 0, 'classes': {}}
    }
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_dir / 'images' / split
        labels_dir = dataset_dir / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        # Count images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        stats[split]['images'] = len(image_files)
        
        # Count labels and classes
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            stats[split]['labels'] = len(label_files)
            
            # Count class occurrences
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                class_name = CLASS_MAPPINGS.get(class_id, f'unknown_{class_id}')
                                stats[split]['classes'][class_name] = stats[split]['classes'].get(class_name, 0) + 1
                except Exception as e:
                    log.warning(f"Failed to read {label_file}: {e}")
    
    return stats


def create_dataset_yaml(dataset_dir: Path, output_path: Optional[Path] = None) -> Path:
    """Create dataset.yaml file from dataset directory.
    
    Args:
        dataset_dir: Path to dataset root directory
        output_path: Output path for dataset.yaml (default: dataset_dir/dataset.yaml)
        
    Returns:
        Path: Path to created dataset.yaml file
    """
    if output_path is None:
        output_path = dataset_dir / 'dataset.yaml'
    
    yaml_content = f"""# YOLOv8 Layout Detection Dataset Configuration
# Auto-generated dataset configuration

# Dataset root directory
path: {dataset_dir.absolute()}

# Training images (relative to path)
train: images/train

# Validation images (relative to path)
val: images/val

# Test images (optional, relative to path)
test: images/test

# Class names (FIXED - DO NOT CHANGE INDICES)
names:
  0: text_block
  1: table
  2: signature
  3: checkbox

# Number of classes
nc: 4
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    log.info(f"Created dataset.yaml: {output_path}")
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate YOLOv8 layout detection dataset"
    )
    
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to dataset root directory'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate dataset structure'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show dataset statistics'
    )
    
    parser.add_argument(
        '--create-yaml',
        action='store_true',
        help='Create dataset.yaml file'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        log.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Validate structure
    if args.validate:
        if not validate_dataset_structure(dataset_dir):
            sys.exit(1)
    
    # Show statistics
    if args.stats:
        stats = count_annotations(dataset_dir)
        log.info("=" * 60)
        log.info("Dataset Statistics")
        log.info("=" * 60)
        for split in ['train', 'val', 'test']:
            if stats[split]['images'] > 0:
                log.info(f"\n{split.upper()}:")
                log.info(f"  Images: {stats[split]['images']}")
                log.info(f"  Labels: {stats[split]['labels']}")
                log.info(f"  Classes:")
                for class_name, count in stats[split]['classes'].items():
                    log.info(f"    {class_name}: {count}")
        log.info("=" * 60)
    
    # Create dataset.yaml
    if args.create_yaml:
        yaml_path = create_dataset_yaml(dataset_dir)
        log.info(f"✅ Created dataset.yaml: {yaml_path}")
    
    if not (args.validate or args.stats or args.create_yaml):
        log.info("No action specified. Use --validate, --stats, or --create-yaml")
        parser.print_help()


if __name__ == "__main__":
    main()



