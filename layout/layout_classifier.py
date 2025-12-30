"""Layout detection combining YOLO and EAST results."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import (
    resolve_canonical_id,
    load_image_from_canonical_id,
    get_intermediate_json_path
)
from layout.yolo_detector import YOLODetector
from layout.east_text_detector import EASTTextDetector


class LayoutClassifier:
    """Combines YOLO and EAST detections into unified layout structure."""
    
    def __init__(self):
        """Initialize layout classifier."""
        self.yolo_detector = YOLODetector(confidence_threshold=0.5)
        self.east_detector = EASTTextDetector(confidence_threshold=0.5)
    
    def detect_layout(self, file_id: Optional[str] = None,
                     pre_0_id: Optional[str] = None,
                     pre_01_id: Optional[str] = None) -> Dict[str, Any]:
        """Detect layout elements in image using canonical ID.
        
        Args:
            file_id: Original file identifier (fallback)
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            dict: Layout detection results with failure detection
        """
        # Load image using canonical ID (with automatic fallback)
        # This will try pre_01_id -> pre_0_id -> file_id in order
        try:
            image = load_image_from_canonical_id(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            log.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            log.error(f"Failed to load image from any available source: {str(e)}")
            raise
        
        # Resolve canonical ID after successful load (for storage purposes)
        # This ensures we use the ID that actually worked
        canonical_id = resolve_canonical_id(
            file_id=file_id,
            pre_0_id=pre_0_id,
            pre_01_id=pre_01_id
        )
        
        log.info(f"Starting SERVICE 1 (layout detection) with canonical_id: {canonical_id}")
        
        # Run YOLO detection
        log.info("Running YOLO layout detection...")
        yolo_detections = self.yolo_detector.detect(image)
        
        # Count total YOLO detections
        total_yolo_detections = sum(len(yolo_detections[key]) for key in yolo_detections)
        
        # Check for YOLO failure
        yolo_failed, yolo_reason = self.yolo_detector.detect_failure(yolo_detections)
        if yolo_failed:
            log.warning(f"YOLO detection returned no results: {yolo_reason}")
            log.info("Note: This is expected with default YOLOv8 (COCO model). EAST will provide text detection.")
        
        # Run EAST detection
        log.info("Running EAST text detection...")
        east_detections = self.east_detector.detect(image)
        
        # Combine and fuse detections
        log.info("Fusing YOLO and EAST detections...")
        fused_layout = self._fuse_detections(yolo_detections, east_detections, image.size)
        
        # Store layout JSON using canonical ID
        layout_path = get_intermediate_json_path(canonical_id, 'layout')
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        
        layout_data = {
            'canonical_id': canonical_id,
            'layout': fused_layout,
            'dimensions': {
                'width': image.size[0],
                'height': image.size[1]
            },
            'detection_metadata': {
                'yolo_detections': {
                    'tables': len(yolo_detections['tables']),
                    'text_blocks': len(yolo_detections['text_blocks']),
                    'signatures': len(yolo_detections['signatures']),
                    'checkboxes': len(yolo_detections['checkboxes'])
                },
                'east_detections': len(east_detections),
                'yolo_failed': yolo_failed,
                'yolo_failure_reason': yolo_reason if yolo_failed else None
            }
        }
        
        # Save to JSON
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Layout detection complete. Saved to: {layout_path}")
        log.info(f"Total detections: {len(fused_layout['tables'])} tables, "
                f"{len(fused_layout['text_blocks'])} text blocks, "
                f"{len(fused_layout['signatures'])} signatures, "
                f"{len(fused_layout['checkboxes'])} checkboxes, "
                f"{len(fused_layout['east_boxes'])} EAST text regions")
        
        # Check overall failure
        # Consider it failed only if:
        # 1. YOLO failed AND no EAST detections (truly no detections)
        # 2. No tables detected AND no text regions (complete failure)
        # Note: EAST detections count as successful text detection
        total_east_detections = len(east_detections)
        
        if yolo_failed and total_east_detections == 0:
            # Truly failed - no detections from either YOLO or EAST
            overall_failed = True
            failure_reason = yolo_reason
        elif yolo_failed and total_east_detections > 0:
            # YOLO failed but EAST detected text regions - partial success
            # This is expected with default YOLOv8 (COCO model)
            overall_failed = False
            failure_reason = None
            log.info(f"YOLO returned no layout detections (using default COCO model), but EAST detected {total_east_detections} text regions - partial success")
        elif len(fused_layout['tables']) == 0 and total_east_detections == 0 and total_yolo_detections == 0:
            # No tables, no text, no YOLO detections - complete failure
            overall_failed = True
            failure_reason = "No tables or text regions detected"
        else:
            # Has detections - success (even if just EAST text regions)
            overall_failed = False
            failure_reason = None
        
        return {
            'canonical_id': canonical_id,
            'layout': fused_layout,
            'dimensions': layout_data['dimensions'],
            'failed': overall_failed,
            'failure_reason': failure_reason,
            'layout_path': str(layout_path.relative_to(Path(settings.STORAGE_ROOT).parent))
        }
    
    def _fuse_detections(self, yolo_detections: Dict[str, List[Dict[str, Any]]],
                        east_detections: List[Dict[str, Any]],
                        image_size: Tuple[int, int]) -> Dict[str, List[Dict[str, Any]]]:
        """Fuse YOLO and EAST detections, removing overlaps.
        
        Args:
            yolo_detections: YOLO detection results
            east_detections: EAST text region detections
            image_size: (width, height) of image
            
        Returns:
            dict: Fused layout structure
        """
        # Start with YOLO detections
        fused = {
            'tables': yolo_detections.get('tables', []).copy(),
            'text_blocks': yolo_detections.get('text_blocks', []).copy(),
            'signatures': yolo_detections.get('signatures', []).copy(),
            'checkboxes': yolo_detections.get('checkboxes', []).copy(),
            'east_boxes': []
        }
        
        # Add EAST detections (filter out overlaps with YOLO text blocks)
        width, height = image_size
        
        for east_box in east_detections:
            # Check if EAST box overlaps significantly with YOLO text blocks
            overlaps = False
            east_bbox = east_box['bbox']
            
            for yolo_text in fused['text_blocks']:
                yolo_bbox = yolo_text['bbox']
                
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(east_bbox, yolo_bbox)
                if iou > 0.5:  # Significant overlap
                    overlaps = True
                    break
            
            # Add EAST box if it doesn't overlap with YOLO text blocks
            if not overlaps:
                fused['east_boxes'].append(east_box)
        
        # Apply NMS (Non-Maximum Suppression) to remove duplicate detections
        # For now, we'll keep all detections but this could be enhanced
        
        return fused
    
    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2] normalized coordinates
            bbox2: [x1, y1, x2, y2] normalized coordinates
            
        Returns:
            float: IoU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union









