"""Fusion engine for merging outputs from all layout detectors.

Handles overlap resolution, confidence normalization, and unified JSON output.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Tuple
import numpy as np
from core.logging import log


class FusionEngine:
    """Fusion engine for merging layout detection results."""
    
    # Priority order for region types (higher = more important)
    PRIORITY = {
        'signature': 4,
        'table': 3,
        'checkbox': 2,
        'handwritten': 2,  # Handwritten has same priority as checkbox
        'text_block': 1,
        'text_block_refined': 1
    }
    
    def __init__(self, iou_threshold: float = 0.85):
        """Initialize fusion engine.
        
        Args:
            iou_threshold: IoU threshold for duplicate removal (default 0.85)
        """
        self.iou_threshold = iou_threshold
    
    def fuse(self, 
             yolo_results: Dict[str, List[Dict[str, Any]]],
             signature_results: List[Dict[str, Any]],
             checkbox_results: List[Dict[str, Any]],
             paddleocr_results: List[Dict[str, Any]],
             image_dims: Tuple[int, int]) -> Dict[str, Any]:
        """Fuse all detector outputs into unified layout.
        
        Args:
            yolo_results: YOLOv8s detections {'tables': [...], 'text_blocks': [...], 'signatures': [...], 'checkboxes': [...]}
            signature_results: GPDS signature detections
            checkbox_results: Checkbox detections
            paddleocr_results: PaddleOCR text detections
            image_dims: (width, height) of image
            
        Returns:
            dict: Unified layout structure
        """
        log.info("Starting fusion of all detector outputs...")
        
        # Step 1: Collect all detections
        all_detections = []
        
        # Add YOLOv8s tables
        for det in yolo_results.get('tables', []):
            all_detections.append({
                'type': 'table',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'YOLOv8s')
            })
        
        # Add YOLOv8s text blocks
        for det in yolo_results.get('text_blocks', []):
            all_detections.append({
                'type': 'text_block',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'YOLOv8s')
            })
        
        # Add YOLOv8s handwritten
        for det in yolo_results.get('handwritten', []):
            all_detections.append({
                'type': 'handwritten',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'YOLOv8s')
            })
        
        # Add YOLOv8s signatures (primary source)
        for det in yolo_results.get('signatures', []):
            all_detections.append({
                'type': 'signature',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'YOLOv8s')
            })
        
        # Add YOLOv8s checkboxes (primary source)
        for det in yolo_results.get('checkboxes', []):
            all_detections.append({
                'type': 'checkbox',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'YOLOv8s')
            })
        
        # Add GPDS signatures (secondary/fallback)
        for det in signature_results:
            all_detections.append({
                'type': 'signature',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'GPDS')
            })
        
        # Add checkbox detector results (secondary/fallback)
        for det in checkbox_results:
            all_detections.append({
                'type': 'checkbox',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'checkbox_detector')
            })
        
        # Add PaddleOCR refined text blocks (ONLY if provided - should be empty when YOLO works)
        if paddleocr_results:
            log.info(f"Adding {len(paddleocr_results)} PaddleOCR detections to fusion")
        for det in paddleocr_results:
            all_detections.append({
                'type': 'text_block_refined',
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'source': det.get('source', 'PaddleOCR')
            })
        else:
            log.debug("No PaddleOCR results to fuse (YOLO is working)")
        
        log.info(f"Total detections before fusion: {len(all_detections)}")
        
        # Step 2: Normalize confidences (if needed)
        all_detections = self._normalize_confidences(all_detections)
        
        # Step 3: Handle overlaps and conflicts
        fused_detections = self._resolve_overlaps(all_detections, image_dims)
        
        # Step 4: Sort by priority and confidence
        fused_detections = self._sort_detections(fused_detections)
        
        # Step 5: Organize by type
        organized = self._organize_by_type(fused_detections)
        
        log.info(f"Fusion complete: {len(organized['tables'])} tables, "
                f"{len(organized['text_blocks'])} text blocks, "
                f"{len(organized['signatures'])} signatures, "
                f"{len(organized['checkboxes'])} checkboxes, "
                f"{len(organized.get('handwritten', []))} handwritten")
        
        return organized
    
    def _normalize_confidences(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize confidence scores to unified scale [0, 1].
        
        Args:
            detections: List of detections
            
        Returns:
            list: Detections with normalized confidences
        """
        if not detections:
            return detections
        
        # Get all confidence scores
        confidences = [d['confidence'] for d in detections]
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Normalize if range is not [0, 1]
        if max_conf > 1.0 or min_conf < 0.0:
            range_conf = max_conf - min_conf
            if range_conf > 0:
                for det in detections:
                    det['confidence'] = (det['confidence'] - min_conf) / range_conf
                    det['confidence'] = max(0.0, min(1.0, det['confidence']))  # Clamp to [0, 1]
        
        return detections
    
    def _resolve_overlaps(self, detections: List[Dict[str, Any]], 
                         image_dims: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Resolve overlaps using priority and IoU rules.
        
        Args:
            detections: List of all detections
            image_dims: (width, height) of image
            
        Returns:
            list: Detections with overlaps resolved
        """
        if not detections:
            return detections
        
        # Sort by priority (descending) so higher priority items are processed first
        sorted_detections = sorted(detections, 
                                  key=lambda x: self.PRIORITY.get(x['type'], 0), 
                                  reverse=True)
        
        fused = []
        used_indices = set()
        
        for i, det1 in enumerate(sorted_detections):
            if i in used_indices:
                continue
            
            bbox1 = det1['bbox']
            type1 = det1['type']
            priority1 = self.PRIORITY.get(type1, 0)
            
            # Check for overlaps with remaining detections
            merged_det = det1.copy()
            merged = False
            
            for j, det2 in enumerate(sorted_detections[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                bbox2 = det2['bbox']
                type2 = det2['type']
                priority2 = self.PRIORITY.get(type2, 0)
                
                # Calculate IoU
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > self.iou_threshold:
                    # High overlap - remove duplicate (keep higher priority)
                    used_indices.add(j)
                    log.debug(f"Removed duplicate: {type2} (IoU={iou:.3f})")
                elif iou > 0.3:  # Significant overlap
                    # Apply overlap rules
                    if type1 == 'table' and type2 == 'text_block':
                        # Text block overlaps with table → assign to table
                        used_indices.add(j)
                        log.debug(f"Text block {j} assigned to table {i}")
                    elif type1 == 'signature' and type2 in ['text_block', 'text_block_refined']:
                        # Signature overlaps with text → signature wins
                        used_indices.add(j)
                        log.debug(f"Text block {j} removed (overlaps with signature {i})")
                    elif type1 == 'checkbox' and type2 == 'table':
                        # Checkbox inside table → mark checkbox
                        if self._is_inside(bbox2, bbox1):
                            merged_det['in_table'] = True
                            merged_det['table_bbox'] = bbox2
                            used_indices.add(j)
                            merged = True
                            log.debug(f"Checkbox {i} marked as inside table {j}")
            
            fused.append(merged_det)
            used_indices.add(i)
        
        return fused
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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
    
    def _is_inside(self, bbox_outer: List[float], bbox_inner: List[float]) -> bool:
        """Check if inner bbox is inside outer bbox.
        
        Args:
            bbox_outer: Outer bounding box [x1, y1, x2, y2]
            bbox_inner: Inner bounding box [x1, y1, x2, y2]
            
        Returns:
            bool: True if inner is inside outer
        """
        x1_o, y1_o, x2_o, y2_o = bbox_outer
        x1_i, y1_i, x2_i, y2_i = bbox_inner
        
        return (x1_i >= x1_o and y1_i >= y1_o and 
                x2_i <= x2_o and y2_i <= y2_o)
    
    def _sort_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort detections by priority, then confidence, then area.
        
        Args:
            detections: List of detections
            
        Returns:
            list: Sorted detections
        """
        def sort_key(det):
            priority = self.PRIORITY.get(det['type'], 0)
            confidence = det['confidence']
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            return (-priority, -confidence, -area)  # Negative for descending
        
        return sorted(detections, key=sort_key)
    
    def _organize_by_type(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize detections by type into final structure.
        
        Args:
            detections: List of all fused detections
            
        Returns:
            dict: Organized layout structure
        """
        organized = {
            'tables': [],
            'text_blocks': [],
            'signatures': [],
            'checkboxes': [],
            'handwritten': []
        }
        
        for det in detections:
            det_type = det['type']
            
            # Map types to organized structure
            if det_type == 'table':
                organized['tables'].append(det)
            elif det_type in ['text_block', 'text_block_refined']:
                # Merge refined and regular text blocks
                organized['text_blocks'].append(det)
            elif det_type == 'handwritten':
                organized['handwritten'].append(det)
            elif det_type == 'signature':
                organized['signatures'].append(det)
            elif det_type == 'checkbox':
                organized['checkboxes'].append(det)
        
        return organized









