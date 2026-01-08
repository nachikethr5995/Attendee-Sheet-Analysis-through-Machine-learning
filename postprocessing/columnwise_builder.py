"""Columnwise builder - builds columnwise output using assigned column_index from detections.

Architecture Rule: Column membership decided using X-axis overlap. OCR only fills content.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from postprocessing.column_anchor import ColumnAnchor


class ColumnwiseBuilder:
    """Builds columnwise output using column_index assigned to detections."""
    
    def __init__(self):
        """Initialize columnwise builder."""
        log.info("Columnwise builder initialized")
    
    def build(self,
              column_anchors: List[ColumnAnchor],
              all_detections: List[Dict[str, Any]],
              row_groups: List[Dict[str, Any]],
              ocr_results: List[Dict[str, Any]],
              signature_results: List[Dict[str, Any]],
              checkbox_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build columnwise output using column_index assigned to detections.
        
        Args:
            column_anchors: Column anchors with headers and X-ranges
            all_detections: All YOLO detections (with column_index already assigned)
            row_groups: Row groups from RowGrouper
            ocr_results: OCR results with text, source, bbox
            signature_results: Signature results with present, bbox
            checkbox_results: Checkbox results with checked, bbox
            
        Returns:
            Columnwise structure with one column = one semantic unit
        """
        if not column_anchors:
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': 0
            }
        
        # Create lookup maps
        ocr_by_bbox = self._create_bbox_lookup(ocr_results)
        signature_by_bbox = self._create_bbox_lookup(signature_results)
        checkbox_by_bbox = self._create_bbox_lookup(checkbox_results)
        
        # Create row_index mapping
        row_index_map = {}
        for row_group in row_groups:
            row_index = row_group.get('row_index', 0)
            row_detections = row_group.get('detections', [])
            for det in row_detections:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    bbox_key = tuple(round(x, 4) for x in bbox[:4])
                    row_index_map[bbox_key] = row_index
        
        # Get all row indices
        all_row_indices = set()
        for row_group in row_groups:
            row_index = row_group.get('row_index', 0)
            if row_index > 0:
                all_row_indices.add(row_index)
        
        # Build columnwise output: group by column_index (already assigned to detections)
        columns_dict = {}
        
        # Initialize columns from anchors
        for anchor in column_anchors:
            columns_dict[anchor.column_index] = {
                'column_index': anchor.column_index,
                'header': anchor.header_text,
                'class': 'Mixed',  # Can contain PrintedText, HandwrittenText, Signature, Checkbox
                'rows': {}
            }
        
        # Aggregate detections by column_index (already assigned)
        for det in all_detections:
            column_index = det.get('column_index')
            if column_index is None or column_index not in columns_dict:
                continue
            
            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            # Get row_index
            bbox_key = tuple(round(x, 4) for x in bbox[:4])
            row_index = row_index_map.get(bbox_key, 0)
            if row_index == 0:
                row_index = self._find_row_index_by_bbox(bbox, row_groups)
            
            if row_index == 0:
                continue
            
            row_key = str(row_index)
            all_row_indices.add(row_index)
            col = columns_dict[column_index]
            
            class_name = det.get('class', '').lower()
            
            # Aggregate by class (all classes allowed in same column)
            if class_name in ['text_block', 'text_box']:
                # PrintedText from PaddleOCR
                ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                if ocr_result:
                    text = ocr_result.get('text', '').strip()
                    ocr_source = ocr_result.get('source', '')
                    confidence = ocr_result.get('confidence', 0.0)
                    
                    if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                        if row_key not in col['rows']:
                            col['rows'][row_key] = []
                        if not isinstance(col['rows'][row_key], list):
                            col['rows'][row_key] = []
                        col['rows'][row_key].append(text)
            
            elif class_name == 'handwritten':
                # HandwrittenText from TrOCR
                ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                if ocr_result:
                    text = ocr_result.get('text', '').strip()
                    ocr_source = ocr_result.get('source', '')
                    confidence = ocr_result.get('confidence', 0.0)
                    
                    if ocr_source == 'trocr' and confidence >= 0.4 and text:
                        if row_key not in col['rows']:
                            col['rows'][row_key] = []
                        if not isinstance(col['rows'][row_key], list):
                            col['rows'][row_key] = []
                        col['rows'][row_key].append(text)
            
            elif class_name == 'signature':
                # Signature presence (boolean)
                signature = self._find_matching_result(bbox, signature_by_bbox)
                if signature and signature.get('present', False):
                    col['rows'][row_key] = True
            
            elif class_name == 'checkbox':
                # Checkbox state (boolean)
                checkbox = self._find_matching_result(bbox, checkbox_by_bbox)
                if checkbox and checkbox.get('present', False):
                    checked_value = checkbox.get('checked', False)
                    col['rows'][row_key] = bool(checked_value) if checked_value is not None else None
        
        # Normalize empty rows explicitly
        for col in columns_dict.values():
            for row_index in all_row_indices:
                row_key = str(row_index)
                if row_key not in col['rows']:
                    # Default to empty list for text columns, False for signature, None for checkbox
                    # Since class is "Mixed", we default to empty list
                    col['rows'][row_key] = []
                elif isinstance(col['rows'][row_key], list):
                    # Deduplicate text lists
                    col['rows'][row_key] = list(dict.fromkeys(col['rows'][row_key]))
        
        # Convert to list and sort by column_index
        columns_list = [col for col in sorted(columns_dict.values(), key=lambda c: c['column_index'])]
        
        total_rows = len(all_row_indices)
        log.info(f"Built {len(columns_list)} columns from {len(column_anchors)} anchors ({total_rows} rows)")
        
        return {
            'columns': columns_list,
            'total_columns': len(columns_list),
            'total_rows': total_rows
        }
    
    def _find_row_index_by_bbox(self,
                                bbox: List[float],
                                row_groups: List[Dict[str, Any]]) -> int:
        """Find row index for a detection by bbox matching."""
        for row_group in row_groups:
            row_detections = row_group.get('detections', [])
            for row_det in row_detections:
                row_bbox = row_det.get('bbox', [])
                if len(row_bbox) >= 4:
                    if self._bboxes_match(bbox, row_bbox, tolerance=0.01):
                        return row_group.get('row_index', 0)
        return 0
    
    def _create_bbox_lookup(self, results: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        """Create bbox-based lookup map."""
        lookup = {}
        for result in results:
            bbox = result.get('bbox', [])
            if len(bbox) >= 4:
                bbox_key = tuple(round(x, 4) for x in bbox[:4])
                lookup[bbox_key] = result
        return lookup
    
    def _find_matching_result(self,
                              bbox: List[float],
                              lookup: Dict[tuple, Dict[str, Any]],
                              tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
        """Find matching result by bbox with tolerance."""
        if len(bbox) < 4:
            return None
        
        bbox_key = tuple(round(x, 4) for x in bbox[:4])
        if bbox_key in lookup:
            return lookup[bbox_key]
        
        for key, result in lookup.items():
            if len(key) >= 4:
                if self._bboxes_match(bbox, list(key), tolerance):
                    return result
        return None
    
    def _bboxes_match(self, bbox1: List[float], bbox2: List[float], tolerance: float = 0.01) -> bool:
        """Check if two bboxes match within tolerance."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        x1_diff = abs(bbox1[0] - bbox2[0])
        y1_diff = abs(bbox1[1] - bbox2[1])
        x2_diff = abs(bbox1[2] - bbox2[2])
        y2_diff = abs(bbox1[3] - bbox2[3])
        
        return (x1_diff < tolerance and y1_diff < tolerance and
                x2_diff < tolerance and y2_diff < tolerance)
