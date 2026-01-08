"""Column assigner - assigns column_index to all detections using X-axis overlap.

Architecture Rule: Column membership decided using X-axis overlap, not text class or row order.
YOLO geometry defines structure. OCR only fills content.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from postprocessing.column_anchor import ColumnAnchor, ColumnAnchorBuilder


class ColumnAssigner:
    """Assigns column_index to all detections using X-axis overlap with column boundaries."""
    
    def __init__(self, header_row_index: int = 1, column_tolerance: float = 0.03):
        """Initialize column assigner.
        
        Args:
            header_row_index: Row index to use as header (1-based, default: 1 = first row)
            column_tolerance: X-axis tolerance for column assignment (default: 0.03 = 3% of image width)
        """
        self.header_row_index = header_row_index
        self.column_tolerance = column_tolerance
        self.anchor_builder = ColumnAnchorBuilder(header_row_index=header_row_index)
        log.info(f"Column assigner initialized (header_row_index: {header_row_index}, tolerance: {column_tolerance})")
    
    def assign_columns(self,
                      all_detections: List[Dict[str, Any]],
                      row_groups: List[Dict[str, Any]],
                      ocr_results: List[Dict[str, Any]]) -> List[ColumnAnchor]:
        """Lock column boundaries from header row and assign column_index to all detections.
        
        Steps:
        1. Find header row
        2. Build column anchors (X-ranges) from header row
        3. Expand X-ranges with tolerance
        4. Assign every detection to a column based on X-axis overlap
        
        Args:
            all_detections: All YOLO detections
            row_groups: Row groups from RowGrouper
            ocr_results: OCR results for header text extraction
            
        Returns:
            List of ColumnAnchor objects with locked X-ranges
        """
        # Step 1: Find header row
        header_row = self._find_header_row(row_groups)
        if not header_row:
            log.warning("No header row found, cannot assign columns")
            return []
        
        # Step 2: Build column anchors from header row (locked X-ranges)
        log.info(f"Locking column boundaries from header row {header_row.get('row_index')}...")
        column_anchors = self.anchor_builder.build_from_header_row(header_row, ocr_results)
        
        if not column_anchors:
            log.error("Failed to build column anchors from header row")
            return []
        
        # Step 3: Expand X-ranges with tolerance
        for anchor in column_anchors:
            x_range = anchor.x_max - anchor.x_min
            anchor.x_min = max(0.0, anchor.x_min - (x_range * self.column_tolerance))
            anchor.x_max = min(1.0, anchor.x_max + (x_range * self.column_tolerance))
            log.debug(f"Column {anchor.column_index} '{anchor.header_text}' → x_range=[{anchor.x_min:.4f}, {anchor.x_max:.4f}]")
        
        # Step 4: Assign EVERY detection to a column based on X-axis overlap
        log.info(f"Assigning {len(all_detections)} detections to {len(column_anchors)} columns...")
        assigned_count = 0
        unassigned_count = 0
        
        for det in all_detections:
            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x_center = (bbox[0] + bbox[2]) / 2.0
            column_index = self._find_column_by_x_center(x_center, column_anchors)
            
            if column_index is not None:
                det['column_index'] = column_index
                assigned_count += 1
                log.debug(f"Assigned {det.get('class', 'unknown')} at x={x_center:.4f} → column {column_index}")
            else:
                det['column_index'] = None
                unassigned_count += 1
                log.warning(f"Unassigned detection {det.get('class', 'unknown')} at x={x_center:.4f} (no column match)")
        
        log.info(f"Column assignment complete: {assigned_count} assigned, {unassigned_count} unassigned")
        return column_anchors
    
    def _find_column_by_x_center(self,
                                 x_center: float,
                                 column_anchors: List[ColumnAnchor]) -> Optional[int]:
        """Find column index for a detection based on X-center overlap.
        
        Args:
            x_center: X-center of detection
            column_anchors: List of column anchors with X-ranges
            
        Returns:
            Column index (1-based) or None if no match
        """
        for anchor in column_anchors:
            if anchor.x_min <= x_center <= anchor.x_max:
                return anchor.column_index
        return None
    
    def _find_header_row(self, row_groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find header row using selection criteria.
        
        Criteria (in priority order):
        1. Row with maximum PrintedText count
        2. No handwritten
        3. No signatures
        4. No checkboxes
        
        Args:
            row_groups: List of row groups
            
        Returns:
            Header row or None
        """
        if not row_groups:
            return None
        
        # Find row with maximum PrintedText and no other classes (smart selection)
        best_row = None
        max_text_count = 0
        
        for row in row_groups:
            detections = row.get('detections', [])
            text_count = sum(1 for d in detections 
                           if d.get('class', '').lower() in ['text_block', 'text_box'])
            handwritten_count = sum(1 for d in detections 
                                  if d.get('class', '').lower() == 'handwritten')
            signature_count = sum(1 for d in detections 
                                if d.get('class', '').lower() == 'signature')
            checkbox_count = sum(1 for d in detections 
                               if d.get('class', '').lower() == 'checkbox')
            
            # Prefer rows with many text blocks and no other classes
            if (text_count > max_text_count and 
                handwritten_count == 0 and 
                signature_count == 0 and 
                checkbox_count == 0):
                max_text_count = text_count
                best_row = row
        
        if best_row:
            log.info(f"Selected header row {best_row.get('row_index')} with {max_text_count} text blocks (smart selection)")
            return best_row
        
        # Fallback 1: Try specified header row index
        for row in row_groups:
            if row.get('row_index') == self.header_row_index:
                log.info(f"Using configured header row {self.header_row_index} (fallback)")
                return row
        
        # Fallback 2: use first row
        if row_groups:
            log.warning("Using first row as header (final fallback)")
            return row_groups[0]
        
        return None


