"""Column schema definition - frozen column boundaries based on header row.

Architecture Rule: Column boundaries are fixed from header row, not inferred per row.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.logging import log


@dataclass
class ColumnDefinition:
    """Frozen column definition with boundaries and header."""
    column_index: int
    x_min: float
    x_max: float
    x_center: float
    header_text: Optional[str] = None


class ColumnSchema:
    """Manages frozen column schema based on header row."""
    
    def __init__(self, header_row_index: int = 1):
        """Initialize column schema.
        
        Args:
            header_row_index: Row index to use as header (1-based, default: 1 = first row)
        """
        self.header_row_index = header_row_index
        self.column_definitions: List[ColumnDefinition] = []
        log.info(f"Column schema initialized (header_row_index: {header_row_index})")
    
    def build_from_header_row(self,
                             header_row: Dict[str, Any],
                             all_detections: List[Dict[str, Any]],
                             ocr_results: Optional[List[Dict[str, Any]]] = None,
                             tables: Optional[List[Dict[str, Any]]] = None) -> List[ColumnDefinition]:
        """Build column schema from header row using resolution rules.
        
        Header Resolution Rules (in order):
        1. If header row exists → use printed text from header row
        2. Else if table detected → use table column labels
        3. Else → fallback to "Column {index}"
        
        Args:
            header_row: Header row with detections
            all_detections: All detections for boundary calculation
            ocr_results: Optional OCR results for header text extraction
            tables: Optional table detections for fallback
            
        Returns:
            List of ColumnDefinition objects
        """
        detections = header_row.get('detections', [])
        if not detections:
            log.warning("No detections in header row, cannot build schema")
            return []
        
        # Create OCR lookup for header text extraction
        ocr_by_bbox = {}
        if ocr_results:
            for ocr_result in ocr_results:
                bbox = ocr_result.get('bbox', [])
                if len(bbox) >= 4:
                    bbox_key = tuple(round(x, 4) for x in bbox[:4])
                    ocr_by_bbox[bbox_key] = ocr_result
        
        # Extract header text boxes (PrintedText from header row)
        header_detections = []
        for det in detections:
            class_name = det.get('class', '').lower()
            if class_name in ['text_block', 'text_box']:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    # Try to get header text from OCR
                    header_text = None
                    bbox_key = tuple(round(x, 4) for x in bbox[:4])
                    if bbox_key in ocr_by_bbox:
                        ocr_result = ocr_by_bbox[bbox_key]
                        header_text = ocr_result.get('text', '').strip()
                    elif 'text' in det:
                        header_text = det.get('text', '').strip()
                    
                    header_detections.append((x_center, bbox, det, header_text))
        
        # Sort by X-center (left to right)
        header_detections.sort(key=lambda x: x[0])
        
        # Rule 2: Fallback to table column labels if no header detections
        if not header_detections and tables:
            log.info("No header detections, attempting to use table column labels")
            # This would require table structure parsing - simplified for now
            # For now, fall through to Rule 3
        
        # Rule 3: Fallback - use all detections if no text boxes
        if not header_detections:
            log.warning("No text boxes in header row, using all detections")
            for det in detections:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    header_detections.append((x_center, bbox, det, None))
            header_detections.sort(key=lambda x: x[0])
        
        # Build column definitions
        column_definitions = []
        for idx, (x_center, bbox, det, header_text) in enumerate(header_detections, start=1):
            x_min = bbox[0]
            x_max = bbox[2]
            
            # Rule 3: Fallback header text
            if not header_text:
                header_text = f"Column {idx}"
            
            # Calculate boundaries (extend to cover adjacent detections)
            if idx > 1:
                # Use midpoint between previous and current
                prev_def = column_definitions[-1]
                x_min = (prev_def.x_max + x_min) / 2.0
            if idx < len(header_detections):
                # Use midpoint between current and next
                next_x_center, next_bbox, _, _ = header_detections[idx]
                x_max = (x_max + next_bbox[0]) / 2.0
            
            col_def = ColumnDefinition(
                column_index=idx,
                x_min=x_min,
                x_max=x_max,
                x_center=x_center,
                header_text=header_text
            )
            column_definitions.append(col_def)
            log.debug(f"Column {idx}: x_min={x_min:.4f}, x_max={x_max:.4f}, header='{header_text}'")
        
        self.column_definitions = column_definitions
        log.info(f"Built column schema with {len(column_definitions)} columns from header row")
        return column_definitions
    
    def find_column_by_x_center(self, x_center: float) -> Optional[int]:
        """Find column index for a given X-center coordinate.
        
        Args:
            x_center: X-center coordinate (normalized 0-1)
            
        Returns:
            Column index (1-based) or None if not found
        """
        if not self.column_definitions:
            return None
        
        # Find column whose boundaries contain x_center
        for col_def in self.column_definitions:
            if col_def.x_min <= x_center <= col_def.x_max:
                return col_def.column_index
        
        # Fallback: find nearest column by distance
        min_distance = float('inf')
        nearest_col = None
        for col_def in self.column_definitions:
            distance = abs(x_center - col_def.x_center)
            if distance < min_distance:
                min_distance = distance
                nearest_col = col_def.column_index
        
        return nearest_col
    
    def get_column_definitions(self) -> List[ColumnDefinition]:
        """Get all column definitions.
        
        Returns:
            List of ColumnDefinition objects
        """
        return self.column_definitions
    
    def get_total_columns(self) -> int:
        """Get total number of columns.
        
        Returns:
            Total column count
        """
        return len(self.column_definitions)

