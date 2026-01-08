"""Header column builder - extracts column anchors from header row.

Architecture Rule: Header row (row_index = header_row_index) defines column structure.
Column assignment = nearest x_center match.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from core.config import settings


class HeaderColumnBuilder:
    """Extracts column anchors from header row in row-wise output."""
    
    def __init__(self, header_row_index: Optional[int] = None):
        """Initialize header column builder.
        
        Args:
            header_row_index: Row index to use as header (1-based). If None, uses settings.HEADER_ROW_INDEX
        """
        self.header_row_index = header_row_index if header_row_index is not None else settings.HEADER_ROW_INDEX
        log.info(f"Header column builder initialized (header_row_index: {self.header_row_index})")
    
    def extract_column_anchors(self, rowwise_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract column anchors from header row.
        
        Rule: Row 2 (header row) defines the column structure.
        
        Algorithm:
        1. Find header row (row_index == header_row_index)
        2. Extract PrintedText items with x_centers
        3. Sort by x_center (left to right)
        4. Create column anchors
        
        Args:
            rowwise_output: Row-wise structured output
            
        Returns:
            List of column anchors with {column_index, header, x_center}
        """
        rows = rowwise_output.get('rows', [])
        if not rows:
            log.warning("No rows in rowwise output, cannot extract column anchors")
            return []
        
        # Find header row
        header_row = None
        for row in rows:
            if row.get('row_index') == self.header_row_index:
                header_row = row
                break
        
        if not header_row:
            log.warning(f"Header row {self.header_row_index} not found in rowwise output")
            return []
        
        # Extract header cells with x_centers
        header_internal = header_row.get('_internal', {})
        header_printed = header_internal.get('PrintedText', [])
        
        if not header_printed:
            log.warning(f"Header row {self.header_row_index} has no PrintedText cells")
            return []
        
        log.info(f"Found header row {self.header_row_index} with {len(header_printed)} header cells")
        
        # Sort by x_center (left to right)
        header_cells = []
        for item in header_printed:
            if isinstance(item, dict) and 'text' in item and 'x_center' in item:
                header_cells.append({
                    'text': item['text'].strip(),
                    'x_center': item['x_center']
                })
            else:
                log.warning(f"Header cell missing x_center: {item}")
        
        if not header_cells:
            log.warning("No valid header cells with x_centers found")
            return []
        
        header_cells.sort(key=lambda x: x['x_center'])
        
        # Build column anchors
        anchors = []
        for idx, cell in enumerate(header_cells, start=1):
            anchors.append({
                'column_index': idx,
                'header': cell['text'] if cell['text'] else f"Column {idx}",
                'x_center': cell['x_center']
            })
            log.debug(f"Column {idx}: '{cell['text']}' at x_center={cell['x_center']:.4f}")
        
        log.info(f"Extracted {len(anchors)} column anchors from header row {self.header_row_index}")
        return anchors
    
    def assign_column(self, x_center: float, anchors: List[Dict[str, Any]]) -> Optional[int]:
        """Assign a detection to a column using nearest x_center match.
        
        Rule: A detection belongs to the column whose anchor x_center is closest.
        
        Args:
            x_center: X-center of detection
            anchors: List of column anchors with x_center
            
        Returns:
            Column index (1-based) or None if no anchors
        """
        if not anchors:
            return None
        
        # Find anchor with closest x_center
        nearest_anchor = min(anchors, key=lambda a: abs(a['x_center'] - x_center))
        return nearest_anchor['column_index']
