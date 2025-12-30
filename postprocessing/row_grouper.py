"""Table-aware row grouping - clusters detections into rows and orders by columns."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple
from core.logging import log
import numpy as np


class RowGrouper:
    """Groups detections into rows based on Y-center clustering, orders by X within rows."""
    
    def __init__(self, row_height_threshold: float = 0.02):
        """Initialize row grouper.
        
        Args:
            row_height_threshold: Normalized height threshold for row clustering
                                (default 0.02 = 2% of image height)
        """
        self.row_height_threshold = row_height_threshold
        log.info(f"Row grouper initialized (row_height_threshold: {row_height_threshold})")
    
    def group_into_rows(self,
                       detections: List[Dict[str, Any]],
                       table_bboxes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Group detections into rows with table awareness.
        
        Args:
            detections: List of detections with 'bbox' (normalized [x1, y1, x2, y2])
            table_bboxes: Optional list of table bboxes for table-aware grouping
            
        Returns:
            List of rows, each containing ordered detections
        """
        if not detections:
            return []
        
        # Step 1: Sort by Y-center
        detections_with_center = []
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                y_center = (bbox[1] + bbox[3]) / 2.0
                detections_with_center.append((y_center, det))
        
        # Sort by Y-center (top to bottom)
        detections_with_center.sort(key=lambda x: x[0])
        
        # Step 2: Cluster into rows
        rows = []
        current_row = []
        current_y_center = None
        
        for y_center, det in detections_with_center:
            if current_y_center is None:
                # First detection - start new row
                current_row = [det]
                current_y_center = y_center
            else:
                # Check if this detection belongs to current row
                y_diff = abs(y_center - current_y_center)
                
                if y_diff <= self.row_height_threshold:
                    # Same row
                    current_row.append(det)
                    # Update row center (weighted average)
                    current_y_center = (current_y_center * (len(current_row) - 1) + y_center) / len(current_row)
                else:
                    # New row - finalize current row
                    if current_row:
                        rows.append(self._finalize_row(current_row))
                    # Start new row
                    current_row = [det]
                    current_y_center = y_center
        
        # Finalize last row
        if current_row:
            rows.append(self._finalize_row(current_row))
        
        log.info(f"Grouped {len(detections)} detections into {len(rows)} rows")
        return rows
    
    def _finalize_row(self, row_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a row by sorting detections by X-center (left to right).
        
        Args:
            row_detections: Detections in the same row
            
        Returns:
            Row dict with sorted detections
        """
        # Calculate Y-center for row
        y_centers = []
        for det in row_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                y_centers.append((bbox[1] + bbox[3]) / 2.0)
        
        row_y_center = np.mean(y_centers) if y_centers else 0.0
        
        # Sort by X-center (left to right)
        detections_with_x = []
        for det in row_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                detections_with_x.append((x_center, det))
        
        detections_with_x.sort(key=lambda x: x[0])
        sorted_detections = [det for _, det in detections_with_x]
        
        return {
            'row_index': None,  # Will be set by caller
            'y_center': row_y_center,
            'detections': sorted_detections,
            'column_count': len(sorted_detections)
        }
    
    def assign_row_indices(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign row indices (1-based) to rows.
        
        Args:
            rows: List of row dicts
            
        Returns:
            Rows with row_index assigned
        """
        for i, row in enumerate(rows, start=1):
            row['row_index'] = i
        return rows


