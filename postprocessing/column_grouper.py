"""Independent column grouping - X-axis clustering (symmetric to row Y-axis clustering).

Architecture Rule: Columns are grouped by X-axis, independent of row grouping.
NEVER infer columns from rows.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from core.config import settings
import numpy as np


class ColumnGrouper:
    """Groups detections into columns based on X-center clustering (independent of rows).
    
    Algorithm: Pure X-center clustering on ALL detections (not anchor-based).
    Symmetric to RowGrouper which uses Y-center clustering.
    """
    
    def __init__(self, column_width_threshold: Optional[float] = None):
        """Initialize column grouper.
        
        Args:
            column_width_threshold: Normalized width threshold for column clustering
                                  (default from settings.COLUMN_WIDTH_THRESHOLD = 3% of image width)
        """
        self.column_width_threshold = column_width_threshold if column_width_threshold is not None else settings.COLUMN_WIDTH_THRESHOLD
        log.info(f"Column grouper initialized (column_width_threshold: {self.column_width_threshold})")
    
    def group_into_columns(self,
                          detections: List[Dict[str, Any]],
                          table_bboxes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Group ALL detections into columns using X-center clustering.
        
        Algorithm:
        1. Compute X-center for each detection
        2. Sort detections left → right
        3. Cluster into columns using threshold
        
        Args:
            detections: List of ALL detections with 'bbox' (normalized [x1, y1, x2, y2])
            table_bboxes: Optional list of table bboxes (for future table-aware grouping)
            
        Returns:
            List of columns, each containing ordered detections
        """
        if not detections:
            return []
        
        # Step 1: Compute X-center for ALL detections
        detections_with_x = []
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                detections_with_x.append((x_center, det))
        
        if not detections_with_x:
            log.warning("No detections with valid bboxes")
            return []
        
        # Step 2: Sort detections left → right by X-center
        detections_with_x.sort(key=lambda x: x[0])
        
        # Step 3: Cluster into columns using threshold
        columns = []
        current_column = []
        current_x_center = None
        
        for x_center, det in detections_with_x:
            if current_x_center is None:
                # First detection - start new column
                current_column = [det]
                current_x_center = x_center
            else:
                # Check if this detection belongs to current column
                x_diff = abs(x_center - current_x_center)
                
                if x_diff <= self.column_width_threshold:
                    # Same column
                    current_column.append(det)
                    # Update column center (weighted average)
                    current_x_center = (current_x_center * (len(current_column) - 1) + x_center) / len(current_column)
                else:
                    # New column - finalize current column
                    if current_column:
                        columns.append(self._finalize_column(current_column))
                    # Start new column
                    current_column = [det]
                    current_x_center = x_center
        
        # Finalize last column
        if current_column:
            columns.append(self._finalize_column(current_column))
        
        log.info(f"Grouped {len(detections)} detections into {len(columns)} columns (X-center clustering)")
        return columns
    
    def _finalize_column(self, column_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a column by calculating boundaries and sorting detections by Y-center (top to bottom).
        
        Args:
            column_detections: Detections in the same column
            
        Returns:
            Column dict with sorted detections and boundaries
        """
        # Calculate X-center and boundaries for column
        x_centers = []
        x_mins = []
        x_maxs = []
        
        for det in column_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                x_centers.append(x_center)
                x_mins.append(bbox[0])
                x_maxs.append(bbox[2])
        
        column_x_center = np.mean(x_centers) if x_centers else 0.0
        column_x_min = min(x_mins) if x_mins else 0.0
        column_x_max = max(x_maxs) if x_maxs else 1.0
        
        # Sort by Y-center (top to bottom) within column
        detections_with_y = []
        for det in column_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                y_center = (bbox[1] + bbox[3]) / 2.0
                detections_with_y.append((y_center, det))
        
        detections_with_y.sort(key=lambda x: x[0])
        sorted_detections = [det for _, det in detections_with_y]
        
        return {
            'column_index': None,  # Will be set by caller
            'x_center': column_x_center,
            'x_min': column_x_min,
            'x_max': column_x_max,
            'detections': sorted_detections,
            'row_count': len(sorted_detections)
        }
    
    def assign_column_indices(self, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign column indices (1-based) to columns.
        
        Args:
            columns: List of column dicts
            
        Returns:
            Columns with column_index assigned
        """
        # Sort columns by X-center to ensure consistent ordering (left to right)
        columns.sort(key=lambda col: col.get('x_center', 0.0))
        
        for i, column in enumerate(columns, start=1):
            column['column_index'] = i
        return columns
