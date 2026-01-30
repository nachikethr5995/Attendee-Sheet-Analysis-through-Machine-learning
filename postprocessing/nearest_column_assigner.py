"""Nearest column assigner - assigns detections to columns using X-center proximity.

Architecture Rule: Column assignment = nearest X-center match. Never infer columns per row.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log


class NearestColumnAssigner:
    """Assigns text items to columns using nearest X-center match."""
    
    def __init__(self):
        """Initialize nearest column assigner."""
        log.info("Nearest column assigner initialized")
    
    def assign_column(self, x_center: float, column_anchors: List[Dict[str, Any]]) -> Optional[int]:
        """Assign a detection to a column using nearest X-center match.
        
        Args:
            x_center: X-center of detection
            column_anchors: List of column anchors with x_center
            
        Returns:
            Column index (1-based) or None if no match
        """
        if not column_anchors:
            return None
        
        # Filter out columns without x_center
        valid_columns = [c for c in column_anchors if c.get('x_center') is not None]
        
        if not valid_columns:
            return None
        
        # Find nearest column by X-center distance
        nearest_column = min(valid_columns, key=lambda c: abs(c['x_center'] - x_center))
        
        return nearest_column.get('column_index')
    
    def assign_text_items(self,
                         text_items: List[Dict[str, Any]],
                         column_anchors: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Assign multiple text items to columns.
        
        Args:
            text_items: List of {text, x_center} dicts
            column_anchors: List of column anchors
            
        Returns:
            Dict mapping column_index to list of text values
        """
        column_texts = {}
        
        for item in text_items:
            if isinstance(item, dict) and 'text' in item and 'x_center' in item:
                x_center = item['x_center']
                text = item['text']
                
                column_index = self.assign_column(x_center, column_anchors)
                if column_index:
                    if column_index not in column_texts:
                        column_texts[column_index] = []
                    column_texts[column_index].append(text)
        
        return column_texts











