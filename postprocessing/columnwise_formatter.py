"""Column-wise structured output formatter - derived from row-wise (source of truth).

Architecture Rule: Column-wise is a pure pivot of row-wise. Columns are anchored to header x-positions.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from core.config import settings


class ColumnwiseFormatter:
    """Formats row-wise output into column-wise structured JSON output.
    
    Architecture:
    - Columns are anchored to header x-positions (x_center)
    - Cells are assigned to nearest column based on x_center
    - Signature and Checkbox are flags within existing columns
    - Column class derived from YOLO distribution
    """
    
    def __init__(self):
        """Initialize column-wise formatter."""
        log.info("Columnwise formatter initialized")
    
    def format_columns(self, rowwise_output: Dict[str, Any]) -> Dict[str, Any]:
        """Format row-wise output into column-wise structure.
        
        Args:
            rowwise_output: Row-wise structured output (source of truth)
            
        Returns:
            Column-wise structured output with columns anchored to header positions
        """
        rows = rowwise_output.get('rows', [])
        total_rows = rowwise_output.get('total_rows', 0)
        
        if not rows:
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': 0
            }
        
        # Step 1: Build column anchors from header row
        column_anchors = self._build_column_anchors(rows)
        
        if not column_anchors:
            log.warning("No column anchors found - cannot build column-wise output")
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': total_rows
            }
        
        # Step 2: Build column-wise structure
        columns = self._build_columnwise(rows, column_anchors, total_rows)
        
        # Step 3: Classify columns based on YOLO distribution
        for col in columns:
            col['class'] = self._classify_column(col, rowwise_output)
        
        # Step 4: Clean up internal fields (after classification)
        for col in columns:
            if 'x_center' in col:
                del col['x_center']
            if '_yolo_classes' in col:
                del col['_yolo_classes']
        
        # Step 4: Validate columns
        self._validate_columns(columns, column_anchors, total_rows)
        
        log.info(f"Formatted {len(columns)} columns from {total_rows} rows")
        
        return {
            'columns': columns,
            'total_columns': len(columns),
            'total_rows': total_rows
        }
    
    def _build_column_anchors(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build column anchors from header row (row_index == HEADER_ROW_INDEX).
        
        Args:
            rows: List of row dictionaries
            
        Returns:
            List of column anchors with column_index, header, and x_center
        """
        # Find header row
        header_row = None
        for row in rows:
            if row.get('row_index') == settings.HEADER_ROW_INDEX:
                header_row = row
                break
        
        if not header_row:
            log.warning(f"Header row (row_index={settings.HEADER_ROW_INDEX}) not found")
            return []
        
        # Extract header cells from _internal.PrintedText
        header_internal = header_row.get('_internal', {})
        header_cells = header_internal.get('PrintedText', [])
        
        if not header_cells:
            log.warning("Header row has no PrintedText cells")
            return []
        
        # Build column anchors sorted by x_center
        column_anchors = []
        for idx, cell in enumerate(header_cells, start=1):
            if isinstance(cell, dict) and 'text' in cell and 'x_center' in cell:
                column_anchors.append({
                    'column_index': idx,
                    'header': cell['text'],
                    'x_center': cell['x_center']
                })
        
        # Sort by x_center to ensure correct order
        column_anchors.sort(key=lambda x: x['x_center'])
        
        # Reassign column_index after sorting
        for idx, anchor in enumerate(column_anchors, start=1):
            anchor['column_index'] = idx
        
        log.info(f"Built {len(column_anchors)} column anchors from header row")
        return column_anchors
    
    def _nearest_column(self, x_center: float, column_anchors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find nearest column anchor for given x_center.
        
        Args:
            x_center: X-center coordinate of cell
            column_anchors: List of column anchors
            
        Returns:
            Nearest column anchor or None
        """
        if not column_anchors:
            return None
        
        return min(column_anchors, key=lambda col: abs(col['x_center'] - x_center))
    
    def _classify_column(self, column: Dict[str, Any], rowwise_output: Dict[str, Any]) -> str:
        """Classify column based on YOLO class distribution.
        
        Priority:
        1. Signature (if Signature class present)
        2. Checkbox (if Checkbox class present)
        3. Mixed (if both PrintedText and HandwrittenText present)
        4. Handwritten (if only HandwrittenText present)
        5. Text (if only PrintedText present)
        
        Args:
            column: Column dictionary with _yolo_classes set
            rowwise_output: Row-wise output for context
            
        Returns:
            Column class string
        """
        yolo_classes = column.get('_yolo_classes', set())
        
        if 'Signature' in yolo_classes:
            return 'Signature'
        elif 'Checkbox' in yolo_classes:
            return 'Checkbox'
        elif 'Text_box' in yolo_classes and 'Handwritten' in yolo_classes:
            return 'Mixed'
        elif 'Handwritten' in yolo_classes:
            return 'Handwritten'
        elif 'Text_box' in yolo_classes:
            return 'Text'
        else:
            return 'Text'  # Default
    
    def _build_columnwise(self, 
                          rows: List[Dict[str, Any]], 
                          column_anchors: List[Dict[str, Any]],
                          total_rows: int) -> List[Dict[str, Any]]:
        """Build column-wise structure from rows.
        
        Args:
            rows: List of row dictionaries
            column_anchors: Column anchors from header
            total_rows: Total number of rows
            
        Returns:
            List of column dictionaries
        """
        # Step 1: Initialize columns from anchors
        columns = []
        for col in column_anchors:
            columns.append({
                'column_index': col['column_index'],
                'header': col['header'],
                'x_center': col['x_center'],  # Store for distance matching
                'rows': {},
                '_yolo_classes': set()  # Track YOLO classes for column classification
            })
        
        # Step 2: Find Signature and Checkbox header columns
        signature_col = None
        checkbox_col = None
        for col in columns:
            header_lower = col['header'].lower()
            if 'signature' in header_lower:
                signature_col = col
            if 'meal' in header_lower and 'opt' in header_lower:
                checkbox_col = col
        
        # Step 3: Assign every cell to nearest column
        for row in rows:
            row_id = row.get('row_index', 0)
            if row_id == settings.HEADER_ROW_INDEX:  # Skip header row from column data
                continue
            
            row_internal = row.get('_internal', {})
            row_columns = row.get('columns', {})
            
            # Assign PrintedText and HandwrittenText using nearest x_center
            for class_name in ['PrintedText', 'HandwrittenText']:
                items = row_internal.get(class_name, [])
                for item in items:
                    if isinstance(item, dict) and 'text' in item and 'x_center' in item:
                        x_center = item['x_center']
                        text = item['text']
                        
                        # Find nearest column
                        nearest_col = self._nearest_column(x_center, column_anchors)
                        if nearest_col:
                            col_idx = nearest_col['column_index']
                            target = columns[col_idx - 1]  # Convert to 0-based index
                            
                            # Skip if this is a Signature/Checkbox column (they only contain boolean/None)
                            header_lower = target['header'].lower()
                            is_signature_col = 'signature' in header_lower
                            is_checkbox_col = 'meal' in header_lower and 'opt' in header_lower
                            
                            if not is_signature_col and not is_checkbox_col:
                                target['rows'].setdefault(str(row_id), {'PrintedText': [], 'HandwrittenText': []})
                                if class_name == 'PrintedText':
                                    target['rows'][str(row_id)]['PrintedText'].append(text)
                                    target['_yolo_classes'].add('Text_box')
                                elif class_name == 'HandwrittenText':
                                    target['rows'][str(row_id)]['HandwrittenText'].append(text)
                                    target['_yolo_classes'].add('Handwritten')
            
            # Step 4: Assign Signature & Checkbox to their header columns
            if signature_col and row_columns.get('Signature', False):
                signature_col['rows'][str(row_id)] = True
                signature_col['_yolo_classes'].add('Signature')
            
            if checkbox_col and row_columns.get('Checkbox') is not None:
                checkbox_col['rows'][str(row_id)] = row_columns['Checkbox']
                checkbox_col['_yolo_classes'].add('Checkbox')
        
        # Step 5: Ensure all data rows are present (from HEADER_ROW_INDEX + 1 to total_rows)
        for col in columns:
            # Determine column type from header
            header_lower = col['header'].lower()
            is_signature_col = 'signature' in header_lower
            is_checkbox_col = 'meal' in header_lower and 'opt' in header_lower
            
            for i in range(settings.HEADER_ROW_INDEX + 1, total_rows + 1):
                row_key = str(i)
                if row_key not in col['rows']:
                    if is_signature_col:
                        col['rows'][row_key] = False
                    elif is_checkbox_col:
                        col['rows'][row_key] = None
                    else:  # Text, Handwritten, Mixed
                        col['rows'][row_key] = {'PrintedText': [], 'HandwrittenText': []}
        
        return columns
    
    def _validate_columns(self, 
                         columns: List[Dict[str, Any]], 
                         column_anchors: List[Dict[str, Any]],
                         total_rows: int):
        """Validate column-wise structure.
        
        Args:
            columns: List of column dictionaries
            column_anchors: Column anchors used
            total_rows: Total number of rows
        """
        # Validation: Column count == header count
        assert len(columns) == len(column_anchors), \
            f"Column count mismatch: {len(columns)} columns vs {len(column_anchors)} anchors"
        
        # Validation: Every non-header row has the same column count
        data_rows = total_rows - 1  # Exclude header row
        for col in columns:
            row_count = len(col.get('rows', {}))
            assert row_count == data_rows, \
                f"Column {col.get('column_index')} has {row_count} rows, expected {data_rows}"
        
        log.debug(f"Column validation passed: {len(columns)} columns, {data_rows} data rows each")
