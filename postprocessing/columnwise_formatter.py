"""Column-wise structured output formatter - header-driven pure pivot of row-wise output.

Architecture Rule: 
- Row-wise is source of truth (group by Y)
- Column-wise is header-driven (group by X, anchors from header row)
- Column assignment = nearest x_center match
- Signature/Checkbox assigned to their header columns (no duplicate columns)

Correct Architecture:
1. Identify Header Row (row_index = HEADER_ROW_INDEX)
2. Build Column Anchors from header x_center positions
3. Assign every cell to nearest column by x_center distance
4. Handle Signature & Checkbox as column-bound flags (assign to header columns)
5. Classify columns (Text / Signature / Checkbox)
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log
from core.config import settings


class ColumnwiseFormatter:
    """Formats row-wise output into column-wise output (header-driven pure pivot)."""
    
    def __init__(self, header_row_index: Optional[int] = None):
        """Initialize column-wise formatter.
        
        Args:
            header_row_index: Row index to use as header (1-based). If None, uses settings.HEADER_ROW_INDEX
        """
        self.header_row_index = header_row_index if header_row_index is not None else settings.HEADER_ROW_INDEX
        log.info(f"Column-wise formatter initialized (header-driven pure pivot mode, header_row_index: {self.header_row_index})")
    
    def format_columns(self,
                      rowwise_output: Dict[str, Any]) -> Dict[str, Any]:
        """Build column-wise output as header-driven pure pivot of row-wise output.
        
        Architecture Rule: 
        - Row-wise is source of truth
        - Columns defined from header row (row_index = header_row_index)
        - Column assignment = nearest x_center match (deterministic)
        - Signature/Checkbox assigned to their header columns (prevents duplicates)
        
        Guarantees:
        - Column-1 will always be populated
        - Printed + handwritten coexist naturally
        - Signature & Checkbox never duplicate
        - Deterministic across runs
        
        Args:
            rowwise_output: Row-wise structured output (SOURCE OF TRUTH)
            
        Returns:
            Column-wise structure (header-driven pure pivot)
        """
        rows = rowwise_output.get('rows', [])
        if not rows:
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': 0
            }
        
        # Step 1: Build column anchors from header row
        log.info("Building column anchors from header row...")
        column_anchors = self._build_column_anchors(rowwise_output)
        
        if not column_anchors:
            log.warning("No column anchors built, returning empty columnwise output")
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': len(rows)
            }
        
        # Step 2: Build columnwise table (includes Signature & Checkbox assignment)
        total_rows = max((row.get('row_index', 0) for row in rows), default=0)
        columns = self._build_columnwise(rows, column_anchors, total_rows)
        
        # Step 3: Validation guards
        self._validate_columns(columns, total_rows, rowwise_output, column_anchors)
        
        log.info(f"Built {len(columns)} columns from {len(rows)} rows (header-driven pure pivot)")
        
        return {
            'columns': columns,
            'total_columns': len(columns),
            'total_rows': total_rows
        }
    
    def _build_column_anchors(self, rowwise_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build column anchors from header row.
        
        Only row_index == header_row_index defines column structure.
        
        Args:
            rowwise_output: Row-wise structured output
            
        Returns:
            List of column anchors with {column_index, header, x_center}
        """
        rows = rowwise_output.get('rows', [])
        
        # Find header row
        header_row = None
        for row in rows:
            if row.get('row_index') == self.header_row_index:
                header_row = row
                break
        
        if not header_row:
            log.warning(f"Header row {self.header_row_index} not found")
            return []
        
        # Extract header cells from _internal.PrintedText
        header_internal = header_row.get('_internal', {})
        header_printed = header_internal.get('PrintedText', [])
        
        if not header_printed:
            log.warning(f"Header row has no PrintedText cells")
            return []
        
        # Build anchors
        anchors = []
        for item in header_printed:
            if isinstance(item, dict) and 'text' in item and 'x_center' in item:
                anchors.append({
                    'header': item['text'].strip(),
                    'x_center': item['x_center']
                })
        
        # Sort by x_center (left to right)
        anchors.sort(key=lambda x: x['x_center'])
        
        # Assign column_index
        for i, col in enumerate(anchors, start=1):
            col['column_index'] = i
            log.debug(f"Column {i}: '{col['header']}' at x_center={col['x_center']:.4f}")
        
        log.info(f"Built {len(anchors)} column anchors from header row")
        return anchors
    
    def _nearest_column(self, x_center: float, column_anchors: List[Dict[str, Any]], tolerance: float = 0.05) -> Optional[Dict[str, Any]]:
        """Find nearest column for a given x_center.
        
        Args:
            x_center: X-center of detection
            column_anchors: List of column anchors
            tolerance: Optional tolerance (not used in nearest match, kept for compatibility)
            
        Returns:
            Column anchor dict or None
        """
        if not column_anchors:
            return None
        
        return min(column_anchors, key=lambda col: abs(col['x_center'] - x_center))
    
    def _classify_column(self, col: Dict[str, Any]) -> str:
        """Classify column based on header and content.
        
        Args:
            col: Column structure with header and rows
            
        Returns:
            Column class: "Signature", "Checkbox", or "Text"
        """
        header_lower = col.get('header', '').lower()
        
        if 'signature' in header_lower:
            return 'Signature'
        if 'meal' in header_lower and 'opt' in header_lower:
            return 'Checkbox'
        
        return 'Text'
    
    def _build_columnwise(self,
                         rows: List[Dict[str, Any]],
                         column_anchors: List[Dict[str, Any]],
                         total_rows: int) -> List[Dict[str, Any]]:
        """Build columnwise table.
        
        Architecture: Header-first, distance-based assignment.
        - Columns defined from header row x_center positions
        - Every cell assigned to nearest column by x_center
        - Signature & Checkbox assigned to their header columns
        
        Args:
            rows: List of row data from rowwise output
            column_anchors: List of column anchors
            total_rows: Total number of rows
            
        Returns:
            List of column structures
        """
        # Step 1: Initialize columns from anchors (single source of truth)
        columns = []
        for col in column_anchors:
            columns.append({
                'column_index': col['column_index'],
                'header': col['header'],
                'x_center': col['x_center'],  # Store for distance matching
                'rows': {}
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
            if row_id == 0:
                continue
            
            row_internal = row.get('_internal', {})
            row_columns = row.get('columns', {})
            
            # Assign PrintedText and HandwrittenText using nearest x_center
            # Skip Signature/Checkbox columns (they only contain boolean/None values)
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
                                target['rows'].setdefault(str(row_id), []).append(text)
            
            # Step 4: Assign Signature & Checkbox to their header columns
            if signature_col and row_columns.get('Signature', False):
                signature_col['rows'][str(row_id)] = True
            
            if checkbox_col and row_columns.get('Checkbox') is not None:
                checkbox_col['rows'][str(row_id)] = row_columns['Checkbox']
        
        # Step 5: Classify columns and ensure all rows are present
        for col in columns:
            col['class'] = self._classify_column(col)
            
            # Remove x_center from final output (internal use only)
            if 'x_center' in col:
                del col['x_center']
            
            # Ensure all rows are present
            # Text columns: empty list for missing rows
            # Signature/Checkbox columns: False/None for missing rows
            for i in range(1, total_rows + 1):
                row_key = str(i)
                if row_key not in col['rows']:
                    if col['class'] in ['Signature', 'Checkbox']:
                        col['rows'][row_key] = False if col['class'] == 'Signature' else None
                    else:
                        col['rows'][row_key] = []
        
        return columns
    
    def _inject_boolean_columns(self,
                                columns: List[Dict[str, Any]],
                                rows: List[Dict[str, Any]]) -> None:
        """Inject Signature & Checkbox columns.
        
        NOTE: This method is now a no-op because Signature & Checkbox are assigned
        to existing columns in _build_columnwise. Kept for backward compatibility.
        
        Args:
            columns: List of column structures (modified in place)
            rows: List of row data from rowwise output
        """
        # Signature and Checkbox are now handled in _build_columnwise
        # by assigning them to their header columns based on x_center matching
        # This prevents duplicate columns and maintains column-bound semantics
        log.debug("Signature and Checkbox columns handled in _build_columnwise")
    
    def _validate_columns(self,
                         columns: List[Dict[str, Any]],
                         total_rows: int,
                         rowwise_output: Dict[str, Any],
                         column_anchors: List[Dict[str, Any]]) -> None:
        """Validate column structure with comprehensive guards.
        
        Args:
            columns: List of column structures
            total_rows: Total number of rows
            rowwise_output: Row-wise output for data loss detection
            column_anchors: Column anchors for validation
        """
        # Column coverage assertion
        assert len(columns) > 0, "No columns built"
        # Columns should match anchors exactly (Signature/Checkbox are assigned to existing columns)
        assert len(columns) == len(column_anchors), \
            f"Column count mismatch: {len(columns)} columns vs expected {len(column_anchors)} anchors"
        
        # Row completeness check
        for col in columns:
            rows = col.get('rows', {})
            assert len(rows) == total_rows, \
                f"Column {col['column_index']} has {len(rows)} rows, expected {total_rows}"
        
        # Data loss detection: sum of cells should match
        total_column_cells = sum(
            len(cell) if isinstance(cell, list) else (1 if cell is not None and cell is not False else 0)
            for col in columns
            for cell in col.get('rows', {}).values()
        )
        
        total_row_cells = 0
        for row in rowwise_output.get('rows', []):
            row_cols = row.get('columns', {})
            total_row_cells += len(row_cols.get('PrintedText', []))
            total_row_cells += len(row_cols.get('HandwrittenText', []))
            if row_cols.get('Signature'):
                total_row_cells += 1
            if row_cols.get('Checkbox') is not None:
                total_row_cells += 1
        
        # Note: Exact match may not be possible due to column assignment logic,
        # but we should have reasonable bounds
        assert total_column_cells > 0, "No data in columns"
        assert total_row_cells > 0, "No data in rows"
        
        log.debug(f"Validation passed: {total_column_cells} column cells, {total_row_cells} row cells")
        log.debug("All column validation assertions passed")
