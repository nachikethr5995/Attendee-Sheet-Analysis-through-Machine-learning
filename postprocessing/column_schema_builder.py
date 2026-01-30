"""Column schema builder - builds column schema from header row in row-wise output.

Architecture Rule: Row-wise is source of truth. Column schema built from header row only.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log


class ColumnSchemaBuilder:
    """Builds column schema from header row in row-wise output."""
    
    def __init__(self, min_header_text_count: int = 2):
        """Initialize column schema builder.
        
        Args:
            min_header_text_count: Minimum PrintedText values to consider a row as header (default: 2)
        """
        self.min_header_text_count = min_header_text_count
        log.info(f"Column schema builder initialized (min_header_text_count: {min_header_text_count})")
    
    def build_from_rowwise(self, rowwise_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build column schema from header row in row-wise output.
        
        Rule: Header row = first row with ≥2 PrintedText values
        
        Args:
            rowwise_output: Row-wise structured output
            
        Returns:
            List of column definitions with index and header
        """
        rows = rowwise_output.get('rows', [])
        if not rows:
            log.warning("No rows in rowwise output, cannot build column schema")
            return []
        
        # Find header row: first row with ≥2 PrintedText values
        header_row = None
        for row in rows:
            printed_text = row.get('columns', {}).get('PrintedText', [])
            if len(printed_text) >= self.min_header_text_count:
                header_row = row
                break
        
        if not header_row:
            log.warning(f"No header row found (no row with ≥{self.min_header_text_count} PrintedText values)")
            return []
        
        header_row_index = header_row.get('row_index', 0)
        column_headers = header_row.get('columns', {}).get('PrintedText', [])
        
        log.info(f"Found header row {header_row_index} with {len(column_headers)} columns")
        
        # Build column schema
        columns = []
        for idx, header_text in enumerate(column_headers, start=1):
            columns.append({
                'index': idx,
                'header': header_text.strip() if header_text else f"Column {idx}"
            })
            log.debug(f"Column {idx}: '{header_text}'")
        
        log.info(f"Built column schema with {len(columns)} columns from header row {header_row_index}")
        return columns











