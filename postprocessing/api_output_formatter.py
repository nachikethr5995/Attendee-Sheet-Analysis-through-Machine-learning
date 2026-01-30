"""API Output Formatter - Shapes rowwise/columnwise data into final API response.

Architecture Rule: NO NULL IF DATA EXISTS
- PrintedText[0] always wins over HandwrittenText[0]
- Null only if both are empty
- No OCR re-running, no YOLO changes, no inference
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import Dict, Any, List, Optional
from core.logging import log


# Column Header → Output Field Mapping (LOCKED)
COLUMN_TO_FIELD = {
    "Last Name": "last_name",
    "First Name": "first_name",
    "Attendee Type": "attendee_type",
    "Credential or Title": "credential",
    "State of License": "state_of_license",
    "NPI or State License#": "license_number",
    "Signature": "signature",
    "Checkbox": "checkbox"
}


def pick_value(column: Dict[str, Any]) -> Optional[str]:
    """Canonical value picker (MANDATORY).
    
    Precedence Rule (STRICT & FINAL):
    1. PrintedText[0]
    2. HandwrittenText[0]
    3. None (ONLY if both are empty)
    
    Printed text ALWAYS wins over handwritten text.
    
    Args:
        column: Column dict with 'PrintedText' and 'HandwrittenText' lists
        
    Returns:
        First available text value or None
    """
    printed = column.get("PrintedText", [])
    handwritten = column.get("HandwrittenText", [])
    
    # Extract text from dicts if needed
    if printed:
        if isinstance(printed[0], dict):
            return printed[0].get("text")
        return printed[0]
    
    if handwritten:
        if isinstance(handwritten[0], dict):
            return handwritten[0].get("text")
        return handwritten[0]
    
    return None


def build_api_row(row_index: int, cols_by_header: Dict[str, Dict[str, Any]], row_columns: Dict[str, Any]) -> Dict[str, Any]:
    """Build one API row from columnwise data organized by header.
    
    Args:
        row_index: Row index (from rowwise)
        cols_by_header: Dict mapping column headers to column data (from columnwise)
        row_columns: Row columns dict from rowwise (for Signature/Checkbox)
        
    Returns:
        API row dict with field names
    """
    row_id_str = str(row_index)
    
    # Build API row using column-header lookup from columnwise
    # Columnwise is derived from rowwise, so this is still using rowwise as source of truth
    api_row = {
        "last_name": pick_value(cols_by_header.get("Last Name", {}).get("rows", {}).get(row_id_str, {})),
        "first_name": pick_value(cols_by_header.get("First Name", {}).get("rows", {}).get(row_id_str, {})),
        "attendee_type": pick_value(cols_by_header.get("Attendee Type", {}).get("rows", {}).get(row_id_str, {})),
        "credential": pick_value(cols_by_header.get("Credential or Title", {}).get("rows", {}).get(row_id_str, {})),
        "state_of_license": pick_value(cols_by_header.get("State of License", {}).get("rows", {}).get(row_id_str, {})),
        "license_number": pick_value(cols_by_header.get("NPI or State License#", {}).get("rows", {}).get(row_id_str, {})),
        "signature": row_columns.get("Signature", False) is True,
        "checkbox": row_columns.get("Checkbox")
    }
    
    return api_row


def build_api_response(rowwise: Dict[str, Any], columnwise: Dict[str, Any]) -> Dict[str, Any]:
    """Build final API response from rowwise and columnwise data.
    
    Architecture:
    - Source of truth: rowwise.rows[i].columns
    - Column-header lookup from columnwise.columns
    - Skip header row (row_index == 1)
    - One API row per table row
    
    Args:
        rowwise: Row-wise structured output
        columnwise: Column-wise structured output
        
    Returns:
        API response dict with 'rows' list
    """
    rows = rowwise.get("rows", [])
    
    if not rows:
        log.warning("No rows in rowwise output - returning empty response")
        return {"rows": []}
    
    # Build column-header lookup (ONCE PER REQUEST)
    cols_by_header = {}
    columns = columnwise.get("columns", [])
    for column in columns:
        header = column.get("header", "")
        if header:
            cols_by_header[header] = column
    
    log.debug(f"Built column-header lookup with {len(cols_by_header)} columns")
    
    # Build API rows (skip header row)
    api_rows = []
    for row in rows:
        row_index = row.get("row_index", 0)
        
        # Skip header row
        if row_index == 1:
            log.debug(f"Skipping header row (row_index=1)")
            continue
        
        # Get row columns from rowwise (source of truth)
        row_columns = row.get("columns", {})
        
        # Build API row using columnwise data (organized by header) and rowwise columns (for Signature/Checkbox)
        api_row = build_api_row(row_index, cols_by_header, row_columns)
        api_rows.append(api_row)
        
        log.debug(f"Built API row for row_index={row_index}")
    
    # Safety guards (non-breaking)
    for i, r in enumerate(api_rows):
        # Ensure no field is accidentally dropped
        assert "last_name" in r, f"Row {i}: missing 'last_name' field"
        assert "first_name" in r, f"Row {i}: missing 'first_name' field"
        assert "attendee_type" in r, f"Row {i}: missing 'attendee_type' field"
        assert "credential" in r, f"Row {i}: missing 'credential' field"
        assert "state_of_license" in r, f"Row {i}: missing 'state_of_license' field"
        assert "license_number" in r, f"Row {i}: missing 'license_number' field"
        assert "signature" in r, f"Row {i}: missing 'signature' field"
        assert "checkbox" in r, f"Row {i}: missing 'checkbox' field"
        
        # Optional warning for empty rows
        if all(v is None or v is False for v in r.values()):
            log.warning(f"Empty output row detected | row_index={rows[i+1].get('row_index') if i+1 < len(rows) else 'unknown'}")
    
    log.info(f"Built {len(api_rows)} API rows from {len(rows)} rowwise rows")
    
    return {"rows": api_rows}

