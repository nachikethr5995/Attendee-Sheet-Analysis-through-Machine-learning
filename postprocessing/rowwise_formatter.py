"""Row-wise structured output formatter - converts grouped rows to final API format.

Architecture Rule: YOLO defines structure; OCR only fills content within YOLO regions
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import re
from typing import List, Dict, Any, Optional
from core.logging import log


class RowwiseFormatter:
    """Formats grouped rows into structured row-wise JSON output."""
    
    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """Initialize row-wise formatter.
        
        Args:
            column_mapping: Optional mapping from detected text to semantic column names
                           e.g., {'First Name': 'first_name', 'Last Name': 'last_name'}
        """
        self.column_mapping = column_mapping or {}
        log.info("Row-wise formatter initialized")
    
    def format_rows(self,
                   rows: List[Dict[str, Any]],
                   ocr_results: List[Dict[str, Any]],
                   signatures: List[Dict[str, Any]],
                   checkboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format grouped rows into final structured output with OCR text aggregation.
        
        Architecture Rule: YOLO defines structure; OCR only fills content within YOLO regions
        
        Args:
            rows: List of grouped rows from RowGrouper (already assigned to rows)
            ocr_results: OCR results from ClassBasedOCRRouter (with 'class', 'text', 'source', 'bbox')
            signatures: Signature results from SignatureHandler
            checkboxes: Checkbox results from CheckboxHandler
            
        Returns:
            Structured output with 'rows' array containing PrintedText, HandwrittenText lists
        """
        # Create lookup maps for quick access
        ocr_by_bbox = self._create_bbox_lookup(ocr_results)
        signature_by_bbox = self._create_bbox_lookup(signatures)
        checkbox_by_bbox = self._create_bbox_lookup(checkboxes)
        
        formatted_rows = []
        
        for row in rows:
            detections = row.get('detections', [])
            row_index = row.get('row_index', 0)
            
            # Initialize row columns with OCR text lists
            columns = {
                'PrintedText': [],      # PaddleOCR results (Text_box class)
                'HandwrittenText': [],  # TrOCR results (Handwritten class)
                'Signature': False,     # Presence flag
                'Checkbox': None        # Checked/unchecked or None
            }
            
            # Process each detection in this row
            for det in detections:
                bbox = det.get('bbox', [])
                class_name = det.get('class', '').lower()
                source = det.get('source', '')
                
                # Safety assertion: All detections must come from YOLO
                if source and source != 'YOLOv8s':
                    log.warning(f"⚠️  Non-YOLO detection in row {row_index} (source: {source}) - skipping")
                    continue
                
                # Find matching OCR result
                ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                
                # Find matching signature
                signature = self._find_matching_result(bbox, signature_by_bbox)
                
                # Find matching checkbox
                checkbox = self._find_matching_result(bbox, checkbox_by_bbox)
                
                # STRICT CLASS-BASED ROUTING (No OCR inference outside YOLO regions)
                if class_name in ['text_block', 'text_box']:
                    # Text_box → PaddleOCR text
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        # Verify it came from PaddleOCR and meets confidence threshold (0.6)
                        # Note: Router already applies 0.6 threshold, but double-check here
                        if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                            normalized_text = self._normalize_text(text)
                            if normalized_text:
                                columns['PrintedText'].append(normalized_text)
                                log.debug(f"Row {row_index}: Added PrintedText '{normalized_text}' (conf: {confidence:.3f})")
                
                elif class_name == 'handwritten':
                    # Handwritten → TrOCR text
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        # Verify it came from TrOCR and meets confidence threshold (0.4 for handwriting)
                        # Note: Router already applies 0.4 threshold, but double-check here
                        if ocr_source == 'trocr' and confidence >= 0.4 and text:
                            normalized_text = self._normalize_text(text)
                            if normalized_text:
                                columns['HandwrittenText'].append(normalized_text)
                                log.debug(f"Row {row_index}: Added HandwrittenText '{normalized_text}' (conf: {confidence:.3f})")
                
                elif class_name == 'signature':
                    # Signature → Presence flag only (NO OCR)
                    if signature and signature.get('present', False):
                        columns['Signature'] = True
                        log.debug(f"Row {row_index}: Signature present")
                
                elif class_name == 'checkbox':
                    # Checkbox → Presence + checked/unchecked (NO OCR)
                    if checkbox and checkbox.get('present', False):
                        columns['Checkbox'] = checkbox.get('checked', False)
                        log.debug(f"Row {row_index}: Checkbox present, checked={columns['Checkbox']}")
                
                # Other classes (Table, etc.) - no processing needed here
            
            # Post-processing: Deduplication per row
            columns['PrintedText'] = list(dict.fromkeys(columns['PrintedText']))  # Preserve order, remove duplicates
            columns['HandwrittenText'] = list(dict.fromkeys(columns['HandwrittenText']))  # Preserve order, remove duplicates
            
            formatted_rows.append({
                'row_index': row_index,
                'columns': columns
            })
        
        log.info(f"Formatted {len(formatted_rows)} rows into structured output")
        log.info(f"  Rows with PrintedText: {sum(1 for r in formatted_rows if r['columns']['PrintedText'])}")
        log.info(f"  Rows with HandwrittenText: {sum(1 for r in formatted_rows if r['columns']['HandwrittenText'])}")
        log.info(f"  Rows with Signature: {sum(1 for r in formatted_rows if r['columns']['Signature'])}")
        log.info(f"  Rows with Checkbox: {sum(1 for r in formatted_rows if r['columns']['Checkbox'] is not None)}")
        
        return {
            'rows': formatted_rows,
            'total_rows': len(formatted_rows),
            'total_columns': sum(len(r.get('columns', {})) for r in formatted_rows)
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by collapsing whitespace.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text (collapsed whitespace, stripped)
        """
        if not text:
            return ""
        # Collapse multiple whitespace to single space
        normalized = re.sub(r"\s+", " ", text.strip())
        return normalized
    
    def _create_bbox_lookup(self, results: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        """Create bbox-based lookup map for quick matching.
        
        Args:
            results: List of results with 'bbox' field
            
        Returns:
            Dict mapping bbox tuple to result
        """
        lookup = {}
        for result in results:
            bbox = result.get('bbox', [])
            if len(bbox) >= 4:
                # Use rounded bbox for matching (to handle floating point differences)
                bbox_key = tuple(round(x, 4) for x in bbox[:4])
                lookup[bbox_key] = result
        return lookup
    
    def _find_matching_result(self,
                              bbox: List[float],
                              lookup: Dict[tuple, Dict[str, Any]],
                              tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
        """Find matching result by bbox with tolerance.
        
        Args:
            bbox: Bounding box to match
            lookup: Bbox lookup dictionary
            tolerance: Matching tolerance (normalized)
            
        Returns:
            Matching result or None
        """
        if len(bbox) < 4:
            return None
        
        # Try exact match first
        bbox_key = tuple(round(x, 4) for x in bbox[:4])
        if bbox_key in lookup:
            return lookup[bbox_key]
        
        # Try approximate match (within tolerance)
        for key, result in lookup.items():
            if len(key) >= 4:
                x1_diff = abs(bbox[0] - key[0])
                y1_diff = abs(bbox[1] - key[1])
                x2_diff = abs(bbox[2] - key[2])
                y2_diff = abs(bbox[3] - key[3])
                
                if (x1_diff < tolerance and y1_diff < tolerance and
                    x2_diff < tolerance and y2_diff < tolerance):
                    return result
        
        return None
    
    def _infer_column_name(self,
                           text: str,
                           existing_columns: Dict[str, Any],
                           default: Optional[str] = None) -> str:
        """Infer column name from text content.
        
        Args:
            text: Detected text
            existing_columns: Already assigned columns in this row
            default: Default column name if inference fails
            
        Returns:
            Inferred column name
        """
        # Check if text matches a known column mapping
        text_lower = text.lower().strip()
        for key, mapped_name in self.column_mapping.items():
            if text_lower == key.lower():
                return mapped_name
        
        # Use text as column name (capitalize first letter)
        if text:
            return text.strip()
        
        # Fallback to default or generic name
        if default:
            return default
        
        # Generate generic column name
        col_num = len(existing_columns) + 1
        return f"Column_{col_num}"
    
    def _infer_checkbox_label(self,
                              checkbox: Dict[str, Any],
                              existing_columns: Dict[str, Any]) -> str:
        """Infer checkbox label from context.
        
        Args:
            checkbox: Checkbox result
            existing_columns: Already assigned columns in this row
            
        Returns:
            Checkbox label
        """
        # Try to find a nearby text column that might be the label
        # For now, use a generic label
        # TODO: Implement spatial relationship detection
        
        # Check if there's a common checkbox label pattern
        for col_name in existing_columns.keys():
            if any(keyword in col_name.lower() for keyword in ['opt', 'meal', 'diet', 'preference']):
                return col_name
        
        # Default checkbox label
        return "Checkbox"

