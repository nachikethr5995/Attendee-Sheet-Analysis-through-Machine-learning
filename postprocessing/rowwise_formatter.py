"""Row-wise structured output formatter - SOURCE OF TRUTH for document structure.

Architecture Rule: YOLO defines structure; OCR only fills content within YOLO regions.
Row-wise is the source of truth. Column-wise is a pure pivot of row-wise.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import re
from typing import List, Dict, Any, Optional
from core.logging import log


class RowwiseFormatter:
    """Formats grouped rows into structured row-wise JSON output (SOURCE OF TRUTH)."""
    
    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """Initialize row-wise formatter.
        
        Args:
            column_mapping: Optional mapping from detected text to semantic column names
                           e.g., {'First Name': 'first_name', 'Last Name': 'last_name'}
        """
        self.column_mapping = column_mapping or {}

    
    def format_rows(self,
                   rows: List[Dict[str, Any]],
                   ocr_results: List[Dict[str, Any]],
                   signatures: List[Dict[str, Any]],
                   checkboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format grouped rows into structured row-wise JSON output.
        
        Architecture Rule: This is the SOURCE OF TRUTH. Column-wise is a pure pivot of this.
        
        Args:
            rows: List of row groups from RowGrouper
            ocr_results: OCR results from ClassBasedOCRRouter
            signatures: Signature results from SignatureHandler
            checkboxes: Checkbox results from CheckboxHandler
            
        Returns:
            Structured row-wise output with X-centers stored for column assignment
        """
        if not rows:
            return {
                'rows': [],
                'total_rows': 0,
                'total_columns': 0
            }
        
        # Create lookup maps for quick matching
        ocr_by_bbox = self._create_bbox_lookup(ocr_results)
        signature_by_bbox = self._create_bbox_lookup(signatures)
        checkbox_by_bbox = self._create_bbox_lookup(checkboxes)
        
        formatted_rows = []
        
        for row in rows:
            detections = row.get('detections', [])
            row_index = row.get('row_index', 0)
            
            # Initialize row columns with OCR text lists (with X-centers)
            columns = {
                'PrintedText': [],      # List of {text, x_center} dicts
                'HandwrittenText': [],  # List of {text, x_center} dicts
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
                
                # Compute X-center for column assignment
                x_center = None
                if len(bbox) >= 4:
                    x_center = (bbox[0] + bbox[2]) / 2.0
                
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
                        if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                            normalized_text = self._normalize_text(text)
                            if normalized_text and x_center is not None:
                                columns['PrintedText'].append({
                                    'text': normalized_text,
                                    'x_center': x_center
                                })
                                log.debug(f"Row {row_index}: Added PrintedText '{normalized_text}' at x={x_center:.4f}")
                
                elif class_name == 'handwritten':
                    # Handwritten → PARSeq text
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        # Verify it came from PARSeq (replaces TrOCR) and meets confidence threshold (0.4 for handwriting)
                        if ocr_source == 'parseq' and confidence >= 0.4 and text:
                            normalized_text = self._normalize_text(text)
                            if normalized_text and x_center is not None:
                                columns['HandwrittenText'].append({
                                    'text': normalized_text,
                                    'x_center': x_center
                                })
                                log.debug(f"Row {row_index}: Added HandwrittenText '{normalized_text}' at x={x_center:.4f}")
                
                elif class_name == 'signature':
                    # Signature → Presence flag only (NO OCR)
                    if signature and signature.get('present', False):
                        columns['Signature'] = True
                        log.debug(f"Row {row_index}: Signature present")
                
                elif class_name == 'checkbox':
                    # Checkbox → Presence + checked/unchecked (NO OCR)
                    if checkbox and checkbox.get('present', False):
                        checked_value = checkbox.get('checked', False)
                        columns['Checkbox'] = bool(checked_value) if checked_value is not None else None
                        log.debug(f"Row {row_index}: Checkbox present, checked={columns['Checkbox']}")
                
                # Other classes (Table, etc.) - no processing needed here
            
            # Post-processing: Deduplication per row (preserve X-centers)
            columns['PrintedText'] = self._deduplicate_with_x_center(columns['PrintedText'])
            columns['HandwrittenText'] = self._deduplicate_with_x_center(columns['HandwrittenText'])
            
            # Convert X-center dicts to simple lists for final output (backward compatibility)
            # X-centers are stored internally but output as simple lists
            output_columns = {
                'PrintedText': [item['text'] if isinstance(item, dict) else item 
                                for item in columns['PrintedText']],
                'HandwrittenText': [item['text'] if isinstance(item, dict) else item 
                                   for item in columns['HandwrittenText']],
                'Signature': columns['Signature'],
                'Checkbox': columns['Checkbox']
            }
            
            formatted_rows.append({
                'row_index': row_index,
                'columns': output_columns,
                '_internal': {  # Internal data with X-centers for column assignment
                    'PrintedText': columns['PrintedText'],
                    'HandwrittenText': columns['HandwrittenText']
                }
            })
        
        # Calculate total columns (max number of PrintedText items across all rows)
        max_columns = 0
        for row in formatted_rows:
            printed_count = len(row.get('columns', {}).get('PrintedText', []))
            handwritten_count = len(row.get('columns', {}).get('HandwrittenText', []))
            max_columns = max(max_columns, printed_count, handwritten_count)
        
        log.info(f"Formatted {len(formatted_rows)} rows (max columns: {max_columns})")
        
        return {
            'rows': formatted_rows,
            'total_rows': len(formatted_rows),
            'total_columns': max_columns
        }
    
    def _deduplicate_with_x_center(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate items while preserving order and X-centers.
        
        Args:
            items: List of {text, x_center} dicts
            
        Returns:
            Deduplicated list preserving order
        """
        seen = set()
        result = []
        for item in items:
            text = item.get('text', '')
            if text and text not in seen:
                seen.add(text)
                result.append(item)
        return result
    
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
