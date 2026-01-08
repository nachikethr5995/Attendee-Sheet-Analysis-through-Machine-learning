"""Column class aggregator - independent X-axis column grouping with class isolation.

Architecture Rule: 
- Columns grouped by X-axis clustering (independent of rows)
- Classes isolated per column (PrintedText, HandwrittenText, Signature, Checkbox)
- Structure: column → classes → rows
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from core.logging import log


class ColumnClassAggregator:
    """Aggregates detections into class-aware column structure using independent column grouping.
    
    Structure: column → classes → rows (not column → rows → classes)
    """
    
    def __init__(self, header_row_index: int = 1):
        """Initialize column class aggregator.
        
        Args:
            header_row_index: Row index to use as header (1-based, default: 1 = first row)
        """
        self.header_row_index = header_row_index
        log.info(f"Column class aggregator initialized (header_row_index: {header_row_index})")
    
    def aggregate(self,
                  column_groups: List[Dict[str, Any]],
                  row_groups: List[Dict[str, Any]],
                  ocr_results: List[Dict[str, Any]],
                  signature_results: List[Dict[str, Any]],
                  checkbox_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate independently grouped columns into class-aware structure.
        
        Args:
            column_groups: Independently grouped columns from ColumnGrouper (X-axis clustering)
            row_groups: Row groups from RowGrouper (Y-axis clustering)
            ocr_results: OCR results with text, source, bbox
            signature_results: Signature results with present, bbox
            checkbox_results: Checkbox results with checked, bbox
            
        Returns:
            Class-aware columnwise structure
        """
        if not column_groups or not row_groups:
            return {
                'columns': [],
                'total_columns': 0,
                'total_rows': 0
            }
        
        # Step 1: Assign row_index to all detections from row_groups
        row_index_map = {}  # bbox -> row_index
        for row_group in row_groups:
            row_index = row_group.get('row_index', 0)
            row_detections = row_group.get('detections', [])
            for det in row_detections:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    bbox_key = tuple(round(x, 4) for x in bbox[:4])
                    row_index_map[bbox_key] = row_index
        
        # Step 2: Create lookup maps for OCR, signatures, checkboxes
        ocr_by_bbox = self._create_bbox_lookup(ocr_results)
        signature_by_bbox = self._create_bbox_lookup(signature_results)
        checkbox_by_bbox = self._create_bbox_lookup(checkbox_results)
        
        # Step 3: Initialize column structure: column → classes → rows
        columns = {}
        all_row_indices = set()
        
        for col_group in column_groups:
            column_index = col_group.get('column_index', 0)
            if column_index == 0:
                continue
            
            # Initialize column with class-segregated storage
            columns[column_index] = {
                'column_index': column_index,
                'header': None,  # Will be set from header detection
                'classes': {
                    'PrintedText': {'rows': {}},
                    'HandwrittenText': {'rows': {}},
                    'Signature': {'rows': {}},
                    'Checkbox': {'rows': {}}
                }
            }
        
        # Step 4: Aggregate detections by column → class → row
        for col_group in column_groups:
            column_index = col_group.get('column_index', 0)
            if column_index == 0 or column_index not in columns:
                continue
            
            detections = col_group.get('detections', [])
            col = columns[column_index]
            
            for det in detections:
                bbox = det.get('bbox', [])
                if len(bbox) < 4:
                    continue
                
                # Get row_index from mapping
                bbox_key = tuple(round(x, 4) for x in bbox[:4])
                row_index = row_index_map.get(bbox_key, 0)
                
                if row_index == 0:
                    # Try to find row by bbox matching
                    row_index = self._find_row_index_by_bbox(bbox, row_groups)
                
                if row_index == 0:
                    continue
                
                row_key = str(row_index)
                all_row_indices.add(row_index)
                
                class_name = det.get('class', '').lower()
                
                # Class-isolated aggregation
                if class_name in ['text_block', 'text_box']:
                    # PrintedText from PaddleOCR
                    ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                            if row_key not in col['classes']['PrintedText']['rows']:
                                col['classes']['PrintedText']['rows'][row_key] = []
                            col['classes']['PrintedText']['rows'][row_key].append(text)
                
                elif class_name == 'handwritten':
                    # HandwrittenText from TrOCR
                    ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        if ocr_source == 'trocr' and confidence >= 0.4 and text:
                            if row_key not in col['classes']['HandwrittenText']['rows']:
                                col['classes']['HandwrittenText']['rows'][row_key] = []
                            col['classes']['HandwrittenText']['rows'][row_key].append(text)
                
                elif class_name == 'signature':
                    # Signature presence
                    signature = self._find_matching_result(bbox, signature_by_bbox)
                    if signature and signature.get('present', False):
                        col['classes']['Signature']['rows'][row_key] = True
                
                elif class_name == 'checkbox':
                    # Checkbox state
                    checkbox = self._find_matching_result(bbox, checkbox_by_bbox)
                    if checkbox and checkbox.get('present', False):
                        col['classes']['Checkbox']['rows'][row_key] = checkbox.get('checked', False)
        
        # Step 5: Detect column headers (printed text in topmost row)
        self._detect_column_headers(columns, column_groups, row_groups, ocr_by_bbox)
        
        # Step 6: Normalize empty rows explicitly (structural completeness)
        for col in columns.values():
            for row_index in all_row_indices:
                row_key = str(row_index)
                
                # Ensure all classes have entries for all rows
                if row_key not in col['classes']['PrintedText']['rows']:
                    col['classes']['PrintedText']['rows'][row_key] = []
                if row_key not in col['classes']['HandwrittenText']['rows']:
                    col['classes']['HandwrittenText']['rows'][row_key] = []
                if row_key not in col['classes']['Signature']['rows']:
                    col['classes']['Signature']['rows'][row_key] = False
                if row_key not in col['classes']['Checkbox']['rows']:
                    col['classes']['Checkbox']['rows'][row_key] = None
        
        # Step 7: Deduplicate text lists
        for col in columns.values():
            for class_name in ['PrintedText', 'HandwrittenText']:
                for row_key, text_list in col['classes'][class_name]['rows'].items():
                    if isinstance(text_list, list):
                        col['classes'][class_name]['rows'][row_key] = list(dict.fromkeys(text_list))
        
        # Step 8: Convert to list and sort by column_index
        columns_list = [col for col in sorted(columns.values(), key=lambda c: c['column_index'])]
        
        # Step 9: Validation assertions
        self._validate_output(columns_list, len(all_row_indices))
        
        total_rows = len(all_row_indices)
        log.info(f"Aggregated {len(column_groups)} columns into class-aware structure ({total_rows} rows)")
        
        return {
            'columns': columns_list,
            'total_columns': len(columns_list),
            'total_rows': total_rows
        }
    
    def _detect_column_headers(self,
                               columns: Dict[int, Dict[str, Any]],
                               column_groups: List[Dict[str, Any]],
                               row_groups: List[Dict[str, Any]],
                               ocr_by_bbox: Dict[tuple, Dict[str, Any]]) -> None:
        """Detect column headers from printed text in topmost row.
        
        Rule: Column header = printed text in topmost row of that column.
        
        Args:
            columns: Column structures to update
            column_groups: Column groups with detections
            row_groups: Row groups for finding topmost row
            ocr_by_bbox: OCR lookup for header text
        """
        if not row_groups:
            return
        
        # Find topmost row (minimum row_index)
        topmost_row_index = min(row.get('row_index', float('inf')) for row in row_groups)
        if topmost_row_index == float('inf'):
            return
        
        # Extract header text from each column's topmost row detections
        for col_group in column_groups:
            column_index = col_group.get('column_index', 0)
            if column_index == 0 or column_index not in columns:
                continue
            
            detections = col_group.get('detections', [])
            header_texts = []
            
            for det in detections:
                # Check if this detection is in the topmost row
                bbox = det.get('bbox', [])
                if len(bbox) < 4:
                    continue
                
                # Find row_index for this detection
                row_index = self._find_row_index_by_bbox(bbox, row_groups)
                if row_index != topmost_row_index:
                    continue
                
                # Only use printed text (text_block/text_box) for headers
                class_name = det.get('class', '').lower()
                if class_name in ['text_block', 'text_box']:
                    # Get text from OCR
                    ocr_result = self._find_matching_result(bbox, ocr_by_bbox)
                    if ocr_result:
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                            header_texts.append(text)
            
            # Set header text (join multiple texts with space)
            if header_texts:
                header_text = " ".join(header_texts).strip()
                columns[column_index]['header'] = header_text
                log.debug(f"Column {column_index} header: '{header_text}'")
            else:
                # Fallback to generic header
                columns[column_index]['header'] = f"Column {column_index}"
        
        log.info(f"Detected headers for {len([c for c in columns.values() if c['header']])} columns")
    
    def _find_row_index_by_bbox(self,
                                bbox: List[float],
                                row_groups: List[Dict[str, Any]]) -> int:
        """Find row index for a detection by bbox matching.
        
        Args:
            bbox: Detection bounding box
            row_groups: List of row groups
            
        Returns:
            Row index (1-based) or 0 if not found
        """
        for row_group in row_groups:
            row_detections = row_group.get('detections', [])
            for row_det in row_detections:
                row_bbox = row_det.get('bbox', [])
                if len(row_bbox) >= 4:
                    if self._bboxes_match(bbox, row_bbox, tolerance=0.01):
                        return row_group.get('row_index', 0)
        return 0
    
    def _create_bbox_lookup(self, results: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        """Create bbox-based lookup map.
        
        Args:
            results: List of results with 'bbox' field
            
        Returns:
            Dict mapping bbox tuple to result
        """
        lookup = {}
        for result in results:
            bbox = result.get('bbox', [])
            if len(bbox) >= 4:
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
        
        # Try approximate match
        for key, result in lookup.items():
            if len(key) >= 4:
                if self._bboxes_match(bbox, list(key), tolerance):
                    return result
        
        return None
    
    def _bboxes_match(self, bbox1: List[float], bbox2: List[float], tolerance: float = 0.01) -> bool:
        """Check if two bboxes match within tolerance.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            tolerance: Matching tolerance (normalized)
            
        Returns:
            True if bboxes match within tolerance
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        x1_diff = abs(bbox1[0] - bbox2[0])
        y1_diff = abs(bbox1[1] - bbox2[1])
        x2_diff = abs(bbox1[2] - bbox2[2])
        y2_diff = abs(bbox1[3] - bbox2[3])
        
        return (x1_diff < tolerance and y1_diff < tolerance and
                x2_diff < tolerance and y2_diff < tolerance)
    
    def _validate_output(self,
                        columns: List[Dict[str, Any]],
                        total_rows: int) -> None:
        """Validate output structure with assertions.
        
        Args:
            columns: List of column structures
            total_rows: Total number of rows
        """
        # Assertion 1: All columns have all classes
        for col in columns:
            assert 'PrintedText' in col['classes'], \
                f"Column {col['column_index']} missing PrintedText class"
            assert 'HandwrittenText' in col['classes'], \
                f"Column {col['column_index']} missing HandwrittenText class"
            assert 'Signature' in col['classes'], \
                f"Column {col['column_index']} missing Signature class"
            assert 'Checkbox' in col['classes'], \
                f"Column {col['column_index']} missing Checkbox class"
        
        # Assertion 2: All rows present in all classes (structural completeness)
        for col in columns:
            for row_index in range(1, total_rows + 1):
                row_key = str(row_index)
                assert row_key in col['classes']['PrintedText']['rows'], \
                    f"Row {row_index} missing in PrintedText for column {col['column_index']}"
                assert row_key in col['classes']['HandwrittenText']['rows'], \
                    f"Row {row_index} missing in HandwrittenText for column {col['column_index']}"
                assert row_key in col['classes']['Signature']['rows'], \
                    f"Row {row_index} missing in Signature for column {col['column_index']}"
                assert row_key in col['classes']['Checkbox']['rows'], \
                    f"Row {row_index} missing in Checkbox for column {col['column_index']}"
        
        log.debug("All validation assertions passed")
