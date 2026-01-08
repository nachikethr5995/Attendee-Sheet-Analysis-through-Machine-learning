"""Column anchor - explicit column definition with locked X-ranges and dominant class.

Architecture Rule: Columns are defined once from header row and reused everywhere.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.logging import log


@dataclass
class ColumnAnchor:
    """Explicit column definition with locked boundaries and dominant class."""
    column_index: int
    header_text: str
    x_center: float
    x_min: float
    x_max: float
    dominant_class: str  # "PrintedText", "HandwrittenText", "Signature", or "Checkbox"


class ColumnAnchorBuilder:
    """Builds explicit column anchors from header row."""
    
    def __init__(self, header_row_index: int = 1, boundary_expansion: float = 0.02):
        """Initialize column anchor builder.
        
        Args:
            header_row_index: Row index to use as header (1-based, default: 1 = first row)
            boundary_expansion: Expansion factor for X boundaries (±2% to absorb noise)
        """
        self.header_row_index = header_row_index
        self.boundary_expansion = boundary_expansion
        log.info(f"Column anchor builder initialized (header_row_index: {header_row_index})")
    
    def build_from_header_row(self,
                              header_row: Dict[str, Any],
                              ocr_results: List[Dict[str, Any]]) -> List[ColumnAnchor]:
        """Build column anchors from header row.
        
        Args:
            header_row: Header row with detections (must have row_index matching header_row_index)
            ocr_results: OCR results for header text extraction
            
        Returns:
            List of ColumnAnchor objects with locked X-ranges
        """
        detections = header_row.get('detections', [])
        if not detections:
            log.warning("No detections in header row, cannot build column anchors")
            return []
        
        # Create OCR lookup for header text
        ocr_by_bbox = {}
        for ocr_result in ocr_results:
            bbox = ocr_result.get('bbox', [])
            if len(bbox) >= 4:
                bbox_key = tuple(round(x, 4) for x in bbox[:4])
                ocr_by_bbox[bbox_key] = ocr_result
        
        # Extract PrintedText detections from header row (these define columns)
        header_detections = []
        for det in detections:
            class_name = det.get('class', '').lower()
            if class_name in ['text_block', 'text_box']:
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    x_min = bbox[0]
                    x_max = bbox[2]
                    
                    # Extract header text from OCR
                    header_text = None
                    bbox_key = tuple(round(x, 4) for x in bbox[:4])
                    
                    # Try exact match first
                    if bbox_key in ocr_by_bbox:
                        ocr_result = ocr_by_bbox[bbox_key]
                        text = ocr_result.get('text', '').strip()
                        ocr_source = ocr_result.get('source', '')
                        confidence = ocr_result.get('confidence', 0.0)
                        
                        if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                            header_text = text
                    
                    # Try approximate match if exact match failed
                    if not header_text:
                        for key, ocr_result in ocr_by_bbox.items():
                            if len(key) >= 4:
                                # Check if bboxes overlap significantly
                                if (abs(bbox[0] - key[0]) < 0.02 and abs(bbox[1] - key[1]) < 0.02 and
                                    abs(bbox[2] - key[2]) < 0.02 and abs(bbox[3] - key[3]) < 0.02):
                                    text = ocr_result.get('text', '').strip()
                                    ocr_source = ocr_result.get('source', '')
                                    confidence = ocr_result.get('confidence', 0.0)
                                    
                                    if ocr_source == 'paddleocr' and confidence >= 0.6 and text:
                                        header_text = text
                                        break
                    
                    # Fallback: use generic header if OCR text not found
                    if not header_text:
                        header_text = f"Column {len(header_detections) + 1}"
                        log.debug(f"No OCR text found for detection at {bbox}, using fallback header '{header_text}'")
                    
                    # Always add detection (even if header_text is fallback)
                    header_detections.append((x_center, x_min, x_max, header_text))
        
        # Sort by X-center (left to right)
        header_detections.sort(key=lambda x: x[0])
        
        if not header_detections:
            log.warning("No PrintedText in header row, cannot build column anchors")
            return []
        
        # Build column anchors with expanded boundaries
        column_anchors = []
        for idx, (x_center, x_min, x_max, header_text) in enumerate(header_detections, start=1):
            # Expand boundaries slightly to absorb noise
            x_range = x_max - x_min
            expanded_x_min = max(0.0, x_min - (x_range * self.boundary_expansion))
            expanded_x_max = min(1.0, x_max + (x_range * self.boundary_expansion))
            
            # Adjust boundaries to avoid overlap with adjacent columns
            if idx > 1:
                prev_anchor = column_anchors[-1]
                # Use midpoint between previous and current
                expanded_x_min = (prev_anchor.x_max + expanded_x_min) / 2.0
            if idx < len(header_detections):
                next_x_center, next_x_min, next_x_max, _ = header_detections[idx]
                # Use midpoint between current and next
                expanded_x_max = (expanded_x_max + next_x_min) / 2.0
            
            anchor = ColumnAnchor(
                column_index=idx,
                header_text=header_text,
                x_center=x_center,
                x_min=expanded_x_min,
                x_max=expanded_x_max,
                dominant_class="PrintedText"  # Will be classified later
            )
            column_anchors.append(anchor)
            log.debug(f"Column {idx}: header='{header_text}', x_range=[{expanded_x_min:.4f}, {expanded_x_max:.4f}]")
        
        log.info(f"Built {len(column_anchors)} column anchors from header row")
        return column_anchors
    
    def classify_column_dominant_class(self,
                                      column_anchor: ColumnAnchor,
                                      all_detections: List[Dict[str, Any]],
                                      row_groups: List[Dict[str, Any]]) -> str:
        """Classify dominant class for a column by scanning detections below header.
        
        Rules (in priority order):
        1. If column contains checkboxes → "Checkbox"
        2. Else if signatures dominate → "Signature"
        3. Else if handwritten present → "HandwrittenText"
        4. Else → "PrintedText"
        
        Args:
            column_anchor: Column anchor to classify
            all_detections: All detections to scan
            row_groups: Row groups for row filtering
            
        Returns:
            Dominant class name
        """
        # Find detections within this column's X-range
        column_detections = []
        for det in all_detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                if column_anchor.x_min <= x_center <= column_anchor.x_max:
                    # Check if below header row
                    row_index = self._get_row_index_for_detection(det, row_groups)
                    if row_index > self.header_row_index:
                        column_detections.append((det, row_index))
        
        if not column_detections:
            return "PrintedText"  # Default
        
        # Count classes
        class_counts = {
            'checkbox': 0,
            'signature': 0,
            'handwritten': 0,
            'text_block': 0,
            'text_box': 0
        }
        
        for det, row_index in column_detections:
            class_name = det.get('class', '').lower()
            if class_name in class_counts:
                class_counts[class_name] += 1
        
        # Apply classification rules
        if class_counts['checkbox'] > 0:
            return "Checkbox"
        elif class_counts['signature'] > class_counts['text_block'] + class_counts['text_box']:
            return "Signature"
        elif class_counts['handwritten'] > 0:
            return "HandwrittenText"
        else:
            return "PrintedText"
    
    def _get_row_index_for_detection(self,
                                    detection: Dict[str, Any],
                                    row_groups: List[Dict[str, Any]]) -> int:
        """Get row index for a detection.
        
        Args:
            detection: Detection dict
            row_groups: List of row groups
            
        Returns:
            Row index (1-based) or 0 if not found
        """
        bbox = detection.get('bbox', [])
        if len(bbox) < 4:
            return 0
        
        for row_group in row_groups:
            row_detections = row_group.get('detections', [])
            for row_det in row_detections:
                row_bbox = row_det.get('bbox', [])
                if len(row_bbox) >= 4:
                    if self._bboxes_match(bbox, row_bbox, tolerance=0.01):
                        return row_group.get('row_index', 0)
        return 0
    
    def _bboxes_match(self, bbox1: List[float], bbox2: List[float], tolerance: float = 0.01) -> bool:
        """Check if two bboxes match within tolerance."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        x1_diff = abs(bbox1[0] - bbox2[0])
        y1_diff = abs(bbox1[1] - bbox2[1])
        x2_diff = abs(bbox1[2] - bbox2[2])
        y2_diff = abs(bbox1[3] - bbox2[3])
        
        return (x1_diff < tolerance and y1_diff < tolerance and
                x2_diff < tolerance and y2_diff < tolerance)

