"""Table-anchored row grouping - YOLO-anchored row construction with out-of-table attachment."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple

from core.logging import log
from core.config import settings
from core.utils import is_bbox_inside_table


# Class definitions (single source of truth)
TEXT_CLASS = "Text_box"
HAND_CLASS = "Handwritten"
SIG_CLASS = "Signature"
CHK_CLASS = "Checkbox"
TABLE_CLASS = "Table"

ROW_ATTACH_CLASSES = {TEXT_CLASS, HAND_CLASS, SIG_CLASS, CHK_CLASS}


def y_center(bbox: list) -> float:
    """Calculate Y-center of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        float: Y-center coordinate
    """
    if len(bbox) < 4:
        return 0.0
    return (bbox[1] + bbox[3]) / 2.0


class RowGrouper:
    """Table-anchored row grouping with out-of-table attachment.
    
    ARCHITECTURE:
    - YOLOv8s is the ONLY detector
    - Table is used as row anchor, not hard boundary
    - Out-of-table detections are attached to nearest row
    - No OCR feedback, no pixel heuristics, no class guessing
    """
    
    def __init__(self, row_y_threshold: Optional[int] = None, 
                 row_attach_max_distance: Optional[int] = None,
                 table_bbox_margin: Optional[int] = None):
        """Initialize row grouper.
        
        Args:
            row_y_threshold: Pixel threshold for row clustering (default: 15)
            row_attach_max_distance: Max pixel distance for out-of-table attachment (default: 30)
            table_bbox_margin: Pixel margin for table containment check (default: 5)
        """
        self.row_y_threshold = row_y_threshold if row_y_threshold is not None else settings.ROW_Y_THRESHOLD
        self.row_attach_max_distance = row_attach_max_distance if row_attach_max_distance is not None else settings.ROW_ATTACH_MAX_DISTANCE
        self.table_bbox_margin = table_bbox_margin if table_bbox_margin is not None else settings.TABLE_BBOX_MARGIN
        
        log.info(
            f"Row grouper initialized: "
            f"row_y_threshold={self.row_y_threshold}px, "
            f"attach_max_distance={self.row_attach_max_distance}px, "
            f"table_margin={self.table_bbox_margin}px"
        )
    
    def group_into_rows(self,
                       detections: List[Dict[str, Any]],
                       table_bboxes: Optional[List[Dict[str, Any]]] = None,
                       image_width: Optional[int] = None,
                       image_height: Optional[int] = None) -> List[Dict[str, Any]]:
        """Group detections into rows using table-anchored construction.
        
        PIPELINE:
        1. Split detections: tables vs content (Text/Hand/Sig/Checkbox)
        2. For each table:
           a. Identify in-table vs out-of-table content
           b. Build rows from in-table content
           c. Attach out-of-table content to nearest row
        3. Assign row_id to all detections
        
        Args:
            detections: List of detections with 'bbox' and 'cls'
            table_bboxes: Optional list of table bboxes (if None, extracts from detections)
            image_width: Image width in pixels (for coordinate conversion)
            image_height: Image height in pixels (for coordinate conversion)
            
        Returns:
            List of rows, each containing ordered detections with row_id
        """
        if not detections:
            return []
        
        # Step 1: Split detections
        tables = []
        content = []
        
        for det in detections:
            cls = det.get('cls', det.get('class', ''))
            if cls == TABLE_CLASS:
                tables.append(det)
            elif cls in ROW_ATTACH_CLASSES:
                content.append(det)
        
        log.info(f"Split detections: {len(tables)} tables, {len(content)} content detections")
        
        if not tables:
            # No tables - fallback to simple Y-center clustering
            log.warning("No tables detected - using simple Y-center clustering")
            return self._simple_row_clustering(content)
        
        # Step 2: Process each table independently
        all_rows = []
        processed_content = set()
        
        for table_idx, table in enumerate(tables):
            table_bbox = table.get('bbox', [])
            if len(table_bbox) < 4:
                continue
            
            log.info(f"Processing table {table_idx + 1}/{len(tables)}")
            
            # Step 2a: Identify in-table vs out-of-table content
            in_table = []
            out_table = []
            
            for det in content:
                if id(det) in processed_content:
                    continue
                
                det_bbox = det.get('bbox', [])
                if len(det_bbox) < 4:
                    continue
                
                # Check if inside table (with margin)
                is_inside = is_bbox_inside_table(
                    det_bbox, table_bbox,
                    margin=self.table_bbox_margin,
                    img_w=image_width,
                    img_h=image_height
                )
                
                if is_inside:
                    in_table.append(det)
                else:
                    out_table.append(det)
            
            log.info(f"  Table {table_idx + 1}: {len(in_table)} in-table, {len(out_table)} out-of-table")
            
            # Step 2b: Build rows from in-table content
            if in_table:
                rows = self._build_rows_from_content(in_table, img_h=image_height)
                
                # Step 2c: Attach out-of-table content to nearest row
                if out_table and rows:
                    rows = self._attach_out_of_table(out_table, rows, img_h=image_height)
                
                all_rows.extend(rows)
                # Mark content as processed
                for det in in_table + out_table:
                    processed_content.add(id(det))
        
        # Step 3: Handle any remaining unprocessed content (shouldn't happen, but safety)
        remaining = [det for det in content if id(det) not in processed_content]
        if remaining:
            log.warning(f"{len(remaining)} detections not assigned to any table - using simple clustering")
            remaining_rows = self._simple_row_clustering(remaining, img_h=image_height)
            all_rows.extend(remaining_rows)
        
        # Step 4: Assign row_id to all detections
        for row_idx, row in enumerate(all_rows, start=1):
            for det in row.get('detections', []):
                det['row_id'] = row_idx
            row['row_index'] = row_idx
        
        log.info(f"Grouped into {len(all_rows)} total rows")
        return all_rows
    
    def _build_rows_from_content(self, content: List[Dict[str, Any]], 
                                  img_h: Optional[int] = None) -> List[Dict[str, Any]]:
        """Build rows from content detections using Y-center clustering.
        
        Args:
            content: List of detections inside a table
            img_h: Image height in pixels (for normalized coordinate conversion)
            
        Returns:
            List of row dicts with detections sorted by X-center
        """
        if not content:
            return []
        
        # Convert pixel threshold to normalized if needed
        if img_h:
            # Threshold is in pixels, convert to normalized
            y_threshold_norm = self.row_y_threshold / img_h
        else:
            # Assume coordinates are already normalized, use normalized threshold
            # Default: 15px / 2000px = 0.0075 (conservative estimate)
            y_threshold_norm = 0.0075
        
        # Sort by Y-center (top to bottom)
        content_with_y = []
        for det in content:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                content_with_y.append((y_center(bbox), det))
        
        content_with_y.sort(key=lambda x: x[0])
        
        # Cluster into rows
        rows = []
        current_row = []
        current_y_center = None
        
        for y_c, det in content_with_y:
            if current_y_center is None:
                # First detection - start new row
                current_row = [det]
                current_y_center = y_c
            else:
                # Check if belongs to current row
                y_diff = abs(y_c - current_y_center)
                
                if y_diff <= y_threshold_norm:
                    # Same row
                    current_row.append(det)
                    # Update row center (weighted average)
                    current_y_center = (current_y_center * (len(current_row) - 1) + y_c) / len(current_row)
                else:
                    # New row - finalize current
                    if current_row:
                        rows.append(self._finalize_row(current_row))
                    # Start new row
                    current_row = [det]
                    current_y_center = y_c
        
        # Finalize last row
        if current_row:
            rows.append(self._finalize_row(current_row))
        
        return rows
    
    def _attach_out_of_table(self, 
                             out_table: List[Dict[str, Any]],
                             rows: List[Dict[str, Any]],
                             img_h: Optional[int] = None) -> List[Dict[str, Any]]:
        """Attach out-of-table detections to nearest row.
        
        Purpose: Recover cells clipped by tight table detection.
        
        Args:
            out_table: List of detections outside table
            rows: List of existing rows
            img_h: Image height in pixels (for normalized coordinate conversion)
            
        Returns:
            Updated rows with attached detections
        """
        if not out_table or not rows:
            return rows
        
        # Convert pixel threshold to normalized if needed
        if img_h:
            attach_threshold_norm = self.row_attach_max_distance / img_h
        else:
            # Assume normalized, use conservative estimate
            attach_threshold_norm = 0.015  # ~30px / 2000px
        
        attached_count = 0
        
        for det in out_table:
            det_bbox = det.get('bbox', [])
            if len(det_bbox) < 4:
                continue
            
            det_y = y_center(det_bbox)
            
            # Find nearest row
            best_row = None
            best_dist = float('inf')
            
            for row in rows:
                # Calculate row Y-center
                row_dets = row.get('detections', [])
                if not row_dets:
                    continue
                
                row_y_centers = [y_center(d.get('bbox', [])) for d in row_dets if d.get('bbox')]
                if not row_y_centers:
                    continue
                
                row_y_center = sum(row_y_centers) / len(row_y_centers)
                dist = abs(det_y - row_y_center)
                
                if dist < best_dist:
                    best_dist = dist
                    best_row = row
            
            # Attach if within threshold (normalized)
            if best_row and best_dist <= attach_threshold_norm:
                best_row['detections'].append(det)
                attached_count += 1
                dist_px = best_dist * img_h if img_h else best_dist
                log.debug(f"Attached out-of-table detection to row (distance: {dist_px:.1f}px)")
        
        if attached_count > 0:
            log.info(f"Attached {attached_count}/{len(out_table)} out-of-table detections to rows")
        
        # Re-sort rows after attachment
        for row in rows:
            row['detections'] = self._sort_detections_by_x(row['detections'])
        
        return rows
    
    def _finalize_row(self, row_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a row by sorting detections by X-center (left to right).
        
        Args:
            row_detections: Detections in the same row
            
        Returns:
            Row dict with sorted detections
        """
        sorted_detections = self._sort_detections_by_x(row_detections)
        
        # Calculate row Y-center
        y_centers = [y_center(d.get('bbox', [])) for d in sorted_detections if d.get('bbox')]
        row_y_center = sum(y_centers) / len(y_centers) if y_centers else 0.0
        
        return {
            'row_index': None,  # Will be set by caller
            'y_center': row_y_center,
            'detections': sorted_detections,
            'column_count': len(sorted_detections)
        }
    
    def _sort_detections_by_x(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort detections by X-center (left to right).
        
        Args:
            detections: List of detections
            
        Returns:
            Sorted list of detections
        """
        detections_with_x = []
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2.0
                detections_with_x.append((x_center, det))
        
        detections_with_x.sort(key=lambda x: x[0])
        return [det for _, det in detections_with_x]
    
    def _simple_row_clustering(self, content: List[Dict[str, Any]], 
                               img_h: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback: Simple Y-center clustering when no tables are present.
        
        Args:
            content: List of detections
            img_h: Image height in pixels (for normalized coordinate conversion)
            
        Returns:
            List of row dicts
        """
        return self._build_rows_from_content(content, img_h=img_h)
    
    def assign_row_indices(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign row indices (1-based) to rows.
        
        Note: This is now redundant as row_id is assigned in group_into_rows,
        but kept for backward compatibility.
        
        Args:
            rows: List of row dicts
            
        Returns:
            Rows with row_index assigned
        """
        for i, row in enumerate(rows, start=1):
            if row.get('row_index') is None:
                row['row_index'] = i
            # Ensure all detections have row_id
            for det in row.get('detections', []):
                if det.get('row_id') is None:
                    det['row_id'] = i
        return rows
