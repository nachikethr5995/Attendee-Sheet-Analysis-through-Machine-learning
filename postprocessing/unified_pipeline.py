"""Unified pipeline - orchestrates YOLOv8s → Class-based OCR → Dual Row/Column grouping → Structured output."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import load_image_from_canonical_id, resolve_text_handwritten_conflicts
from layout.layout_service import LayoutService
from ocr.class_based_router import ClassBasedOCRRouter
from postprocessing.signature_handler import SignatureHandler
from postprocessing.checkbox_handler import CheckboxHandler
from postprocessing.row_grouper import RowGrouper
from postprocessing.rowwise_formatter import RowwiseFormatter
from postprocessing.column_grouper import ColumnGrouper
from postprocessing.columnwise_formatter import ColumnwiseFormatter


class UnifiedPipeline:
    """Unified pipeline orchestrating the complete analysis flow.
    
    Pipeline:
    1. YOLOv8s layout detection (on original image)
    2. Class-based OCR routing (Text_box → PaddleOCR, Handwritten → PARSeq)
    3. Signature handling (presence + crop, no OCR)
    4. Checkbox handling (presence + checked/unchecked)
    5. Dual structural grouping:
       - Table-aware row grouping
       - Column grouping
    6. Dual structured output:
       - Row-wise structured output
       - Column-wise structured output
    """
    
    def __init__(self, use_gpu: Optional[bool] = None):
        """Initialize unified pipeline.
        
        Args:
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        
        log.info("Initializing Unified Pipeline...")
        
        # Initialize components
        self.layout_service = LayoutService()
        self.ocr_router = ClassBasedOCRRouter(use_gpu=self.use_gpu)
        self.signature_handler = SignatureHandler(save_crops=False)  # Can enable if needed
        self.checkbox_handler = CheckboxHandler()
        self.row_grouper = RowGrouper()  # Uses defaults from settings: ROW_Y_THRESHOLD, ROW_ATTACH_MAX_DISTANCE, TABLE_BBOX_MARGIN
        self.rowwise_formatter = RowwiseFormatter()
        self.column_grouper = ColumnGrouper(column_width_threshold=settings.COLUMN_WIDTH_THRESHOLD)
        self.columnwise_formatter = ColumnwiseFormatter()
        
        log.info("Unified Pipeline initialized (with dual row/column grouping)")
    
    def process(self,
                file_id: Optional[str] = None,
                pre_0_id: Optional[str] = None,
                pre_01_id: Optional[str] = None) -> Dict[str, Any]:
        """Process image through complete unified pipeline.
        
        Args:
            file_id: Original file identifier
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            Complete structured output with row-wise data
        """
        log.info("Starting Unified Pipeline processing...")
        
        # Step 1: Load image (use canonical ID resolution)
        try:
            image = load_image_from_canonical_id(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            log.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            log.error(f"Failed to load image: {str(e)}")
            return {
                'rows': [],
                'error': str(e),
                'failed': True
            }
        
        # Step 2: YOLOv8s layout detection
        log.info("Step 1: Running YOLOv8s layout detection...")
        try:
            layout_result = self.layout_service.detect_layout(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            
            if layout_result.get('failed', False):
                log.error(f"YOLOv8s detection failed: {layout_result.get('failure_reason')}")
                return {
                    'rows': [],
                    'error': layout_result.get('failure_reason', 'Layout detection failed'),
                    'failed': True
                }
        except Exception as e:
            log.error(f"YOLOv8s detection error: {str(e)}", exc_info=True)
            return {
                'rows': [],
                'error': f"Layout detection error: {str(e)}",
                'failed': True
            }
        
        # Extract detections by class
        tables = layout_result.get('tables', [])
        text_blocks = layout_result.get('text_blocks', [])
        handwritten = layout_result.get('handwritten', [])
        signatures = layout_result.get('signatures', [])
        checkboxes = layout_result.get('checkboxes', [])
        
        # ✅ FIX: Count handwritten at YOLO stage ONLY (YOLO is single source of truth)
        # This count is authoritative and will NOT be recalculated later
        yolo_handwritten_count = len(handwritten)
        log.info(f"YOLOv8s detected: {len(tables)} tables, {len(text_blocks)} text_blocks, "
                f"{yolo_handwritten_count} handwritten, {len(signatures)} signatures, {len(checkboxes)} checkboxes")
        log.info(f"✅ YOLO handwritten count (authoritative): {yolo_handwritten_count}")
        
        # Debug: Log handwritten detection details
        if handwritten:
            log.info(f"Handwritten detections details: {len(handwritten)} regions detected")
            for i, det in enumerate(handwritten[:3]):  # Log first 3
                bbox = det.get('bbox', [])
                conf = det.get('confidence', 0.0)
                log.debug(f"  Handwritten {i+1}: bbox={bbox}, conf={conf:.2f}")
        else:
            log.warning("⚠️  No handwritten detections from YOLO - check YOLO model or image")
        
        # HARD REJECTION: Text_box vs Handwritten conflict resolution
        # When overlapping (IoU ≥ threshold), lower confidence is COMPLETELY REMOVED
        # Rejected detections do NOT participate in OCR, grouping, stats, or output
        # This runs AFTER YOLO NMS, BEFORE everything else
        if settings.TEXT_HAND_CONFLICT_RESOLUTION and text_blocks and handwritten:
            log.info("Running Text_box ↔ Handwritten hard rejection (lower confidence removed)...")
            original_text_count = len(text_blocks)
            original_hand_count = len(handwritten)
            
            # CRITICAL: After this call, rejected detections are GONE
            # They will NOT appear in OCR, grouping, output, or anywhere downstream
            text_blocks, handwritten = resolve_text_handwritten_conflicts(
                text_blocks,
                handwritten,
                iou_threshold=settings.TEXT_HAND_IOU_THRESHOLD
            )
            
            rejected_count = (original_text_count - len(text_blocks)) + (original_hand_count - len(handwritten))
            if rejected_count > 0:
                log.warning(
                    f"After hard rejection: {len(text_blocks)}/{original_text_count} text_blocks, "
                    f"{len(handwritten)}/{original_hand_count} handwritten survive"
                )
            else:
                log.info("No Text_box ↔ Handwritten conflicts detected")
        
        # TIE-BREAKER: Signature vs Handwritten using column consensus
        # When Signature and Handwritten overlap with EQUAL confidence, use column context
        # This runs AFTER Text_box/Handwritten rejection, BEFORE OCR routing
        if settings.SIG_HAND_TIEBREAK_ENABLED and signatures and handwritten:
            from core.utils import assign_column_ids, resolve_signature_handwritten_ties
            
            log.info("Running Signature ↔ Handwritten tie-breaker (column consensus)...")
            
            # Step 1: Early column assignment using table structure (GEOMETRY ONLY)
            # Build column anchors from table header row (row 2 by y-position)
            column_anchors = None
            if tables:
                # Get table bbox for header row identification
                table_bbox = tables[0].get('bbox', []) if tables else None
                
                if table_bbox and len(table_bbox) >= 4:
                    # Combine all text-like detections for header row extraction
                    all_text_dets = text_blocks + handwritten + signatures
                    
                    # Filter to detections inside table
                    from core.utils import is_center_inside_bbox
                    table_dets = [
                        det for det in all_text_dets
                        if det.get('bbox') and is_center_inside_bbox(det['bbox'], table_bbox)
                    ]
                    
                    if table_dets:
                        # Sort by y-center to find rows
                        table_dets_sorted = sorted(
                            table_dets,
                            key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2.0 if d.get('bbox') else 0
                        )
                        
                        # Header row is typically row 2 (skip title row)
                        # Use y-clustering to identify header row
                        header_row_idx = settings.HEADER_ROW_INDEX - 1  # 0-based
                        
                        # Simple approach: take detections from top portion for header
                        if len(table_dets_sorted) >= 3:
                            # Get y-range of table
                            min_y = min(d['bbox'][1] for d in table_dets_sorted if d.get('bbox'))
                            max_y = max(d['bbox'][3] for d in table_dets_sorted if d.get('bbox'))
                            y_range = max_y - min_y
                            
                            # Header row is approximately in the top 10-20% of table
                            header_y_threshold = min_y + y_range * 0.15
                            header_dets = [
                                d for d in table_dets_sorted
                                if d.get('bbox') and (d['bbox'][1] + d['bbox'][3]) / 2.0 <= header_y_threshold
                            ]
                            
                            if header_dets:
                                # Extract x-centers as column anchors
                                column_anchors = sorted([
                                    (d['bbox'][0] + d['bbox'][2]) / 2.0
                                    for d in header_dets if d.get('bbox')
                                ])
                                log.info(f"Extracted {len(column_anchors)} column anchors from header row")
            
            # Step 2: Assign column_id to signatures and handwritten
            if column_anchors:
                # Add column_id to each detection
                for det in signatures:
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        det_x = (bbox[0] + bbox[2]) / 2.0
                        # Find nearest column
                        min_dist = float('inf')
                        nearest_col = 0
                        for idx, anchor in enumerate(column_anchors):
                            dist = abs(det_x - anchor)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_col = idx
                        det['column_id'] = nearest_col
                
                for det in handwritten:
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        det_x = (bbox[0] + bbox[2]) / 2.0
                        min_dist = float('inf')
                        nearest_col = 0
                        for idx, anchor in enumerate(column_anchors):
                            dist = abs(det_x - anchor)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_col = idx
                        det['column_id'] = nearest_col
                
                for det in text_blocks:
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        det_x = (bbox[0] + bbox[2]) / 2.0
                        min_dist = float('inf')
                        nearest_col = 0
                        for idx, anchor in enumerate(column_anchors):
                            dist = abs(det_x - anchor)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_col = idx
                        det['column_id'] = nearest_col
            
            # Step 3: Run tie-breaker with column consensus
            all_dets_for_stats = text_blocks + handwritten + signatures
            original_sig_count = len(signatures)
            original_hand_count = len(handwritten)
            
            signatures, handwritten = resolve_signature_handwritten_ties(
                signatures,
                handwritten,
                all_dets_for_stats,
                iou_threshold=settings.SIG_HAND_IOU_THRESHOLD,
                conf_epsilon=settings.SIG_HAND_CONF_EPSILON,
                column_conf_threshold=settings.SIG_HAND_COLUMN_CONF_THRESHOLD
            )
            
            reclassified_count = original_sig_count - len(signatures)
            if reclassified_count > 0:
                log.info(
                    f"After tie-breaker: {len(signatures)}/{original_sig_count} signatures, "
                    f"{len(handwritten)}/{original_hand_count + reclassified_count} handwritten"
                )
            else:
                log.info("No Signature ↔ Handwritten ties requiring column consensus")
        
        # Step 3: Class-based OCR routing (YOLO-only authority)
        # Architecture Rule: YOLO defines structure; OCR only fills content within YOLO regions
        log.info("Step 2: Running class-based OCR routing (YOLO-only regions)...")
        ocr_results = []
        
        # CRITICAL: All detections MUST come from YOLOv8s - no other source allowed
        # Route Text_box to PaddleOCR (strict routing, no fallbacks)
        # Filter: Only process text_boxes whose center point lies inside table bbox
        # This matches row/column assignment logic (x_center/y_center) for consistency
        if text_blocks:
            # Filter text_boxes to table-only using center-point containment
            filtered_text_blocks = text_blocks
            if tables:
                from core.utils import is_center_inside_bbox
                table_bboxes = [table.get('bbox', []) for table in tables if table.get('bbox')]
                
                filtered_text_blocks = []
                for det in text_blocks:
                    det_bbox = det.get('bbox', [])
                    if not det_bbox or len(det_bbox) < 4:
                        continue
                    
                    # Check if text_box center point is inside any table
                    is_inside = any(
                        is_center_inside_bbox(det_bbox, table_bbox)
                        for table_bbox in table_bboxes
                    )
                    
                    if is_inside:
                        filtered_text_blocks.append(det)
                
                log.info(
                    f"PaddleOCR routing: {len(filtered_text_blocks)}/{len(text_blocks)} "
                    f"text_boxes passed (center-in-table)"
                )
            else:
                # No tables detected - skip PaddleOCR entirely (strict policy)
                log.warning(
                    f"No tables detected. "
                    f"Skipping PaddleOCR for all {len(text_blocks)} text_boxes."
                )
                filtered_text_blocks = []
            
            if filtered_text_blocks:
                log.info(f"Routing {len(filtered_text_blocks)} Text_box detections (YOLO) to PaddleOCR...")
                # Ensure all detections have YOLO source and class
                for det in filtered_text_blocks:
                    det['source'] = 'YOLOv8s'  # Enforce YOLO source
                    det['class'] = 'text_block'  # Ensure class is set
                text_ocr = self.ocr_router.process_detections(image, filtered_text_blocks)
                ocr_results.extend(text_ocr)
            else:
                log.info("No text_boxes inside table — skipping PaddleOCR")
        
        # Route Handwritten to PARSeq (strict routing, no fallbacks)
        # Phase-1: Each YOLO handwritten box = exactly one OCR call (NO MERGING)
        if handwritten:
            log.info(f"Routing {len(handwritten)} Handwritten detections (YOLO) to PARSeq...")
            # Fail-fast guard: YOLO authority - all detections must be from YOLO
            for det in handwritten:
                det_source = det.get('source', 'YOLOv8s')
                assert det_source == 'YOLOv8s', f"Non-YOLO handwritten detection (source: {det_source}) - pipeline violation"
                # Fail-fast guard: One box → one OCR call
                det_bbox = det.get('bbox', [])
                assert isinstance(det_bbox, list) and len(det_bbox) >= 4, f"Invalid bbox in handwritten detection: {det_bbox}"
                det['source'] = 'YOLOv8s'  # Enforce YOLO source
                det['class'] = 'handwritten'  # Ensure class is set
            
            # Check if PARSeq is available
            if not hasattr(self.ocr_router, 'parseq_recognizer') or not self.ocr_router.parseq_recognizer or not self.ocr_router.parseq_recognizer.is_available():
                log.error("⚠️  PARSeq recognizer not available - handwritten text will not be recognized")
                log.error("   Install PARSeq dependencies: pip install transformers")
                log.error("   Or check PARSeq initialization logs above")
            else:
                log.info("✅ PARSeq recognizer is available")
            
            handwritten_ocr = self.ocr_router.process_detections(image, handwritten)
            log.info(f"PARSeq returned {len(handwritten_ocr)} OCR results from {len(handwritten)} detections")
            
            # Fail-fast guard: Handwritten count sanity
            # Each YOLO handwritten box should produce exactly one OCR result
            if len(handwritten_ocr) != len(handwritten):
                error_msg = (
                    f"Handwritten detections lost - pipeline violation: "
                    f"{len(handwritten)} detections → {len(handwritten_ocr)} OCR results"
                )
                log.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Debug: Log OCR results
            for i, ocr_result in enumerate(handwritten_ocr[:3]):  # Log first 3
                text = ocr_result.get('text', '')
                conf = ocr_result.get('confidence', 0.0)
                source = ocr_result.get('source', '')
                log.debug(f"  PARSeq result {i+1}: text='{text}', conf={conf:.2f}, source={source}")
            
            ocr_results.extend(handwritten_ocr)
        else:
            log.info("No handwritten detections to route to PARSeq")
        
        # Safety assertion: Verify all OCR results have correct source
        for ocr_result in ocr_results:
            ocr_class = ocr_result.get('class', '').lower()
            ocr_source = ocr_result.get('source', '').lower()
            
            if ocr_class == 'text_block' and ocr_source != 'paddleocr':
                log.warning(f"⚠️  Text_block OCR result has wrong source: {ocr_source} (expected: paddleocr)")
            elif ocr_class == 'handwritten' and ocr_source != 'parseq':
                log.warning(f"⚠️  Handwritten OCR result has wrong source: {ocr_source} (expected: parseq)")
        
        # Step 4: Signature handling (presence + crop, no OCR)
        log.info("Step 3: Processing signatures...")
        signature_results = self.signature_handler.process_signatures(
            image, signatures, canonical_id=file_id or pre_0_id or pre_01_id
        )
        
        # Step 5: Checkbox handling (presence + checked/unchecked)
        log.info("Step 4: Processing checkboxes...")
        checkbox_results = self.checkbox_handler.process_checkboxes(image, checkboxes)
        
        # Step 5: Table-anchored row construction
        # This runs AFTER conflict resolution, BEFORE OCR routing
        # Purpose: Build rows anchored to tables, attach out-of-table detections
        log.info("Step 5: Table-anchored row construction...")
        
        # Combine all detections for row grouping
        # CRITICAL: Use 'cls' field (not 'class') for row grouper compatibility
        all_detections = []
        
        # Add tables (needed for row anchoring)
        for table in tables:
            all_detections.append({
                **table,
                'cls': 'Table',
                'class': 'table',  # Keep for backward compatibility
                'source': 'YOLOv8s'
            })
        
        # Add content detections (Text_box, Handwritten, Signature, Checkbox)
        for det in text_blocks:
            all_detections.append({
                **det,
                'cls': 'Text_box',
                'class': 'text_block',  # Keep for backward compatibility
                'source': 'YOLOv8s'
            })
        
        for det in handwritten:
            all_detections.append({
                **det,
                'cls': 'Handwritten',
                'class': 'handwritten',  # Keep for backward compatibility
                'source': 'YOLOv8s'
            })
        
        for det in signatures:
            all_detections.append({
                **det,
                'cls': 'Signature',
                'class': 'signature',  # Keep for backward compatibility
                'source': 'YOLOv8s'
            })
        
        for det in checkboxes:
            all_detections.append({
                **det,
                'cls': 'Checkbox',
                'class': 'checkbox',  # Keep for backward compatibility
                'source': 'YOLOv8s'
            })
        
        # Get image dimensions for coordinate conversion
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]
        
        # Table-anchored row construction with out-of-table attachment
        rows = self.row_grouper.group_into_rows(
            all_detections,
            table_bboxes=tables,
            image_width=img_w,
            image_height=img_h
        )
        rows = self.row_grouper.assign_row_indices(rows)
        
        # Step 8: Formatting (Row-wise is source of truth)
        # Step 8a: Format into row-wise structured output (SOURCE OF TRUTH)
        log.info("Step 6a: Formatting row-wise structured output (SOURCE OF TRUTH)...")
        rowwise_output = self.rowwise_formatter.format_rows(
            rows, ocr_results, signature_results, checkbox_results
        )
        
        # STEP 10: Validate header row exists and has printed text (fail fast)
        rows_list = rowwise_output.get('rows', [])
        header_row = None
        for row in rows_list:
            if row.get('row_index') == settings.HEADER_ROW_INDEX:
                header_row = row
                break
        
        if not header_row:
            error_msg = f"STEP 5 violation: Header row (row_index={settings.HEADER_ROW_INDEX}) not found"
            log.error(error_msg)
            raise ValueError(error_msg)
        
        header_internal = header_row.get('_internal', {})
        header_printed = header_internal.get('PrintedText', [])
        if not header_printed:
            error_msg = f"STEP 5 violation: Header row {settings.HEADER_ROW_INDEX} has no PrintedText cells"
            log.error(error_msg)
            raise ValueError(error_msg)
        
        log.info(f"✓ STEP 5: Header row {settings.HEADER_ROW_INDEX} validated ({len(header_printed)} header cells)")
        
        # STEP 10: Validate no Text_box & Handwritten overlap remains
        overlap_detected = False
        for row in rows_list:
            row_internal = row.get('_internal', {})
            printed = row_internal.get('PrintedText', [])
            handwritten = row_internal.get('HandwrittenText', [])
            
            if printed and handwritten:
                printed_x = {item.get('x_center') for item in printed if isinstance(item, dict) and 'x_center' in item}
                handwritten_x = {item.get('x_center') for item in handwritten if isinstance(item, dict) and 'x_center' in item}
                overlap = printed_x & handwritten_x
                if overlap:
                    log.warning(f"⚠️  STEP 1 violation: Text_box & Handwritten overlap at x_centers: {overlap}")
                    overlap_detected = True
        
        if not overlap_detected:
            log.info("✓ STEP 1: No Text_box & Handwritten overlap detected")
        
        # Step 8b: Format into column-wise structured output (pure pivot of row-wise)
        log.info("Step 6b: Building column-wise structured output (pure pivot of row-wise)...")
        columnwise_output = self.columnwise_formatter.format_columns(rowwise_output)
        
        log.info("Unified Pipeline processing complete")
        
        # ✅ FIX: Use YOLO-authoritative counts (set at YOLO stage, never recalculated)
        # YOLO is the single source of truth for detection counts
        final_text_blocks = len(text_blocks)
        final_signatures = len(signatures)
        final_checkboxes = len(checkboxes)
        # handwritten_count is already set from YOLO stage (yolo_handwritten_count)
        
        # Validation guard: Compare YOLO count with rowwise count (non-blocking)
        rows_list = rowwise_output.get('rows', [])
        handwritten_from_rows = 0
        printed_from_rows = 0
        for row in rows_list:
            row_id = row.get('row_index', 0)
            if row_id == 0 or row_id == 1:  # Skip invalid and header rows
                continue
            row_internal = row.get('_internal', {})
            handwritten_from_rows += len(row_internal.get('HandwrittenText', []))
            printed_from_rows += len(row_internal.get('PrintedText', []))
        
        # Non-blocking validation: Warn if mismatch (helps debugging)
        if handwritten_from_rows != yolo_handwritten_count:
            log.warning(
                f"⚠️  Handwritten count mismatch | YOLO={yolo_handwritten_count} "
                f"| Rowwise={handwritten_from_rows} | "
                f"This is informational - YOLO count is authoritative"
            )
        else:
            log.debug(f"✓ Handwritten count validation: YOLO={yolo_handwritten_count} matches rowwise={handwritten_from_rows}")
        
        log.info(
            f"Layout counters (YOLO-authoritative): "
            f"{final_text_blocks} text_blocks, {yolo_handwritten_count} handwritten, "
            f"{final_signatures} signatures, {final_checkboxes} checkboxes"
        )
        log.debug(
            f"Row-wise counts (informational): {printed_from_rows} printed, {handwritten_from_rows} handwritten"
        )
        
        return {
            'rowwise': rowwise_output,
            'columnwise': columnwise_output,
            'layout': {
                'tables': len(tables),
                'text_blocks': final_text_blocks,
                'handwritten': yolo_handwritten_count,  # ✅ YOLO-authoritative (set at YOLO stage)
                'signatures': final_signatures,
                'checkboxes': final_checkboxes
            },
            'failed': False
        }

