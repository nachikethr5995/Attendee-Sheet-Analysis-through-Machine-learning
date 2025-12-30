"""Unified pipeline - orchestrates YOLOv8s → Class-based OCR → Row grouping → Structured output."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import Dict, Any, Optional
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import load_image_from_canonical_id
from layout.layout_service import LayoutService
from ocr.class_based_router import ClassBasedOCRRouter
from postprocessing.signature_handler import SignatureHandler
from postprocessing.checkbox_handler import CheckboxHandler
from postprocessing.row_grouper import RowGrouper
from postprocessing.rowwise_formatter import RowwiseFormatter


class UnifiedPipeline:
    """Unified pipeline orchestrating the complete analysis flow.
    
    Pipeline:
    1. YOLOv8s layout detection (on original image)
    2. Class-based OCR routing (Text_box → PaddleOCR, Handwritten → TrOCR)
    3. Signature handling (presence + crop, no OCR)
    4. Checkbox handling (presence + checked/unchecked)
    5. Table-aware row grouping
    6. Row-wise structured output
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
        self.row_grouper = RowGrouper()
        self.rowwise_formatter = RowwiseFormatter()
        
        log.info("Unified Pipeline initialized")
    
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
        
        log.info(f"YOLOv8s detected: {len(tables)} tables, {len(text_blocks)} text_blocks, "
                f"{len(handwritten)} handwritten, {len(signatures)} signatures, {len(checkboxes)} checkboxes")
        
        # Step 3: Class-based OCR routing (YOLO-only authority)
        # Architecture Rule: YOLO defines structure; OCR only fills content within YOLO regions
        log.info("Step 2: Running class-based OCR routing (YOLO-only regions)...")
        ocr_results = []
        
        # CRITICAL: All detections MUST come from YOLOv8s - no other source allowed
        # Route Text_box to PaddleOCR (strict routing, no fallbacks)
        if text_blocks:
            log.info(f"Routing {len(text_blocks)} Text_box detections (YOLO) to PaddleOCR...")
            # Ensure all detections have YOLO source and class
            for det in text_blocks:
                det['source'] = 'YOLOv8s'  # Enforce YOLO source
                det['class'] = 'text_block'  # Ensure class is set
            text_ocr = self.ocr_router.process_detections(image, text_blocks)
            ocr_results.extend(text_ocr)
        
        # Route Handwritten to TrOCR (strict routing, no fallbacks)
        if handwritten:
            log.info(f"Routing {len(handwritten)} Handwritten detections (YOLO) to TrOCR...")
            # Ensure all detections have YOLO source and class
            for det in handwritten:
                det['source'] = 'YOLOv8s'  # Enforce YOLO source
                det['class'] = 'handwritten'  # Ensure class is set
            handwritten_ocr = self.ocr_router.process_detections(image, handwritten)
            ocr_results.extend(handwritten_ocr)
        
        # Safety assertion: Verify all OCR results have correct source
        for ocr_result in ocr_results:
            ocr_class = ocr_result.get('class', '').lower()
            ocr_source = ocr_result.get('source', '').lower()
            
            if ocr_class == 'text_block' and ocr_source != 'paddleocr':
                log.warning(f"⚠️  Text_block OCR result has wrong source: {ocr_source} (expected: paddleocr)")
            elif ocr_class == 'handwritten' and ocr_source != 'trocr':
                log.warning(f"⚠️  Handwritten OCR result has wrong source: {ocr_source} (expected: trocr)")
        
        # Step 4: Signature handling (presence + crop, no OCR)
        log.info("Step 3: Processing signatures...")
        signature_results = self.signature_handler.process_signatures(
            image, signatures, canonical_id=file_id or pre_0_id or pre_01_id
        )
        
        # Step 5: Checkbox handling (presence + checked/unchecked)
        log.info("Step 4: Processing checkboxes...")
        checkbox_results = self.checkbox_handler.process_checkboxes(image, checkboxes)
        
        # Step 6: Combine all detections for row grouping
        all_detections = []
        all_detections.extend([{**det, 'class': 'text_block'} for det in text_blocks])
        all_detections.extend([{**det, 'class': 'handwritten'} for det in handwritten])
        all_detections.extend([{**det, 'class': 'signature'} for det in signatures])
        all_detections.extend([{**det, 'class': 'checkbox'} for det in checkboxes])
        
        # Step 7: Table-aware row grouping
        log.info("Step 5: Grouping detections into rows...")
        rows = self.row_grouper.group_into_rows(all_detections, table_bboxes=tables)
        rows = self.row_grouper.assign_row_indices(rows)
        
        # Step 8: Format into row-wise structured output
        log.info("Step 6: Formatting row-wise structured output...")
        formatted_output = self.rowwise_formatter.format_rows(
            rows, ocr_results, signature_results, checkbox_results
        )
        
        log.info("Unified Pipeline processing complete")
        
        return {
            **formatted_output,
            'layout': {
                'tables': len(tables),
                'text_blocks': len(text_blocks),
                'handwritten': len(handwritten),
                'signatures': len(signatures),
                'checkboxes': len(checkboxes)
            },
            'failed': False
        }

