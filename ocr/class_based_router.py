"""Class-based OCR router - routes Text_box to PaddleOCR, Handwritten to TrOCR.

STRICT RULE:
- Text_box → PaddleOCR ONLY
- Handwritten → TrOCR ONLY
- No cross-routing allowed

OCR INPUT QUALITY IMPROVEMENTS (3.1 & 3.2):
- 3.1: YOLO-safe bbox expansion (geometry only, YOLO remains authority)
- 3.2: Resolution normalization (upscale only, aspect preserved)
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import re
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import expand_yolo_bbox, normalize_resolution, image_to_array
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.handwritten.parseq_recognizer import PARSeqRecognizer


class ClassBasedOCRRouter:
    """Routes OCR based on YOLOv8s class labels.
    
    Architecture:
    - Text_box detections → PaddleOCR (printed text)
    - Handwritten detections → PARSeq (handwriting)
    - No fallback between engines (strict routing)
    """
    
    def __init__(self, use_gpu: Optional[bool] = None):
        """Initialize class-based OCR router.
        
        Args:
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        
        log.info("Initializing Class-Based OCR Router...")
        
        # Initialize PaddleOCR (for Text_box only)
        try:
            self.paddle_recognizer = PaddleOCRRecognizer(lang='en', use_gpu=self.use_gpu)
        except Exception as e:
            log.error(f"❌ Failed to initialize PaddleOCR recognizer: {str(e)}")
            log.error("   PaddleOCR is required for Text_box recognition")
            log.error("   Install with: pip install paddlepaddle paddleocr")
            log.error("   Or try: pip install --upgrade paddlepaddle paddleocr")
            raise RuntimeError(f"PaddleOCR initialization failed: {str(e)}") from e
        
        # Initialize PARSeq (for Handwritten only) - uses official PARSeq codebase
        # NOTE: Requires PARSeq repository cloned to ocr/handwritten/parseq/
        self.parseq_recognizer = None
        try:
            self.parseq_recognizer = PARSeqRecognizer(
                checkpoint_path=None,  # Uses settings.PARSEQ_CHECKPOINT_PATH
                use_gpu=self.use_gpu,
                min_short_side=96,  # PARSeq requires higher resolution
                pad_ratio=settings.OCR_HANDWRITTEN_PAD_RATIO  # Phase-1: 0.18 (increased from 0.15)
            )
            if not self.parseq_recognizer.is_available():
                log.warning("⚠️  PARSeq initialized but not available")
                log.warning("   Check: 1) PARSeq codebase cloned, 2) weights downloaded")
                self.parseq_recognizer = None
        except Exception as e:
            log.warning(f"⚠️  PARSeq initialization failed: {str(e)}")
            log.warning("   Handwritten text recognition will be unavailable")
            log.warning("   Setup instructions: See Back_end/PARSEQ_SETUP.md")
            log.warning("   1. Clone: https://github.com/baudm/parseq")
            log.warning("   2. Place in: Back_end/ocr/handwritten/parseq/")
            log.warning("   3. Download weights to: Back_end/ocr/handwritten/parseq/weights/")
            self.parseq_recognizer = None
        
        log.info("Class-Based OCR Router initialized")
        
        # Verify PaddleOCR initialization
        if self.paddle_recognizer is None:
            log.error("❌ PaddleOCR recognizer is None - initialization failed")
            raise RuntimeError("PaddleOCR recognizer must be initialized")
        elif self.paddle_recognizer.ocr is None:
            log.error("❌ PaddleOCR engine is None - initialization failed")
            log.error("   Check PaddleOCR installation: pip install paddlepaddle paddleocr")
            log.error("   Or try: pip install --upgrade paddlepaddle paddleocr")
            raise RuntimeError("PaddleOCR engine must be initialized")
        else:
            log.info(f"✅ PaddleOCR (Text_box): initialized and ready")
        
        # Verify PARSeq initialization
        if self.parseq_recognizer is None:
            log.warning("⚠️  PARSeq recognizer initialization failed - handwritten text will be unavailable")
        elif not self.parseq_recognizer.is_available():
            log.warning("⚠️  PARSeq engine is not available - handwritten text will be unavailable")
        else:
            log.info(f"✅ PARSeq (Handwritten): initialized and ready")
    
    def process_detections(self,
                          image: Image.Image,
                          detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process detections with class-based routing.
        
        ARCHITECTURAL RULE: YOLOv8s is the ONLY authority for regions.
        This function receives ONLY YOLO detections and routes them to appropriate OCR engines.
        
        Args:
            image: Full image (PIL Image) - used only for cropping YOLO regions
            detections: List of detections from YOLOv8s with 'class', 'bbox', 'confidence'
                       MUST come from YOLOv8s - no other source allowed
            
        Returns:
            List of OCR results with 'text', 'confidence', 'source', 'bbox', 'class'
        """
        # Safety assertion: All detections must have YOLO source
        for det in detections:
            source = det.get('source', '')
            if source and source != 'YOLOv8s':
                log.warning(f"⚠️  Non-YOLO detection detected (source: {source}) - this violates architectural rule")
        
        width, height = image.size
        ocr_results = []
        
        for i, detection in enumerate(detections):
            # Fail-fast guard: YOLO authority
            det_source = detection.get('source', 'YOLOv8s')
            assert det_source == 'YOLOv8s', f"Non-YOLO detection in OCR router (source: {det_source}) - pipeline violation"
            
            class_name = detection.get('class', '').lower()
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            
            if not bbox or len(bbox) < 4:
                log.warning(f"Detection {i} has invalid bbox, skipping")
                continue
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            # Ensure valid crop coordinates (before expansion)
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # 3.1: YOLO-SAFE BBOX EXPANSION (OCR input quality fix)
            # Rule: YOLO remains authority - expansion is geometry-only, no new detection
            if settings.OCR_ENABLE_BBOX_EXPANSION:
                if class_name == 'text_block' or class_name == 'text_box':
                    # Text_box: 1.15× expansion
                    x1, y1, x2, y2 = expand_yolo_bbox(
                        (x1, y1, x2, y2), width, height,
                        scale=settings.OCR_TEXT_BOX_BBOX_SCALE
                    )
                    log.debug(f"Text_box bbox expanded by {settings.OCR_TEXT_BOX_BBOX_SCALE}×")
                elif class_name == 'handwritten':
                    # Handwritten: 1.25× expansion (needs more breathing room)
                    x1, y1, x2, y2 = expand_yolo_bbox(
                        (x1, y1, x2, y2), width, height,
                        scale=settings.OCR_HANDWRITTEN_BBOX_SCALE
                    )
                    log.debug(f"Handwritten bbox expanded by {settings.OCR_HANDWRITTEN_BBOX_SCALE}×")
            
            # Crop region
            try:
                region_image = image.crop((x1, y1, x2, y2))
                # Fail-fast guard: OCR never sees full image
                # Verify crop is smaller than full image
                crop_w, crop_h = region_image.size
                full_w, full_h = image.size
                assert crop_w < full_w or crop_h < full_h, (
                    f"OCR crop size ({crop_w}x{crop_h}) >= full image ({full_w}x{full_h}) - pipeline violation"
                )
            except Exception as e:
                log.warning(f"Failed to crop detection {i}: {str(e)}")
                continue
            
            # 3.2: RESOLUTION NORMALIZATION (minimum OCR viability)
            # Rule: Upscale only, preserve aspect ratio, never downsample
            if settings.OCR_ENABLE_RESOLUTION_NORMALIZATION:
                if class_name == 'text_block' or class_name == 'text_box':
                    # PaddleOCR: min 48px short side
                    region_array = image_to_array(region_image)
                    region_array = normalize_resolution(region_array, settings.OCR_TEXT_BOX_MIN_SHORT_SIDE)
                    region_image = Image.fromarray(region_array)
                    log.debug(f"Text_box resolution normalized (min {settings.OCR_TEXT_BOX_MIN_SHORT_SIDE}px)")
                elif class_name == 'handwritten':
                    # TrOCR: min 64px short side (higher res needed)
                    region_array = image_to_array(region_image)
                    region_array = normalize_resolution(region_array, settings.OCR_HANDWRITTEN_MIN_SHORT_SIDE)
                    region_image = Image.fromarray(region_array)
                    log.debug(f"Handwritten resolution normalized (min {settings.OCR_HANDWRITTEN_MIN_SHORT_SIDE}px)")
            
            # STRICT CLASS-BASED ROUTING (No fallbacks, no heuristics, YOLO authority only)
            if class_name == 'text_block' or class_name == 'text_box':
                # SAFETY ASSERTION: Only YOLO-derived bboxes allowed for expansion
                det_source = detection.get('source', 'YOLOv8s')
                assert det_source == 'YOLOv8s', f"Non-YOLO bbox expansion blocked (source: {det_source})"
                
                # Text_box → PaddleOCR ONLY (no exceptions, no fallbacks)
                # Hard assertion: PaddleOCR must be initialized
                assert self.paddle_recognizer is not None, "PaddleOCR recognizer must be initialized before OCR routing"
                assert self.paddle_recognizer.ocr is not None, "PaddleOCR engine must be initialized (check initialization logs)"
                
                log.debug(f"Routing Text_box detection {i+1} to PaddleOCR (YOLO authority)...")
                
                # Run PaddleOCR directly on YOLO Text_box crop
                # Architecture Rule: PaddleOCR must NEVER run on full images, ONLY on YOLO crops
                try:
                    # Use the recognizer's recognize method (handles all PaddleOCR API variations)
                    text, conf = self.paddle_recognizer.recognize(region_image)
                    
                    # Apply confidence threshold (0.6 as per requirements, but keep bbox)
                    if float(conf) >= 0.6 and text.strip():
                        normalized_text = self._normalize_text(text, is_handwritten=False)
                        if normalized_text:
                            ocr_results.append({
                                'bbox': bbox,
                                'text': normalized_text,
                                'confidence': float(conf),
                                'source': 'paddleocr',
                                'class': 'text_block',
                                'detection_confidence': confidence
                            })
                            log.info(f"PaddleOCR text extracted: '{normalized_text}' (conf={conf:.2f})")
                        else:
                            log.debug(f"PaddleOCR text empty after normalization (conf={conf:.2f})")
                    else:
                        log.debug(f"PaddleOCR text dropped: confidence {conf:.2f} < 0.6 or empty text")
                        ocr_results.append({
                            'bbox': bbox,
                            'text': '',
                            'confidence': 0.0,
                            'source': 'paddleocr',
                            'class': 'text_block',
                            'detection_confidence': confidence
                        })
                except Exception as e:
                    log.error(f"PaddleOCR recognition failed for Text_box detection {i+1}: {str(e)}", exc_info=True)
                    ocr_results.append({
                        'bbox': bbox,
                        'text': '',
                        'confidence': 0.0,
                        'source': 'paddleocr',
                        'class': 'text_block',
                        'error': f'PaddleOCR recognition failed: {str(e)}'
                    })
                
            elif class_name == 'handwritten':
                # SAFETY ASSERTION: Only YOLO-derived bboxes allowed for expansion
                det_source = detection.get('source', 'YOLOv8s')
                assert det_source == 'YOLOv8s', f"Non-YOLO bbox expansion blocked (source: {det_source})"
                
                # Phase-1 Fix B: Column-aware PARSeq decoding
                # Get column header from detection (if available) for decode policy
                column_header = detection.get('column_header')  # May be None if not assigned yet
                max_length = None
                if column_header:
                    # Look up decode policy for this column
                    policy = settings.PARSEQ_COLUMN_DECODE_POLICY.get(column_header, {})
                    max_length = policy.get('max_length', settings.PARSEQ_DEFAULT_MAX_LENGTH)
                    log.debug(f"Column-aware decoding: '{column_header}' → max_length={max_length}")
                else:
                    # Use default if no column header
                    max_length = settings.PARSEQ_DEFAULT_MAX_LENGTH
                    log.debug(f"Using default max_length={max_length} (no column header)")
                
                # Handwritten → PARSeq ONLY (no exceptions, no fallbacks)
                # Note: PARSeq is optional, so we allow it to be None (unlike PaddleOCR)
                if self.parseq_recognizer is None or not self.parseq_recognizer.is_available():
                    log.warning(f"PARSeq not available for Handwritten detection {i}, skipping")
                    ocr_results.append({
                        'bbox': bbox,
                        'text': '',
                        'confidence': 0.0,
                        'source': 'parseq',  # Mark as PARSeq
                        'class': 'handwritten',
                        'error': 'PARSeq not available'
                    })
                    continue
                
                log.debug(f"Routing Handwritten detection {i+1} to PARSeq (YOLO authority)...")
                
                # Run PARSeq directly on YOLO Handwritten crop
                # Architecture Rule: PARSeq must NEVER run on full images, ONLY on YOLO crops
                # Phase-1 Fix B: Pass max_length for column-aware decoding
                try:
                    text, conf = self.parseq_recognizer.recognize(region_image, max_length=max_length)
                    
                    # Apply confidence threshold (0.4 for handwriting, but keep bbox)
                    if float(conf) >= 0.4 and text.strip():
                        # Phase-1.5: Pass column_header for charset filtering
                        normalized_text = self._normalize_text(text, is_handwritten=True, column_header=column_header)
                        if normalized_text:
                            ocr_results.append({
                                'bbox': bbox,
                                'text': normalized_text,
                                'confidence': float(conf),
                                'source': 'parseq',  # Changed from 'trocr'
                                'class': 'handwritten',
                                'detection_confidence': confidence
                            })
                            log.info(f"PARSeq text extracted: '{normalized_text}' (conf={conf:.2f})")
                        else:
                            log.debug(f"PARSeq text empty after normalization (conf={conf:.2f})")
                    else:
                        log.debug(f"PARSeq text dropped: confidence {conf:.2f} < 0.4 or empty text")
                        ocr_results.append({
                            'bbox': bbox,
                            'text': '',
                            'confidence': 0.0,
                            'source': 'parseq',
                            'class': 'handwritten',
                            'detection_confidence': confidence
                        })
                except Exception as e:
                    log.error(f"PARSeq recognition failed for Handwritten detection {i+1}: {str(e)}", exc_info=True)
                    ocr_results.append({
                        'bbox': bbox,
                        'text': '',
                        'confidence': 0.0,
                        'source': 'parseq',
                        'class': 'handwritten',
                        'error': f'PARSeq recognition failed: {str(e)}'
                    })
                
            else:
                # Other classes (Table, Signature, Checkbox) - no OCR
                # This is correct - OCR should never process these classes
                log.debug(f"Skipping OCR for class '{class_name}' (not OCR-able per YOLO classification)")
                ocr_results.append({
                    'bbox': bbox,
                    'text': '',
                    'confidence': 0.0,
                    'source': 'none',
                    'class': class_name,
                    'note': f'Class {class_name} does not require OCR (YOLO classification)'
                })
        
        # Log summary
        text_block_count = sum(1 for r in ocr_results if r.get('class') == 'text_block' and r.get('text', '').strip())
        handwritten_count = sum(1 for r in ocr_results if r.get('class') == 'handwritten' and r.get('text', '').strip())
        log.info(f"Class-based OCR routing complete: {len(ocr_results)} detections processed")
        log.info(f"  Text_box with text: {text_block_count}, Handwritten with text: {handwritten_count}")
        return ocr_results
    
    def _sanitize_decoder_artifacts(self, text: str, column_header: Optional[str] = None) -> str:
        """Sanitize PARSeq decoder artifacts (Phase-1.5: Symbol Elimination & OCR Hygiene).
        
        Comprehensive symbol elimination pipeline for handwritten OCR output.
        Symbols are never valid semantic output in handwritten table cells.
        
        Pipeline (5 steps):
        1. Trim obvious edge junk
        2. Remove repeated decoder artifacts
        3. Remove trailing non-alphanumeric characters (CRITICAL)
        4. Column-aware allowed charset filtering (STRICT)
        5. Final whitespace normalization
        
        Rules:
        - Deterministic symbol removal
        - Column-aware charset filtering
        - No spell correction
        - No dictionary lookup
        - No inference
        
        Examples:
        "Cooley&" → "Cooley"
        "Michael&" → "Michael"
        "`" → "" (empty after sanitization)
        "--Chin-" → "Chin"
        "HCP&" → "HCP"
        
        Args:
            text: Raw OCR text from PARSeq
            column_header: Column header for charset filtering (optional)
            
        Returns:
            Sanitized text with all symbols removed
        """
        if not text:
            return ""
        
        # Step 1: Trim obvious edge junk
        text = text.strip(" |-_.,;:")
        
        # Step 2: Remove repeated decoder artifacts
        text = re.sub(r"[|_]{2,}", "", text)
        
        # Step 3: Remove trailing non-alphanumeric characters (CRITICAL)
        # This fixes: Cooley&, Michael&, HCP#
        text = re.sub(r"[^A-Za-z0-9]+$", "", text)
        
        # Step 4: Column-aware allowed charset (STRICT)
        if column_header:
            allowed_charset = settings.PARSEQ_COLUMN_CHARSET.get(column_header)
            if allowed_charset:
                # Filter to only allowed characters
                text = re.sub(f"[^{allowed_charset}]", "", text)
        
        # Step 5: Final whitespace normalization
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        # Hard safety guard: Symbols must be gone
        if re.search(r"[&|_<>]", text):
            log.warning(f"Symbols still present after sanitization: '{text}' - applying aggressive cleanup")
            # Aggressive cleanup: remove all remaining symbols
            text = re.sub(r"[&|_<>]", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
        
        return text
    
    def _normalize_text(self, text: str, is_handwritten: bool = False, column_header: Optional[str] = None) -> str:
        """Normalize text by collapsing whitespace.
        
        Phase-1.5: For handwritten text, apply comprehensive symbol elimination.
        
        Args:
            text: Raw OCR text
            is_handwritten: If True, apply symbol elimination (sanitize artifacts)
            column_header: Column header for charset filtering (optional, for handwritten only)
            
        Returns:
            Normalized text (collapsed whitespace, stripped, sanitized if handwritten)
        """
        if not text:
            return ""
        
        # Phase-1.5: Comprehensive symbol elimination for handwritten text
        if is_handwritten:
            text = self._sanitize_decoder_artifacts(text, column_header=column_header)
            
            # Hard safety guard: Symbols must be gone
            if re.search(r"[&|_<>]", text):
                log.warning(f"Symbols still present after sanitization: '{text}' - this should not happen")
                # This should not happen due to aggressive cleanup in sanitize, but log it
            
            # Hard safety guard: Empty text is allowed but logged
            if text == "":
                log.warning(f"Empty handwritten OCR after sanitization (original: '{text}')")
        
        # Collapse multiple whitespace to single space
        normalized = re.sub(r"\s+", " ", text.strip())
        return normalized

