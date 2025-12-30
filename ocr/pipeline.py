"""OCR Pipeline - Main orchestrator for document OCR.

This module orchestrates the complete OCR pipeline:
1. PaddleOCR detection (uses existing detector from layout module)
2. PaddleOCR recognition (default)
3. TrOCR recognition (fallback for low confidence or handwriting)
4. Result merging with original bounding boxes
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import (
    load_image_from_canonical_id,
    get_intermediate_json_path,
    image_to_array
)

# Import PaddleOCR detector from layout module (for detection only)
from layout.paddleocr_detector import PaddleOCRTextDetector

# Import recognition engines (isolated modules)
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.trocr_recognizer import TrOCRRecognizer
from ocr.handwriting_classifier import HandwritingClassifier


class OCRPipeline:
    """DEPRECATED: Legacy OCR pipeline using PaddleOCR detection.
    
    ⚠️ WARNING: This pipeline violates the architectural rule that YOLO is the only detector.
    Use UnifiedPipeline or ClassBasedOCRRouter instead.
    
    This class is kept for backward compatibility but should NOT be used for new code.
    """
    
    def __init__(self,
                 paddle_confidence_threshold: float = 0.5,
                 use_gpu: Optional[bool] = None):
        """Initialize OCR pipeline.
        
        Args:
            paddle_confidence_threshold: Confidence threshold for PaddleOCR fallback
                                        If PaddleOCR confidence < this, use TrOCR
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        log.warning("⚠️  OCRPipeline is DEPRECATED - uses PaddleOCR detection (violates YOLO-only rule)")
        log.warning("   Use UnifiedPipeline or ClassBasedOCRRouter instead")
        
        self.paddle_confidence_threshold = paddle_confidence_threshold
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        
        log.info("Initializing OCR Pipeline (DEPRECATED)...")
        
        # Initialize PaddleOCR detector (for detection only) - DEPRECATED
        self.paddle_detector = PaddleOCRTextDetector(lang='en')
        
        # Initialize PaddleOCR recognizer (isolated module)
        self.paddle_recognizer = PaddleOCRRecognizer(lang='en', use_gpu=self.use_gpu)
        
        # Initialize TrOCR recognizer (isolated module)
        self.trocr_recognizer = TrOCRRecognizer(
            model_name="microsoft/trocr-base-handwritten",
            use_gpu=self.use_gpu
        )
        
        # Initialize handwriting classifier
        self.handwriting_classifier = HandwritingClassifier()
        
        log.info("OCR Pipeline initialized (DEPRECATED)")
        log.info(f"  PaddleOCR detector: {'available' if self.paddle_detector.ocr else 'unavailable'}")
        log.info(f"  PaddleOCR recognizer: {'available' if self.paddle_recognizer.is_available() else 'unavailable'}")
        log.info(f"  TrOCR recognizer: {'available' if self.trocr_recognizer.is_available() else 'unavailable'}")
        log.info(f"  Confidence threshold: {self.paddle_confidence_threshold}")
    
    def process_image(self,
                     file_id: Optional[str] = None,
                     pre_0_id: Optional[str] = None,
                     pre_01_id: Optional[str] = None) -> Dict[str, Any]:
        """Process image through complete OCR pipeline.
        
        Args:
            file_id: Raw file identifier
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            dict: OCR results with bounding boxes, text, source, confidence
        """
        # Step 1: Load image
        log.info("Loading image for OCR pipeline...")
        try:
            image = load_image_from_canonical_id(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            width, height = image.size
            log.info(f"Image loaded: {width}x{height}")
        except Exception as e:
            # TASK 3: Make OCR failure non-fatal - return valid response
            log.error(f"Failed to load image: {str(e)}")
            return {
                'text_regions': [],
                'dimensions': {'width': 0, 'height': 0},
                'error': str(e),
                'failed': True,
                'metadata': {'error_type': type(e).__name__}
            }
        
        # Step 2: PaddleOCR detection (get bounding boxes)
        log.info("Running PaddleOCR detection...")
        try:
            detections = self.paddle_detector.detect(image)
            log.info(f"PaddleOCR detected {len(detections)} text regions")
        except Exception as e:
            # TASK 3: Make OCR failure non-fatal - return valid response
            log.error(f"PaddleOCR detection failed: {str(e)}")
            return {
                'text_regions': [],
                'dimensions': {'width': width, 'height': height},
                'error': f"Detection failed: {str(e)}",
                'failed': True,
                'metadata': {'error_type': type(e).__name__}
            }
        
        if not detections:
            log.warning("No text regions detected")
            return {
                'text_regions': [],
                'failed': False
            }
        
        # Step 3: Process each detected region
        log.info(f"Processing {len(detections)} text regions...")
        ocr_results = []
        
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [])
            if not bbox or len(bbox) < 4:
                continue
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            # Crop region from image
            try:
                region_image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                log.warning(f"Failed to crop region {i}: {str(e)}")
                continue
            
            # Step 3a: PaddleOCR recognition (default)
            log.debug(f"Processing region {i+1}/{len(detections)} with PaddleOCR...")
            paddle_text, paddle_confidence = self.paddle_recognizer.recognize(region_image)
            
            # Step 3b: Check if fallback is needed
            use_fallback = False
            fallback_reason = None
            
            # Check handwriting classification
            is_handwriting = self.handwriting_classifier.is_handwriting(region_image, bbox)
            if is_handwriting:
                use_fallback = True
                fallback_reason = "handwriting_detected"
            
            # Check confidence threshold
            elif paddle_confidence < self.paddle_confidence_threshold:
                use_fallback = True
                fallback_reason = f"low_confidence_{paddle_confidence:.2f}"
            
            # Step 3c: TrOCR recognition (if fallback needed)
            final_text = paddle_text
            final_confidence = paddle_confidence
            recognizer_source = "PaddleOCR"
            
            if use_fallback and self.trocr_recognizer.is_available():
                log.debug(f"Using TrOCR fallback for region {i+1} (reason: {fallback_reason})...")
                trocr_text, trocr_confidence = self.trocr_recognizer.recognize(region_image)
                
                # Use TrOCR result if it's better or if PaddleOCR failed
                if trocr_text.strip() and (not paddle_text.strip() or trocr_confidence > paddle_confidence):
                    final_text = trocr_text
                    final_confidence = trocr_confidence
                    recognizer_source = "TrOCR"
                    log.debug(f"TrOCR result selected: '{final_text}' (confidence: {final_confidence:.2f})")
                else:
                    log.debug(f"Keeping PaddleOCR result: '{final_text}' (confidence: {final_confidence:.2f})")
            
            # TASK 4: Confidence filtering - drop results with confidence < 0.5
            # But keep raw bbox even if text is empty (for row grouping)
            final_confidence_float = float(final_confidence)
            
            # If confidence is too low, set text to empty but keep bbox
            if final_confidence_float < 0.5:
                if final_text.strip():  # Only log if we had text
                    log.debug(f"Region {i+1}: Dropping low-confidence text (conf={final_confidence_float:.3f}): '{final_text[:50]}...'")
                final_text = ""  # Empty text for low confidence
                final_confidence_float = 0.0  # Set confidence to 0.0 for empty text
            
            # Step 3d: Create result entry
            # ⚠️ DEPRECATED: This format includes fallback fields which violate clean architecture
            ocr_results.append({
                'bbox': bbox,  # Preserve original normalized bounding box (always keep for row grouping)
                'text': final_text,  # May be empty if confidence < 0.5
                'confidence': final_confidence_float,  # 0.0 if text was dropped
                'source': recognizer_source,
                # ❌ DEPRECATED FIELDS (kept for backward compatibility):
                'is_handwriting': False,  # Handwriting should come from YOLO class, not OCR heuristics
                'fallback_used': use_fallback,
                'fallback_reason': fallback_reason if use_fallback else None,
                'paddle_text': paddle_text,  # Keep original for reference
                'paddle_confidence': float(paddle_confidence)
            })
        
        # TASK 4: Filter out results with empty text and confidence < 0.5
        # But keep bboxes for row grouping even if text is empty
        filtered_results = []
        for result in ocr_results:
            # Keep result if it has text OR if we want to keep empty bboxes for row grouping
            # For now, we keep all results (even empty text) to preserve bboxes for row grouping
            # The confidence is already set to 0.0 for empty text
            filtered_results.append(result)
        
        # Count statistics
        total_regions = len(filtered_results)
        regions_with_text = sum(1 for r in filtered_results if r['text'].strip())
        regions_empty_text = total_regions - regions_with_text
        
        log.info(f"OCR pipeline complete: {total_regions} regions processed ({regions_with_text} with text, {regions_empty_text} empty text kept for bbox)")
        
        # Step 4: Save results
        canonical_id = pre_01_id or pre_0_id or file_id
        if canonical_id:
            try:
                output_path = get_intermediate_json_path(canonical_id, 'ocr')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'text_regions': filtered_results,
                        'dimensions': {'width': width, 'height': height},
                        'metadata': {
                            'paddle_confidence_threshold': self.paddle_confidence_threshold,
                            'total_regions': total_regions,
                            'regions_with_text': regions_with_text,
                            'regions_empty_text': regions_empty_text,
                            'paddleocr_count': sum(1 for r in filtered_results if r['source'] == 'PaddleOCR'),
                            'trocr_count': sum(1 for r in filtered_results if r['source'] == 'TrOCR'),
                            'handwriting_count': sum(1 for r in filtered_results if r['is_handwriting'])
                        }
                    }, f, indent=2, ensure_ascii=False)
                log.info(f"OCR results saved to: {output_path}")
            except Exception as e:
                log.warning(f"Failed to save OCR results: {str(e)}")
        
        return {
            'text_regions': filtered_results,
            'dimensions': {'width': width, 'height': height},
            'failed': False,
            'metadata': {
                'total_regions': total_regions,
                'regions_with_text': regions_with_text,
                'regions_empty_text': regions_empty_text
            }
        }




