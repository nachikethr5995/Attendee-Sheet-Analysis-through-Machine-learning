"""Class-based OCR router - routes Text_box to PaddleOCR, Handwritten to TrOCR.

STRICT RULE:
- Text_box → PaddleOCR ONLY
- Handwritten → TrOCR ONLY
- No cross-routing allowed
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import re
from typing import List, Dict, Any, Optional
from PIL import Image
from core.logging import log
from core.config import settings
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.trocr_recognizer import TrOCRRecognizer


class ClassBasedOCRRouter:
    """Routes OCR based on YOLOv8s class labels.
    
    Architecture:
    - Text_box detections → PaddleOCR (printed text)
    - Handwritten detections → TrOCR (handwriting)
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
        
        # Initialize TrOCR (for Handwritten only)
        try:
            self.trocr_recognizer = TrOCRRecognizer(
                model_name="microsoft/trocr-base-handwritten",
                use_gpu=self.use_gpu
            )
        except Exception as e:
            log.warning(f"⚠️  TrOCR initialization failed: {str(e)}")
            log.warning("   Handwritten text recognition will be unavailable")
            self.trocr_recognizer = None
        
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
        
        # Verify TrOCR initialization
        if self.trocr_recognizer is None:
            log.warning("⚠️  TrOCR recognizer initialization failed - handwritten text will be unavailable")
        elif not self.trocr_recognizer.is_available():
            log.warning("⚠️  TrOCR engine is not available - handwritten text will be unavailable")
        else:
            log.info(f"✅ TrOCR (Handwritten): initialized and ready")
    
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
            
            # Ensure valid crop coordinates
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # Crop region
            try:
                region_image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                log.warning(f"Failed to crop detection {i}: {str(e)}")
                continue
            
            # STRICT CLASS-BASED ROUTING (No fallbacks, no heuristics, YOLO authority only)
            if class_name == 'text_block' or class_name == 'text_box':
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
                        normalized_text = self._normalize_text(text)
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
                # Handwritten → TrOCR ONLY (no exceptions, no fallbacks)
                # Note: TrOCR is optional, so we allow it to be None (unlike PaddleOCR)
                if self.trocr_recognizer is None or not self.trocr_recognizer.is_available():
                    log.warning(f"TrOCR not available for Handwritten detection {i}, skipping")
                    ocr_results.append({
                        'bbox': bbox,
                        'text': '',
                        'confidence': 0.0,
                        'source': 'trocr',  # Still mark as TrOCR even if unavailable
                        'class': 'handwritten',
                        'error': 'TrOCR not available'
                    })
                    continue
                
                log.debug(f"Routing Handwritten detection {i+1} to TrOCR (YOLO authority)...")
                
                # Run TrOCR directly on YOLO Handwritten crop
                # Architecture Rule: TrOCR must NEVER run on full images, ONLY on YOLO crops
                try:
                    text, conf = self.trocr_recognizer.recognize(region_image)
                    
                    # Apply confidence threshold (0.4 for handwriting, but keep bbox)
                    if float(conf) >= 0.4 and text.strip():
                        normalized_text = self._normalize_text(text)
                        if normalized_text:
                            ocr_results.append({
                                'bbox': bbox,
                                'text': normalized_text,
                                'confidence': float(conf),
                                'source': 'trocr',
                                'class': 'handwritten',
                                'detection_confidence': confidence
                            })
                            log.info(f"TrOCR text extracted: '{normalized_text}' (conf={conf:.2f})")
                        else:
                            log.debug(f"TrOCR text empty after normalization (conf={conf:.2f})")
                    else:
                        log.debug(f"TrOCR text dropped: confidence {conf:.2f} < 0.4 or empty text")
                        ocr_results.append({
                            'bbox': bbox,
                            'text': '',
                            'confidence': 0.0,
                            'source': 'trocr',
                            'class': 'handwritten',
                            'detection_confidence': confidence
                        })
                except Exception as e:
                    log.error(f"TrOCR recognition failed for Handwritten detection {i+1}: {str(e)}", exc_info=True)
                    ocr_results.append({
                        'bbox': bbox,
                        'text': '',
                        'confidence': 0.0,
                        'source': 'trocr',
                        'class': 'handwritten',
                        'error': f'TrOCR recognition failed: {str(e)}'
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

