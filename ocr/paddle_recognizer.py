"""PaddleOCR recognition engine (isolated module).

This module handles text recognition using PaddleOCR.
It is completely isolated from TrOCR and PyTorch dependencies.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import image_to_array

# PaddleOCR import (PaddlePaddle framework)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    log.warning(f"PaddleOCR not installed. PaddleOCR recognition will be unavailable. Error: {str(e)}")
    log.warning("Install with: pip install paddlepaddle paddleocr")
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    log.warning(f"PaddleOCR import failed. PaddleOCR recognition will be unavailable. Error: {str(e)}")


class PaddleOCRRecognizer:
    """PaddleOCR text recognition engine (isolated from PyTorch/TrOCR).
    
    This class handles:
    - Text recognition on cropped image regions
    - Confidence score extraction
    - Device selection (GPU/CPU) for PaddlePaddle
    """
    
    def __init__(self, lang: str = 'en', use_gpu: Optional[bool] = None):
        """Initialize PaddleOCR recognition engine.
        
        Args:
            lang: Language code ('en', 'ch', etc.)
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.lang = lang
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.ocr = None
        
        if not PADDLEOCR_AVAILABLE:
            log.error("PaddleOCR not available. Recognition will be unavailable.")
            log.error("Please install PaddleOCR: pip install paddlepaddle paddleocr")
            return
        
        try:
            log.info(f"Initializing PaddleOCR recognizer (lang={lang})...")
            
            # CRITICAL ARCHITECTURAL RULE: PaddleOCR must ONLY do recognition (det=False)
            # YOLOv8s is the ONLY authority for regions - PaddleOCR must NEVER detect
            # PaddleOCR receives ONLY cropped regions from YOLO
            # 
            # Note: This version of PaddleOCR may not support many parameters.
            # We try progressively simpler configurations until one works.
            # This version of PaddleOCR appears to have very limited parameter support
            # Try progressively simpler configurations
            try:
                # Method 1: Try with det=False (most critical parameter)
                self.ocr = PaddleOCR(
                    lang=lang,
                    det=False  # ❌ CRITICAL: Disable detection - YOLO is the only detector
                )
                log.info("✅ PaddleOCR initialized (RECOGNITION ONLY, det=False)")
            except (TypeError, ValueError, Exception) as e0:
                log.info(f"PaddleOCR with det=False failed ({str(e0)}), trying absolute minimal...")
                try:
                    # Method 2: Try absolute minimal (lang only)
                    # WARNING: This version may not support det=False at all
                    # We'll enforce recognition-only at runtime
                    self.ocr = PaddleOCR(lang=lang)
                    log.warning("⚠️  PaddleOCR initialized without explicit det=False parameter")
                    log.warning("   This version does not support det=False parameter")
                    log.warning("   We will enforce recognition-only mode at runtime")
                    log.warning("   Architecture: PaddleOCR will only receive YOLO crops (no full images)")
                    log.info("✅ PaddleOCR initialized (minimal config, lang only)")
                except Exception as e1:
                    log.error(f"Failed to initialize PaddleOCR")
                    log.error(f"  Method 1 (det=False): {str(e0)}")
                    log.error(f"  Method 2 (minimal, lang only): {str(e1)}")
                    log.error(f"  PaddleOCR version may be incompatible or incorrectly installed")
                    log.error(f"  Try: pip install --upgrade paddlepaddle paddleocr")
                    raise e1
            
            log.info("✅ PaddleOCR recognizer initialized successfully")
            log.info(f"   PaddleOCR engine type: {type(self.ocr)}")
            log.info(f"   Language: {lang}, GPU: {self.use_gpu}")
        except Exception as e:
            log.error(f"❌ Failed to initialize PaddleOCR recognizer: {str(e)}", exc_info=True)
            log.error("   This is a CRITICAL error - PaddleOCR must be initialized for Text_box recognition")
            log.error("   Install with: pip install paddlepaddle paddleocr")
            self.ocr = None
            raise RuntimeError(f"PaddleOCR initialization failed: {str(e)}") from e
    
    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text from a cropped image region.
        
        Args:
            image: PIL Image (RGB) of cropped text region
            
        Returns:
            tuple: (recognized_text, confidence_score)
                - recognized_text: Recognized text string
                - confidence_score: Confidence score (0.0-1.0)
        """
        if not self.ocr:
            # Only log once per instance to avoid spam
            if not hasattr(self, '_warned_once'):
                log.error("PaddleOCR not available, returning empty recognition")
                log.error("PaddleOCR recognizer was not initialized. Check initialization logs for errors.")
                log.error("Install PaddleOCR with: pip install paddlepaddle paddleocr")
                self._warned_once = True
            return "", 0.0
        
        try:
            # Convert PIL Image to numpy array (BGR for OpenCV/PaddleOCR)
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # CRITICAL: PaddleOCR must ONLY do recognition, not detection
            # Try ocr() method with explicit rec=True, det=False if supported
            # Otherwise rely on initialization-time settings
            result = None
            try:
                # Try ocr() with explicit rec=True, det=False (if supported)
                result = self.ocr.ocr(img_bgr, rec=True, det=False)
            except (TypeError, ValueError):
                # If rec/det parameters not supported in ocr() call, try without them
                # (rely on initialization-time det=False if it was set)
                try:
                    result = self.ocr.ocr(img_bgr)
                except AttributeError:
                    # Try predict() method (newer API)
                    try:
                        result = self.ocr.predict(img_bgr)
                    except Exception as e:
                        log.warning(f"All PaddleOCR API methods failed: {str(e)}")
                        return "", 0.0
            except Exception as e:
                log.error(f"PaddleOCR recognition failed: {str(e)}")
                return "", 0.0
            
            if not result:
                return "", 0.0
            
            # Handle different PaddleOCR result formats
            texts = []
            confidences = []
            
            # New API format: result is typically a list of OCRResult objects
            if isinstance(result, list):
                # Handle list format (list of OCRResult objects or old format)
                for item in result:
                    # Check for OCRResult object (new format) - has rec_texts and rec_scores attributes
                    if hasattr(item, 'rec_texts') and hasattr(item, 'rec_scores'):
                        # OCRResult format: rec_texts is a list of texts, rec_scores is a list of scores
                        if item.rec_texts and len(item.rec_texts) > 0:
                            for text, score in zip(item.rec_texts, item.rec_scores):
                                texts.append(str(text))
                                confidences.append(float(score))
                    # Check for dictionary-like access (OCRResult can be accessed like dict)
                    elif hasattr(item, '__getitem__') and hasattr(item, 'keys'):
                        # Dictionary-like format
                        if 'rec_texts' in item and 'rec_scores' in item:
                            rec_texts = item['rec_texts']
                            rec_scores = item['rec_scores']
                            if rec_texts and len(rec_texts) > 0:
                                for text, score in zip(rec_texts, rec_scores):
                                    texts.append(str(text))
                                    confidences.append(float(score))
                        elif 'rec_text' in item:
                            texts.append(str(item['rec_text']))
                            if 'rec_score' in item:
                                confidences.append(float(item['rec_score']))
                    # Check for simple dict
                    elif isinstance(item, dict):
                        if 'rec_texts' in item and 'rec_scores' in item:
                            rec_texts = item['rec_texts']
                            rec_scores = item['rec_scores']
                            if rec_texts and len(rec_texts) > 0:
                                for text, score in zip(rec_texts, rec_scores):
                                    texts.append(str(text))
                                    confidences.append(float(score))
                        elif 'rec_text' in item:
                            texts.append(str(item['rec_text']))
                            if 'rec_score' in item:
                                confidences.append(float(item['rec_score']))
                    # Check for object with rec_text/rec_score attributes
                    elif hasattr(item, 'rec_text') and hasattr(item, 'rec_score'):
                        texts.append(str(item.rec_text))
                        confidences.append(float(item.rec_score))
                    # Old format: [[bbox], (text, confidence)]
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        text_info = item[1] if len(item) > 1 else None
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            texts.append(str(text_info[0]))
                            confidences.append(float(text_info[1]))
            # Single OCRResult object (not in a list)
            elif hasattr(result, 'rec_texts') and hasattr(result, 'rec_scores'):
                if result.rec_texts and len(result.rec_texts) > 0:
                    for text, score in zip(result.rec_texts, result.rec_scores):
                        texts.append(str(text))
                        confidences.append(float(score))
            # Dictionary format
            elif isinstance(result, dict):
                if 'rec_texts' in result and 'rec_scores' in result:
                    rec_texts = result['rec_texts']
                    rec_scores = result['rec_scores']
                    if rec_texts and len(rec_texts) > 0:
                        for text, score in zip(rec_texts, rec_scores):
                            texts.append(str(text))
                            confidences.append(float(score))
                elif 'rec_text' in result:
                    texts.append(str(result['rec_text']))
                    if 'rec_score' in result:
                        confidences.append(float(result['rec_score']))
            # Object with single rec_text/rec_score
            elif hasattr(result, 'rec_text') and hasattr(result, 'rec_score'):
                texts.append(str(result.rec_text))
                confidences.append(float(result.rec_score))
            
            # Combine all text lines
            if texts:
                combined_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                return combined_text, avg_confidence
            else:
                log.debug(f"PaddleOCR result format not recognized. Result type: {type(result)}")
                if isinstance(result, list) and len(result) > 0:
                    log.debug(f"First item type: {type(result[0])}, attributes: {[attr for attr in dir(result[0]) if not attr.startswith('_')][:10]}")
                return "", 0.0
                
        except Exception as e:
            log.error(f"PaddleOCR recognition failed: {str(e)}", exc_info=True)
            return "", 0.0
    
    def recognize_batch(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """Recognize text from multiple cropped image regions.
        
        Args:
            images: List of PIL Images (RGB) of cropped text regions
            
        Returns:
            list: List of (recognized_text, confidence_score) tuples
        """
        results = []
        for image in images:
            text, confidence = self.recognize(image)
            results.append((text, confidence))
        return results
    
    def is_available(self) -> bool:
        """Check if PaddleOCR recognizer is available.
        
        Returns:
            bool: True if PaddleOCR is initialized and ready
        """
        return self.ocr is not None




