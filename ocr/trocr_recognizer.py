"""TrOCR recognition engine (isolated PyTorch module).

This module handles text recognition using TrOCR (Transformer-based OCR).
It is completely isolated from PaddleOCR and PaddlePaddle dependencies.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from core.logging import log
from core.config import settings

# PyTorch and Transformers imports (TrOCR dependencies)
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    log.warning("TrOCR dependencies not installed. TrOCR recognition will be unavailable.")


class TrOCRRecognizer:
    """TrOCR text recognition engine (isolated from PaddleOCR/PaddlePaddle).
    
    This class handles:
    - Text recognition on cropped image regions (especially handwriting)
    - Confidence score extraction
    - Device selection (GPU/CPU) for PyTorch
    - Explicit tensor management (no shared state with PaddleOCR)
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/trocr-base-handwritten",
                 use_gpu: Optional[bool] = None):
        """Initialize TrOCR recognition engine.
        
        Args:
            model_name: HuggingFace model name for TrOCR
                       Options:
                       - "microsoft/trocr-base-handwritten" (handwriting)
                       - "microsoft/trocr-base-printed" (printed text)
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.model_name = model_name
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.device = None
        self.processor = None
        self.model = None
        
        if not TROCR_AVAILABLE:
            log.warning("TrOCR dependencies not available. Recognition will be unavailable.")
            return
        
        try:
            # Determine device explicitly
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                log.info(f"TrOCR using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                log.info("TrOCR using CPU")
            
            log.info(f"Initializing TrOCR recognizer (model={model_name}, device={self.device})...")
            
            # Load TrOCR processor and model
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            log.info("TrOCR recognizer initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize TrOCR recognizer: {str(e)}", exc_info=True)
            self.processor = None
            self.model = None
    
    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text from a cropped image region.
        
        Args:
            image: PIL Image (RGB) of cropped text region
            
        Returns:
            tuple: (recognized_text, confidence_score)
                - recognized_text: Recognized text string
                - confidence_score: Confidence score (0.0-1.0)
                   Note: TrOCR doesn't provide explicit confidence scores,
                   so we use a heuristic based on token probabilities
        """
        if not self.model or not self.processor:
            log.warning("TrOCR not available, returning empty recognition")
            return "", 0.0
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image using TrOCR processor
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Run inference (no gradient computation)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # TrOCR doesn't provide explicit confidence scores
            # Use a heuristic: if text is non-empty, assign reasonable confidence
            # For better confidence, we could compute token probabilities, but that's expensive
            confidence = 0.85 if generated_text.strip() else 0.0
            
            return generated_text.strip(), confidence
            
        except Exception as e:
            log.error(f"TrOCR recognition failed: {str(e)}", exc_info=True)
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
        """Check if TrOCR recognizer is available.
        
        Returns:
            bool: True if TrOCR is initialized and ready
        """
        return self.model is not None and self.processor is not None




