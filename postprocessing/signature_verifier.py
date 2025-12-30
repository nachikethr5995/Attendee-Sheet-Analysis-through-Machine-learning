"""GPDS Signature Verification Module.

This module provides post-detection signature verification.
It validates whether detected signature regions are genuine handwritten signatures
versus noise, printed text, or artifacts.

IMPORTANT: This is NOT a detector. It only verifies already-detected signature regions.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import image_to_array

# PyTorch imports for GPDS model
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available. GPDS signature verification will be unavailable.")


class GPDSSignatureVerifier:
    """GPDS signature verification model.
    
    This class verifies whether cropped signature regions are genuine signatures.
    It does NOT detect signatures - it only validates already-detected regions.
    
    Role: Signature authenticity filter
    Input: Cropped signature images (from detection)
    Output: Verification score and validity classification
    """
    
    # Model input size (GPDS standard)
    DEFAULT_INPUT_SIZE = (150, 220)  # width, height
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 verification_threshold: float = 0.7,
                 input_size: Optional[Tuple[int, int]] = None,
                 use_gpu: Optional[bool] = None):
        """Initialize GPDS signature verifier.
        
        Args:
            model_path: Path to GPDS verification model weights (.pt file)
                       If None, looks for default path
            verification_threshold: Confidence threshold for valid signature (default: 0.7)
            input_size: Model input size (width, height). Default: (150, 220)
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.verification_threshold = verification_threshold
        self.input_size = input_size or self.DEFAULT_INPUT_SIZE
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.model = None
        self.device = None
        
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available. GPDS signature verification will be unavailable.")
            return
        
        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            log.info(f"GPDS verifier using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            log.info("GPDS verifier using CPU")
        
        # Load model
        try:
            default_model_path = Path(settings.MODELS_ROOT) / "signature_model" / "gpds_verification.pt"
            
            if model_path:
                model_path = Path(model_path)
            elif default_model_path.exists():
                model_path = default_model_path
            else:
                # Try alternative name
                alt_path = Path(settings.MODELS_ROOT) / "signature_model" / "gpds_signature.pt"
                if alt_path.exists():
                    model_path = alt_path
                    log.info(f"Using alternative model path: {alt_path}")
                else:
                    log.warning(f"GPDS verification model not found at: {default_model_path}")
                    log.warning("GPDS signature verification will be unavailable.")
                    log.warning("Download GPDS verification model and place at: models/signature_model/gpds_verification.pt")
                    return
            
            if model_path.exists():
                log.info(f"Loading GPDS verification model from: {model_path}")
                # Load YOLO model (GPDS models are typically YOLOv5/YOLOv8 format)
                self.model = YOLO(str(model_path))
                # Move model to device if needed
                if hasattr(self.model, 'model'):
                    self.model.model.to(self.device)
                log.info("GPDS verification model loaded successfully")
            else:
                log.warning(f"GPDS verification model not found: {model_path}")
                
        except Exception as e:
            log.error(f"Failed to load GPDS verification model: {str(e)}", exc_info=True)
            self.model = None
    
    def preprocess_crop(self, image: Image.Image) -> np.ndarray:
        """Preprocess cropped signature image for verification.
        
        Args:
            image: PIL Image (RGB) of cropped signature region
            
        Returns:
            np.ndarray: Preprocessed image ready for model input
        """
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Convert to numpy array
        img_array = np.array(gray, dtype=np.uint8)
        
        # Resize to model input size
        width, height = self.input_size
        img_resized = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)
        
        # Optional: Binarization (Otsu thresholding)
        # This helps with contrast enhancement
        _, img_binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize to [0, 1] range
        img_normalized = img_binary.astype(np.float32) / 255.0
        
        # Convert to 3-channel if model expects RGB
        # Most GPDS models expect grayscale, but some expect RGB
        # We'll provide both formats
        img_rgb = np.stack([img_normalized] * 3, axis=-1)  # Convert to RGB
        
        return img_rgb
    
    def verify(self, signature_crop: Image.Image) -> Dict[str, Any]:
        """Verify if a cropped signature region is a genuine signature.
        
        Args:
            signature_crop: PIL Image (RGB) of cropped signature region
            
        Returns:
            dict: Verification results with:
                - verification_score: float (0-1)
                - is_valid_signature: bool
                - model: str ("GPDS")
                - notes: Optional[str] (error messages or warnings)
        """
        if not self.model:
            return {
                'verification_score': 0.0,
                'is_valid_signature': False,
                'model': 'GPDS',
                'notes': 'GPDS model not available',
                'status': 'skipped'
            }
        
        try:
            # Check crop quality
            width, height = signature_crop.size
            if width < 20 or height < 20:
                return {
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'model': 'GPDS',
                    'notes': 'poor_crop_quality: crop too small',
                    'status': 'invalid'
                }
            
            # Check if crop is blank
            img_array = image_to_array(signature_crop)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Check if image has content (not blank)
            if np.std(gray) < 5:  # Very low variance = blank image
                return {
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'model': 'GPDS',
                    'notes': 'poor_crop_quality: blank or uniform image',
                    'status': 'invalid'
                }
            
            # Preprocess crop
            processed_img = self.preprocess_crop(signature_crop)
            
            # Convert to BGR for YOLO (OpenCV format)
            img_bgr = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Run inference
            # Note: GPDS verification models typically output a single score
            # If the model is a classification model, we'll get class probabilities
            results = self.model(img_bgr, conf=0.1, verbose=False)
            
            verification_score = 0.0
            is_valid = False
            
            if results and len(results) > 0:
                result = results[0]
                
                # Extract confidence from detection
                # For verification models, we typically get:
                # - A single detection with confidence = verification score
                # - Or class probabilities
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Detection-based model
                    max_confidence = 0.0
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > max_confidence:
                            max_confidence = conf
                    verification_score = max_confidence
                elif hasattr(result, 'probs'):
                    # Classification model
                    probs = result.probs
                    # Assuming class 0 = invalid, class 1 = valid signature
                    if len(probs) >= 2:
                        verification_score = float(probs[1])  # Valid signature probability
                    else:
                        verification_score = float(probs[0]) if len(probs) > 0 else 0.0
                else:
                    # Fallback: use first available confidence
                    log.warning("Unknown GPDS model output format, using default score")
                    verification_score = 0.5  # Default neutral score
            
            # Determine validity based on threshold
            is_valid = verification_score >= self.verification_threshold
            
            return {
                'verification_score': float(verification_score),
                'is_valid_signature': is_valid,
                'model': 'GPDS',
                'notes': None if is_valid else f'low_confidence: {verification_score:.2f} < {self.verification_threshold}',
                'status': 'verified' if is_valid else 'low_confidence'
            }
            
        except Exception as e:
            log.error(f"GPDS signature verification failed: {str(e)}", exc_info=True)
            return {
                'verification_score': 0.0,
                'is_valid_signature': False,
                'model': 'GPDS',
                'notes': f'verification_error: {str(e)}',
                'status': 'error'
            }
    
    def verify_batch(self, signature_crops: List[Image.Image]) -> List[Dict[str, Any]]:
        """Verify multiple signature crops in batch.
        
        Args:
            signature_crops: List of PIL Images (cropped signature regions)
            
        Returns:
            list: List of verification results
        """
        results = []
        for crop in signature_crops:
            result = self.verify(crop)
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """Check if GPDS verifier is available.
        
        Returns:
            bool: True if model is loaded and ready
        """
        return self.model is not None




