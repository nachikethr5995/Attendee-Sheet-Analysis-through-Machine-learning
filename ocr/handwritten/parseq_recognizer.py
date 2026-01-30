"""PARSeq recognition engine using official PARSeq codebase.

This implementation uses the official PARSeq repository from:
https://github.com/baudm/parseq

NOT HuggingFace - pure PyTorch implementation.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import normalize_resolution

# PyTorch imports (no transformers needed)
try:
    import torch
    PARSEQ_AVAILABLE = True
except ImportError:
    PARSEQ_AVAILABLE = False
    log.warning("PARSeq dependencies not installed. PARSeq recognition will be unavailable.")
    log.warning("Install with: pip install torch torchvision timm einops pillow pyyaml")


class PARSeqRecognizer:
    """PARSeq text recognition engine using official codebase.
    
    This class handles:
    - Text recognition on cropped image regions (especially handwriting)
    - Confidence score extraction
    - Device selection (GPU/CPU) for PyTorch
    - PARSeq-specific preprocessing (resolution normalization, padding)
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 use_gpu: Optional[bool] = None,
                 min_short_side: int = 96,
                 pad_ratio: float = 0.18):
        """Initialize PARSeq recognition engine.
        
        Args:
            checkpoint_path: Path to PARSeq checkpoint file (.pt) or 'pretrained=parseq'
                           If None, uses settings.PARSEQ_CHECKPOINT_PATH
                           Can also use 'pretrained=parseq' or 'pretrained=parseq-tiny'
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
            min_short_side: Minimum short side in pixels (default: 96 for PARSeq)
            pad_ratio: Padding ratio for crops (default: 0.15)
        """
        self.checkpoint_path = checkpoint_path or settings.PARSEQ_CHECKPOINT_PATH
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.min_short_side = min_short_side
        self.pad_ratio = pad_ratio
        self.device = None
        self.model = None
        self.img_transform = None
        self.available = False
        
        # STEP 4: Explicit init logging (MANDATORY)
        if not PARSEQ_AVAILABLE:
            log.error("❌ PARSeq initialization failed: Missing dependencies (torch)")
            log.error("   Install with: pip install torch torchvision timm einops pillow pyyaml")
            return
        
        try:
            # STEP 6: Device compatibility check
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                log.info(f"PARSeq device: GPU ({torch.cuda.get_device_name(0)})")
            else:
                self.device = torch.device("cpu")
                if self.use_gpu:
                    log.warning("PARSeq: GPU requested but CUDA not available, using CPU")
                else:
                    log.info("PARSeq device: CPU")
            
            log.info(f"Initializing PARSeq recognizer (checkpoint={self.checkpoint_path}, device={self.device})...")
            
            # Import PARSeq utilities from official codebase
            try:
                # Add parseq directory to path (use absolute path)
                import sys
                import os
                
                # Get absolute path to parseq directory
                current_file = Path(__file__).resolve()
                parseq_path = current_file.parent / "parseq"
                parseq_path = parseq_path.resolve()  # Ensure absolute path
                parseq_path_str = str(parseq_path)
                
                log.info(f"PARSeq codebase path: {parseq_path_str}")
                
                # Verify path exists
                if not parseq_path.exists():
                    raise ImportError(f"PARSeq directory not found: {parseq_path}")
                
                # Verify strhub module exists
                strhub_path = parseq_path / "strhub"
                if not strhub_path.exists():
                    raise ImportError(f"PARSeq strhub module not found: {strhub_path}")
                
                # Verify strhub/models/utils.py exists
                utils_path = parseq_path / "strhub" / "models" / "utils.py"
                if not utils_path.exists():
                    raise ImportError(f"PARSeq utils.py not found: {utils_path}")
                
                # Verify strhub/data/module.py exists
                module_path = parseq_path / "strhub" / "data" / "module.py"
                if not module_path.exists():
                    raise ImportError(f"PARSeq module.py not found: {module_path}")
                
                # Add to sys.path if not already there (must be first to take precedence)
                if parseq_path_str not in sys.path:
                    sys.path.insert(0, parseq_path_str)
                    log.debug(f"Added PARSeq path to sys.path: {parseq_path_str}")
                
                # Now try to import (must happen after sys.path is set)
                from strhub.models.utils import load_from_checkpoint, _get_model_class, create_model
                from strhub.data.module import SceneTextDataModule
                log.info("✓ PARSeq utilities imported successfully")
                
                # Store imports for later use
                self._load_from_checkpoint = load_from_checkpoint
                self._get_model_class = _get_model_class
                self._create_model = create_model
            except ImportError as e:
                log.error("❌ Failed to import PARSeq utilities")
                log.error(f"   Error type: {type(e).__name__}")
                log.error(f"   Error message: {str(e)}")
                if 'parseq_path' in locals():
                    log.error(f"   Expected path: {parseq_path}")
                    log.error(f"   Path exists: {parseq_path.exists()}")
                    if parseq_path.exists():
                        log.error(f"   Contents: {list(parseq_path.iterdir())[:5]}")
                import traceback
                log.error(f"   Traceback: {traceback.format_exc()}")
                log.error("   PARSeq codebase not found in ocr/handwritten/parseq/")
                log.error("   Clone from: https://github.com/baudm/parseq")
                log.error("   Place in: Back_end/ocr/handwritten/parseq/")
                raise ImportError("PARSeq utilities not found - clone official repository") from e
            
            # STEP 5: Load model from checkpoint (LOCAL FILES ONLY - NO AUTO-DOWNLOAD)
            # Resolve checkpoint path to absolute
            checkpoint_file = Path(self.checkpoint_path)
            if not checkpoint_file.is_absolute():
                # If relative, resolve from current file location
                current_file = Path(__file__).resolve()
                base_path = current_file.parent.parent.parent  # Back_end/
                checkpoint_file = (base_path / self.checkpoint_path).resolve()
            
            log.info(f"Loading PARSeq model from local checkpoint: {checkpoint_file}")
            
            # Verify checkpoint exists
            if not checkpoint_file.exists():
                log.error(f"❌ PARSeq checkpoint not found: {checkpoint_file}")
                log.error("   Expected location: Back_end/ocr/handwritten/parseq/weights/parseq-bb5792a6.pt")
                log.error("   Download from: https://github.com/baudm/parseq/releases/tag/v1.0.0")
                log.error("   Place the .pt file in: Back_end/ocr/handwritten/parseq/weights/")
                raise FileNotFoundError(f"PARSeq checkpoint not found: {checkpoint_file}")
            
            log.info(f"   Checkpoint file exists: {checkpoint_file}")
            log.info(f"   File size: {checkpoint_file.stat().st_size / (1024*1024):.2f} MB")
            
            # Load model using official PARSeq loader (local file path only)
            # Handle both full PyTorch Lightning checkpoints and state dict-only checkpoints
            try:
                self.model = self._load_from_checkpoint(str(checkpoint_file)).eval().to(self.device)
            except (KeyError, RuntimeError) as e:
                error_msg = str(e)
                if 'pytorch-lightning_version' in error_msg or 'state_dict' in error_msg.lower():
                    # Checkpoint might be state dict only - load it manually
                    log.info("   Loading checkpoint as state dict (not full PyTorch Lightning checkpoint)...")
                    checkpoint_data = torch.load(str(checkpoint_file), map_location='cpu')
                    
                    # Check if it's a state dict or full checkpoint
                    if 'state_dict' in checkpoint_data or 'hyper_parameters' not in checkpoint_data:
                        # It's a state dict - create model with default config and load state dict
                        # Create model with 'parseq' config (default)
                        model = self._create_model('parseq', pretrained=False)
                        # Load state dict (handle both 'state_dict' key and direct state dict)
                        state_dict = checkpoint_data.get('state_dict', checkpoint_data)
                        # For PARSeq, the state dict goes into model.model
                        model.model.load_state_dict(state_dict, strict=False)
                        self.model = model.eval().to(self.device)
                    else:
                        # It's a full checkpoint but missing version - patch it
                        log.warning("   Checkpoint missing version metadata, applying compatibility patch...")
                        checkpoint_data['pytorch-lightning_version'] = '2.2.0'
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as tmp:
                            torch.save(checkpoint_data, tmp.name)
                            tmp_path = tmp.name
                        try:
                            self.model = self._load_from_checkpoint(tmp_path).eval().to(self.device)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                else:
                    raise
            
            log.info("✓ PARSeq model loaded successfully from local checkpoint")
            
            # Get image transform from model (PARSeq provides this)
            self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
            log.info(f"✓ PARSeq image transform configured (img_size={self.model.hparams.img_size})")
            
            self.available = True
            log.info("✅ PARSeq initialized successfully")
            log.info("Handwritten OCR: PARSeq active")
            
        except Exception as e:
            self.available = False
            self.model = None
            self.img_transform = None
            log.exception("❌ PARSeq initialization failed")
            log.error(f"   Error type: {type(e).__name__}")
            log.error(f"   Error message: {str(e)}")
            
            # Provide specific guidance
            if isinstance(e, ImportError):
                log.error("   PARSeq codebase not found - clone official repository")
            elif isinstance(e, FileNotFoundError):
                log.error("   Model checkpoint not found - download weights or use pretrained=parseq")
            elif "cuda" in str(e).lower() or "gpu" in str(e).lower():
                log.error("   CUDA/GPU error - try setting use_gpu=False")
            else:
                log.error("   Check: 1) PARSeq codebase cloned, 2) weights/pretrained available, 3) dependencies installed")
    
    def _preprocess_crop(self, image: Image.Image) -> Image.Image:
        """Preprocess crop for PARSeq (MANDATORY for handwriting).
        
        Steps:
        1. Resolution normalization (min 96px short side)
        2. Padding (18% ratio - Phase-1: increased for better recognition)
        3. Light background normalization (grayscale + CLAHE)
        
        Args:
            image: PIL Image (RGB) of cropped text region
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to numpy for preprocessing
        img_array = np.array(image.convert('RGB'))
        
        # Step 1: Resolution normalization
        img_array = normalize_resolution(img_array, self.min_short_side)
        
        # Step 2: Padding
        h, w = img_array.shape[:2]
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        
        # Add padding (white background)
        img_array = cv2.copyMakeBorder(
            img_array,
            pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]  # White padding
        )
        
        # Step 3: Light background normalization
        # Convert to grayscale, apply CLAHE, convert back to RGB
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    def recognize(self, image: Image.Image, max_length: Optional[int] = None) -> Tuple[str, float]:
        """Recognize text from a cropped image region.
        
        Args:
            image: PIL Image (RGB) of cropped text region
            max_length: Maximum sequence length for decoding (Phase-1: column-aware)
                        If None, uses model default
            
        Returns:
            tuple: (recognized_text, confidence_score)
                - recognized_text: Recognized text string
                - confidence_score: Confidence score (0.0-1.0)
        """
        if not self.model or not self.available or not self.img_transform:
            log.warning("PARSeq not available, returning empty recognition")
            return "", 0.0
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess crop (resolution normalization + padding)
            preprocessed_image = self._preprocess_crop(image)
            
            # Apply PARSeq transform (resize, normalize, etc.)
            img_tensor = self.img_transform(preprocessed_image).unsqueeze(0).to(self.device)
            
            # Run inference (deterministic, no beam search hacks)
            with torch.no_grad():
                # PARSeq inference: model returns logits, then decode with tokenizer
                # Phase-1 Fix B: Apply max_length constraint if provided (column-aware decoding)
                p = self.model(img_tensor, max_length=max_length).softmax(-1)
                pred, p = self.model.tokenizer.decode(p)
                
                # Extract text and confidence
                text = pred[0] if isinstance(pred, (list, tuple)) else str(pred)
                # Confidence is the probability of the predicted sequence
                confidence = float(p[0].mean().item()) if hasattr(p, 'mean') else 0.9
            
            return text.strip(), confidence
            
        except Exception as e:
            log.error(f"PARSeq recognition failed: {str(e)}", exc_info=True)
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
        """Check if PARSeq recognizer is available.
        
        Returns:
            bool: True if PARSeq is initialized and ready
        """
        return self.available and self.model is not None and self.img_transform is not None
