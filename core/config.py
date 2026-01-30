"""Configuration management for AIME Backend."""

import os
from pathlib import Path
from typing import Optional, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Storage Paths
    STORAGE_ROOT: str = "./storage"
    STORAGE_RAW: str = "./storage/raw"
    STORAGE_PROCESSED_BASIC: str = "./storage/processed_basic"
    STORAGE_PROCESSED_ADVANCED: str = "./storage/processed_advanced"
    STORAGE_PROCESSED: str = "./storage/processed"  # Legacy, kept for compatibility
    STORAGE_INTERMEDIATE: str = "./storage/intermediate"
    STORAGE_OUTPUT: str = "./storage/output"
    
    # Model Paths
    MODELS_ROOT: str = "./models"
    YOLO_WEIGHTS_PATH: str = "./models/yolo_weights"
    YOLO_LAYOUT_MODEL_PATH: str = "./models/yolo_layout"  # Training-ready YOLOv8s models
    EAST_MODEL_PATH: str = "./models/east"
    SIGNATURE_MODEL_PATH: str = "./models/signature_model"
    
    # Processing Settings
    MAX_FILE_SIZE_MB: int = 50
    TARGET_IMAGE_SIZE: int = 2048
    IMAGE_QUALITY_THRESHOLD: float = 0.7
    
    # GPU Settings
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    
    # OCR Settings
    OCR_PADDLE_CONFIDENCE_THRESHOLD: float = 0.5  # Threshold for TrOCR fallback
    OCR_PADDLE_LANG: str = "en"  # PaddleOCR language
    OCR_TROCR_MODEL: str = "microsoft/trocr-base-handwritten"  # TrOCR model name
    OCR_TABLE_FILTER_MODE: str = "center"  # Filter mode: "center" (center-point containment) or "none" (process all)
    
    # OCR Input Quality Settings (3.1 & 3.2)
    # These improve OCR reliability without changing detection or routing
    OCR_ENABLE_BBOX_EXPANSION: bool = True  # Enable YOLO-safe bbox expansion
    OCR_ENABLE_RESOLUTION_NORMALIZATION: bool = True  # Enable minimum resolution normalization
    
    # Text_box (PaddleOCR) settings
    OCR_TEXT_BOX_BBOX_SCALE: float = 1.15  # Expansion factor for Text_box (15% expansion)
    OCR_TEXT_BOX_MIN_SHORT_SIDE: int = 48  # Minimum short side in pixels for PaddleOCR
    
    # Handwritten (PARSeq) settings
    OCR_HANDWRITTEN_BBOX_SCALE: float = 1.25  # Expansion factor for Handwritten (25% - needs more breathing room)
    OCR_HANDWRITTEN_MIN_SHORT_SIDE: int = 96  # Minimum short side in pixels for PARSeq (higher res needed)
    OCR_HANDWRITTEN_PAD_RATIO: float = 0.18  # Padding ratio for handwritten crops (Phase-1: increased from 0.15)
    PARSEQ_CHECKPOINT_PATH: str = "./ocr/handwritten/parseq/weights/parseq-bb5792a6.pt"  # PARSeq checkpoint: local file path
    
    # Phase-1: Handwritten Box Merging (DISABLED - NO MERGING)
    # Each YOLO handwritten box = exactly one OCR call (architectural rule)
    HANDWRITTEN_MERGE_ENABLED: bool = False  # Merging disabled - each box processed separately
    HANDWRITTEN_MERGE_X_GAP_THRESHOLD: float = 0.03  # Not used (merging disabled)
    
    # Phase-1: Column-Aware PARSeq Decoding (Fix B)
    # Static map: column header → decode policy (max_length only, no vocab/language rules)
    # Deterministic: Uses layout knowledge, no text guessing
    PARSEQ_COLUMN_DECODE_POLICY: Dict[str, Dict[str, int]] = {
        "Last Name": {"max_length": 15},
        "First Name": {"max_length": 15},
        "Attendee Type": {"max_length": 30},
        "Credential or Title": {"max_length": 20},
        "State of License": {"max_length": 10},
        "NPI or State License#": {"max_length": 20},
        "Name": {"max_length": 20},  # Generic name column
        "Guest": {"max_length": 30},  # Guest-related columns
    }
    PARSEQ_DEFAULT_MAX_LENGTH: int = 25  # Default max_length if column not in policy
    
    # Phase-1.5: Column-Aware Allowed Charset (STRICT)
    # Symbols are never valid semantic output in handwritten table cells
    # This enables deterministic symbol elimination
    PARSEQ_COLUMN_CHARSET: Dict[str, str] = {
        "Last Name": r"A-Za-z",
        "First Name": r"A-Za-z",
        "Attendee Type": r"A-Za-z ",
        "Credential or Title": r"A-Za-z ",
        "State of License": r"A-Za-z",
        "NPI or State License#": r"A-Za-z0-9",
    }
    PARSEQ_COLUMN_CHARSET: Dict[str, str] = {
        "Last Name": r"A-Za-z",  # Letters only
        "First Name": r"A-Za-z",  # Letters only
        "Attendee Type": r"A-Za-z ",  # Letters and spaces
        "Credential or Title": r"A-Za-z ",  # Letters and spaces
        "State of License": r"A-Za-z",  # Letters only
        "NPI or State License#": r"A-Za-z0-9",  # Alphanumeric
        "Name": r"A-Za-z ",  # Generic name column
        "Guest": r"A-Za-z ",  # Guest-related columns
    }
    PARSEQ_DEFAULT_ALLOWED_CHARS: str = r"A-Za-z0-9 "  # Default: alphanumeric and spaces
    
    # Text_box vs Handwritten Conflict Resolution
    # When Text_box and Handwritten overlap, higher confidence wins
    TEXT_HAND_CONFLICT_RESOLUTION: bool = True  # Enable conflict resolution
    TEXT_HAND_IOU_THRESHOLD: float = 0.30  # Minimum IoU to consider as conflict
    
    # Signature vs Handwritten Tie-Breaker (Column Consensus)
    # When Signature and Handwritten overlap with EQUAL confidence, use column context
    SIG_HAND_TIEBREAK_ENABLED: bool = True  # Enable tie-breaker
    SIG_HAND_IOU_THRESHOLD: float = 0.30  # Minimum IoU to consider as conflict
    SIG_HAND_CONF_EPSILON: float = 0.01  # Confidence difference to be considered "equal"
    SIG_HAND_COLUMN_CONF_THRESHOLD: float = 0.60  # Minimum confidence for column stats
    
    # Row/Column Grouping Settings
    ROW_HEIGHT_THRESHOLD: float = 0.02  # Normalized height threshold for row clustering (2% of image height)
    COLUMN_WIDTH_THRESHOLD: float = 0.03  # Normalized width threshold for column clustering (3% of image width)
    HEADER_ROW_INDEX: int = 1  # Row index to use as header (1-based, HARD-LOCK: must be row_index == 1)
    
    # Table-Anchored Row Construction
    ROW_Y_THRESHOLD: int = 15  # Pixel threshold for row clustering (absolute pixels, not normalized)
    ROW_ATTACH_MAX_DISTANCE: int = 30  # Maximum pixel distance for out-of-table row attachment
    TABLE_BBOX_MARGIN: int = 5  # Pixel margin for table bbox containment check
    
    # Signature Verification Settings
    SIGNATURE_VERIFICATION_THRESHOLD: float = 0.7  # GPDS verification threshold
    SIGNATURE_VERIFICATION_MODEL_PATH: Optional[str] = None  # Custom model path
    
    # Preprocessing Settings (OPTIONAL - defaults to disabled)
    PREPROCESSING_ENABLED: bool = False  # Default: preprocessing is disabled
    PREPROCESSING_MODE: str = "none"  # none | basic | advanced
    PREPROCESSING_TRIGGER_ON_YOLO_FAILURE: bool = True  # Trigger on YOLOv8s 0 detections
    PREPROCESSING_TRIGGER_ON_YOLO_LOW_CONFIDENCE: bool = True  # Trigger on low YOLOv8s confidence
    PREPROCESSING_YOLO_CONFIDENCE_THRESHOLD: float = 0.3  # Minimum mean confidence to skip preprocessing
    PREPROCESSING_TRIGGER_ON_OCR_LOW_CONFIDENCE: bool = True  # Trigger on low OCR confidence
    PREPROCESSING_OCR_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum OCR confidence to skip preprocessing
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        settings.STORAGE_RAW,
        settings.STORAGE_PROCESSED_BASIC,
        settings.STORAGE_PROCESSED_ADVANCED,
        settings.STORAGE_PROCESSED,  # Legacy
        settings.STORAGE_INTERMEDIATE,
        settings.STORAGE_OUTPUT,
        settings.MODELS_ROOT,
        settings.YOLO_WEIGHTS_PATH,
        settings.YOLO_LAYOUT_MODEL_PATH,
        settings.EAST_MODEL_PATH,
        settings.SIGNATURE_MODEL_PATH,
        Path(settings.LOG_FILE).parent,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()










