"""Configuration management for AIME Backend."""

import os
from pathlib import Path
from typing import Optional
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
    
    # Row/Column Grouping Settings
    ROW_HEIGHT_THRESHOLD: float = 0.02  # Normalized height threshold for row clustering (2% of image height)
    COLUMN_WIDTH_THRESHOLD: float = 0.03  # Normalized width threshold for column clustering (3% of image width)
    HEADER_ROW_INDEX: int = 2  # Row index to use as header (1-based, default: 2 = second row)
    
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










