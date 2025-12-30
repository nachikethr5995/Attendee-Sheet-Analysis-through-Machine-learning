"""Utility functions for AIME Backend."""

import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import numpy as np
from PIL import Image


def generate_file_id() -> str:
    """Generate a unique file ID.
    
    Format: file_{timestamp}_{random}
    
    Returns:
        str: Unique file identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"file_{timestamp}_{random_str}"


def generate_processed_id() -> str:
    """Generate a unique processed image ID (legacy).
    
    Format: pre_{timestamp}_{random}
    
    Returns:
        str: Unique processed identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_{timestamp}_{random_str}"


def generate_pre_0_id() -> str:
    """Generate a unique basic preprocessing ID (SERVICE 0).
    
    Format: pre_0_{timestamp}_{random}
    
    Returns:
        str: Unique basic preprocessing identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_0_{timestamp}_{random_str}"


def generate_pre_01_id() -> str:
    """Generate a unique advanced preprocessing ID (SERVICE 0.1).
    
    Format: pre_01_{timestamp}_{random}
    
    Returns:
        str: Unique advanced preprocessing identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"pre_01_{timestamp}_{random_str}"


def generate_analysis_id() -> str:
    """Generate a unique analysis ID.
    
    Format: analysis_{timestamp}_{random}
    
    Returns:
        str: Unique analysis identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = secrets.token_hex(4)
    return f"analysis_{timestamp}_{random_str}"


def get_raw_file_path(file_id: str, extension: str) -> Path:
    """Get the storage path for a raw uploaded file.
    
    Args:
        file_id: File identifier
        extension: File extension (e.g., 'jpg', 'png', 'pdf')
        
    Returns:
        Path: Full path to the raw file
    """
    from core.config import settings
    return Path(settings.STORAGE_RAW) / f"{file_id}.{extension}"


def get_processed_image_path(processed_id: str) -> Path:
    """Get the storage path for a processed image (legacy).
    
    Args:
        processed_id: Processed image identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED) / f"{processed_id}.png"


def get_basic_processed_path(pre_0_id: str) -> Path:
    """Get the storage path for a basic processed image (SERVICE 0).
    
    Args:
        pre_0_id: Basic preprocessing identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED_BASIC) / f"{pre_0_id}.png"


def get_advanced_processed_path(pre_01_id: str) -> Path:
    """Get the storage path for an advanced processed image (SERVICE 0.1).
    
    Args:
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        Path: Full path to the processed PNG image
    """
    from core.config import settings
    return Path(settings.STORAGE_PROCESSED_ADVANCED) / f"{pre_01_id}.png"


def resolve_canonical_id(file_id: Optional[str] = None, 
                         pre_0_id: Optional[str] = None, 
                         pre_01_id: Optional[str] = None) -> str:
    """Resolve canonical ID from available IDs, checking file existence.
    
    Priority: pre_01_id > pre_0_id > file_id
    Only returns an ID if the corresponding file actually exists.
    
    Args:
        file_id: Original file identifier
        pre_0_id: Basic preprocessing identifier
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        str: Canonical ID to use for processing (first available file)
        
    Raises:
        ValueError: If no valid ID is provided or no files exist
    """
    from core.logging import log
    
    # Try pre_01_id first (advanced preprocessing) - check if file exists
    if pre_01_id:
        path = get_advanced_processed_path(pre_01_id)
        if path.exists():
            log.info(f"Resolved canonical_id: pre_01_id ({pre_01_id})")
            return pre_01_id
        else:
            log.warning(f"pre_01_id provided but file not found: {path}, falling back...")
    
    # Try pre_0_id (basic preprocessing) - check if file exists
    if pre_0_id:
        path = get_basic_processed_path(pre_0_id)
        if path.exists():
            log.info(f"Resolved canonical_id: pre_0_id ({pre_0_id})")
            return pre_0_id
        else:
            log.warning(f"pre_0_id provided but file not found: {path}, falling back...")
    
    # Fallback to file_id (raw file) - check if file exists
    if file_id:
        from ingestion.file_handler import FileHandler
        from core.utils import get_raw_file_path
        # Try common extensions
        for ext in ['png', 'jpg', 'jpeg', 'heic', 'heif', 'avif', 'pdf']:
            test_path = get_raw_file_path(file_id, ext)
            if test_path.exists():
                log.info(f"Resolved canonical_id: file_id ({file_id})")
                return file_id
        log.warning(f"file_id provided but file not found: {file_id}")
    
    raise ValueError("At least one ID (file_id, pre_0_id, or pre_01_id) must be provided and the file must exist")


def load_image_from_canonical_id(file_id: Optional[str] = None,
                                  pre_0_id: Optional[str] = None,
                                  pre_01_id: Optional[str] = None) -> Image.Image:
    """Load processed image from canonical ID.
    
    Priority: pre_01_id > pre_0_id > file_id
    
    Args:
        file_id: Original file identifier (fallback)
        pre_0_id: Basic preprocessing identifier
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        Image.Image: Loaded image in RGB format
        
    Raises:
        FileNotFoundError: If no image found for any ID
        ValueError: If no valid ID is provided
    """
    from PIL import Image
    from core.logging import log
    
    # Try pre_01_id first (advanced preprocessing)
    if pre_01_id:
        path = get_advanced_processed_path(pre_01_id)
        if path.exists():
            log.info(f"Loading image from pre_01_id: {pre_01_id}")
            return Image.open(path).convert('RGB')
        else:
            log.warning(f"pre_01_id image not found: {path}")
    
    # Try pre_0_id (basic preprocessing)
    if pre_0_id:
        path = get_basic_processed_path(pre_0_id)
        if path.exists():
            log.info(f"Loading image from pre_0_id: {pre_0_id}")
            return Image.open(path).convert('RGB')
        else:
            log.warning(f"pre_0_id image not found: {path}")
    
    # Fallback to file_id (raw file)
    if file_id:
        from ingestion.file_handler import FileHandler
        # Try common extensions
        for ext in ['png', 'jpg', 'jpeg', 'heic', 'pdf']:
            try:
                return FileHandler.load_image(file_id, ext)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Could not find file for file_id: {file_id}")
    
    raise ValueError("At least one ID (file_id, pre_0_id, or pre_01_id) must be provided")


def get_intermediate_json_path(processed_id: str, suffix: str) -> Path:
    """Get the storage path for intermediate JSON files.
    
    Args:
        processed_id: Processed image identifier
        suffix: File suffix (e.g., 'layout', 'tables', 'ocr')
        
    Returns:
        Path: Full path to the intermediate JSON file
    """
    from core.config import settings
    return Path(settings.STORAGE_INTERMEDIATE) / f"{processed_id}_{suffix}.json"


def get_output_json_path(analysis_id: str) -> Path:
    """Get the storage path for final output JSON.
    
    Args:
        analysis_id: Analysis identifier
        
    Returns:
        Path: Full path to the output JSON file
    """
    from core.config import settings
    return Path(settings.STORAGE_OUTPUT) / f"{analysis_id}.json"


def validate_file_size(file_path: Path, max_size_mb: int) -> bool:
    """Validate that a file is within size limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum size in megabytes
        
    Returns:
        bool: True if file is within limits
    """
    size_mb = file_path.stat().st_size / (1024 * 1024)
    return size_mb <= max_size_mb


def validate_image_format(extension: str) -> bool:
    """Validate that file extension is supported.
    
    Args:
        extension: File extension (lowercase, without dot)
        
    Returns:
        bool: True if format is supported
    """
    supported_formats = {'jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf'}
    return extension.lower() in supported_formats


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image object
        
    Returns:
        np.ndarray: Image as numpy array (RGB)
    """
    return np.array(image.convert('RGB'))


def array_to_image(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array (RGB)
        
    Returns:
        Image.Image: PIL Image object
    """
    return Image.fromarray(array.astype(np.uint8))


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON-serializable types to native Python types.
    
    This function ensures that all data structures can be safely serialized to JSON,
    which is required for FastAPI/Pydantic v2 responses.
    
    Converts:
    - numpy.bool_ → bool
    - numpy.integer → int
    - numpy.floating → float
    - numpy.ndarray → list
    - Recursively processes dicts and lists
    
    Args:
        obj: Object to sanitize (can be dict, list, or any type)
        
    Returns:
        JSON-safe version of the object with all numpy types converted
    """
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    
    # Handle numpy types
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Return as-is for other types (str, int, float, bool, None, etc.)
    return obj










