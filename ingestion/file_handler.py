"""File format detection and loading."""

from pathlib import Path
from typing import Tuple, Optional
import mimetypes
from PIL import Image
from core.logging import log
from core.utils import get_raw_file_path

# Optional imports
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    log.warning("pillow-heif not installed. HEIC/HEIF support will be limited.")

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    log.warning("pdf2image not installed. PDF support will be unavailable.")


class FileHandler:
    """Handles file format detection and loading."""
    
    SUPPORTED_FORMATS = {
        'jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf'
    }
    
    @staticmethod
    def detect_format(file_path: Path) -> Tuple[str, str]:
        """Detect file format from path and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[str, str]: (mime_type, extension)
        """
        extension = file_path.suffix[1:].lower()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = f"image/{extension}" if extension != 'pdf' else "application/pdf"
        
        return mime_type, extension
    
    @staticmethod
    def load_image(file_id: str, extension: str) -> Image.Image:
        """Load image from file ID.
        
        Supports: JPG, PNG, HEIC, HEIF, AVIF, PDF (first page)
        
        Args:
            file_id: File identifier
            extension: File extension
            
        Returns:
            Image.Image: Loaded image in RGB format
            
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be loaded
        """
        file_path = get_raw_file_path(file_id, extension)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension_lower = extension.lower()
        
        try:
            if extension_lower == 'pdf':
                if not PDF_AVAILABLE:
                    raise ImportError("pdf2image is not installed. Install it with: pip install pdf2image")
                # Convert PDF first page to image
                log.info(f"Converting PDF first page: {file_id}")
                images = convert_from_path(str(file_path), first_page=1, last_page=1, dpi=300)
                if not images:
                    raise ValueError("PDF conversion returned no images")
                image = images[0]
            elif extension_lower in {'heic', 'heif'}:
                if not HEIF_AVAILABLE:
                    raise ImportError("pillow-heif is not installed. Install it with: pip install pillow-heif")
                # Load HEIC/HEIF using pillow-heif
                log.info(f"Loading HEIC/HEIF: {file_id}")
                image = Image.open(file_path)
            elif extension_lower == 'avif':
                # Load AVIF (requires pillow-avif-plugin)
                log.info(f"Loading AVIF: {file_id}")
                try:
                    image = Image.open(file_path)
                except Exception as e:
                    raise ImportError(f"AVIF support requires pillow-avif-plugin. Install with: pip install pillow-avif-plugin. Error: {str(e)}")
            elif extension_lower in {'jpg', 'jpeg', 'png'}:
                # Standard image formats
                log.info(f"Loading standard image: {file_id}")
                image = Image.open(file_path)
            else:
                raise ValueError(f"Unsupported format: {extension}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                log.info(f"Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            log.info(f"Image loaded: {image.size}, mode: {image.mode}")
            return image
            
        except Exception as e:
            log.error(f"Failed to load image {file_id}: {str(e)}")
            raise IOError(f"Failed to load image: {str(e)}")










