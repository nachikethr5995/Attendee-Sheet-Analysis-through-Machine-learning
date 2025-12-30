"""File upload endpoint."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from core.utils import (
    generate_file_id,
    get_raw_file_path,
    validate_file_size,
    validate_image_format,
)
from core.config import settings
from core.logging import log

router = APIRouter(prefix="/api", tags=["upload"])


class UploadResponse(BaseModel):
    """Upload response model."""
    file_id: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing.
    
    Supports: JPG, PNG, HEIC, HEIF, AVIF, PDF
    
    Args:
        file: Uploaded file
        
    Returns:
        UploadResponse: File ID and status message
        
    Raises:
        HTTPException: If file is invalid or too large
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix[1:].lower()
        if not validate_image_format(file_extension):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Supported: JPG, PNG, HEIC, HEIF, AVIF, PDF"
            )
        
        # Generate file ID
        file_id = generate_file_id()
        file_path = get_raw_file_path(file_id, file_extension)
        
        # Read file content
        content = await file.read()
        
        # Check file size
        temp_path = file_path.parent / f"temp_{file_id}.{file_extension}"
        temp_path.write_bytes(content)
        
        if not validate_file_size(temp_path, settings.MAX_FILE_SIZE_MB):
            temp_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Move to final location
        temp_path.rename(file_path)
        
        log.info(f"File uploaded: {file_id} ({file.filename})")
        
        return UploadResponse(
            file_id=file_id,
            message="File uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/upload/{file_id}/image")
async def get_uploaded_image(file_id: str):
    """Get uploaded raw image by file_id.
    
    Args:
        file_id: File identifier
        
    Returns:
        FileResponse: The uploaded image file
        
    Raises:
        HTTPException: If image not found
    """
    try:
        # Try to find the file with any supported extension
        file_path = None
        for ext in ['jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf']:
            potential_path = get_raw_file_path(file_id, ext)
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded file not found: {file_id}"
            )
        
        # Determine media type
        ext = file_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.heic': 'image/heic',
            '.heif': 'image/heif',
            '.avif': 'image/avif',
            '.pdf': 'application/pdf',
        }
        media_type = media_types.get(ext, 'application/octet-stream')
        
        log.info(f"Serving uploaded file: {file_id}")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error serving uploaded file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serve uploaded file: {str(e)}"
        )










