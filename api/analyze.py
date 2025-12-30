"""Analysis endpoints for preprocessing and pipeline."""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from core.logging import log
from core.config import settings
from core.utils import (
    get_raw_file_path,
    get_basic_processed_path,
    get_advanced_processed_path,
    resolve_canonical_id,
    json_safe
)
from preprocessing.basic import BasicPreprocessor
from preprocessing.advanced import AdvancedPreprocessor
from layout.layout_service import LayoutService
from ocr.pipeline import OCRPipeline
from postprocessing.signature_verification_service import SignatureVerificationService
from postprocessing.unified_pipeline import UnifiedPipeline
from postprocessing.unified_pipeline import UnifiedPipeline

router = APIRouter(prefix="/api", tags=["analyze"])


class PreprocessRequest(BaseModel):
    """Request model for basic preprocessing (SERVICE 0)."""
    file_id: str


class PreprocessResponse(BaseModel):
    """Response model for basic preprocessing."""
    pre_0_id: str
    clean_image_path: str
    metadata: Dict[str, Any]
    
    class Config:
        extra = "allow"


class AdvancedPreprocessRequest(BaseModel):
    """Request model for advanced preprocessing (SERVICE 0.1)."""
    file_id: Optional[str] = None
    pre_0_id: Optional[str] = None


class AdvancedPreprocessResponse(BaseModel):
    """Response model for advanced preprocessing."""
    pre_01_id: str
    clean_image_path: str
    metadata: Dict[str, Any]
    
    class Config:
        extra = "allow"


class AnalyzeRequest(BaseModel):
    """Request model for complete analysis."""
    file_id: str


class LayoutRequest(BaseModel):
    """Request model for layout detection (SERVICE 1)."""
    file_id: Optional[str] = None
    pre_0_id: Optional[str] = None
    pre_01_id: Optional[str] = None


class LayoutResponse(BaseModel):
    """Response model for layout detection."""
    canonical_id: str
    tables: List[Dict[str, Any]]
    text_blocks: List[Dict[str, Any]]
    signatures: List[Dict[str, Any]]
    checkboxes: List[Dict[str, Any]]
    handwritten: List[Dict[str, Any]] = []  # Handwritten text regions
    dimensions: Dict[str, int]
    failed: bool
    failure_reason: Optional[str] = None
    layout_path: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


class OCRRequest(BaseModel):
    """Request model for OCR pipeline (SERVICE 3)."""
    file_id: Optional[str] = None
    pre_0_id: Optional[str] = None
    pre_01_id: Optional[str] = None
    paddle_confidence_threshold: Optional[float] = None  # Override default threshold


class OCRResponse(BaseModel):
    """Response model for OCR pipeline."""
    text_regions: List[Dict[str, Any]]
    dimensions: Dict[str, int]
    failed: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


class RowwiseRequest(BaseModel):
    """Request model for unified row-wise pipeline."""
    file_id: Optional[str] = None
    pre_0_id: Optional[str] = None
    pre_01_id: Optional[str] = None


class RowwiseResponse(BaseModel):
    """Response model for row-wise structured output."""
    rows: List[Dict[str, Any]]
    total_rows: int
    total_columns: int
    layout: Dict[str, int]
    failed: bool
    error: Optional[str] = None
    
    class Config:
        extra = "allow"


class SignatureVerificationRequest(BaseModel):
    """Request model for signature verification."""
    file_id: Optional[str] = None
    pre_0_id: Optional[str] = None
    pre_01_id: Optional[str] = None
    canonical_id: Optional[str] = None  # Alternative: direct canonical_id
    verification_threshold: Optional[float] = None  # Override default threshold


class SignatureVerificationResponse(BaseModel):
    """Response model for signature verification."""
    signatures: List[Dict[str, Any]]
    total_signatures: int
    valid_signatures: int
    failed: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


@router.get("/analyze/preprocess")
async def preprocess_basic_get():
    """Helper endpoint to provide information about the POST endpoint.
    
    This endpoint exists to provide helpful information when users try to access
    the preprocessing endpoint via GET (e.g., typing URL in browser).
    
    Returns:
        JSONResponse: Information about how to use the endpoint
    """
    return JSONResponse(
        status_code=405,
        content={
            "detail": "Method Not Allowed",
            "message": "This endpoint requires a POST request, not GET.",
            "usage": {
                "method": "POST",
                "url": "/api/analyze/preprocess",
                "content_type": "application/json",
                "body": {
                    "file_id": "your_file_id_here"
                }
            },
            "swagger_ui": "http://localhost:8000/docs",
            "example_curl": 'curl -X POST "http://localhost:8000/api/analyze/preprocess" -H "Content-Type: application/json" -d \'{"file_id": "your_file_id"}\''
        }
    )


@router.get("/analyze/preprocess/advanced")
async def preprocess_advanced_get():
    """Helper endpoint to provide information about the POST endpoint.
    
    This endpoint exists to provide helpful information when users try to access
    the advanced preprocessing endpoint via GET.
    
    Returns:
        JSONResponse: Information about how to use the endpoint
    """
    return JSONResponse(
        status_code=405,
        content={
            "detail": "Method Not Allowed",
            "message": "This endpoint requires a POST request, not GET.",
            "usage": {
                "method": "POST",
                "url": "/api/analyze/preprocess/advanced",
                "content_type": "application/json",
                "body": {
                    "file_id": "your_file_id_here",
                    "pre_0_id": "your_pre_0_id_here (optional)"
                }
            },
            "swagger_ui": "http://localhost:8000/docs",
            "example_curl": 'curl -X POST "http://localhost:8000/api/analyze/preprocess/advanced" -H "Content-Type: application/json" -d \'{"file_id": "your_file_id"}\''
        }
    )


@router.post("/analyze/preprocess", response_model=PreprocessResponse)
async def preprocess_basic(request: PreprocessRequest):
    """SERVICE 0: Basic preprocessing (always runs).
    
    Runs lightweight preprocessing on every uploaded image:
    1. Format conversion (HEIC/AVIF/PDF → PNG)
    2. Light denoising
    3. Image resize to 2048px long edge
    4. Soft deskew (minor angle corrections only)
    5. Light contrast normalization
    6. Minimal artifact removal
    
    Args:
        request: PreprocessRequest with file_id
        
    Returns:
        PreprocessResponse: pre_0_id and metadata
        
    Raises:
        HTTPException: If file not found or processing fails
    """
    try:
        # Validate request
        if not request.file_id:
            raise HTTPException(
                status_code=400,
                detail="file_id is required"
            )
        file_id = request.file_id
        
        # Find the uploaded file
        file_path = None
        file_extension = None
        for ext in ['jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf']:
            potential_path = get_raw_file_path(file_id, ext)
            if potential_path.exists():
                file_path = potential_path
                file_extension = ext
                break
        
        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_id}"
            )
        
        log.info(f"Starting SERVICE 0 (basic preprocessing) for file_id: {file_id}")
        
        # Run basic preprocessing
        preprocessor = BasicPreprocessor()
        result = preprocessor.process(file_id, file_extension)
        
        log.info(f"SERVICE 0 complete. pre_0_id: {result['pre_0_id']}")
        
        # Validate result structure before creating response
        if 'pre_0_id' not in result:
            raise ValueError("pre_0_id missing from preprocessing result")
        if 'clean_image_path' not in result:
            raise ValueError("clean_image_path missing from preprocessing result")
        if 'metadata' not in result:
            raise ValueError("metadata missing from preprocessing result")
        
        try:
            response = PreprocessResponse(**result)
            return response
        except Exception as e:
            log.error(f"Response validation error: {str(e)}")
            log.error(f"Result structure: {result}")
            raise HTTPException(
                status_code=500,
                detail=f"Response validation failed: {str(e)}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        log.error(f"File not found in SERVICE 0: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        log.error(f"Error in SERVICE 0 preprocessing: {str(e)}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Full traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {str(e)}"
        )


@router.post("/analyze/preprocess/advanced", response_model=AdvancedPreprocessResponse)
async def preprocess_advanced(request: AdvancedPreprocessRequest):
    """SERVICE 0.1: Advanced preprocessing (conditional/fallback only).
    
    Runs heavy CV enhancements as a fallback for difficult images:
    1. Strong denoising
    2. Shadow removal
    3. Full perspective correction
    4. Orientation correction
    5. CLAHE++ tuning
    6. Handwriting enhancement
    7. Adaptive binarization
    8. Strong structural cleanup
    
    Args:
        request: AdvancedPreprocessRequest with file_id or pre_0_id
        
    Returns:
        AdvancedPreprocessResponse: pre_01_id and metadata
        
    Raises:
        HTTPException: If file not found or processing fails
    """
    try:
        # Validate request
        if not request.file_id and not request.pre_0_id:
            raise HTTPException(
                status_code=400,
                detail="Either file_id or pre_0_id must be provided"
            )
        file_id = request.file_id
        pre_0_id = request.pre_0_id
        
        log.info("Starting SERVICE 0.1 (advanced preprocessing)")
        
        # Run advanced preprocessing
        preprocessor = AdvancedPreprocessor()
        result = preprocessor.process(file_id=file_id, pre_0_id=pre_0_id)
        
        log.info(f"SERVICE 0.1 complete. pre_01_id: {result['pre_01_id']}")
        
        # Validate result structure before creating response
        if 'pre_01_id' not in result:
            raise ValueError("pre_01_id missing from advanced preprocessing result")
        if 'clean_image_path' not in result:
            raise ValueError("clean_image_path missing from advanced preprocessing result")
        if 'metadata' not in result:
            raise ValueError("metadata missing from advanced preprocessing result")
        
        try:
            response = AdvancedPreprocessResponse(**result)
            return response
        except Exception as e:
            log.error(f"Response validation error: {str(e)}")
            log.error(f"Result structure: {result}")
            raise HTTPException(
                status_code=500,
                detail=f"Response validation failed: {str(e)}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        log.error(f"File not found in SERVICE 0.1: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except ValueError as e:
        log.error(f"Invalid request in SERVICE 0.1: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        log.error(f"Error in SERVICE 0.1 preprocessing: {str(e)}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Full traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced preprocessing failed: {str(e)}"
        )


@router.get("/analyze/preprocess/{pre_0_id}/image")
async def get_basic_processed_image(pre_0_id: str):
    """Get basic processed image (SERVICE 0 output).
    
    Args:
        pre_0_id: Basic preprocessing identifier
        
    Returns:
        FileResponse: Processed PNG image
        
    Raises:
        HTTPException: If image not found
    """
    image_path = get_basic_processed_path(pre_0_id)
    
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed image not found: {pre_0_id}"
        )
    
    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"{pre_0_id}.png"
    )


@router.get("/analyze/preprocess/advanced/{pre_01_id}/image")
async def get_advanced_processed_image(pre_01_id: str):
    """Get advanced processed image (SERVICE 0.1 output).
    
    Args:
        pre_01_id: Advanced preprocessing identifier
        
    Returns:
        FileResponse: Processed PNG image
        
    Raises:
        HTTPException: If image not found
    """
    image_path = get_advanced_processed_path(pre_01_id)
    
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Advanced processed image not found: {pre_01_id}"
        )
    
    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"{pre_01_id}.png"
    )


@router.post("/analyze/layout", response_model=LayoutResponse)
async def detect_layout(request: LayoutRequest):
    """SERVICE 1: Layout detection using YOLOv8s.
    
    Detects layout elements using:
    - YOLOv8s for tables, text blocks, signatures, and checkboxes
    - GPDS signature detector for handwritten signatures
    - Custom checkbox detector (tiny YOLOv8) for checkboxes
    - PaddleOCR text detector for refined text regions
    - Fusion engine to unify all detections
    
    Uses canonical ID resolution: pre_01_id > pre_0_id > file_id
    
    Args:
        request: LayoutRequest with file_id, pre_0_id, or pre_01_id
        
    Returns:
        LayoutResponse: Layout detection results
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Validate request
        if not request.file_id and not request.pre_0_id and not request.pre_01_id:
            raise HTTPException(
                status_code=400,
                detail="At least one ID (file_id, pre_0_id, or pre_01_id) must be provided"
            )
        
        log.info("Starting SERVICE 1 (hybrid layout detection)")
        
        # Run hybrid layout detection
        layout_service = LayoutService()
        result = layout_service.detect_layout(
            file_id=request.file_id,
            pre_0_id=request.pre_0_id,
            pre_01_id=request.pre_01_id
        )
        
        log.info(f"SERVICE 1 complete. canonical_id: {result['canonical_id']}, failed: {result['failed']}")
        
        return LayoutResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "file must exist" in error_msg or "not found" in error_msg.lower():
            log.error(f"File not found in SERVICE 1: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=404,
                detail=f"Image file not found. Please ensure at least one of the following exists: file_id, pre_0_id, or pre_01_id. Error: {error_msg}"
            )
        else:
            log.error(f"Invalid request in SERVICE 1: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {error_msg}"
            )
    except FileNotFoundError as e:
        log.error(f"File not found in SERVICE 1: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found. The provided IDs (file_id, pre_0_id, or pre_01_id) do not correspond to existing files. Error: {str(e)}"
        )
    except Exception as e:
        log.error(f"Error in SERVICE 1 layout detection: {str(e)}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Full traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Layout detection failed: {str(e)}"
        )


@router.post("/analyze/ocr", response_model=OCRResponse)
async def ocr_pipeline(request: OCRRequest):
    """OCR pipeline endpoint (SERVICE 3).
    
    Executes complete OCR pipeline:
    1. PaddleOCR detection → bounding boxes
    2. PaddleOCR recognition (default)
    3. TrOCR recognition (fallback for low confidence or handwriting)
    4. Merge results preserving original bounding boxes
    
    Args:
        request: OCRRequest with file_id, pre_0_id, or pre_01_id
        
    Returns:
        OCRResponse: OCR results with bounding boxes, text, source, confidence
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Validate request
        if not request.file_id and not request.pre_0_id and not request.pre_01_id:
            raise HTTPException(
                status_code=400,
                detail="At least one ID (file_id, pre_0_id, or pre_01_id) must be provided"
            )
        
        log.info("Starting SERVICE 3 (OCR pipeline)")
        
        # Initialize OCR pipeline
        confidence_threshold = request.paddle_confidence_threshold
        if confidence_threshold is None:
            confidence_threshold = settings.OCR_PADDLE_CONFIDENCE_THRESHOLD
        
        ocr_pipeline = OCRPipeline(
            paddle_confidence_threshold=confidence_threshold,
            use_gpu=settings.USE_GPU
        )
        
        # Run OCR pipeline
        result = ocr_pipeline.process_image(
            file_id=request.file_id,
            pre_0_id=request.pre_0_id,
            pre_01_id=request.pre_01_id
        )
        
        log.info(f"SERVICE 3 complete. Processed {len(result.get('text_regions', []))} text regions")
        
        # TASK 1: Sanitize result to ensure JSON-safe types (convert numpy types)
        sanitized_result = json_safe(result)
        
        return OCRResponse(**sanitized_result)
        
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "file must exist" in error_msg or "not found" in error_msg.lower():
            log.error(f"File not found in SERVICE 3: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=404,
                detail=f"Image file not found. Please ensure at least one of the following exists: file_id, pre_0_id, or pre_01_id. Error: {error_msg}"
            )
        else:
            log.error(f"Invalid request in SERVICE 3: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {error_msg}"
            )
    except FileNotFoundError as e:
        log.error(f"File not found in SERVICE 3: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found. The provided IDs (file_id, pre_0_id, or pre_01_id) do not correspond to existing files. Error: {str(e)}"
        )
    except Exception as e:
        # TASK 3: Make OCR failure non-fatal - always return valid JSON
        log.error(f"OCR pipeline error: {str(e)}", exc_info=True)
        # Return empty but valid response instead of crashing
        sanitized_error_response = json_safe({
            'text_regions': [],
            'dimensions': {'width': 0, 'height': 0},
            'failed': True,
            'error': str(e),
            'metadata': {'error_type': type(e).__name__}
        })
        return OCRResponse(**sanitized_error_response)
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Full traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR pipeline failed: {str(e)}"
        )


@router.post("/analyze/signatures/verify", response_model=SignatureVerificationResponse)
async def verify_signatures(request: SignatureVerificationRequest):
    """Signature verification endpoint (Post-detection verification).
    
    This endpoint verifies already-detected signatures using GPDS verification model.
    It does NOT detect signatures - it only validates detected regions.
    
    Args:
        request: SignatureVerificationRequest with canonical_id or file_id/pre_0_id/pre_01_id
        
    Returns:
        SignatureVerificationResponse: Verified signatures with verification scores
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Determine canonical_id
        canonical_id = request.canonical_id
        if not canonical_id:
            if request.pre_01_id:
                canonical_id = request.pre_01_id
            elif request.pre_0_id:
                canonical_id = request.pre_0_id
            elif request.file_id:
                canonical_id = request.file_id
            else:
                raise HTTPException(
                    status_code=400,
                    detail="At least one ID (canonical_id, file_id, pre_0_id, or pre_01_id) must be provided"
                )
        
        log.info(f"Starting signature verification for canonical_id: {canonical_id}")
        
        # Initialize verification service
        threshold = request.verification_threshold
        if threshold is None:
            threshold = settings.SIGNATURE_VERIFICATION_THRESHOLD
        
        verification_service = SignatureVerificationService(
            verification_threshold=threshold
        )
        
        # Verify signatures from layout JSON
        verified_signatures = verification_service.verify_from_layout(
            canonical_id=canonical_id,
            file_id=request.file_id,
            pre_0_id=request.pre_0_id,
            pre_01_id=request.pre_01_id
        )
        
        # Count valid signatures
        valid_count = sum(1 for sig in verified_signatures if sig.get('is_valid_signature', False))
        
        log.info(f"Signature verification complete: {valid_count}/{len(verified_signatures)} valid signatures")
        
        return SignatureVerificationResponse(
            signatures=verified_signatures,
            total_signatures=len(verified_signatures),
            valid_signatures=valid_count,
            failed=False,
            metadata={
                'verification_threshold': threshold,
                'canonical_id': canonical_id
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        log.error(f"Invalid request in signature verification: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {error_msg}"
        )
    except FileNotFoundError as e:
        log.error(f"File not found in signature verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=f"Layout file not found. Please run layout detection first. Error: {str(e)}"
        )
    except Exception as e:
        log.error(f"Error in signature verification: {str(e)}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Full traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Signature verification failed: {str(e)}"
        )


def _calculate_mean_confidence(detections: Dict[str, List[Dict[str, Any]]]) -> float:
    """Calculate mean confidence across all detections.
    
    Args:
        detections: Detection results from YOLOv8s
        
    Returns:
        float: Mean confidence (0.0 if no detections)
    """
    all_confidences = []
    for detection_list in detections.values():
        for detection in detection_list:
            conf = detection.get('confidence', 0.0)
            if conf > 0:
                all_confidences.append(conf)
    
    if not all_confidences:
        return 0.0
    return sum(all_confidences) / len(all_confidences)


def _should_trigger_preprocessing(layout_result: Dict[str, Any], 
                                  ocr_result: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Determine if preprocessing should be triggered based on results.
    
    Args:
        layout_result: YOLOv8s layout detection results
        ocr_result: Optional OCR results for confidence evaluation
        
    Returns:
        tuple[bool, str]: (should_trigger, reason)
    """
    # Check if preprocessing is explicitly enabled
    if settings.PREPROCESSING_ENABLED and settings.PREPROCESSING_MODE != "none":
        return True, f"Preprocessing explicitly enabled (mode: {settings.PREPROCESSING_MODE})"
    
    # Check YOLOv8s failure conditions
    if settings.PREPROCESSING_TRIGGER_ON_YOLO_FAILURE:
        layout_failed = layout_result.get('failed', False)
        if layout_failed:
            return True, "YOLOv8s returned failed status"
        
        # Count total detections
        total_detections = (
            len(layout_result.get('tables', [])) +
            len(layout_result.get('text_blocks', [])) +
            len(layout_result.get('signatures', [])) +
            len(layout_result.get('checkboxes', [])) +
            len(layout_result.get('handwritten', []))
        )
        
        if total_detections == 0:
            return True, "YOLOv8s returned 0 detections"
    
    # Check YOLOv8s confidence threshold
    if settings.PREPROCESSING_TRIGGER_ON_YOLO_LOW_CONFIDENCE:
        detections = {
            'tables': layout_result.get('tables', []),
            'text_blocks': layout_result.get('text_blocks', []),
            'signatures': layout_result.get('signatures', []),
            'checkboxes': layout_result.get('checkboxes', []),
            'handwritten': layout_result.get('handwritten', [])
        }
        mean_conf = _calculate_mean_confidence(detections)
        if mean_conf < settings.PREPROCESSING_YOLO_CONFIDENCE_THRESHOLD:
            return True, f"YOLOv8s mean confidence ({mean_conf:.3f}) below threshold ({settings.PREPROCESSING_YOLO_CONFIDENCE_THRESHOLD})"
    
    # Check OCR confidence threshold (if OCR has been run)
    if ocr_result and settings.PREPROCESSING_TRIGGER_ON_OCR_LOW_CONFIDENCE:
        text_regions = ocr_result.get('text_regions', [])
        if text_regions:
            confidences = [r.get('confidence', 0.0) for r in text_regions if r.get('confidence', 0.0) > 0]
            if confidences:
                mean_ocr_conf = sum(confidences) / len(confidences)
                if mean_ocr_conf < settings.PREPROCESSING_OCR_CONFIDENCE_THRESHOLD:
                    return True, f"OCR mean confidence ({mean_ocr_conf:.3f}) below threshold ({settings.PREPROCESSING_OCR_CONFIDENCE_THRESHOLD})"
    
    return False, "No preprocessing trigger conditions met"


@router.post("/analyze/rowwise", response_model=RowwiseResponse)
async def analyze_rowwise(request: RowwiseRequest):
    """Unified pipeline endpoint returning row-wise structured output.
    
    Architecture:
    1. YOLOv8s layout detection (on original image, no preprocessing)
    2. Class-based OCR routing:
       - Text_box → PaddleOCR ONLY
       - Handwritten → TrOCR ONLY
    3. Signature handling (presence + crop, NO OCR)
    4. Checkbox handling (presence + checked/unchecked state)
    5. Table-aware row grouping (Y-center clustering, X-ordering)
    6. Row-wise structured JSON output
    
    Args:
        request: RowwiseRequest with file_id, pre_0_id, or pre_01_id
        
    Returns:
        RowwiseResponse: Row-wise structured output with columns per row
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Validate request
        if not request.file_id and not request.pre_0_id and not request.pre_01_id:
            raise HTTPException(
                status_code=400,
                detail="At least one ID (file_id, pre_0_id, or pre_01_id) must be provided"
            )
        
        log.info("Starting Unified Row-wise Pipeline")
        
        # Initialize unified pipeline
        pipeline = UnifiedPipeline(use_gpu=settings.USE_GPU)
        
        # Process through unified pipeline
        result = pipeline.process(
            file_id=request.file_id,
            pre_0_id=request.pre_0_id,
            pre_01_id=request.pre_01_id
        )
        
        # Sanitize result
        sanitized_result = json_safe(result)
        
        log.info(f"Unified Pipeline complete: {result.get('total_rows', 0)} rows, {result.get('total_columns', 0)} total columns")
        
        return RowwiseResponse(**sanitized_result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unified Pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Row-wise analysis failed: {str(e)}"
        )


@router.post("/analyze")
async def analyze_complete(request: AnalyzeRequest):
    """Complete analysis endpoint running all services in sequence.
    
    NEW FLOW (Preprocessing is OPTIONAL):
    1. Run SERVICE 1 (YOLOv8s) on original file_id (NO preprocessing)
    2. Evaluate YOLOv8s results
    3. Conditionally trigger SERVICE 0 (basic) or SERVICE 0.1 (advanced) if needed
    4. If preprocessing triggered: rerun SERVICE 1 on processed image
    5. Run SERVICE 3 (OCR pipeline)
    6. Run SERVICE 4 (signature verification)
    
    Args:
        request: AnalyzeRequest with file_id
        
    Returns:
        dict: Complete analysis results with preprocessing status and all service outputs
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        file_id = request.file_id
        
        log.info(f"Starting complete analysis for file_id: {file_id}")
        log.info(f"Preprocessing enabled: {settings.PREPROCESSING_ENABLED}, mode: {settings.PREPROCESSING_MODE}")
        
        # Find file extension
        file_extension = None
        for ext in ['jpg', 'jpeg', 'png', 'heic', 'heif', 'avif', 'pdf']:
            potential_path = get_raw_file_path(file_id, ext)
            if potential_path.exists():
                file_extension = ext
                break
        
        if not file_extension:
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_id}"
            )
        
        # Track preprocessing status
        preprocessing_applied = False
        pre_0_id = None
        pre_01_id = None
        canonical_id = file_id  # Default to original file
        services_used = []
        fallbacks_used = []
        
        # STEP 1: Run YOLOv8s on ORIGINAL image (NO preprocessing)
        log.info("Step 1: Running SERVICE 1 (YOLOv8s layout detection) on ORIGINAL image (no preprocessing)...")
        layout_service = LayoutService()
        layout_result = layout_service.detect_layout(file_id=file_id)
        services_used.append("yolo")
        
        # Evaluate YOLOv8s results
        should_preprocess, preprocess_reason = _should_trigger_preprocessing(layout_result)
        
        if should_preprocess:
            log.warning(f"⚠️  Preprocessing trigger condition met: {preprocess_reason}")
            log.info("Step 2: Triggering preprocessing fallback...")
            
            # Determine preprocessing mode
            preprocess_mode = settings.PREPROCESSING_MODE
            if preprocess_mode == "none":
                # Auto-select: try basic first, then advanced if needed
                preprocess_mode = "basic"
            
            # STEP 2a: Run basic preprocessing
            if preprocess_mode in ["basic", "advanced"]:
                log.info("Step 2a: Running SERVICE 0 (basic preprocessing)...")
                basic_preprocessor = BasicPreprocessor()
                basic_result = basic_preprocessor.process(file_id, file_extension)
                pre_0_id = basic_result['pre_0_id']
                preprocessing_applied = True
                services_used.append("preprocessing_basic")
                log.info(f"SERVICE 0 complete. pre_0_id: {pre_0_id}")
                
                # STEP 2b: Rerun YOLOv8s on preprocessed image
                log.info("Step 2b: Rerunning SERVICE 1 (YOLOv8s) on preprocessed image...")
                layout_result = layout_service.detect_layout(pre_0_id=pre_0_id)
                canonical_id = pre_0_id
                
                # Check if we still need advanced preprocessing
                if preprocess_mode == "advanced" or (
                    settings.PREPROCESSING_TRIGGER_ON_YOLO_FAILURE and 
                    layout_result.get('failed', False)
                ):
                    log.warning("Basic preprocessing insufficient, triggering advanced preprocessing...")
                    fallbacks_used.append("advanced_preprocessing")
                    
                    # STEP 2c: Run advanced preprocessing
                    log.info("Step 2c: Running SERVICE 0.1 (advanced preprocessing)...")
                    advanced_preprocessor = AdvancedPreprocessor()
                    advanced_result = advanced_preprocessor.process(pre_0_id=pre_0_id)
                    pre_01_id = advanced_result['pre_01_id']
                    services_used.append("preprocessing_advanced")
                    log.info(f"SERVICE 0.1 complete. pre_01_id: {pre_01_id}")
                    
                    # STEP 2d: Rerun YOLOv8s on advanced preprocessed image
                    log.info("Step 2d: Rerunning SERVICE 1 (YOLOv8s) on advanced preprocessed image...")
                    layout_result = layout_service.detect_layout(pre_01_id=pre_01_id)
                    canonical_id = pre_01_id
        else:
            log.info("✅ YOLOv8s results acceptable - skipping preprocessing")
        
        # Log YOLOv8s results
        services_failed = layout_result.get('failed', False)
        failure_reason = layout_result.get('failure_reason')
        
        if services_failed:
            log.warning(f"SERVICE 1 failed: {failure_reason}")
        else:
            log.info(f"SERVICE 1 complete. Detected {len(layout_result.get('tables', []))} tables, "
                    f"{len(layout_result.get('text_blocks', []))} text blocks, "
                    f"{len(layout_result.get('signatures', []))} signatures, "
                    f"{len(layout_result.get('checkboxes', []))} checkboxes, "
                    f"{len(layout_result.get('handwritten', []))} handwritten")
        
        services_failed = layout_result.get('failed', False)
        failure_reason = layout_result.get('failure_reason')
        
        if services_failed:
            log.warning(f"SERVICE 1 failed: {failure_reason}")
        else:
            log.info(f"SERVICE 1 complete. Detected {len(layout_result.get('tables', []))} tables, "
                    f"{len(layout_result.get('text_blocks', []))} text blocks, "
                    f"{len(layout_result.get('signatures', []))} signatures, "
                    f"{len(layout_result.get('checkboxes', []))} checkboxes")
        
        # STEP 3: Run SERVICE 3 (OCR Pipeline) - PaddleOCR detection and extraction
        log.info("Step 3: Running SERVICE 3 (OCR pipeline - PaddleOCR detection and extraction)...")
        ocr_result = None
        ocr_failed = False
        ocr_error = None
        
        try:
            ocr_pipeline = OCRPipeline(
                paddle_confidence_threshold=settings.OCR_PADDLE_CONFIDENCE_THRESHOLD,
                use_gpu=settings.USE_GPU
            )
            # Use canonical_id (which may be file_id, pre_0_id, or pre_01_id)
            ocr_result = ocr_pipeline.process_image(
                file_id=file_id if canonical_id == file_id else None,
                pre_0_id=pre_0_id if canonical_id == pre_0_id else None,
                pre_01_id=pre_01_id if canonical_id == pre_01_id else None
            )
            services_used.append("paddleocr")
            
            if ocr_result.get('failed', False):
                ocr_failed = True
                ocr_error = ocr_result.get('error', 'Unknown OCR error')
                log.warning(f"SERVICE 3 (OCR) failed: {ocr_error}")
            else:
                log.info(f"SERVICE 3 complete. Processed {len(ocr_result.get('text_regions', []))} text regions")
                
                # Check if OCR results should trigger preprocessing (post-OCR evaluation)
                if not preprocessing_applied:
                    should_preprocess_ocr, preprocess_reason_ocr = _should_trigger_preprocessing(
                        layout_result, ocr_result
                    )
                    if should_preprocess_ocr and preprocess_reason_ocr.startswith("OCR"):
                        log.warning(f"⚠️  Post-OCR preprocessing trigger: {preprocess_reason_ocr}")
                        # Note: We don't rerun everything here, just log it for future optimization
        except Exception as e:
            ocr_failed = True
            ocr_error = str(e)
            log.error(f"SERVICE 3 (OCR) error: {str(e)}", exc_info=True)
        
        # STEP 4: Run SERVICE 4 (Signature Verification) - GPDS verification
        log.info("Step 4: Running SERVICE 4 (signature verification - GPDS)...")
        signature_verification_result = None
        signature_verification_failed = False
        signature_verification_error = None
        
        try:
            verification_service = SignatureVerificationService(
                verification_threshold=settings.SIGNATURE_VERIFICATION_THRESHOLD
            )
            signature_verification_result = verification_service.verify_from_layout(
                canonical_id=canonical_id,
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            services_used.append("signature_verification")
            
            valid_count = sum(1 for sig in signature_verification_result if sig.get('is_valid_signature', False))
            log.info(f"SERVICE 4 complete. Verified {valid_count}/{len(signature_verification_result)} signatures")
        except Exception as e:
            signature_verification_failed = True
            signature_verification_error = str(e)
            log.warning(f"SERVICE 4 (signature verification) error: {str(e)}", exc_info=True)
            # Signature verification failure is not critical - continue
        
        # Prepare final result with preprocessing status
        result = {
            "file_id": file_id,
            "processed_id": canonical_id if preprocessing_applied else None,  # processed_id only if preprocessing was applied
            "preprocessing_applied": preprocessing_applied,
            "pre_0_id": pre_0_id,
            "pre_01_id": pre_01_id,
            "canonical_id": canonical_id,
            "status": "complete" if not services_failed else "complete_with_fallback",
            "message": "Analysis complete" if not services_failed else "Analysis complete with preprocessing fallback",
            "services_used": services_used,
            "fallbacks_used": fallbacks_used,
            "layout": {
                "tables": layout_result.get('tables', []),
                "text_blocks": layout_result.get('text_blocks', []),
                "signatures": layout_result.get('signatures', []),
                "checkboxes": layout_result.get('checkboxes', []),
                "handwritten": layout_result.get('handwritten', [])
            },
            "layout_failed": services_failed,
            "layout_failure_reason": failure_reason if services_failed else None,
            "ocr": {
                "text_regions": json_safe(ocr_result.get('text_regions', [])) if ocr_result else [],
                "dimensions": json_safe(ocr_result.get('dimensions', {})) if ocr_result else {},
                "failed": ocr_failed,
                "error": ocr_error if ocr_failed else None,
                "metadata": json_safe(ocr_result.get('metadata', {})) if ocr_result else {}
            },
            "signature_verification": {
                "signatures": signature_verification_result if signature_verification_result else [],
                "total_signatures": len(signature_verification_result) if signature_verification_result else 0,
                "valid_signatures": sum(1 for sig in (signature_verification_result or []) if sig.get('is_valid_signature', False)),
                "failed": signature_verification_failed,
                "error": signature_verification_error if signature_verification_failed else None
            }
        }
        
        log.info(f"Complete analysis finished for file_id: {file_id}")
        return result
        
    except Exception as e:
        log.error(f"Error in complete analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )








