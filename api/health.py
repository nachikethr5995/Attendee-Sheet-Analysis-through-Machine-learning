"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import torch

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    dependencies: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        HealthResponse: System status and dependency checks
    """
    dependencies = {
        "opencv": "available",
        "pillow": "available",
        "torch": "available",
        "gpu": "available" if torch.cuda.is_available() else "unavailable",
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        dependencies=dependencies,
    )










