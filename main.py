"""Main FastAPI application for AIME Backend."""

# Import core first to fix Python path (must be before any other imports)
import core  # This runs the path fix in core/__init__.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings, ensure_directories
from core.logging import log
from api import health, upload, analyze

# Ensure directories exist
ensure_directories()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    log.info("AIME Backend starting up...")
    log.info(f"Storage root: {settings.STORAGE_ROOT}")
    log.info(f"API running on {settings.API_HOST}:{settings.API_PORT}")
    yield
    # Shutdown
    log.info("AIME Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AIME Backend",
    description="OCR & Computer Vision Pipeline for Life Sciences",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc",  # ReDoc documentation
    openapi_url="/openapi.json",  # OpenAPI schema
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(analyze.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )










