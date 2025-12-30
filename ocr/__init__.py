"""OCR module (Service 3).

This module provides:
- PaddleOCR recognition engine (isolated)
- TrOCR recognition engine (isolated)
- Handwriting classifier
- OCR pipeline orchestrator
"""

from ocr.pipeline import OCRPipeline
from ocr.paddle_recognizer import PaddleOCRRecognizer
from ocr.trocr_recognizer import TrOCRRecognizer
from ocr.handwriting_classifier import HandwritingClassifier

__all__ = [
    'OCRPipeline',
    'PaddleOCRRecognizer',
    'TrOCRRecognizer',
    'HandwritingClassifier'
]










