"""Unified layout service integrating all layout detectors.

This is the main SERVICE 1 implementation using hybrid layout detection.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import (
    resolve_canonical_id,
    load_image_from_canonical_id,
    get_intermediate_json_path
)
from layout.yolov8_layout_detector import YOLOv8LayoutDetector
from layout.signature_detector import SignatureDetector
from layout.checkbox_detector import CheckboxDetector
from layout.paddleocr_detector import PaddleOCRTextDetector
from layout.fusion_engine import FusionEngine


class LayoutService:
    """Unified layout detection service using YOLOv8s.
    
    Pipeline:
    1. YOLOv8s (primary, trainable) - detects all layout elements
    2. Check acceptance rules
    3. Fusion and output
    """
    
    def __init__(self, yolo_model_version: str = "v1.0"):
        """Initialize layout service with YOLOv8s.
        
        Args:
            yolo_model_version: YOLOv8 model version string (default: "v1.0")
        """
        log.info("Initializing Layout Service (SERVICE 1) - Training-Ready Architecture...")
        
        # PRIMARY: YOLOv8s layout detector (training-ready)
        log.info("Initializing YOLOv8s layout detector...")
        self.yolo_detector = YOLOv8LayoutDetector(
            model_version=yolo_model_version,
            confidence_threshold=0.01,  # TASK 3: Lowered to 0.01 for debugging
            iou_threshold=0.7,  # TASK 3: NMS threshold
            use_gpu=settings.USE_GPU
        )
        
        # Legacy detectors (kept for compatibility, may be replaced by YOLOv8s)
        self.signature_detector = SignatureDetector(confidence_threshold=0.5)
        self.checkbox_detector = CheckboxDetector(confidence_threshold=0.4)
        self.paddleocr_detector = PaddleOCRTextDetector(use_angle_cls=False, lang='en')
        
        # Initialize fusion engine
        self.fusion_engine = FusionEngine(iou_threshold=0.85)
        
        log.info("Layout Service initialized")
        log.info(f"  YOLOv8s: {'available' if self.yolo_detector.is_available() else 'unavailable'}")
    
    def detect_layout(self, file_id: Optional[str] = None,
                     pre_0_id: Optional[str] = None,
                     pre_01_id: Optional[str] = None) -> Dict[str, Any]:
        """Detect layout elements in image using hybrid ensemble.
        
        Args:
            file_id: Original file identifier (fallback)
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            dict: Layout detection results with failure detection
        """
        # Load image using canonical ID (with automatic fallback)
        try:
            image = load_image_from_canonical_id(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            log.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            log.error(f"Failed to load image from any available source: {str(e)}")
            raise
        
        # Resolve canonical ID after successful load (for storage purposes)
        canonical_id = resolve_canonical_id(
            file_id=file_id,
            pre_0_id=pre_0_id,
            pre_01_id=pre_01_id
        )
        
        log.info(f"Starting SERVICE 1 (YOLOv8s) with canonical_id: {canonical_id}")
        
        # Track which models were used
        models_used = []
        detection_times = {}
        errors = {}
        fallback_used = False
        
        # STEP 1: Run YOLOv8s
        log.info("Step 1: Running YOLOv8s layout detector...")
        yolo_results = {
            'tables': [],
            'text_blocks': [],
            'signatures': [],
            'checkboxes': [],
            'handwritten': []  # Handwritten text regions
        }
        
        try:
            if self.yolo_detector.is_available():
                yolo_results = self.yolo_detector.detect(image)
                models_used.append("YOLOv8s")
                
                log.info(f"YOLOv8s detected: {len(yolo_results['tables'])} tables, "
                        f"{len(yolo_results['text_blocks'])} text blocks, "
                        f"{len(yolo_results['signatures'])} signatures, "
                        f"{len(yolo_results['checkboxes'])} checkboxes, "
                        f"{len(yolo_results['handwritten'])} handwritten")
                
                # TASK 6: Check acceptance rules (informational - won't block pipeline)
                is_accepted, reason = self.yolo_detector.is_output_accepted(yolo_results)
                log.info(f"YOLOv8s acceptance check: {reason}")
                
                if not is_accepted:
                    log.warning(f"⚠️  YOLOv8s returned 0 detections: {reason}")
                    log.warning("   Will use PaddleOCR fallback for layout detection (TASK 5)")
            else:
                log.error("YOLOv8s not available - cannot proceed")
                errors['YOLOv8s'] = "YOLOv8s detector not available"
                
        except Exception as e:
            log.error(f"YOLOv8s detection failed: {str(e)}", exc_info=True)
            errors['YOLOv8s'] = str(e)
        
        # Count YOLO detections for fallback decision
        total_yolo_detections = (
            len(yolo_results['tables']) +
            len(yolo_results['text_blocks']) +
            len(yolo_results['signatures']) +
            len(yolo_results['checkboxes']) +
            len(yolo_results.get('handwritten', []))
        )
        
        # Use YOLOv8s results
        merged_results = {
            'tables': yolo_results['tables'],
            'text_blocks': yolo_results['text_blocks'],
            'signatures': yolo_results['signatures'],
            'checkboxes': yolo_results['checkboxes'],
            'handwritten': yolo_results.get('handwritten', [])
        }
        
        # Step 2: Run GPDS signature detector (secondary detector)
        log.info("Running GPDS signature detector...")
        try:
            signature_results = self.signature_detector.detect(image)
            models_used.append("GPDS")
            log.info(f"GPDS: {len(signature_results)} signatures")
            if len(signature_results) == 0:
                log.info("   No signatures detected by GPDS (may not exist in image)")
        except Exception as e:
            log.warning(f"GPDS signature detection failed: {str(e)}")
            signature_results = []
            errors['GPDS'] = str(e)
            # GPDS failure is not critical - continue
        
        # Step 3: Run checkbox detector (secondary detector)
        log.info("Running checkbox detector...")
        try:
            checkbox_results = self.checkbox_detector.detect(image)
            models_used.append("checkbox_detector")
            log.info(f"Checkbox detector: {len(checkbox_results)} checkboxes")
            if len(checkbox_results) == 0:
                log.info("   No checkboxes detected (may not exist in image)")
        except Exception as e:
            log.warning(f"Checkbox detection failed: {str(e)}")
            checkbox_results = []
            errors['checkbox_detector'] = str(e)
            # Checkbox failure is not critical - continue
        
        # Step 4: PaddleOCR - ONLY for fallback when YOLO fails (NOT for layout fusion)
        # PaddleOCR is used in OCR pipeline, not layout detection
        # Only use it as fallback when YOLO returns 0 detections
        if total_yolo_detections == 0:
            log.warning("⚠️  YOLOv8s returned 0 detections, using PaddleOCR fallback for layout")
            try:
                paddleocr_results = self.paddleocr_detector.detect(image)
                paddleocr_results = self.paddleocr_detector.merge_small_regions(paddleocr_results)
                models_used.append("PaddleOCR_fallback")
                log.info(f"PaddleOCR fallback: {len(paddleocr_results)} text regions")
                
                # Convert PaddleOCR detections to text_block layout elements
                for i, paddle_detection in enumerate(paddleocr_results):
                    text_block = {
                        'bbox': paddle_detection.get('bbox', []),
                        'confidence': paddle_detection.get('confidence', 0.7),
                        'class': 'text_block',
                        'source': 'PaddleOCR_fallback',
                        'original_source': paddle_detection.get('source', 'PaddleOCR'),
                        'type': 'text_block',
                        'model_version': 'fallback',
                        'is_custom_model': False,
                        'fallback_reason': 'YOLOv8s_zero_detections'
                    }
                    merged_results['text_blocks'].append(text_block)
                
                log.info(f"✅ PaddleOCR fallback: Added {len(paddleocr_results)} text_block(s) to layout")
                fallback_used = True
            except Exception as e:
                log.warning(f"PaddleOCR fallback failed: {str(e)}")
                errors['PaddleOCR'] = str(e)
        else:
            log.info(f"YOLOv8s has {total_yolo_detections} detections - PaddleOCR not needed for layout")
        
        # Step 5: Fuse all results (EXCLUDE PaddleOCR - it's only used as fallback)
        log.info("Fusing all detector outputs...")
        
        # PaddleOCR is NOT included in fusion - it's only used as fallback when YOLO fails
        # When YOLO works, PaddleOCR is skipped entirely
        # When YOLO fails, PaddleOCR results are already added to merged_results['text_blocks']
        # So we pass empty list to fusion engine
        try:
            fused_layout = self.fusion_engine.fuse(
                yolo_results=merged_results,
                signature_results=signature_results,
                checkbox_results=checkbox_results,
                paddleocr_results=[],  # Never include PaddleOCR in fusion (only fallback)
                image_dims=image.size
            )
        except Exception as e:
            log.error(f"Fusion failed: {str(e)}", exc_info=True)
            # Fallback to YOLOv8s results only
            fused_layout = {
                'tables': merged_results.get('tables', []),
                'text_blocks': merged_results.get('text_blocks', []),
                'signatures': signature_results,
                'checkboxes': checkbox_results,
                'handwritten': merged_results.get('handwritten', [])
            }
            errors['fusion'] = str(e)
        
        # Step 6: Store layout JSON using canonical ID
        layout_path = get_intermediate_json_path(canonical_id, 'layout')
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate confidence summary
        all_confidences = []
        for det_list in fused_layout.values():
            for det in det_list:
                all_confidences.append(det.get('confidence', 0.0))
        
        confidence_summary = {
            'mean': float(np.mean(all_confidences)) if all_confidences else 0.0,
            'min': float(np.min(all_confidences)) if all_confidences else 0.0,
            'max': float(np.max(all_confidences)) if all_confidences else 0.0,
            'count': len(all_confidences)
        }
        
        layout_data = {
            'canonical_id': canonical_id,
            'tables': fused_layout['tables'],
            'text_blocks': fused_layout['text_blocks'],
            'signatures': fused_layout['signatures'],
            'checkboxes': fused_layout['checkboxes'],
            'handwritten': fused_layout.get('handwritten', []),
            'dimensions': {
                'width': image.size[0],
                'height': image.size[1]
            },
            'metadata': {
                'models_used': models_used,
                'detection_times': detection_times,
                'confidence_summary': confidence_summary,
                'errors': errors if errors else None
            }
        }
        
        # Save to JSON
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Layout detection complete. Saved to: {layout_path}")
        log.info(f"Total detections: {len(fused_layout['tables'])} tables, "
                f"{len(fused_layout['text_blocks'])} text blocks, "
                f"{len(fused_layout['signatures'])} signatures, "
                f"{len(fused_layout['checkboxes'])} checkboxes, "
                f"{len(fused_layout.get('handwritten', []))} handwritten")
        
        # Check overall failure
        # Consider it failed if:
        # 1. YOLOv8s failed
        # 2. No detections at all from any model
        total_detections = (len(fused_layout['tables']) + 
                           len(fused_layout['text_blocks']) + 
                           len(fused_layout['signatures']) + 
                           len(fused_layout['checkboxes']) +
                           len(fused_layout.get('handwritten', [])))
        
        if 'YOLOv8s' in errors:
            overall_failed = True
            failure_reason = f"YOLOv8s failed: {errors.get('YOLOv8s', 'unknown')}"
        elif total_detections == 0:
            overall_failed = True
            failure_reason = "No layout elements detected from any model"
        else:
            overall_failed = False
            failure_reason = None
        
        return {
            'canonical_id': canonical_id,
            'tables': fused_layout['tables'],
            'text_blocks': fused_layout['text_blocks'],
            'signatures': fused_layout['signatures'],
            'checkboxes': fused_layout['checkboxes'],
            'handwritten': fused_layout.get('handwritten', []),
            'dimensions': layout_data['dimensions'],
            'failed': overall_failed,
            'failure_reason': failure_reason,
            'layout_path': str(layout_path.relative_to(Path(settings.STORAGE_ROOT).parent)),
            'metadata': layout_data['metadata']
        }









