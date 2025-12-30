"""Training-Ready YOLOv8s Layout Detector.

This module implements a production-grade, training-ready YOLOv8s detector
for document layout analysis. It supports:
- Custom model training and fine-tuning
- Model versioning and swapping
- Fixed class mappings (never change)
- Fallback detection logic
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from core.logging import log
from core.config import settings
from core.utils import image_to_array

# YOLOv8 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("ultralytics not installed. YOLOv8s detection will be unavailable.")


# FIXED CLASS MAPPINGS - NEVER CHANGE
# These indices must remain consistent across all training runs
CLASS_MAPPINGS = {
    0: 'text_block',
    1: 'table',
    2: 'signature',
    3: 'checkbox'
}

# Reverse mapping for class name to ID
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAPPINGS.items()}

# Acceptance thresholds for YOLO output
# If any condition is met, YOLO output is accepted
# Note: These are informational - results are used regardless of acceptance
ACCEPTANCE_THRESHOLDS = {
    'text_block': 0.35,  # Lowered from 0.45
    'table': 0.30,       # Lowered from 0.40
    'signature': 0.30,   # Lowered from 0.40
    'checkbox': 0.30     # Lowered from 0.40
}


class YOLOv8LayoutDetector:
    """Training-ready YOLOv8s layout detector.
    
    This detector:
    - Uses YOLOv8s as primary model (trainable)
    - Supports custom trained models
    - Maintains fixed class mappings
    - Implements acceptance rules
    - Supports model versioning
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_version: str = "v1.0",
                 confidence_threshold: float = 0.01,  # Lowered to 0.01 for debugging
                 iou_threshold: float = 0.7,  # NMS threshold
                 use_gpu: Optional[bool] = None):
        """Initialize YOLOv8s layout detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
                       If None, uses default path: models/yolo_layout/yolov8s_layout.pt
                       Falls back to yolov8s.pt (COCO pretrained) if custom not found
            model_version: Model version string (e.g., "v1.0", "v2.1")
            confidence_threshold: Minimum confidence for detections (default: 0.01 for debugging)
            iou_threshold: NMS IoU threshold (default: 0.7)
            use_gpu: Whether to use GPU. If None, uses settings.USE_GPU
        """
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.model = None
        self.device = None
        self.is_custom_model = False
        self.is_obb_model = False  # Track if model is OBB type
        
        if not YOLO_AVAILABLE:
            log.warning("YOLOv8 not available. Install with: pip install ultralytics")
            return
        
        # Determine device
        try:
            import torch
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                log.info(f"YOLOv8s using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                log.info("YOLOv8s using CPU")
        except ImportError:
            self.device = "cpu"
            log.info("YOLOv8s using CPU (PyTorch not available for device check)")
        
        # Load model
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load YOLOv8 model with fallback strategy.
        
        Strategy:
        1. Try custom model path (if provided)
        2. Try default custom model location
        3. Fall back to COCO pretrained yolov8s.pt
        """
        try:
            # Step 1: Try provided model path
            if model_path and Path(model_path).exists():
                log.info(f"Loading YOLOv8 model from: {model_path}")
                self.model = YOLO(str(model_path))
                self.is_custom_model = True
                log.info(f"✅ Custom YOLOv8 model loaded (version: {self.model_version})")
                return
            
            # Step 2: Try default custom model locations (multiple possible names)
            model_dir = Path(settings.MODELS_ROOT) / "yolo_layout"
            possible_model_names = [
                "yolov8s_layout.pt",
                "best.pt",
                f"yolov8s_layout_{self.model_version}.pt"
            ]
            
            for model_name in possible_model_names:
                model_path = model_dir / model_name
                if model_path.exists():
                    log.info(f"Loading custom YOLOv8 model from: {model_path}")
                    self.model = YOLO(str(model_path))
                    self.is_custom_model = True
                    log.info(f"✅ Custom YOLOv8 model loaded: {model_name} (version: {self.model_version})")
                    
                    # Check model type (TASK 1: Verify model-task compatibility)
                    if hasattr(self.model, 'model'):
                        model_type = type(self.model.model).__name__
                        log.info(f"Model type: {model_type}")
                        if 'OBB' in model_type or 'OBBModel' in str(type(self.model.model)):
                            self.is_obb_model = True
                            log.warning("⚠️  Model is an OBB (Oriented Bounding Box) model!")
                            log.warning("   OBB models output rotated boxes, not axis-aligned boxes")
                            log.warning("   Code will convert OBB boxes to axis-aligned boxes")
                        else:
                            self.is_obb_model = False
                            log.info("✅ Model is standard bbox (axis-aligned) model")
                    
                    # Check if model task matches (detect vs obb)
                    if hasattr(self.model, 'task'):
                        task = self.model.task
                        log.info(f"Model task: {task}")
                        if task == 'obb' and not self.is_obb_model:
                            log.warning("⚠️  Model task is 'obb' but model type doesn't match!")
                        elif task != 'obb' and self.is_obb_model:
                            log.warning("⚠️  Model type is OBB but task is not 'obb'!")
                    
                    # Verify model classes
                    if hasattr(self.model, 'names'):
                        log.info(f"Model classes: {self.model.names}")
                        # Check if it matches our expected classes
                        expected_classes = set(CLASS_MAPPINGS.values())
                        model_classes = set(self.model.names.values()) if isinstance(self.model.names, dict) else set()
                        
                        # Check if we can map model classes to expected classes
                        class_name_mapping = {
                            'Checkbox': 'checkbox',
                            'Table': 'table',
                            'Signature': 'signature',
                            'Text_box': 'text_block',
                            'Handwritten': 'text_block'
                        }
                        mappable_classes = set(class_name_mapping.keys())
                        if mappable_classes.intersection(model_classes):
                            log.info("✅ Model classes can be mapped to expected layout classes")
                        else:
                            log.warning(f"⚠️  Model classes may not match expected layout classes")
                            log.warning(f"Expected: {expected_classes}, Model has: {model_classes}")
                    
                    return
            
            # Step 3: Fall back to COCO pretrained (NOT RECOMMENDED)
            log.error("❌ Custom YOLOv8 layout model not found!")
            log.error(f"   Searched in: {model_dir}")
            log.error(f"   Looked for: {', '.join(possible_model_names)}")
            log.error("⚠️  Falling back to COCO pretrained yolov8s.pt")
            log.error("⚠️  COCO model is NOT trained for document layout detection!")
            log.error("⚠️  It will NOT detect tables, text blocks, signatures, or checkboxes correctly!")
            log.error("⚠️  Train a custom model using TRAINING_GUIDE.md")
            log.info("Loading COCO pretrained yolov8s.pt...")
            try:
                self.model = YOLO('yolov8s.pt')
                self.is_custom_model = False
                log.warning("⚠️  COCO pretrained YOLOv8s loaded (WILL NOT WORK FOR LAYOUT DETECTION)")
            except Exception as e:
                log.error(f"Failed to load COCO pretrained model: {str(e)}")
                self.model = None
            
        except Exception as e:
            log.error(f"Failed to load YOLOv8 model: {str(e)}", exc_info=True)
            self.model = None
    
    def detect(self, image: Image.Image) -> Dict[str, List[Dict[str, Any]]]:
        """Detect layout elements in image using YOLOv8s.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            dict: Detection results with keys: 'tables', 'text_blocks', 'signatures', 'checkboxes'
            Each list contains dicts with 'bbox', 'confidence', 'class', 'source', 'model_version'
        """
        if not self.model:
            log.warning("YOLOv8 model not available, returning empty detections")
            return self._empty_detections()
        
        try:
            # TASK 2: Inspect input preprocessing
            # Convert PIL to numpy array (BGR for YOLO)
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            width, height = image.size
            log.info(f"Input image: {width}x{height}, mode: {image.mode}, BGR shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
            log.info(f"BGR value range: [{img_bgr.min()}, {img_bgr.max()}]")
            
            # Save debug image (TASK 2: Visual inspection)
            try:
                debug_dir = Path(settings.STORAGE_ROOT) / "debug" / "yolo_input"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"yolo_input_{id(image)}.jpg"
                cv2.imwrite(str(debug_path), img_bgr)
                log.info(f"Saved debug input image to: {debug_path}")
            except Exception as e:
                log.warning(f"Failed to save debug image: {e}")
            
            # TASK 3: Lower confidence and NMS thresholds
            # Run YOLO inference with configured thresholds
            log.info(f"Running YOLOv8s inference: conf={self.confidence_threshold}, iou={self.iou_threshold}")
            try:
                results = self.model(
                    img_bgr, 
                    conf=self.confidence_threshold, 
                    iou=self.iou_threshold,
                    verbose=False, 
                    device=self.device
                )
                log.info(f"Inference completed, results type: {type(results)}, length: {len(results) if results else 0}")
            except Exception as e:
                log.error(f"YOLOv8s inference failed: {str(e)}", exc_info=True)
                return self._empty_detections()
            
            # Initialize detection results
            detections = self._empty_detections()
            
            # TASK 3: Log raw predictions BEFORE filtering
            # Also run with very low threshold for debugging
            try:
                results_low_conf = self.model(img_bgr, conf=0.01, iou=0.7, verbose=False, device=self.device)
                if results_low_conf and len(results_low_conf) > 0:
                    result_low = results_low_conf[0]
                    # Check both boxes and obb
                    boxes_low = result_low.boxes if hasattr(result_low, 'boxes') else None
                    obb_low = result_low.obb if hasattr(result_low, 'obb') else None
                    
                    low_conf_count = 0
                    if boxes_low is not None:
                        low_conf_count = len(boxes_low)
                    elif obb_low is not None:
                        low_conf_count = len(obb_low)
                    
                    log.info(f"YOLOv8s RAW detections (conf≥0.01, BEFORE filtering): {low_conf_count} boxes")
                    if low_conf_count > 0:
                        # Log class distribution
                        class_counts = {}
                        boxes_to_check = boxes_low if boxes_low is not None else obb_low
                        for box in boxes_to_check:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = self.model.names.get(class_id, f"class_{class_id}")
                            if class_name not in class_counts:
                                class_counts[class_name] = {'count': 0, 'confidences': []}
                            class_counts[class_name]['count'] += 1
                            class_counts[class_name]['confidences'].append(confidence)
                        
                        log.info("Raw class distribution (conf≥0.01):")
                        for class_name, data in class_counts.items():
                            avg_conf = sum(data['confidences']) / len(data['confidences'])
                            min_conf = min(data['confidences'])
                            max_conf = max(data['confidences'])
                            log.info(f"  {class_name}: {data['count']} detections, confidence: {min_conf:.3f}-{max_conf:.3f} (avg: {avg_conf:.3f})")
                            
                            # Check if signatures/checkboxes are being detected but filtered
                            if class_name in ['Signature', 'Checkbox']:
                                log.warning(f"⚠️  {class_name} detected at low confidence - check if they're being filtered out")
                    else:
                        log.error("⚠️  YOLOv8s detected 0 boxes even at conf=0.01!")
                        log.error("   This indicates MODEL-DATA MISMATCH or PREPROCESSING ISSUE")
                        log.error("   Possible causes:")
                        log.error("   1. Model is OBB but dataset was standard bbox (or vice versa)")
                        log.error("   2. Image preprocessing doesn't match training")
                        log.error("   3. Model not trained for this document type")
                        log.error("   4. Model weights corrupted")
            except Exception as e:
                log.warning(f"Low confidence inference failed (non-critical): {str(e)}")
            
            if results and len(results) > 0:
                result = results[0]
                
                # TASK 1: Handle OBB vs standard bbox
                is_obb = self.is_obb_model or (hasattr(result, 'obb') and result.obb is not None)
                boxes_to_process = None
                
                if is_obb:
                    boxes_to_process = result.obb if hasattr(result, 'obb') and result.obb is not None else None
                    log.info("Processing OBB (Oriented Bounding Box) detections")
                else:
                    boxes_to_process = result.boxes if hasattr(result, 'boxes') and result.boxes is not None else None
                    log.info("Processing standard bbox detections")
                
                if boxes_to_process is None or len(boxes_to_process) == 0:
                    log.warning(f"YOLOv8s filtered detections (conf≥{self.confidence_threshold}): 0 boxes")
                    log.warning(f"   Try lowering confidence threshold (current: {self.confidence_threshold})")
                    return detections
                
                filtered_count = len(boxes_to_process)
                log.info(f"YOLOv8s filtered detections (conf≥{self.confidence_threshold}): {filtered_count} boxes")
                
                # Process each detection
                processed_count = 0
                skipped_count = 0
                boxes_to_process = result.obb if is_obb else result.boxes
                
                if boxes_to_process is None or len(boxes_to_process) == 0:
                    log.warning("No boxes to process in result")
                    return detections
                
                for box in boxes_to_process:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # TASK 1: Get bounding box coordinates (handle OBB vs standard)
                    if is_obb:
                        # OBB format: get oriented box and convert to axis-aligned
                        # OBB boxes have xyxyxyxy format (4 points) or rbox format
                        if hasattr(box, 'xyxyxyxy') and box.xyxyxyxy is not None:
                            # Oriented box with 4 corner points
                            points = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
                            x_coords = points[:, 0]
                            y_coords = points[:, 1]
                            x1, y1 = float(x_coords.min()), float(y_coords.min())
                            x2, y2 = float(x_coords.max()), float(y_coords.max())
                        elif hasattr(box, 'xywhr') and box.xywhr is not None:
                            # Rotated box format (center_x, center_y, width, height, rotation)
                            # Convert to axis-aligned by taking bounding box of rotated rectangle
                            xywhr = box.xywhr[0].cpu().numpy()
                            cx, cy, w, h, r = xywhr[0], xywhr[1], xywhr[2], xywhr[3], xywhr[4]
                            # Approximate: use width/height as axis-aligned box
                            x1, y1 = float(cx - w/2), float(cy - h/2)
                            x2, y2 = float(cx + w/2), float(cy + h/2)
                        elif hasattr(box, 'xyxy') and box.xyxy is not None:
                            # Fallback: try xyxy (some OBB models still provide this)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        else:
                            log.warning(f"OBB box has no recognizable format, skipping")
                            skipped_count += 1
                            continue
                    else:
                        # Standard format: xyxy
                        if hasattr(box, 'xyxy') and box.xyxy is not None:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        else:
                            log.warning(f"Standard box has no xyxy format, skipping")
                            skipped_count += 1
                            continue
                    
                    # Normalize coordinates to 0-1 range and clamp to valid range
                    bbox_normalized = [
                        max(0.0, min(1.0, float(x1 / width))),   # Clamp x1 to [0, 1]
                        max(0.0, min(1.0, float(y1 / height))),  # Clamp y1 to [0, 1]
                        max(0.0, min(1.0, float(x2 / width))),   # Clamp x2 to [0, 1]
                        max(0.0, min(1.0, float(y2 / height)))   # Clamp y2 to [0, 1]
                    ]
                    
                    # Validate bbox (x2 > x1, y2 > y1)
                    if bbox_normalized[2] <= bbox_normalized[0] or bbox_normalized[3] <= bbox_normalized[1]:
                        log.warning(f"Invalid bbox after normalization: {bbox_normalized}, skipping")
                        skipped_count += 1
                        continue
                    
                    # TASK 4: Validate class index mapping
                    if self.is_custom_model:
                        # Get the actual class name from the model
                        model_class_name = self.model.names.get(class_id, 'unknown')
                        
                        # TASK 4: Map model's class names to our standard class names
                        # Expected model classes: {0: 'Checkbox', 1: 'Handwritten', 2: 'Signature', 3: 'Table', 4: 'Text_box'}
                        # We want: text_block, table, signature, checkbox, handwritten
                        class_name_mapping = {
                            'Checkbox': 'checkbox',
                            'Table': 'table',
                            'Signature': 'signature',
                            'Text_box': 'text_block',
                            'Handwritten': 'handwritten'  # Keep handwritten as separate class
                        }
                        
                        class_name = class_name_mapping.get(model_class_name, 'unknown')
                        if class_name == 'unknown':
                            log.warning(f"⚠️  Unknown class mapping: class_id={class_id}, model_class={model_class_name}")
                            log.warning(f"   Model classes: {self.model.names}")
                            log.warning(f"   This detection will be skipped")
                        log.debug(f"Custom model: class_id={class_id} ({model_class_name}) -> {class_name}, confidence={confidence:.3f}")
                    else:
                        # COCO model - map to text_block as placeholder
                        # This is only for testing - custom model required for production
                        coco_class = result.names.get(class_id, 'unknown')
                        log.warning(f"⚠️  Using COCO model! Detected COCO class: {coco_class} (ID: {class_id})")
                        log.warning("⚠️  COCO model is NOT trained for document layout - train a custom model!")
                        class_name = 'text_block'  # Placeholder
                    
                    # Create detection entry
                    detection = {
                        'bbox': bbox_normalized,
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id,
                        'source': 'YOLOv8s',
                        'model_version': self.model_version,
                        'is_custom_model': self.is_custom_model
                    }
                    
                    # Add to appropriate list
                    try:
                        if class_name == 'text_block':
                            detections['text_blocks'].append(detection)
                            processed_count += 1
                        elif class_name == 'table':
                            detections['tables'].append(detection)
                            processed_count += 1
                        elif class_name == 'signature':
                            detections['signatures'].append(detection)
                            processed_count += 1
                            log.info(f"✅ Signature detected: confidence={confidence:.3f}, bbox={bbox_normalized}")
                        elif class_name == 'checkbox':
                            detections['checkboxes'].append(detection)
                            processed_count += 1
                            log.info(f"✅ Checkbox detected: confidence={confidence:.3f}, bbox={bbox_normalized}")
                        elif class_name == 'handwritten':
                            detections['handwritten'].append(detection)
                            processed_count += 1
                            log.info(f"✅ Handwritten detected: confidence={confidence:.3f}, bbox={bbox_normalized}")
                        elif class_name == 'unknown':
                            skipped_count += 1
                            log.warning(f"⚠️  Skipping unknown class: class_id={class_id}, model_class={model_class_name if self.is_custom_model else 'N/A'}, confidence={confidence:.3f}")
                    except Exception as e:
                        log.error(f"Error processing detection: {str(e)}", exc_info=True)
                        skipped_count += 1
                
                if skipped_count > 0:
                    log.warning(f"Skipped {skipped_count} detections due to unknown class or processing errors")
            
            # Log detection summary with class breakdown
            total_detections = (len(detections['tables']) + len(detections['text_blocks']) + 
                              len(detections['signatures']) + len(detections['checkboxes']) + 
                              len(detections['handwritten']))
            log.info(f"YOLOv8s detected: {len(detections['tables'])} tables, "
                    f"{len(detections['text_blocks'])} text blocks, "
                    f"{len(detections['signatures'])} signatures, "
                    f"{len(detections['checkboxes'])} checkboxes, "
                    f"{len(detections['handwritten'])} handwritten "
                    f"(total: {total_detections}, processed: {processed_count}, skipped: {skipped_count})")
            
            # Log confidence ranges for each class to diagnose why signatures/checkboxes aren't detected
            if len(detections['tables']) > 0:
                table_confs = [d['confidence'] for d in detections['tables']]
                log.info(f"  Tables: confidence range [{min(table_confs):.3f}, {max(table_confs):.3f}]")
            if len(detections['text_blocks']) > 0:
                text_confs = [d['confidence'] for d in detections['text_blocks']]
                log.info(f"  Text blocks: confidence range [{min(text_confs):.3f}, {max(text_confs):.3f}]")
            if len(detections['handwritten']) > 0:
                hw_confs = [d['confidence'] for d in detections['handwritten']]
                log.info(f"  Handwritten: confidence range [{min(hw_confs):.3f}, {max(hw_confs):.3f}]")
            if len(detections['signatures']) > 0:
                sig_confs = [d['confidence'] for d in detections['signatures']]
                log.info(f"  Signatures: confidence range [{min(sig_confs):.3f}, {max(sig_confs):.3f}]")
            else:
                log.warning("  ⚠️  No signatures detected - check raw detections above to see if Signature class exists")
            if len(detections['checkboxes']) > 0:
                cb_confs = [d['confidence'] for d in detections['checkboxes']]
                log.info(f"  Checkboxes: confidence range [{min(cb_confs):.3f}, {max(cb_confs):.3f}]")
            else:
                log.warning("  ⚠️  No checkboxes detected - check raw detections above to see if Checkbox class exists")
            
            if total_detections == 0 and filtered_count > 0:
                log.error("⚠️  CRITICAL: YOLOv8s found boxes but none were processed!")
                log.error(f"   Found {filtered_count} boxes but 0 were added to results")
                log.error("   This indicates a class mapping or processing issue")
            
            return detections
            
        except Exception as e:
            log.error(f"YOLOv8 detection failed: {str(e)}", exc_info=True)
            return self._empty_detections()
    
    def is_output_accepted(self, detections: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, str]:
        """Check if YOLO output meets acceptance criteria.
        
        TASK 6: Fixed acceptance logic - accept if ANY detection exists (less strict)
        
        YOLO output is ACCEPTED if ANY of these conditions hold:
        - ≥1 detection of ANY layout class exists (regardless of confidence)
        - OR ≥1 text_block with confidence ≥ threshold
        - OR ≥1 signature with confidence ≥ threshold
        - OR ≥1 checkbox with confidence ≥ threshold
        - OR ≥1 table with confidence ≥ threshold
        - OR ≥1 handwritten with confidence ≥ threshold
        
        Args:
            detections: Detection results from detect()
            
        Returns:
            Tuple[bool, str]: (is_accepted, reason)
        """
        # TASK 6: Count total detections first (any class, any confidence)
        total_detections = (
            len(detections.get('text_blocks', [])) +
            len(detections.get('tables', [])) +
            len(detections.get('signatures', [])) +
            len(detections.get('checkboxes', [])) +
            len(detections.get('handwritten', []))
        )
        
        # TASK 6: Accept if ANY detection exists (less strict)
        if total_detections > 0:
            return True, f"Accepted: {total_detections} total detection(s) found (any class, any confidence)"
        
        # Also check high-confidence detections (original logic)
        text_blocks = detections.get('text_blocks', [])
        high_conf_text = [d for d in text_blocks if d.get('confidence', 0) >= ACCEPTANCE_THRESHOLDS['text_block']]
        if len(high_conf_text) >= 1:
            return True, f"Accepted: {len(high_conf_text)} text_block(s) with confidence ≥ {ACCEPTANCE_THRESHOLDS['text_block']}"
        
        signatures = detections.get('signatures', [])
        high_conf_sig = [d for d in signatures if d.get('confidence', 0) >= ACCEPTANCE_THRESHOLDS['signature']]
        if len(high_conf_sig) >= 1:
            return True, f"Accepted: {len(high_conf_sig)} signature(s) with confidence ≥ {ACCEPTANCE_THRESHOLDS['signature']}"
        
        checkboxes = detections.get('checkboxes', [])
        high_conf_check = [d for d in checkboxes if d.get('confidence', 0) >= ACCEPTANCE_THRESHOLDS['checkbox']]
        if len(high_conf_check) >= 1:
            return True, f"Accepted: {len(high_conf_check)} checkbox(es) with confidence ≥ {ACCEPTANCE_THRESHOLDS['checkbox']}"
        
        tables = detections.get('tables', [])
        high_conf_tables = [d for d in tables if d.get('confidence', 0) >= ACCEPTANCE_THRESHOLDS.get('table', 0.30)]
        if len(high_conf_tables) >= 1:
            return True, f"Accepted: {len(high_conf_tables)} table(s) with confidence ≥ {ACCEPTANCE_THRESHOLDS.get('table', 0.30)}"
        
        # Not accepted (but this won't block pipeline - see TASK 6)
        return False, "YOLO output: 0 detections found (will use PaddleOCR fallback)"
    
    def _empty_detections(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return empty detection structure.
        
        Returns:
            dict: Empty detection results with all keys
        """
        return {
            'tables': [],
            'text_blocks': [],
            'signatures': [],
            'checkboxes': [],
            'handwritten': []  # Handwritten text regions
        }
    
    def is_available(self) -> bool:
        """Check if detector is available.
        
        Returns:
            bool: True if model is loaded and ready
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            dict: Model metadata
        """
        return {
            'model_type': 'YOLOv8s',
            'model_version': self.model_version,
            'is_custom_model': self.is_custom_model,
            'device': self.device,
            'class_mappings': CLASS_MAPPINGS,
            'acceptance_thresholds': ACCEPTANCE_THRESHOLDS
        }



