"""PaddleOCR text detector (DBNet/SAST) for refined text regions.

Uses PaddleOCR's text detection model to refine text block regions.
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

# Optional PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    log.warning("PaddleOCR not installed. Text refinement will be unavailable.")


class PaddleOCRTextDetector:
    """PaddleOCR text detector for refined text regions."""
    
    def __init__(self, use_angle_cls: bool = False, lang: str = 'en'):
        """Initialize PaddleOCR text detector.
        
        Args:
            use_angle_cls: Deprecated - kept for compatibility (not used)
            lang: Language code ('en', 'ch', etc.)
        """
        self.lang = lang
        self.ocr = None
        
        if not PADDLEOCR_AVAILABLE:
            log.warning("PaddleOCR not available. Text refinement will be unavailable.")
            return
        
        try:
            log.info("Initializing PaddleOCR text detector...")
            # TASK 2: Try PP-OCRv4 configuration first (upgraded, stronger configuration)
            try:
                # Method 1: Try PP-OCRv4 using ocr_version parameter (simplest, if supported)
                self.ocr = PaddleOCR(
                    ocr_version='PP-OCRv4',  # Use PP-OCRv4 models
                    use_angle_cls=True,  # Enable angle classifier
                    lang=lang,
                    det_db_box_thresh=0.3,  # Lower threshold for better detection
                    det_db_unclip_ratio=1.8,  # Tuned for form-style documents
                    rec_algorithm="SVTR_LCNet",  # Stronger recognition algorithm
                    use_space_char=True,  # Better handling of spaces
                    show_log=False
                )
                log.info("✅ PaddleOCR text detector initialized with PP-OCRv4 (ocr_version parameter)")
            except (TypeError, ValueError) as e0:
                log.info(f"ocr_version parameter not supported ({str(e0)}), trying explicit model directories...")
                try:
                    # Method 2: Try PP-OCRv4 with explicit model directories and tuned parameters
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,  # Enable angle classifier
                        lang=lang,
                        det_model_dir="PP-OCRv4_det",  # PP-OCRv4 detection model
                        rec_model_dir="PP-OCRv4_rec",  # PP-OCRv4 recognition model
                        cls_model_dir="ch_ppocr_mobile_v2.0_cls",  # Angle classifier model
                        det_db_box_thresh=0.3,  # Lower threshold for better detection
                        det_db_unclip_ratio=1.8,  # Tuned for form-style documents
                        rec_algorithm="SVTR_LCNet",  # Stronger recognition algorithm
                        use_space_char=True,  # Better handling of spaces
                        show_log=False
                    )
                    log.info("✅ PaddleOCR text detector initialized with PP-OCRv4 configuration (explicit directories)")
                except Exception as e1:
                    log.info(f"PP-OCRv4 explicit directories failed ({str(e1)}), trying with use_angle_cls...")
                    try:
                        # Method 3: Try with use_angle_cls (angle classifier enabled)
                        self.ocr = PaddleOCR(
                            use_angle_cls=True,
                            lang=lang,
                            show_log=False
                        )
                        log.info("PaddleOCR text detector initialized with use_angle_cls parameter")
                    except Exception as e2:
                        log.info(f"use_angle_cls failed ({str(e2)}), trying minimal initialization...")
                        try:
                            # Method 4: Try absolute minimal (most compatible)
                            self.ocr = PaddleOCR(lang=lang)
                            log.info("PaddleOCR text detector initialized with minimal parameters")
                        except Exception as e3:
                            log.info(f"Minimal initialization failed ({str(e3)}), trying with use_textline_orientation...")
                            try:
                                # Method 5: Try with use_textline_orientation (newer versions)
                                self.ocr = PaddleOCR(lang=lang, use_textline_orientation=False)
                                log.info("PaddleOCR text detector initialized with use_textline_orientation")
                            except Exception as e4:
                                log.info(f"use_textline_orientation not supported ({str(e4)}), trying with show_log...")
                                try:
                                    # Method 6: Try with show_log (older versions)
                                    self.ocr = PaddleOCR(lang=lang, show_log=False)
                                    log.info("PaddleOCR text detector initialized with show_log")
                                except Exception as e5:
                                    log.error(f"All PaddleOCR initialization attempts failed.")
                                    log.error(f"  Method 1 (ocr_version='PP-OCRv4'): {str(e0)}")
                                    log.error(f"  Method 2 (PP-OCRv4 explicit): {str(e1)}")
                                    log.error(f"  Method 3 (use_angle_cls): {str(e2)}")
                                    log.error(f"  Method 4 (minimal): {str(e3)}")
                                    log.error(f"  Method 5 (use_textline_orientation): {str(e4)}")
                                    log.error(f"  Method 6 (show_log): {str(e5)}")
                                    raise e5
        except Exception as e:
            log.error(f"Failed to initialize PaddleOCR: {str(e)}", exc_info=True)
            self.ocr = None
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect text regions in image using PaddleOCR.
        
        Args:
            image: Input PIL Image (RGB)
            
        Returns:
            list: List of text region detections with 'bbox', 'confidence', 'type', 'source'
        """
        if not self.ocr:
            log.warning("PaddleOCR not available, returning empty detections")
            return []
        
        try:
            # Convert PIL to numpy array
            img_array = image_to_array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            width, height = image.size
            
            # Run PaddleOCR detection using new API (predict method)
            # Newer PaddleOCR versions use predict() instead of ocr()
            # Try new API first, fallback to old API for compatibility
            result = None
            try:
                # New API: use predict() method
                result = self.ocr.predict(img_bgr)
                log.debug(f"PaddleOCR result with predict(): type={type(result)}, len={len(result) if isinstance(result, list) else 'N/A'}")
            except AttributeError:
                # Old API: use ocr() method (deprecated but may still work)
                try:
                    result = self.ocr.ocr(img_bgr)
                    log.debug(f"PaddleOCR result with ocr(): type={type(result)}, len={len(result) if result else 0}")
                except Exception as e:
                    log.error(f"PaddleOCR detection failed: {str(e)}", exc_info=True)
                    return []
            except Exception as e:
                log.error(f"PaddleOCR detection failed: {str(e)}", exc_info=True)
                return []
            
            text_regions = []
            
            log.debug(f"PaddleOCR raw result: {result}")
            log.debug(f"Result check: result={result is not None}, len(result)={len(result) if result else 0}")
            
            if result and len(result) > 0:
                ocr_result = result[0]
                log.debug(f"OCR result type: {type(ocr_result)}")
                
                # Handle new PaddleOCR format (OCRResult object - dictionary-like)
                if hasattr(ocr_result, 'keys') and hasattr(ocr_result, '__getitem__'):
                    log.debug("Detected OCRResult object (dictionary-like, new PaddleOCR format)")
                    
                    # Try to find detection polygons/boxes in the dictionary
                    polygons = None
                    for key in ['dt_polys', 'dt_boxes', 'boxes', 'polys', 'det_polys', 'det_boxes']:
                        if key in ocr_result:
                            polygons = ocr_result[key]
                            log.debug(f"Found detection data in key '{key}': {len(polygons) if hasattr(polygons, '__len__') else 'N/A'} items")
                            break
                    
                    # Process polygons if found
                    if polygons and hasattr(polygons, '__iter__') and not isinstance(polygons, str):
                        for polygon in polygons:
                            if polygon is None:
                                continue
                            
                            try:
                                # Handle numpy arrays - convert to list of points
                                import numpy as np
                                if isinstance(polygon, np.ndarray):
                                    # numpy array: shape (n, 2) where n is number of points
                                    polygon_points = polygon.tolist()
                                elif isinstance(polygon, (list, tuple)):
                                    # Already a list/tuple
                                    polygon_points = list(polygon)
                                else:
                                    log.warning(f"Unknown polygon type: {type(polygon)}")
                                    continue
                                
                                if len(polygon_points) < 3:  # Need at least 3 points for a polygon
                                    continue
                                
                                # Extract x and y coordinates
                                x_coords = [float(p[0]) for p in polygon_points]
                                y_coords = [float(p[1]) for p in polygon_points]
                                
                                x_min = min(x_coords)
                                y_min = min(y_coords)
                                x_max = max(x_coords)
                                y_max = max(y_coords)
                                
                                # Normalize coordinates
                                bbox_normalized = [
                                    float(x_min / width),
                                    float(y_min / height),
                                    float(x_max / width),
                                    float(y_max / height)
                                ]
                                
                                text_regions.append({
                                    'bbox': bbox_normalized,
                                    'confidence': 0.9,  # Default confidence
                                    'type': 'text_block_refined',
                                    'source': 'PaddleOCR',
                                    'polygon': [[float(p[0]/width), float(p[1]/height)] for p in polygon_points]
                                })
                            except Exception as e:
                                log.warning(f"Failed to process polygon: {e}")
                                continue
                
                # Handle OCRResult with dt_polys/dt_boxes attributes (older new format)
                elif hasattr(ocr_result, 'dt_polys') or hasattr(ocr_result, 'dt_boxes'):
                    log.debug("Detected OCRResult object (attribute-based, new PaddleOCR format)")
                    
                    # Extract polygons from OCRResult
                    if hasattr(ocr_result, 'dt_polys'):
                        polygons = ocr_result.dt_polys
                        log.debug(f"Found dt_polys: {len(polygons) if hasattr(polygons, '__len__') else 'N/A'}")
                        
                        # Convert to list if needed
                        if hasattr(polygons, '__iter__') and not isinstance(polygons, str):
                            for polygon in polygons:
                                if polygon and len(polygon) > 0:
                                    # polygon is already a list of points
                                    x_coords = [float(p[0]) for p in polygon]
                                    y_coords = [float(p[1]) for p in polygon]
                                    
                                    x_min = min(x_coords)
                                    y_min = min(y_coords)
                                    x_max = max(x_coords)
                                    y_max = max(y_coords)
                                    
                                    # Normalize coordinates
                                    bbox_normalized = [
                                        float(x_min / width),
                                        float(y_min / height),
                                        float(x_max / width),
                                        float(y_max / height)
                                    ]
                                    
                                    text_regions.append({
                                        'bbox': bbox_normalized,
                                        'confidence': 0.9,  # Default confidence
                                        'type': 'text_block_refined',
                                        'source': 'PaddleOCR',
                                        'polygon': [[float(p[0]/width), float(p[1]/height)] for p in polygon]
                                    })
                    
                    # Fallback: try dt_boxes if dt_polys not available
                    elif hasattr(ocr_result, 'dt_boxes'):
                        boxes = ocr_result.dt_boxes
                        log.debug(f"Found dt_boxes: {len(boxes) if hasattr(boxes, '__len__') else 'N/A'}")
                        # Similar processing for boxes...
                
                # Handle old format (list of lists)
                elif isinstance(ocr_result, list):
                    log.debug(f"Detected list format (old PaddleOCR format): {len(ocr_result)} items")
                    for line in ocr_result:
                        if line:
                            # line[0] contains polygon coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            # line[1] contains confidence (but we're not using recognition)
                            polygon = line[0] if isinstance(line, list) else line
                            
                            # Convert polygon to bounding box
                            x_coords = [point[0] for point in polygon]
                            y_coords = [point[1] for point in polygon]
                            
                            x_min = min(x_coords)
                            y_min = min(y_coords)
                            x_max = max(x_coords)
                            y_max = max(y_coords)
                            
                            # Normalize coordinates to 0-1 range
                            bbox_normalized = [
                                float(x_min / width),
                                float(y_min / height),
                                float(x_max / width),
                                float(y_max / height)
                            ]
                            
                            # Get confidence if available
                            confidence = float(line[1]) if len(line) > 1 and isinstance(line[1], (int, float)) else 0.9
                            
                            text_regions.append({
                                'bbox': bbox_normalized,
                                'confidence': confidence,
                                'type': 'text_block_refined',
                                'source': 'PaddleOCR',
                                'polygon': [[float(p[0]/width), float(p[1]/height)] for p in polygon]  # Keep polygon for reference
                            })
                else:
                    log.warning(f"Unknown PaddleOCR result format: {type(ocr_result)}")
                    log.warning(f"Available attributes: {[attr for attr in dir(ocr_result) if not attr.startswith('_')]}")
            
            log.info(f"PaddleOCR detected {len(text_regions)} text regions")
            if len(text_regions) == 0:
                log.warning("PaddleOCR returned 0 text regions. Raw result structure:")
                log.warning(f"  result type: {type(result)}")
                log.warning(f"  result length: {len(result) if result else 0}")
                if result and len(result) > 0:
                    log.warning(f"  result[0] type: {type(result[0])}")
                    log.warning(f"  result[0] length: {len(result[0]) if result[0] else 0}")
                    if result[0] and len(result[0]) > 0:
                        log.warning(f"  result[0][0] sample: {result[0][0]}")
            return text_regions
            
        except Exception as e:
            log.error(f"PaddleOCR text detection failed: {str(e)}", exc_info=True)
            return []
    
    def merge_small_regions(self, text_regions: List[Dict[str, Any]], 
                           min_area: float = 0.001) -> List[Dict[str, Any]]:
        """Merge small text regions into larger blocks.
        
        Args:
            text_regions: List of text region detections
            min_area: Minimum area threshold (normalized) for merging
            
        Returns:
            list: Merged text regions
        """
        if not text_regions:
            return []
        
        # Group nearby regions
        # Simple implementation: merge regions that are close together
        merged = []
        used = set()
        
        for i, region in enumerate(text_regions):
            if i in used:
                continue
            
            bbox1 = region['bbox']
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            
            # If region is too small, try to merge with nearby regions
            if area1 < min_area:
                # Find nearby regions to merge with
                merged_bbox = bbox1.copy()
                merged_confidence = region['confidence']
                merged_count = 1
                
                for j, other_region in enumerate(text_regions[i+1:], start=i+1):
                    if j in used:
                        continue
                    
                    bbox2 = other_region['bbox']
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    
                    # Check if regions are close (simple distance check)
                    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
                    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
                    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                    
                    if distance < 0.1:  # Close regions
                        # Merge bounding boxes
                        merged_bbox[0] = min(merged_bbox[0], bbox2[0])
                        merged_bbox[1] = min(merged_bbox[1], bbox2[1])
                        merged_bbox[2] = max(merged_bbox[2], bbox2[2])
                        merged_bbox[3] = max(merged_bbox[3], bbox2[3])
                        merged_confidence = max(merged_confidence, other_region['confidence'])
                        merged_count += 1
                        used.add(j)
                
                merged.append({
                    'bbox': merged_bbox,
                    'confidence': merged_confidence,
                    'type': 'text_block_refined',
                    'source': 'PaddleOCR',
                    'merged_count': merged_count
                })
                used.add(i)
            else:
                # Keep large regions as-is
                merged.append(region)
                used.add(i)
        
        return merged









