"""Visualization tool for table-only OCR filtering.

Creates debug visualization showing:
- Table bounding boxes (green)
- Text_boxes inside tables (blue) - will be processed by PaddleOCR
- Text_boxes outside tables (red) - will be skipped
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import json

# Add Back_end to path
back_end_dir = Path(__file__).parent.parent
if str(back_end_dir) not in sys.path:
    sys.path.insert(0, str(back_end_dir))

from core.logging import log
from core.config import settings
from core.utils import is_bbox_inside_table, load_image_from_canonical_id


def visualize_table_ocr_filtering(
    image_path: Optional[str] = None,
    file_id: Optional[str] = None,
    pre_0_id: Optional[str] = None,
    pre_01_id: Optional[str] = None,
    layout_json_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Image.Image:
    """Create visualization of table-only OCR filtering.
    
    Args:
        image_path: Direct path to image file
        file_id: File identifier (alternative to image_path)
        pre_0_id: Basic preprocessing ID (alternative to image_path)
        pre_01_id: Advanced preprocessing ID (alternative to image_path)
        layout_json_path: Path to layout JSON file (if None, will run layout detection)
        output_path: Path to save visualization (if None, returns PIL Image)
        
    Returns:
        PIL Image with visualization overlay
    """
    # Load image
    if image_path:
        image = Image.open(image_path).convert('RGB')
    elif file_id or pre_0_id or pre_01_id:
        image = load_image_from_canonical_id(
            file_id=file_id,
            pre_0_id=pre_0_id,
            pre_01_id=pre_01_id
        )
    else:
        raise ValueError("Must provide image_path or file_id/pre_0_id/pre_01_id")
    
    width, height = image.size
    
    # Load layout detections
    if layout_json_path and Path(layout_json_path).exists():
        with open(layout_json_path, 'r') as f:
            layout_data = json.load(f)
    else:
        # Run layout detection
        log.info("Running layout detection...")
        from layout.layout_service import LayoutService
        layout_service = LayoutService()
        layout_result = layout_service.detect_layout(
            file_id=file_id,
            pre_0_id=pre_0_id,
            pre_01_id=pre_01_id
        )
        layout_data = layout_result
    
    # Extract detections
    tables = layout_data.get('tables', [])
    text_blocks = layout_data.get('text_blocks', [])
    
    log.info(f"Found {len(tables)} tables and {len(text_blocks)} text_blocks")
    
    # Create visualization
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Try to load font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw tables (green)
    table_bboxes = []
    for i, table in enumerate(tables):
        bbox = table.get('bbox', [])
        if len(bbox) >= 4:
            table_bboxes.append(bbox)
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            # Draw table bounding box (green, thick)
            draw.rectangle([x1, y1, x2, y2], outline='green', width=4)
            draw.text((x1 + 5, y1 + 5), f"Table {i+1}", fill='green', font=font)
    
    # Filter text_blocks
    inside_count = 0
    outside_count = 0
    
    if settings.OCR_LIMIT_TO_TABLES and table_bboxes:
        for det in text_blocks:
            det_bbox = det.get('bbox', [])
            if not det_bbox or len(det_bbox) < 4:
                continue
            
            # Check if inside any table
            is_inside = any(
                is_bbox_inside_table(
                    det_bbox,
                    table_bbox,
                    min_overlap=settings.OCR_TABLE_OVERLAP_THRESHOLD
                )
                for table_bbox in table_bboxes
            )
            
            x1 = int(det_bbox[0] * width)
            y1 = int(det_bbox[1] * height)
            x2 = int(det_bbox[2] * width)
            y2 = int(det_bbox[3] * height)
            
            if is_inside:
                # Inside table - blue (will be processed)
                draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
                inside_count += 1
            else:
                # Outside table - red (will be skipped)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                outside_count += 1
    else:
        # Filtering disabled or no tables - all will be processed
        for det in text_blocks:
            det_bbox = det.get('bbox', [])
            if not det_bbox or len(det_bbox) < 4:
                continue
            
            x1 = int(det_bbox[0] * width)
            y1 = int(det_bbox[1] * height)
            x2 = int(det_bbox[2] * width)
            y2 = int(det_bbox[3] * height)
            
            # All blue (all will be processed)
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            inside_count += 1
    
    # Add legend
    legend_y = 10
    legend_x = 10
    
    # Table legend
    draw.rectangle([legend_x, legend_y, legend_x + 30, legend_y + 20], outline='green', width=3)
    draw.text((legend_x + 35, legend_y + 2), "Table", fill='green', font=font_small)
    
    # Inside table legend
    legend_y += 30
    draw.rectangle([legend_x, legend_y, legend_x + 30, legend_y + 20], outline='blue', width=2)
    draw.text((legend_x + 35, legend_y + 2), f"Inside (processed): {inside_count}", fill='blue', font=font_small)
    
    # Outside table legend
    legend_y += 30
    draw.rectangle([legend_x, legend_y, legend_x + 30, legend_y + 20], outline='red', width=2)
    draw.text((legend_x + 35, legend_y + 2), f"Outside (skipped): {outside_count}", fill='red', font=font_small)
    
    # Add stats
    stats_text = (
        f"OCR_LIMIT_TO_TABLES: {settings.OCR_LIMIT_TO_TABLES}\n"
        f"Overlap threshold: {settings.OCR_TABLE_OVERLAP_THRESHOLD}\n"
        f"Total text_blocks: {len(text_blocks)}\n"
        f"Inside tables: {inside_count}\n"
        f"Outside tables: {outside_count}"
    )
    
    # Draw stats background
    stats_y = height - 100
    draw.rectangle([10, stats_y, 400, height - 10], fill='white', outline='black', width=2)
    draw.text((15, stats_y + 5), stats_text, fill='black', font=font_small)
    
    log.info(f"Visualization complete: {inside_count} inside, {outside_count} outside")
    
    # Save or return
    if output_path:
        vis_image.save(output_path)
        log.info(f"Visualization saved to: {output_path}")
    
    return vis_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize table-only OCR filtering")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--file-id", type=str, help="File ID")
    parser.add_argument("--pre-0-id", type=str, help="Basic preprocessing ID")
    parser.add_argument("--pre-01-id", type=str, help="Advanced preprocessing ID")
    parser.add_argument("--layout-json", type=str, help="Path to layout JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output path for visualization")
    
    args = parser.parse_args()
    
    visualize_table_ocr_filtering(
        image_path=args.image,
        file_id=args.file_id,
        pre_0_id=args.pre_0_id,
        pre_01_id=args.pre_01_id,
        layout_json_path=args.layout_json,
        output_path=args.output
    )










