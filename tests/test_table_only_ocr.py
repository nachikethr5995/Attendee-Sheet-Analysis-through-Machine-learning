"""Unit tests for table-only OCR filtering feature (center-point based)."""

import pytest
from core.utils import is_center_inside_bbox
from core.config import settings


class TestCenterPointContainment:
    """Test center-point containment utility function."""
    
    def test_center_fully_inside_table(self):
        """Test text_box center fully inside table."""
        # Table: [0.1, 0.1, 0.9, 0.9]
        # Text_box: [0.2, 0.2, 0.8, 0.8] - center at (0.5, 0.5), fully inside
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.2, 0.2, 0.8, 0.8]
        
        assert is_center_inside_bbox(text_bbox, table_bbox) == True
    
    def test_center_partially_outside_table(self):
        """Test text_box with center outside table (left edge)."""
        # Table: [0.1, 0.1, 0.9, 0.9]
        # Text_box: [0.05, 0.2, 0.8, 0.8] - center at (0.425, 0.5), x-center outside
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.05, 0.2, 0.8, 0.8]
        
        # Center x (0.425) < table x_min (0.1) - should fail
        assert is_center_inside_bbox(text_bbox, table_bbox) == False
    
    def test_center_completely_outside_table(self):
        """Test text_box center completely outside table."""
        # Table: [0.1, 0.1, 0.9, 0.9]
        # Text_box: [0.0, 0.0, 0.05, 0.05] - center at (0.025, 0.025), outside
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.0, 0.0, 0.05, 0.05]
        
        assert is_center_inside_bbox(text_bbox, table_bbox) == False
    
    def test_center_on_table_edge(self):
        """Test text_box center exactly on table edge (boundary case)."""
        # Table: [0.1, 0.1, 0.9, 0.9]
        # Text_box: [0.0, 0.0, 0.2, 0.2] - center at (0.1, 0.1), exactly on edge
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.0, 0.0, 0.2, 0.2]
        
        # Center (0.1, 0.1) is on table boundary - should pass (inclusive)
        assert is_center_inside_bbox(text_bbox, table_bbox) == True
    
    def test_center_near_table_boundary(self):
        """Test text_box center near table boundary (narrow column case)."""
        # Table: [0.1, 0.1, 0.9, 0.9]
        # Text_box: [0.08, 0.5, 0.12, 0.6] - center at (0.1, 0.55), just inside
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.08, 0.5, 0.12, 0.6]
        
        # Center x (0.1) == table x_min (0.1) - should pass (inclusive)
        assert is_center_inside_bbox(text_bbox, table_bbox) == True
    
    def test_invalid_bbox(self):
        """Test with invalid bbox (too few coordinates)."""
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.2, 0.2]  # Only 2 coordinates
        
        assert is_bbox_inside_table(text_bbox, table_bbox) == False
    
    def test_zero_area_bbox(self):
        """Test with zero-area bbox."""
        table_bbox = [0.1, 0.1, 0.9, 0.9]
        text_bbox = [0.2, 0.2, 0.2, 0.2]  # Zero area
        
        assert is_bbox_inside_table(text_bbox, table_bbox) == False
    
    def test_multiple_tables(self):
        """Test filtering with multiple tables."""
        # Table 1: [0.1, 0.1, 0.5, 0.5]
        # Table 2: [0.5, 0.5, 0.9, 0.9]
        # Text_box 1: [0.2, 0.2, 0.4, 0.4] - center (0.3, 0.3), inside table 1
        # Text_box 2: [0.6, 0.6, 0.8, 0.8] - center (0.7, 0.7), inside table 2
        # Text_box 3: [0.0, 0.0, 0.05, 0.05] - center (0.025, 0.025), outside both
        
        table1 = [0.1, 0.1, 0.5, 0.5]
        table2 = [0.5, 0.5, 0.9, 0.9]
        
        text1 = [0.2, 0.2, 0.4, 0.4]
        text2 = [0.6, 0.6, 0.8, 0.8]
        text3 = [0.0, 0.0, 0.05, 0.05]
        
        assert is_center_inside_bbox(text1, table1) == True
        assert is_center_inside_bbox(text1, table2) == False
        assert is_center_inside_bbox(text2, table1) == False
        assert is_center_inside_bbox(text2, table2) == True
        assert is_center_inside_bbox(text3, table1) == False
        assert is_center_inside_bbox(text3, table2) == False


class TestTableOnlyOCRFiltering:
    """Test table-only OCR filtering logic."""
    
    def test_filter_text_blocks_with_tables(self):
        """Test filtering text_blocks when tables are present."""
        from postprocessing.unified_pipeline import UnifiedPipeline
        from unittest.mock import Mock, patch
        
        # Mock data
        tables = [
            {'bbox': [0.1, 0.1, 0.9, 0.9], 'class': 'table', 'confidence': 0.95}
        ]
        text_blocks = [
            {'bbox': [0.2, 0.2, 0.3, 0.3], 'class': 'text_block', 'confidence': 0.9},  # Inside
            {'bbox': [0.0, 0.0, 0.05, 0.05], 'class': 'text_block', 'confidence': 0.9},  # Outside
            {'bbox': [0.5, 0.5, 0.6, 0.6], 'class': 'text_block', 'confidence': 0.9},  # Inside
        ]
        
        # Enable table-only filtering
        with patch('core.config.settings.OCR_LIMIT_TO_TABLES', True):
            with patch('core.config.settings.OCR_TABLE_OVERLAP_THRESHOLD', 0.8):
                from core.utils import is_center_inside_bbox
                
                table_bboxes = [t.get('bbox', []) for t in tables if t.get('bbox')]
                filtered = []
                
                for det in text_blocks:
                    det_bbox = det.get('bbox', [])
                    if not det_bbox or len(det_bbox) < 4:
                        continue
                    
                    is_inside = any(
                        is_center_inside_bbox(det_bbox, table_bbox)
                        for table_bbox in table_bboxes
                    )
                    
                    if is_inside:
                        filtered.append(det)
                
                # Should filter to 2 text_blocks (inside table)
                assert len(filtered) == 2
                assert filtered[0]['bbox'] == [0.2, 0.2, 0.3, 0.3]
                assert filtered[1]['bbox'] == [0.5, 0.5, 0.6, 0.6]
    
    def test_filter_text_blocks_no_tables(self):
        """Test filtering when no tables detected (strict policy)."""
        from unittest.mock import patch
        
        tables = []
        text_blocks = [
            {'bbox': [0.2, 0.2, 0.3, 0.3], 'class': 'text_block', 'confidence': 0.9},
        ]
        
        # Enable table-only filtering
        with patch('core.config.settings.OCR_LIMIT_TO_TABLES', True):
            # No tables - should return empty list (strict policy)
            filtered = []
            
            # Should skip all text_blocks
            assert len(filtered) == 0
    
    def test_filter_disabled_processes_all(self):
        """Test that disabling filter processes all text_blocks."""
        from unittest.mock import patch
        
        tables = [
            {'bbox': [0.1, 0.1, 0.9, 0.9], 'class': 'table', 'confidence': 0.95}
        ]
        text_blocks = [
            {'bbox': [0.2, 0.2, 0.3, 0.3], 'class': 'text_block', 'confidence': 0.9},  # Inside
            {'bbox': [0.0, 0.0, 0.05, 0.05], 'class': 'text_block', 'confidence': 0.9},  # Outside
        ]
        
        # Disable table-only filtering
        with patch('core.config.settings.OCR_LIMIT_TO_TABLES', False):
            # Should process all text_blocks
            filtered = text_blocks
            assert len(filtered) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

