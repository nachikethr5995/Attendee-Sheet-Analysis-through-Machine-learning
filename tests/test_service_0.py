"""Unit tests for Service 0 (Preprocessing)."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from core.utils import (
    generate_file_id,
    generate_processed_id,
    validate_file_size,
    validate_image_format,
)
from ingestion.file_handler import FileHandler
from preprocessing.converter import ImageConverter
from preprocessing.enhancer import ImageEnhancer
from preprocessing.quality import QualityScorer
from preprocessing.deskew import DeskewCorrector
from preprocessing.perspective import PerspectiveCorrector


class TestUtils:
    """Test utility functions."""
    
    def test_generate_file_id(self):
        """Test file ID generation."""
        file_id = generate_file_id()
        assert file_id.startswith("file_")
        assert len(file_id) > 10
    
    def test_generate_processed_id(self):
        """Test processed ID generation."""
        processed_id = generate_processed_id()
        assert processed_id.startswith("pre_")
        assert len(processed_id) > 10
    
    def test_validate_image_format(self):
        """Test image format validation."""
        assert validate_image_format("jpg")
        assert validate_image_format("png")
        assert validate_image_format("heic")
        assert validate_image_format("pdf")
        assert not validate_image_format("txt")
        assert not validate_image_format("docx")


class TestImageConverter:
    """Test image conversion and normalization."""
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image (3000x2000)
        test_image = Image.new('RGB', (3000, 2000), color='red')
        converter = ImageConverter(target_size=2048)
        normalized = converter.normalize(test_image)
        
        # Should be resized to 2048px long edge
        assert max(normalized.size) == 2048
        assert normalized.mode == 'RGB'
    
    def test_save_processed(self, tmp_path):
        """Test saving processed image."""
        test_image = Image.new('RGB', (1000, 1000), color='blue')
        converter = ImageConverter()
        
        # Mock the path function
        import core.utils as utils_module
        original_func = utils_module.get_processed_image_path
        
        def mock_path(pid):
            return tmp_path / f"{pid}.png"
        
        utils_module.get_processed_image_path = mock_path
        
        try:
            output_path = converter.save_processed(test_image, "test_id")
            assert output_path.exists()
            assert output_path.suffix == '.png'
        finally:
            utils_module.get_processed_image_path = original_func


class TestImageEnhancer:
    """Test image enhancement."""
    
    def test_enhance_pipeline(self):
        """Test full enhancement pipeline."""
        # Create test image
        test_image = Image.new('RGB', (1000, 1000), color='green')
        enhancer = ImageEnhancer(use_gpu=False)
        enhanced = enhancer.enhance(test_image)
        
        assert enhanced.size == test_image.size
        assert enhanced.mode == 'RGB'


class TestQualityScorer:
    """Test quality scoring."""
    
    def test_blur_detection(self):
        """Test blur detection."""
        # Create sharp image (high frequency)
        sharp = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        is_blurred, variance = QualityScorer.detect_blur(sharp)
        
        assert isinstance(is_blurred, bool)
        assert variance > 0
    
    def test_quality_score(self):
        """Test quality score calculation."""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = QualityScorer.calculate_quality_score(test_image)
        
        assert 0.0 <= score <= 1.0


class TestDeskewCorrector:
    """Test deskew correction."""
    
    def test_rotate_image(self):
        """Test image rotation."""
        test_image = Image.new('RGB', (100, 200), color='red')
        corrector = DeskewCorrector()
        
        rotated = corrector.rotate_image(test_image, 90)
        assert rotated.size == (200, 100)  # Dimensions swapped


class TestPerspectiveCorrector:
    """Test perspective correction."""
    
    def test_order_points(self):
        """Test point ordering."""
        # Create test points
        pts = np.array([[100, 100], [200, 50], [200, 150], [100, 150]], dtype=np.float32)
        corrector = PerspectiveCorrector()
        ordered = corrector.order_points(pts)
        
        assert len(ordered) == 4
        assert ordered.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])










