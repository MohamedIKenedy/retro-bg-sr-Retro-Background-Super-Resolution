"""Smoke tests for data loading and transforms."""

import pytest
import numpy as np
from pathlib import Path
from src.data.loader import Loader
from src.data.transforms import Transforms


class TestLoader:
    """Test data loading functionality."""
    
    def test_loader_init(self, temp_dir):
        """Test Loader initialization."""
        loader = Loader(str(temp_dir), str(temp_dir / "output"))
        assert loader is not None
        assert loader.input_path == temp_dir
    
    def test_loader_extract_images_empty(self, temp_dir):
        """Test that loader handles empty directories."""
        loader = Loader(str(temp_dir), str(temp_dir / "output"))
        files = loader.extract_images()
        assert isinstance(files, list)
        # Empty directory should return empty or skip gracefully
        assert len(files) == 0
    
    def test_loader_extract_images(self, sample_images_dir, temp_dir):
        """Test successful image extraction from directory."""
        loader = Loader(str(sample_images_dir), str(temp_dir / "output"))
        files = loader.extract_images()
        assert len(files) == 2
        assert all(f.exists() for f in files)
        assert all(f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'] for f in files)


class TestTransforms:
    """Test image transformation pipeline."""
    
    def test_transforms_init(self, temp_dir):
        """Test Transforms initialization."""
        transforms = Transforms(str(temp_dir))
        assert transforms is not None
    
    def test_transforms_call_grayscale(self, sample_gray_image, temp_dir):
        """Test transform on grayscale image."""
        transforms = Transforms(str(temp_dir))
        # Convert to BGR for processing
        img_bgr = np.stack([sample_gray_image] * 3, axis=-1)
        tensor = transforms(img_bgr)
        assert tensor.shape[0] == 1  # Batch size
        assert tensor.shape[1] == 3  # Channels (RGB)
        assert tensor.shape[2] > 0 and tensor.shape[3] > 0  # Height, Width
    
    def test_transforms_call_color(self, sample_color_image, temp_dir):
        """Test transform on color image."""
        transforms = Transforms(str(temp_dir))
        tensor = transforms(sample_color_image)
        assert tensor.shape[0] == 1  # Batch size
        assert tensor.shape[1] == 3  # RGB channels
        assert tensor.dtype == np.float32
    
    def test_transforms_preserves_aspect_ratio(self, temp_dir):
        """Test that transforms preserve aspect ratio without padding."""
        transforms = Transforms(str(temp_dir))
        # Test various aspect ratios
        for h, w in [(128, 256), (256, 128), (512, 512)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            tensor = transforms(img)
            assert tensor.shape[0] == 1
            assert tensor.shape[1] == 3
            assert tensor.shape[2] == h
            assert tensor.shape[3] == w
