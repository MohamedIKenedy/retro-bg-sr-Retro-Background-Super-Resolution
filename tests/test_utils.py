"""Checks for metrics, viz, and logging helpers."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.utils.logging import setup_logging
from src.utils.viz import plot_comparison


class TestMetrics:
    """Test image quality metrics."""
    
    def test_psnr_identical_images(self, sample_color_image):
        """Test PSNR of identical images (should be infinite/high)."""
        psnr = calculate_psnr(sample_color_image, sample_color_image)
        assert psnr == float('inf') or psnr > 100  # Identical images = very high PSNR
    
    def test_psnr_different_images(self, sample_color_image):
        """Test PSNR of different images."""
        # Create slightly different image
        noisy = sample_color_image.astype(np.float32) + np.random.normal(0, 5, sample_color_image.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        psnr = calculate_psnr(sample_color_image, noisy)
        assert 0 < psnr < 100
    
    def test_ssim_identical_images(self, sample_color_image):
        """Test SSIM of identical images (should be 1.0)."""
        ssim = calculate_ssim(sample_color_image, sample_color_image)
        assert ssim == 1.0 or abs(ssim - 1.0) < 0.001
    
    def test_ssim_noisy_images(self, sample_color_image):
        """Test SSIM of noisy images (should be < 1)."""
        noisy = sample_color_image.astype(np.float32) + np.random.normal(0, 10, sample_color_image.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        ssim = calculate_ssim(sample_color_image, noisy)
        assert 0 < ssim < 1.0
    
    def test_metrics_uint8_input(self, sample_color_image):
        """Test metrics accept uint8 images."""
        psnr = calculate_psnr(sample_color_image, sample_color_image)
        ssim = calculate_ssim(sample_color_image, sample_color_image)
        assert isinstance(psnr, (int, float))
        assert isinstance(ssim, (int, float))


class TestLogging:
    """Test logging functionality."""
    
    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(str(log_file))
        assert logger is not None
        # Log something
        logger.info("Test log message")
        # Check log file was created
        assert log_file.exists()
    
    def test_logging_file_content(self, temp_dir):
        """Test that logging writes to file."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(str(log_file))
        test_message = "Test message for logging"
        logger.info(test_message)
        # Read log file
        with open(log_file) as f:
            content = f.read()
        assert test_message in content


class TestVisualization:
    """Test visualization utilities."""
    
    def test_plot_comparison(self, sample_color_image, temp_dir):
        """Test comparison plot generation."""
        output_file = temp_dir / "comparison.png"

        before = sample_color_image
        after = cv2.GaussianBlur(sample_color_image, (5, 5), 1)  # Blurred "enhancement"
        result = plot_comparison(before, after, str(output_file), "Test Comparison")

        assert result is None or result is True
    
    def test_plot_saves_file(self, sample_texture_image, temp_dir):
        """Test that plot saves output file."""
        output_file = temp_dir / "comparison.png"
        before = sample_texture_image
        after = sample_texture_image.copy()
        plot_comparison(before, after, str(output_file), "Test")
        assert output_file.exists()
