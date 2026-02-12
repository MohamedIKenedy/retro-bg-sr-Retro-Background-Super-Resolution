"""Tests for inference and post-processing steps."""

import pytest
import numpy as np
import cv2
from src.enhancement.detail_enhancer import DetailEnhancer
from src.enhancement.atmospheric import AtmosphericEnhancer
from src.enhancement.postprocess import Postprocess


class TestDetailEnhancer:
    """Test detail enhancement functionality."""
    
    def test_detail_enhancer_init(self, temp_dir):
        """Test DetailEnhancer initialization."""
        enhancer = DetailEnhancer(str(temp_dir))
        assert enhancer is not None
    
    def test_enhance_clahe(self, sample_texture_image, temp_dir):
        """Test CLAHE enhancement method."""
        enhancer = DetailEnhancer(str(temp_dir))
        result = enhancer.enhance(sample_texture_image, method="clahe", strength=1.0)
        assert result.shape == sample_texture_image.shape
        assert result.dtype == np.uint8
        assert result.max() <= 255 and result.min() >= 0
    
    def test_enhance_detail_layer(self, sample_texture_image, temp_dir):
        """Test detail layer enhancement method."""
        enhancer = DetailEnhancer(str(temp_dir))
        result = enhancer.enhance(sample_texture_image, method="detail_layer", strength=1.0)
        assert result.shape == sample_texture_image.shape
        assert result.dtype == np.uint8
    
    def test_enhance_combined(self, sample_texture_image, temp_dir):
        """Test combined enhancement method."""
        enhancer = DetailEnhancer(str(temp_dir))
        result = enhancer.enhance(sample_texture_image, method="combined", strength=1.0)
        assert result.shape == sample_texture_image.shape
        assert result.dtype == np.uint8
    
    def test_enhance_strength_scaling(self, sample_texture_image, temp_dir):
        """Test that strength parameter affects output."""
        enhancer = DetailEnhancer(str(temp_dir))
        result_low = enhancer.enhance(sample_texture_image.copy(), method="clahe", strength=0.5)
        result_high = enhancer.enhance(sample_texture_image.copy(), method="clahe", strength=2.0)
        # High strength should produce more contrast
        assert result_high.std() > result_low.std()


class TestAtmosphericEnhancer:
    """Test atmospheric color grading functionality."""
    
    def test_atmospheric_init(self, temp_dir):
        """Test AtmosphericEnhancer initialization."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        assert enhancer is not None
    
    def test_apply_no_effects(self, sample_color_image, temp_dir):
        """Test apply_eerie_atmosphere with minimal effects."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result = enhancer.apply_eerie_atmosphere(
            sample_color_image,
            blur_strength=0, haze=0, temp=0, tint=0,
            saturation=0, brightness=0, contrast=0, grain=0
        )
        assert result.shape == sample_color_image.shape
        assert result.dtype == np.uint8
        # With no effects, should be very similar to original
        diff = cv2.absdiff(result, sample_color_image).astype(float).mean()
        assert diff < 5  # Small difference due to float32 conversion
    
    def test_apply_temperature(self, sample_color_image, temp_dir):
        """Test temperature adjustment effect."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result_cool = enhancer.apply_eerie_atmosphere(sample_color_image.copy(), temp=-20)
        result_warm = enhancer.apply_eerie_atmosphere(sample_color_image.copy(), temp=20)
        assert result_cool.shape == sample_color_image.shape
        assert result_warm.shape == sample_color_image.shape
        # Cool should have more blue, warm should have more red
        assert result_cool[:, :, 0].mean() > result_warm[:, :, 0].mean()  # Blue channel
    
    def test_apply_saturation(self, sample_color_image, temp_dir):
        """Test saturation adjustment."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result = enhancer.apply_eerie_atmosphere(sample_color_image, saturation=20)
        assert result.shape == sample_color_image.shape
        # Result should be valid
        assert result.min() >= 0 and result.max() <= 255
    
    def test_apply_grain(self, sample_color_image, temp_dir):
        """Test grain addition effect."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result_no_grain = enhancer.apply_eerie_atmosphere(sample_color_image, grain=0)
        result_grain = enhancer.apply_eerie_atmosphere(sample_color_image, grain=20)
        # With grain, image should have more noise/variation
        assert result_grain.std() > result_no_grain.std()
    
    def test_apply_haze(self, sample_color_image, temp_dir):
        """Test haze/fog effect."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result = enhancer.apply_eerie_atmosphere(sample_color_image, haze=30)
        assert result.shape == sample_color_image.shape
        # Haze should reduce contrast slightly
        assert result.std() < sample_color_image.std()
    
    def test_apply_all_effects(self, sample_color_image, temp_dir):
        """Test combining all effects."""
        enhancer = AtmosphericEnhancer(str(temp_dir))
        result = enhancer.apply_eerie_atmosphere(
            sample_color_image,
            blur_strength=20, haze=15, temp=-10, tint=5,
            saturation=10, brightness=5, contrast=5, grain=15
        )
        assert result.shape == sample_color_image.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255


class TestPostprocess:
    """Test postprocessing functionality."""
    
    def test_postprocess_init(self, temp_dir):
        """Test Postprocess initialization."""
        post = Postprocess(str(temp_dir), str(temp_dir / "output"))
        assert post is not None
    
    def test_denoise(self, sample_texture_image, temp_dir):
        """Test denoising."""
        post = Postprocess(str(temp_dir), str(temp_dir / "output"))
        # Add noise first
        noisy = sample_texture_image.astype(np.float32) + np.random.normal(0, 10, sample_texture_image.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        result = post.denoise(noisy)
        assert result.shape == noisy.shape
        # Denoising should reduce noise
        assert result.std() < noisy.std()
    
    def test_sharpen(self, sample_texture_image, temp_dir):
        """Test sharpening."""
        post = Postprocess(str(temp_dir), str(temp_dir / "output"))
        result = post.sharpen(sample_texture_image)
        assert result.shape == sample_texture_image.shape
        # Sharpening should increase contrast
        assert result.std() > sample_texture_image.std()
    
    def test_apply_all(self, sample_color_image, temp_dir):
        """Test applying all postprocessing steps."""
        post = Postprocess(str(temp_dir), str(temp_dir / "output"))
        result = post.apply_all(sample_color_image)
        assert result.shape == sample_color_image.shape
        assert result.dtype == np.uint8
