"""Fixtures for sample image sets and configs."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gray_image():
    """Create a 256x256 grayscale test image."""
    # Create gradient pattern
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        img[i, :] = i
    return img


@pytest.fixture
def sample_color_image():
    """Create a 256x256 BGR color test image."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Red channel: horizontal gradient
    for i in range(256):
        img[:, i, 2] = i
    # Green channel: vertical gradient
    for i in range(256):
        img[i, :, 1] = i
    # Blue channel: constant
    img[:, :, 0] = 128
    return img


@pytest.fixture
def sample_texture_image():
    """Create a test image with texture patterns."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add checkerboard pattern
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if ((i // 32) + (j // 32)) % 2 == 0:
                img[i:i+32, j:j+32] = [200, 150, 100]  # BGR
            else:
                img[i:i+32, j:j+32] = [100, 150, 200]
    return img


@pytest.fixture
def sample_images_dir(temp_dir, sample_color_image, sample_texture_image):
    """Create directory with sample images."""
    img_dir = temp_dir / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "test1.png"), sample_color_image)
    cv2.imwrite(str(img_dir / "test2.png"), sample_texture_image)
    return img_dir
