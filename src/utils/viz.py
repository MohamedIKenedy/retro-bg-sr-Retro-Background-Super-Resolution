"""Grid and montage rendering helpers for before/after views."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union, Tuple
import logging


def plot_comparison(before_img: np.ndarray,
                   after_img: np.ndarray,
                   output_path: Union[str, Path],
                   title: str = "Before vs After") -> bool:
    """
    Create side-by-side comparison of before/after images.

    Args:
        before_img: Original image (BGR or RGB)
        after_img: Enhanced image (BGR or RGB)
        output_path: Path to save the comparison plot
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert BGR to RGB for matplotlib
        if len(before_img.shape) == 3:
            before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
            after_rgb = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        else:
            before_rgb = before_img
            after_rgb = after_img

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(before_rgb)
        ax1.set_title("Before")
        ax1.axis('off')

        ax2.imshow(after_rgb)
        ax2.set_title("After")
        ax2.axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save to logs folder
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logging.error(f"Failed to create comparison plot: {e}")
        return False


def plot_grid_comparisons(image_pairs: List[Tuple[np.ndarray, np.ndarray, str]],
                         output_path: Union[str, Path],
                         title: str = "Image Enhancement Results",
                         cols: int = 2) -> bool:
    """
    Create a grid of before/after image comparisons.

    Args:
        image_pairs: List of (before_img, after_img, filename) tuples
        output_path: Path to save the grid plot
        title: Overall plot title
        cols: Number of columns in grid

    Returns:
        True if successful, False otherwise
    """
    try:
        n_images = len(image_pairs)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows * 2, cols, figsize=(6*cols, 4*rows))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(2, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (before_img, after_img, filename) in enumerate(image_pairs):
            row = (idx // cols) * 2
            col = idx % cols

            # Convert BGR to RGB
            if len(before_img.shape) == 3:
                before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
                after_rgb = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
            else:
                before_rgb = before_img
                after_rgb = after_img

            # Before image
            axes[row, col].imshow(before_rgb)
            axes[row, col].set_title(f"{Path(filename).stem}\n(Before)")
            axes[row, col].axis('off')

            # After image
            axes[row+1, col].imshow(after_rgb)
            axes[row+1, col].set_title("(After)")
            axes[row+1, col].axis('off')

        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row = (idx // cols) * 2
            col = idx % cols
            axes[row, col].axis('off')
            axes[row+1, col].axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save to logs folder
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logging.error(f"Failed to create grid comparison: {e}")
        return False


def visualize_sample_enhancements(sample_paths: List[Union[str, Path]],
                                 output_dir: Union[str, Path] = "logs/visualizations",
                                 max_samples: int = 4) -> List[str]:
    """
    Visualize before/after for a sample of images.

    Args:
        sample_paths: List of image file paths to visualize
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize

    Returns:
        List of paths to created visualization files
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    image_pairs = []

    # Load and sample images
    for img_path in sample_paths[:max_samples]:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                # For demo, create "enhanced" version by applying slight blur
                # In real usage, this would be the actual enhanced image
                enhanced = cv2.GaussianBlur(img, (3, 3), 0.5)
                image_pairs.append((img, enhanced, str(img_path)))
            else:
                logger.warning(f"Could not load image: {img_path}")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    if not image_pairs:
        logger.warning("No valid images found for visualization")
        return created_files

    # Create individual comparisons
    for i, (before, after, filename) in enumerate(image_pairs):
        output_path = output_dir / f"comparison_{i+1}_{Path(filename).stem}.png"
        if plot_comparison(before, after, output_path, f"Enhancement: {Path(filename).name}"):
            created_files.append(str(output_path))
            logger.info(f"Created comparison: {output_path}")

    # Create grid comparison if multiple images
    if len(image_pairs) > 1:
        grid_path = output_dir / "grid_comparison.png"
        if plot_grid_comparisons(image_pairs, grid_path, "Sample Enhancement Results"):
            created_files.append(str(grid_path))
            logger.info(f"Created grid comparison: {grid_path}")

    return created_files


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Find some sample images
    sample_dir = Path("data/raw/BioHazard_2")
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
        if image_files:
            created = visualize_sample_enhancements(image_files[:3])
            print(f"Created {len(created)} visualization files in logs/visualizations/")
        else:
            print("No sample images found in data/raw/BioHazard_2/")
    else:
        print("Sample directory not found")
