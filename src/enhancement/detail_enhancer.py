"""Fine detail enhancement using CLAHE and adaptive techniques."""

import cv2
import numpy as np
from typing import Union
from pathlib import Path


class DetailEnhancer:
    """Enhance fine details like textures, tires, materials using CLAHE."""
    
    def __init__(self, output_path: Union[str, Path] = "data/enhanced/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def enhance(self, image: np.ndarray, method: str = "clahe", strength: float = 1.0) -> np.ndarray:
        """
        Enhance fine details in the image.
        
        Args:
            image: Input BGR image
            method: Enhancement method - 'clahe', 'detail_layer', 'combined', or 'text_sharp'
            strength: Enhancement strength multiplier (1.0 = default)
        
        Returns:
            Detailed enhanced image
        """
        if method == "clahe":
            return self._clahe_enhance(image, strength)
        elif method == "detail_layer":
            return self._detail_layer_enhance(image, strength)
        elif method == "combined":
            # Apply CLAHE first, then detail layer
            enhanced = self._clahe_enhance(image, strength * 0.7)
            return self._detail_layer_enhance(enhanced, strength * 0.3)
        elif method == "text_sharp":
            # Aggressive sharpening for text clarity
            return self._text_aware_sharpening(image, strength)
        else:
            raise ValueError(f"Unknown method: {method}. Choose: clahe, detail_layer, combined, text_sharp")
    
    def _clahe_enhance(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Enhances local contrast without affecting global brightness.
        Perfect for: tires, metal, concrete, wood textures, floor details.
        
        Args:
            image: BGR image
            strength: Amplification factor for clip limit (default 2.0)
        
        Returns:
            Enhanced image
        """
        # Convert to LAB for better contrast control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clip_limit = 2.0 * strength
        tile_size = 8
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _detail_layer_enhance(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Enhance fine details by decomposing into texture and smooth layers.
        
        Uses bilateral filtering to separate edge structure from texture details,
        then amplifies and recombines them.
        
        Args:
            image: BGR image
            strength: Detail amplification factor (1.5 = 50% stronger details)
        
        Returns:
            Enhanced image with recovered micro-details
        """
        # Bilateral filter: separates edges from texture while smoothing
        # d=9: diameter of filter kernel
        # sigmaColor=75: color similarity threshold
        # sigmaSpace=75: geometric (spatial) similarity threshold
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Extract detail layer (high-frequency component)
        detail = cv2.subtract(image.astype(np.float32), bilateral.astype(np.float32))
        
        # Amplify detail layer
        detail_amplified = detail * (1.0 + strength * 0.5)
        
        # Recombine smooth layer with amplified details
        result = bilateral.astype(np.float32) + detail_amplified
        
        # Clip and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _text_aware_sharpening(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Aggressive sharpening optimized for text clarity.
        
        Detects high-contrast edges (where text is) and applies selective sharpening.
        Combines CLAHE for contrast + Strong unsharp mask for edges.
        
        Args:
            image: BGR image
            strength: Sharpening intensity (1.0 = default, 2.0+ = very aggressive)
        
        Returns:
            Text-enhanced image with sharp edges
        """
        # Step 1: Apply CLAHE for local contrast (helps text stand out)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        image_contrast = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
        
        # Step 2: Edge detection to identify text regions
        gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to get text-adjacent regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_mask = cv2.dilate(edges, kernel, iterations=2)
        
        # Convert mask to 3-channel for blending
        edge_mask_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # Step 3: Aggressive unsharp sharpening
        # Use smaller kernel for sharper results
        blurred = cv2.GaussianBlur(image_contrast, (3, 3), 0.5)
        detail = image_contrast.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply strong sharpening amount
        sharpen_amount = 2.0 * strength  # 2.0 = very strong sharpening
        sharpened = image_contrast.astype(np.float32) + detail * sharpen_amount
        
        # Step 4: Blend sharpening selectively - only apply where edges are
        # Sharp texturally, but also sharpen smooth text areas
        result = image_contrast.astype(np.float32) * (1.0 - edge_mask_3ch * 0.7)
        result = result + sharpened * (edge_mask_3ch * 0.7)
        
        # Step 5: Additional morphological cleanup for text
        # Ensure text is well-defined
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        
        # Gentle contrast boost specifically for low-contrast areas
        result_uint8 = cv2.convertScaleAbs(result_uint8, alpha=1.1, beta=5)
        
        return result_uint8
    
    def enhance_with_edge_preservation(self, image: np.ndarray, 
                                      sigma_spatial: float = 10.0,
                                      sigma_range: float = 0.1) -> np.ndarray:
        """
        Domain Transform edge-preserving filter.
        
        Smooths image while preserving sharp edges and details.
        Better than bilateral for large-scale detail preservation.
        
        Args:
            image: BGR image
            sigma_spatial: Edge-awareness parameter (higher = preserve more edges)
            sigma_range: Color sensitivity (higher = more color preservation)
        
        Returns:
            Edge-preserving filtered image
        """
        # Domain transform edge-preserving filter
        # This is slower but better quality than bilateral
        result = cv2.dtFilter1D(image, image, sigma_spatial, sigma_range)
        return result
    
    def unsharp_mask_advanced(self, image: np.ndarray, 
                             kernel_size: int = 5,
                             sigma: float = 1.0, 
                             amount: float = 0.5) -> np.ndarray:
        """
        Advanced unsharp masking for controlled sharpening.
        
        Works by extracting details at specific scale and selectively reinforcing them.
        
        Args:
            image: BGR image
            kernel_size: Gaussian blur kernel size
            sigma: Gaussian sigma
            amount: Sharpening amount (0.3-1.0 for subtle, 1.0+ for aggressive)
        
        Returns:
            Sharpened image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur at specific scale
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Extract detail (high-frequency)
        detail = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))
        
        # Reinforce details
        sharpened = image.astype(np.float32) + detail * amount
        
        # Clip and convert
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return result
    
    def adaptive_unsharp(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive unsharp masking that respects image structure.
        
        Applies stronger sharpening to textured areas and lighter to smooth areas.
        Great for tires, floor textures without over-sharpening smooth surfaces.
        
        Returns:
            Adaptively sharpened image
        """
        # Detect texture regions using Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = np.abs(laplacian)
        
        # Normalize texture map to [0, 1]
        texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-6)
        
        # Apply unsharp mask
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        detail = image.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply variable sharpening based on texture map
        # Higher texture = more sharpening
        strength = (texture_map * 0.8 + 0.2)  # Range [0.2, 1.0]
        strength = cv2.cvtColor((strength * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        result = image.astype(np.float32) + detail * strength * 0.5
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
