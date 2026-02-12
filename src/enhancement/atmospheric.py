"""Atmospheric and color grading enhancements for eerie, moody, or atmospheric effects."""

import cv2
import numpy as np
from typing import Union, Literal
from pathlib import Path


class AtmosphericEnhancer:
    """Apply atmospheric effects for eerie, cinematic, or moody looks."""
    
    def __init__(self, output_path: Union[str, Path] = "data/enhanced/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def apply_eerie_atmosphere(self, image: np.ndarray, 
                              blur_strength: float = 0,
                              haze: float = 8,
                              temp: float = -12,  # Negative = cooler/bluer
                              tint: float = 5,    # Positive = greener
                              saturation: float = -3,
                              brightness: float = 0,
                              contrast: float = 5,
                              grain: float = 10,
                              fog_color: tuple = (100, 115, 105)) -> np.ndarray:  # BGR - very subtle cool fog
        """
        Apply eerie atmospheric color grading to enhance mood.
        
        Args:
            image: BGR image
            blur_strength: Atmospheric blur amount (0-100)
            haze: Fog/haze density (0-100)
            temp: Color temperature (-50 to +50, negative=cooler/blue, positive=warmer/orange)
            tint: Color tint (-50 to +50, negative=magenta, positive=green)
            saturation: Saturation adjustment (-50 to +50)
            brightness: Brightness adjustment (-50 to +50)
            contrast: Contrast adjustment (-50 to +50)
            grain: Film grain amount (0-100)
            fog_color: RGB color of fog (default: cool greenish-blue for RE2)
        
        Returns:
            Atmospherically enhanced image
        """
        result = image.astype(np.float32) / 255.0
        
        # Apply color grading first (before haze)
        if temp != 0:
            result = self._adjust_temperature(result, temp)
        
        if tint != 0:
            result = self._apply_tint(result, tint)
        
        if saturation != 0:
            result = self._adjust_saturation(result, saturation)
        
        if contrast != 0:
            result = self._adjust_contrast(result, contrast)
        
        if brightness != 0:
            result = self._adjust_brightness(result, brightness)
        
        # Apply atmospheric effects
        if blur_strength > 0 or haze > 0:
            result = self._add_atmospheric_haze(result, blur_strength, haze, fog_color)
        
        if grain > 0:
            result = self._add_grain(result, grain)
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    def _add_atmospheric_haze(self, image: np.ndarray, blur_strength: float, 
                             haze: float, fog_color: tuple) -> np.ndarray:
        """Add atmospheric fog/haze effect with depth - SUBTLE version."""
        haze_strength = np.clip(haze / 100.0, 0, 1)
        
        if blur_strength > 0:
            # Very subtle atmospheric blur
            blur_kernel = max(3, int(blur_strength / 15) * 2 + 1)
            blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), blur_strength/30)
            # Blend original with blurred (keep mostly original)
            image = image * 0.7 + blurred * 0.3
        
        if haze_strength > 0:
            # Create fog layer with specified color
            fog = np.full_like(image, 1.0)
            fog[:, :, 0] = fog_color[0] / 255.0  # B
            fog[:, :, 1] = fog_color[1] / 255.0  # G
            fog[:, :, 2] = fog_color[2] / 255.0  # R
            
            # Create very subtle depth map
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            depth_map = (255 - gray).astype(np.float32) / 255.0
            depth_map = cv2.GaussianBlur(depth_map, (51, 51), 15)
            depth_map = np.expand_dims(depth_map, axis=2)
            
            # Very subtle fog blend (max 15% fog even in darkest areas)
            fog_amount = depth_map * haze_strength * 0.15
            image = image * (1 - fog_amount) + fog * fog_amount
        
        return np.clip(image, 0, 1)
    
    def _adjust_temperature(self, image: np.ndarray, temp: float) -> np.ndarray:
        """
        Adjust color temperature (cool/blue vs warm/orange) - SUBTLE version.
        FIXED: Negative = cooler (more blue), Positive = warmer (more orange)
        """
        temp_norm = np.clip(temp / 50.0, -1, 1)
        
        if temp_norm < 0:
            # Cool (blue) - subtle increase blue, slight reduce red/green
            b_mult = 1 + abs(temp_norm) * 0.15
            g_mult = 1 - abs(temp_norm) * 0.08
            r_mult = 1 - abs(temp_norm) * 0.12
        else:
            # Warm (orange) - subtle increase red/green, slight reduce blue
            r_mult = 1 + temp_norm * 0.15
            g_mult = 1 + temp_norm * 0.08
            b_mult = 1 - temp_norm * 0.12
        
        # Apply to BGR channels
        image[:, :, 0] = np.clip(image[:, :, 0] * b_mult, 0, 1)  # B
        image[:, :, 1] = np.clip(image[:, :, 1] * g_mult, 0, 1)  # G
        image[:, :, 2] = np.clip(image[:, :, 2] * r_mult, 0, 1)  # R
        
        return image
    
    def _apply_tint(self, image: np.ndarray, tint: float) -> np.ndarray:
        """
        Apply color tint (green/magenta) - SUBTLE version.
        FIXED: Negative = magenta, Positive = green
        """
        tint_norm = np.clip(tint / 50.0, -1, 1)
        
        if tint_norm > 0:
            # Green tint - subtle increase green, slight reduce red/blue
            g_mult = 1 + tint_norm * 0.12
            r_mult = 1 - tint_norm * 0.06
            b_mult = 1 - tint_norm * 0.06
        else:
            # Magenta tint - subtle increase red/blue, slight reduce green
            r_mult = 1 + abs(tint_norm) * 0.08
            b_mult = 1 + abs(tint_norm) * 0.08
            g_mult = 1 - abs(tint_norm) * 0.12
        
        image[:, :, 0] = np.clip(image[:, :, 0] * b_mult, 0, 1)  # B
        image[:, :, 1] = np.clip(image[:, :, 1] * g_mult, 0, 1)  # G
        image[:, :, 2] = np.clip(image[:, :, 2] * r_mult, 0, 1)  # R
        
        return image
    
    def _adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        """Adjust color saturation."""
        sat_norm = np.clip(saturation / 50.0, -1, 1)
        factor = 1 + sat_norm  # 0 to 2
        
        # Convert to HSV
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Convert back
        hsv = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr.astype(np.float32) / 255.0
    
    def _adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """Adjust contrast - SUBTLE version."""
        contrast_norm = np.clip(contrast / 50.0, -1, 1)
        factor = 1 + contrast_norm * 0.3  # Reduced from 1.0
        
        # Contrast around midpoint (0.5)
        result = (image - 0.5) * factor + 0.5
        
        return np.clip(result, 0, 1)
    
    def _adjust_brightness(self, image: np.ndarray, brightness: float) -> np.ndarray:
        """Adjust overall brightness - SUBTLE version."""
        bright_norm = np.clip(brightness / 50.0, -1, 1)
        return np.clip(image + bright_norm * 0.15, 0, 1)  # Reduced from 0.3
    
    def _add_grain(self, image: np.ndarray, grain: float) -> np.ndarray:
        """Add film grain/noise for texture - SUBTLE version."""
        grain_norm = grain / 100.0
        
        # Generate very fine grain noise
        noise = np.random.normal(0, grain_norm * 0.02, image.shape)  # Reduced from 0.05
        
        # Add grain
        result = image + noise
        
        return np.clip(result, 0, 1)


# Example usage for Resident Evil 2
if __name__ == "__main__":
    enhancer = AtmosphericEnhancer()
    
    # Load image
    img = cv2.imread("/mnt/user-data/uploads/1770836062289_image.png")
    
    enhanced = enhancer.apply_eerie_atmosphere(
        img,
        blur_strength=0,      
        haze=12,              
        temp=-18,             
        tint=6,               
        saturation=-5,        
        brightness=-3,        
        contrast=8,           
        grain=12,             
        fog_color=(95, 115, 105)  
    )
    
    # Save
    cv2.imwrite("/mnt/user-data/outputs/re2_enhanced_subtle.png", enhanced)
    print("Subtle enhanced image saved!")