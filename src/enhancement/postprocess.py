"""Sharpening, denoising, and color correction helpers."""


import cv2 as cv
import numpy as np
from typing import Union 
from pathlib import Path


class Postprocess:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents = True, exist_ok = True)
        
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        ''' Applies a sharpening filter to enhance edges and details in the image.'''
        
        # Apply Gaussian blur for the mask
        blurred = cv.GaussianBlur(image, (0, 0), sigmaX=2)
        # Unsharp mask: original + (original - blurred) * amount
        amount = 0.5  # Strength; reduced to prevent over-sharpening artifacts
        
        sharpened = cv.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        ''' Applies a denoising filter to reduce noise while preserving edges.'''
        # Denoising using fastNlMeansDenoisingColored with gentler parameters
        denoised = cv.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        return denoised
    
    def color_correct(self, image: np.ndarray) -> np.ndarray:
        ''' Adjusts the color balance of the image to enhance visual appeal.'''
        
        # Convert to YUV for equalization
        img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        
        # Equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
        
        # Convert back to BGR color space
        corrected = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        
        return corrected
        
    
    def apply_all(self, img: np.ndarray) -> np.ndarray:
        """ Apply all postprocessing steps in sequence: denoise -> sharpen -> color_correct."""
        img = self.denoise(img)
        img = self.sharpen(img)
        img = self.color_correct(img)
        return img