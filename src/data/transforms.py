"""Resize, normalise, and tile augmentations for ESRGAN."""

# TODO: add resize/normalize helpers tailored to ESRGAN inputs

import os
from PIL import Image
import cv2 as cv  
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

import torch




class Transforms:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents = True, exist_ok = True)
        
    
    def __call__(self, input_data: Union[str, Path, np.ndarray]) -> torch.Tensor:
        ''' Applies the transformations to the input image and returns a tensor.'''
        # Load the images with cv2 if path is provided
        if isinstance(input_data, (str, Path)):
            img = cv.imread(str(input_data))
            if img is None:
                raise ValueError(f"Could not read image: {input_data}")
        else:
            img = input_data
        
        # Convert BGR to RGB     
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Scale to -1 to 1
        img = img * 2 - 1 
        
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # Convert to 1xCxHxW
        
        return tensor
    
    def save_tensor(self, tensor: torch.Tensor, filename: str):
        ''' Saves the tensor as an image file.'''
        output_file = self.output_path / filename
        torch.save(tensor, output_file)
        print(f"Saved tensor to {output_file}")
        
        
        

        
        