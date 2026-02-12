"""Inference wrapper around Real-ESRGAN models."""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class Upscaler:
    """
    Real-ESRGAN upscaler using the official basicsr library.
    Handles model loading, inference, and image saving.
    """
    
    def __init__(self, model_name: str = 'RealESRGAN_x4plus', output_path: Union[str, Path] = 'data/enhanced/'):
        """
        Initialize the upscaler.
        
        Args:
            model_name: Model variant ('RealESRGAN_x4plus', 'RealESRGAN_x2plus', etc.)
            output_path: Directory to save enhanced images
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        model_configs = {
            'RealESRGAN_x4plus': {
                'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 
                'num_block': 23, 'num_grow_ch': 32, 'scale': 4
            },
            'RealESRGAN_x2plus': {
                'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 
                'num_block': 23, 'num_grow_ch': 32, 'scale': 2
            },
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_configs.keys())}")
        
        # Build model
        self.model_name = model_name
        config = model_configs[model_name]
        self.scale = config['scale']
        
        model = RRDBNet(
            num_in_ch=config['num_in_ch'],
            num_out_ch=config['num_out_ch'],
            num_feat=config['num_feat'],
            num_block=config['num_block'],
            num_grow_ch=config['num_grow_ch'],
            scale=config['scale']
        )
        
        # Initialize RealESRGANer
        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=f'weights/{model_name}.pth',
            model=model,  # Use 'model' not 'upscale_model'
            tile=256,  # Tile size for memory efficiency
            tile_pad=10,  # Padding between tiles
            pre_pad=0,
            half=torch.cuda.is_available(),  # Use FP16 if on GPU
            device=self.device
        )
    
    def upscale_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Upscale a tensor directly (from transforms).
        Input: [1, 3, H, W] or [3, H, W] in [-1, 1] range
        Output: [1, 3, H*scale, W*scale] in [-1, 1] range
        """
        # Convert tensor to numpy uint8 BGR format
        img = self._tensor_to_uint8(tensor)
        
        # Upscale using RealESRGANer (returns BGR uint8 numpy array)
        output_img, _ = self.upsampler.enhance(img, outscale=self.scale)
        
        # Convert back to tensor
        output_tensor = self._uint8_to_tensor(output_img)
        return output_tensor
    
    def upscale_image(self, input_path: Union[str, Path]) -> np.ndarray:
        """
        Upscale an image file directly.
        Input: Path to image file
        Output: BGR upscaled image (numpy uint8)
        """
        input_path = Path(input_path)
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")
        
        # Upscale (RealESRGAN handles BGR format natively)
        output_img, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output_img
    
    def _tensor_to_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor [-1, 1] to uint8 BGR image [0, 255]."""
        # Handle both [1, 3, H, W] and [3, H, W]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # [3, H, W] -> [H, W, 3] and [-1, 1] -> [0, 1]
        img = (tensor.permute(1, 2, 0) + 1) / 2
        
        # [0, 1] -> [0, 255]
        img = (img.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        
        # RGB -> BGR (RealESRGAN expects BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def _uint8_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 BGR image [0, 255] to tensor [-1, 1]."""
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # [0, 1] -> [-1, 1]
        img = img * 2 - 1
        
        # [H, W, 3] -> [1, 3, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def upscale_array(self, img: np.ndarray) -> np.ndarray:
        """
        Public helper: upscale a BGR uint8 numpy image in-memory and return upscaled BGR uint8 numpy image.

        Args:
            img: BGR uint8 image

        Returns:
            upscaled BGR uint8 image
        """
        if img is None:
            raise ValueError("img must be a valid numpy array")

        # RealESRGANer accepts BGR uint8 arrays directly
        output_img, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output_img
    
    def save_enhanced(self, output_img: np.ndarray, filename: str):
        """Save upscaled image to output path."""
        output_file = self.output_path / f"{filename}_enhanced.png"
        cv2.imwrite(str(output_file), output_img)
        print(f"Saved enhanced image: {output_file}")