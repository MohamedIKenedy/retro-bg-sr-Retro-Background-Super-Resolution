#!/bin/bash

# Script to download Real-ESRGAN models into weights folder
# Official models from: https://github.com/xinntao/Real-ESRGAN

set -e  # Exit on any error

echo "Creating weights directory..."
mkdir -p weights

echo "Downloading RealESRGAN models..."

# RealESRGAN x4+ model (recommended for 4x upscaling)
if [ ! -f "weights/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading RealESRGAN_x4plus.pth..."
    curl -L -o weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    echo "Downloaded RealESRGAN_x4plus.pth"
else
    echo "RealESRGAN_x4plus.pth already exists, skipping..."
fi

# RealESRGAN x2+ model (for 2x upscaling)
if [ ! -f "weights/RealESRGAN_x2plus.pth" ]; then
    echo "Downloading RealESRGAN_x2plus.pth..."
    curl -L -o weights/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth
    echo "Downloaded RealESRGAN_x2plus.pth"
else
    echo "RealESRGAN_x2plus.pth already exists, skipping..."
fi

echo "Model download complete!"
echo "Available models in weights/:"
ls -lh weights/

