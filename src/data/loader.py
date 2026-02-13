"""Placeholder for the dataset preparation pipeline."""

# TODO: unzip archives, validate contents, and feed to torch DataLoader

import os
import rarfile
from pathlib import Path
from typing import List
from PIL import Image


class Loader:
    def __init__(self, file_path: str | Path, output_path: str | Path):
        self.file_path = Path(file_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents = True, exist_ok = True)
        
    
    def extract_images(self) -> List[Path]:
        ''' Extracts images from the rar file or directory and returns a list of paths.'''
        if not self.file_path.exists():
            raise FileNotFoundError(f"File or directory {self.file_path} does not exist.")
        
        extracted_files = []
        
        # Input is a directory
        if self.file_path.is_dir():
            print(f"Loading images from directory: {self.file_path}")
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
                extracted_files.extend(list(self.file_path.rglob(f"*{ext}")))
                extracted_files.extend(list(self.file_path.rglob(f"*{ext.upper()}")))
            return extracted_files

        # Input is a RAR file
        print(f"Extracting images from archive: {self.file_path}")
        try:
            with rarfile.RarFile(self.file_path) as rar:
                for file_info in rar.infolist():
                    if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        rar.extract(file_info, path=self.output_path)
                        extracted_files.append(self.output_path / file_info.filename)
        except rarfile.RarCannotExec:
            raise RuntimeError(
                "Cannot find 'unrar' executable. To extract RAR files, please install unrar "
                "and add it to your PATH, or provide a directory path instead."
            )
                    
        return extracted_files
    

    def _validate_images(self) -> bool:
        ''' Validates the extracted images by checking if they can be opened.'''

        for image_path in self.output_path.glob('*'):
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Invalid image file: {image_path} - {e}")
                return False
        return True            


        
                