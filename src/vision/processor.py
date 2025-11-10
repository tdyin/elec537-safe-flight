"""Image preprocessing and augmentation utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """Handles image preprocessing and enhancement."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image dimensions (width, height)
        """
        self.target_size = target_size
    
    def resize(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            size: Target size (width, height). Uses default if None.
            
        Returns:
            Resized image
        """
        size = size or self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def preprocess(self, image: np.ndarray, 
                   enhance: bool = True, 
                   denoise: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Input image
            enhance: Whether to enhance contrast
            denoise: Whether to apply denoising
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        if denoise:
            processed = self.denoise(processed)
        
        if enhance:
            processed = self.enhance_contrast(processed)
        
        processed = self.resize(processed)
        processed = self.normalize(processed)
        
        return processed
