"""Vision-based obstacle detector using deep learning and classical CV."""

import torch
import torch.nn as nn
import cv2
import numpy as np
from loguru import logger


class ObstacleDetector:
    """Detects small obstacles like wires and cables using computer vision."""
    
    def __init__(self, model_path=None, use_gpu=True):
        """
        Initialize the obstacle detector.
        
        Args:
            model_path: Path to pretrained model weights
            use_gpu: Whether to use GPU acceleration
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model path provided. Initialize model before inference.")
    
    def load_model(self, model_path):
        """Load pretrained model weights."""
        try:
            # Placeholder for actual model loading
            # self.model = torch.load(model_path, map_location=self.device)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image):
        """
        Detect obstacles in the input image.
        
        Args:
            image: Input image (numpy array or torch tensor)
            
        Returns:
            Detection results with bounding boxes and confidence scores
        """
        if self.model is None:
            logger.warning("Model not initialized. Returning empty detections.")
            return []
        
        # Preprocess image
        processed = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            detections = self.model(processed)
        
        # Post-process detections
        results = self._postprocess(detections)
        
        return results
    
    def _preprocess(self, image):
        """Preprocess image for model input."""
        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def _postprocess(self, detections):
        """Post-process model outputs."""
        # Placeholder for post-processing logic
        return detections
    
    def detect_wires_classical(self, image):
        """
        Detect wires using classical computer vision techniques.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Binary mask of detected wires
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        # Create mask
        mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
        return mask
