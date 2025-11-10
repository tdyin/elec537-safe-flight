"""Unit tests for vision module."""

import pytest
import numpy as np
import cv2

from vision import ObstacleDetector, ImageProcessor


class TestImageProcessor:
    """Test ImageProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = ImageProcessor(target_size=(224, 224))
        assert processor.target_size == (224, 224)
    
    def test_resize(self):
        """Test image resizing."""
        processor = ImageProcessor(target_size=(224, 224))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = processor.resize(image)
        assert resized.shape == (224, 224, 3)
    
    def test_normalize(self):
        """Test image normalization."""
        processor = ImageProcessor()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = processor.normalize(image)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
    
    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        processor = ImageProcessor()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = processor.enhance_contrast(image)
        assert enhanced.shape == image.shape
    
    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        processor = ImageProcessor(target_size=(224, 224))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = processor.preprocess(image)
        assert processed.shape == (224, 224, 3)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0


class TestObstacleDetector:
    """Test ObstacleDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = ObstacleDetector(use_gpu=False)
        assert detector.device.type == 'cpu'
    
    def test_classical_wire_detection(self):
        """Test classical wire detection."""
        detector = ObstacleDetector(use_gpu=False)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = detector.detect_wires_classical(image)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
