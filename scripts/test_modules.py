"""Example script for testing obstacle detection."""

import cv2
import numpy as np
from pathlib import Path

from vision import ObstacleDetector, ImageProcessor
from lidar import LidarProcessor, LidarObstacleDetector
from fusion import SensorFusion


def test_vision_detection():
    """Test vision-based obstacle detection."""
    print("Testing vision detection...")
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector and processor
    detector = ObstacleDetector(use_gpu=False)
    processor = ImageProcessor()
    
    # Process image
    processed = processor.preprocess(image, enhance=True)
    
    # Test classical detection (no model needed)
    wire_mask = detector.detect_wires_classical(image)
    
    print(f"Image shape: {image.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Wire mask shape: {wire_mask.shape}")
    print("Vision detection test passed!\n")


def test_lidar_detection():
    """Test LiDAR-based obstacle detection."""
    print("Testing LiDAR detection...")
    
    # Create dummy point cloud
    points = np.random.randn(1000, 3) * 2.0
    
    # Initialize processor and detector
    processor = LidarProcessor()
    detector = LidarObstacleDetector()
    
    # Process point cloud
    processed = processor.process(points)
    
    # Detect obstacles
    obstacles = detector.detect(processed)
    
    print(f"Input points: {len(points)}")
    print(f"Processed points: {len(processed)}")
    print(f"Detected obstacles: {len(obstacles)}")
    print("LiDAR detection test passed!\n")


def test_sensor_fusion():
    """Test sensor fusion."""
    print("Testing sensor fusion...")
    
    # Create dummy detections
    vision_detections = [
        {'position': np.array([1.0, 0.0, 0.5]), 'confidence': 0.8},
        {'position': np.array([2.0, 1.0, 0.3]), 'confidence': 0.6}
    ]
    
    lidar_detections = [
        {'centroid': np.array([1.1, 0.1, 0.5]), 'size': np.array([0.1, 0.1, 0.3])},
        {'centroid': np.array([3.0, 0.0, 0.4]), 'size': np.array([0.2, 0.2, 0.5])}
    ]
    
    # Initialize fusion
    fusion = SensorFusion()
    
    # Fuse detections
    fused = fusion.fuse(vision_detections, lidar_detections)
    
    print(f"Vision detections: {len(vision_detections)}")
    print(f"LiDAR detections: {len(lidar_detections)}")
    print(f"Fused detections: {len(fused)}")
    print("Sensor fusion test passed!\n")


if __name__ == '__main__':
    print("Running Safe Path module tests...\n")
    
    test_vision_detection()
    test_lidar_detection()
    test_sensor_fusion()
    
    print("All tests completed successfully!")
