"""Unit tests for sensor fusion module."""

import pytest
import numpy as np

from fusion import SensorFusion


class TestSensorFusion:
    """Test SensorFusion class."""
    
    def test_initialization(self):
        """Test fusion initialization."""
        fusion = SensorFusion(vision_weight=0.6, lidar_weight=0.4)
        assert abs(fusion.vision_weight + fusion.lidar_weight - 1.0) < 1e-6
    
    def test_fuse_empty(self):
        """Test fusion with empty detections."""
        fusion = SensorFusion()
        result = fusion.fuse([], [])
        assert len(result) == 0
    
    def test_fuse_vision_only(self):
        """Test fusion with only vision detections."""
        fusion = SensorFusion()
        vision_detections = [
            {'position': np.array([1.0, 0.0, 0.5]), 'confidence': 0.8}
        ]
        result = fusion.fuse(vision_detections, [])
        assert len(result) == 1
        assert result[0]['type'] == 'vision_only'
    
    def test_fuse_lidar_only(self):
        """Test fusion with only LiDAR detections."""
        fusion = SensorFusion()
        lidar_detections = [
            {'centroid': np.array([1.0, 0.0, 0.5]), 'size': np.array([0.1, 0.1, 0.3])}
        ]
        result = fusion.fuse([], lidar_detections)
        assert len(result) == 1
        assert result[0]['type'] == 'lidar_only'
    
    def test_fuse_both(self):
        """Test fusion with both modalities."""
        fusion = SensorFusion(confidence_threshold=0.0)
        vision_detections = [
            {'position': np.array([1.0, 0.0, 0.5]), 'confidence': 0.8}
        ]
        lidar_detections = [
            {'centroid': np.array([1.1, 0.1, 0.5]), 'size': np.array([0.1, 0.1, 0.3])}
        ]
        result = fusion.fuse(vision_detections, lidar_detections)
        assert len(result) > 0
    
    def test_update_weights(self):
        """Test weight updating."""
        fusion = SensorFusion()
        fusion.update_weights(0.7, 0.3)
        assert abs(fusion.vision_weight - 0.7) < 1e-6
        assert abs(fusion.lidar_weight - 0.3) < 1e-6
