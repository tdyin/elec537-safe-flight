"""Unit tests for LiDAR module."""

import pytest
import numpy as np

from lidar import LidarProcessor, LidarObstacleDetector


class TestLidarProcessor:
    """Test LidarProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = LidarProcessor(max_range=10.0, min_range=0.1)
        assert processor.max_range == 10.0
        assert processor.min_range == 0.1
    
    def test_filter_range(self):
        """Test range filtering."""
        processor = LidarProcessor(max_range=5.0, min_range=0.5)
        points = np.array([
            [0.1, 0.0, 0.0],  # Too close
            [1.0, 0.0, 0.0],  # OK
            [10.0, 0.0, 0.0]  # Too far
        ])
        filtered = processor.filter_range(points)
        assert len(filtered) == 1
    
    def test_voxel_downsample(self):
        """Test voxel downsampling."""
        processor = LidarProcessor()
        points = np.random.randn(1000, 3)
        downsampled = processor.voxel_downsample(points, voxel_size=0.1)
        assert len(downsampled) < len(points)
        assert downsampled.shape[1] == 3
    
    def test_segment_ground(self):
        """Test ground segmentation."""
        processor = LidarProcessor()
        # Create points with clear ground plane
        ground = np.random.randn(100, 3) * 0.05
        non_ground = np.random.randn(100, 3) * 0.5 + np.array([0, 0, 1.0])
        points = np.vstack([ground, non_ground])
        
        ground_pts, non_ground_pts = processor.segment_ground(points, threshold=0.2)
        assert len(ground_pts) > 0
        assert len(non_ground_pts) > 0


class TestLidarObstacleDetector:
    """Test LidarObstacleDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = LidarObstacleDetector(
            cluster_tolerance=0.1,
            min_cluster_size=10,
            max_cluster_size=1000
        )
        assert detector.cluster_tolerance == 0.1
        assert detector.min_cluster_size == 10
    
    def test_detect_empty(self):
        """Test detection with empty point cloud."""
        detector = LidarObstacleDetector()
        obstacles = detector.detect(np.array([]).reshape(0, 3))
        assert len(obstacles) == 0
    
    def test_extract_obstacle_info(self):
        """Test obstacle information extraction."""
        detector = LidarObstacleDetector()
        cluster = np.random.randn(50, 3)
        info = detector._extract_obstacle_info(cluster)
        
        assert 'centroid' in info
        assert 'min_bounds' in info
        assert 'max_bounds' in info
        assert 'size' in info
        assert info['num_points'] == 50
