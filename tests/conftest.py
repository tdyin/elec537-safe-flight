"""Pytest configuration file."""

import pytest
import numpy as np


@pytest.fixture
def sample_image():
    """Fixture providing a sample image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_point_cloud():
    """Fixture providing a sample point cloud."""
    return np.random.randn(1000, 3) * 2.0


@pytest.fixture
def sample_obstacles():
    """Fixture providing sample obstacle detections."""
    return [
        {
            'centroid': np.array([1.0, 0.0, 0.5]),
            'size': np.array([0.1, 0.1, 0.3]),
            'num_points': 50
        },
        {
            'centroid': np.array([2.0, 1.0, 0.3]),
            'size': np.array([0.2, 0.2, 0.5]),
            'num_points': 75
        }
    ]
