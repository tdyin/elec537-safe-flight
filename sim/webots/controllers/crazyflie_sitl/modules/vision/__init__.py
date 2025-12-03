"""
Vision submodule for depth estimation and object detection.

Components:
- estimator: Depth estimation using MiDaS ONNX model
- processor: Object detection processing and wire detection
"""

from .estimator import SimpleDepthEstimator, analyze_depth_map, VISION_AVAILABLE
from .processor import VisionProcessor, DETECTOR_AVAILABLE

__all__ = [
    # Depth estimation
    'SimpleDepthEstimator',
    'analyze_depth_map',
    'VISION_AVAILABLE',
    # Object detection
    'VisionProcessor',
    'DETECTOR_AVAILABLE',
]
