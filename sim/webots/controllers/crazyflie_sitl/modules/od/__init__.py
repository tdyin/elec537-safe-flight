"""
Obstacle detection submodule for vision-based obstacle avoidance.

Components:
- zones: Obstacle zone classification (CLEAR, FAR, CAUTION, CLOSE, CRITICAL)
- depth_analyzer: Depth map analysis for obstacle detection
- detector: Main obstacle detector using MiDaS depth estimation
"""

from .zones import ObstacleZone
from .depth_analyzer import DepthAnalyzer
from .detector import ObstacleDetector, SimpleObstacleDetector

__all__ = [
    'ObstacleZone',
    'DepthAnalyzer',
    'ObstacleDetector',
    'SimpleObstacleDetector',
]
