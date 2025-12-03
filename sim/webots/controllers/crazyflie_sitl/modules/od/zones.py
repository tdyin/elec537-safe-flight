"""
Obstacle zone classification.

Defines the zone classification system for obstacle detection.
"""

from enum import Enum


class ObstacleZone(Enum):
    """Classification of obstacle zones."""
    CLEAR = "clear"           # No obstacle in detection range
    FAR = "far"               # Obstacle detected but far away
    CAUTION = "caution"       # Obstacle approaching, prepare to maneuver
    CLOSE = "close"           # Obstacle close, maneuver required
    CRITICAL = "critical"     # Obstacle very close, emergency action
