"""
Navigation submodule for path planning and control.

Components:
- path_planner: Waypoint-based path planning with deviation recovery
- state_machine: Navigation mode and avoidance phase enums
- velocity_controller: Velocity smoothing and stuck detection
- avoidance_handler: Obstacle avoidance state machine
- visualizer: Live navigation visualization
"""

from .path_planner import PathPlanner, Waypoint
from .state_machine import NavigationMode, AvoidancePhase, PathState
from .velocity_controller import VelocityController, StuckDetector, StuckDetectorConfig
from .avoidance_handler import AvoidanceHandler, AvoidanceConfig, AvoidanceState
from .visualizer import NavigationVisualizer

__all__ = [
    # Path planning
    'PathPlanner',
    'PathState',
    'Waypoint',
    # State machine
    'NavigationMode',
    'AvoidancePhase',
    # Velocity control
    'VelocityController',
    'StuckDetector',
    'StuckDetectorConfig',
    # Avoidance
    'AvoidanceHandler',
    'AvoidanceConfig',
    'AvoidanceState',
    # Visualization
    'NavigationVisualizer',
]
