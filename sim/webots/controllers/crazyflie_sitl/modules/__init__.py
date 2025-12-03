"""
Crazyflie SITL Controller Modules

Modular components for the unified Webots controller:
- communication: TCP bridge for external SITL control
- sensors: Sensor data collection (IMU, GPS, camera, range)
- safety_monitor: Crash detection and safety monitoring
- keyboard_handler: Manual keyboard control input

Submodules:
- vision/: Depth estimation and object detection
- nav/: Path planning and navigation state machine
- od/: Obstacle detection and depth analysis
- modes/: Control mode handlers (external, vision)
"""

from .communication import CommunicationBridge
from .sensors import SensorManager
from .safety_monitor import SafetyMonitor
from .keyboard_handler import KeyboardHandler

# Vision modules from vision/ subfolder
from .vision import SimpleDepthEstimator, analyze_depth_map, VISION_AVAILABLE
from .vision import VisionProcessor, DETECTOR_AVAILABLE

# Navigation modules from nav/ subfolder
from .nav import PathPlanner, PathState, Waypoint
from .nav.state_machine import NavigationMode, AvoidancePhase

# Obstacle detection from od/ subfolder
from .od import ObstacleDetector, ObstacleZone, SimpleObstacleDetector

# Navigation controller
from .navigation_controller import NavigationController, SimpleNavigationController

# Control mode handlers
from .modes import ExternalModeHandler, VisionModeHandler

__all__ = [
    # Core modules
    'CommunicationBridge',
    'SensorManager',
    'SafetyMonitor',
    'KeyboardHandler',
    'SimpleDepthEstimator',
    'analyze_depth_map',
    'VISION_AVAILABLE',
    'VisionProcessor',
    'DETECTOR_AVAILABLE',
    # Navigation modules
    'PathPlanner',
    'PathState',
    'Waypoint',
    'NavigationMode',
    'AvoidancePhase',
    # Obstacle detection
    'ObstacleDetector',
    'ObstacleZone',
    'SimpleObstacleDetector',
    # Navigation controller
    'NavigationController',
    'SimpleNavigationController',
    # Mode handlers
    'ExternalModeHandler',
    'VisionModeHandler',
]
