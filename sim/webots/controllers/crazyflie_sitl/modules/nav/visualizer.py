"""
Navigation Visualizer.

Wraps the live depth visualization for navigation feedback.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .state_machine import NavigationMode, AvoidancePhase

# Optional visualization support
try:
    import sys
    utils_dir = Path(__file__).parent.parent.parent.parent / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from viz_depth_live import LiveDepthVisualizer
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    LiveDepthVisualizer = None

# Import logger
try:
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")


class NavigationVisualizer:
    """
    Handles visualization for navigation system.
    
    Provides live depth map visualization with navigation overlays.
    """
    
    def __init__(self, 
                 enabled: bool = False,
                 save_frames: bool = False,
                 window_name: str = 'Navigation - Depth Analysis',
                 display_size: Tuple[int, int] = (800, 480)):
        """
        Initialize navigation visualizer.
        
        Args:
            enabled: Enable visualization on startup
            save_frames: Save visualization frames to disk
            window_name: Window title
            display_size: Display window size (width, height)
        """
        self.enabled = enabled
        self.save_frames = save_frames
        self.window_name = window_name
        self.display_size = display_size
        
        self._visualizer: Optional[LiveDepthVisualizer] = None
        self._update_count = 0
        
        if enabled:
            self._create_visualizer()
    
    @property
    def is_available(self) -> bool:
        """Check if visualization library is available."""
        return VIZ_AVAILABLE
    
    @property
    def is_active(self) -> bool:
        """Check if visualizer is currently active."""
        return self._visualizer is not None
    
    def _create_visualizer(self):
        """Create the visualizer instance."""
        if not VIZ_AVAILABLE:
            log("[NAV] Visualization not available (missing dependencies)", "WARNING")
            return
        
        try:
            save_dir = None
            if self.save_frames:
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
                save_dir = str(project_root / 'data' / 'visualization' / 'nav_frames')
            
            self._visualizer = LiveDepthVisualizer(
                window_name=self.window_name,
                display_size=self.display_size,
                save_dir=save_dir
            )
            log("[NAV] Live visualization enabled", "SUCCESS")
        except Exception as e:
            log(f"[NAV] Visualization failed: {e}", "WARNING")
            self._visualizer = None
    
    def enable(self, save_frames: bool = False):
        """Enable visualization."""
        self.enabled = True
        self.save_frames = save_frames
        if self._visualizer is None:
            self._create_visualizer()
    
    def disable(self):
        """Disable visualization."""
        self.enabled = False
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
            log("[NAV] Live visualization disabled", "INFO")
    
    def update(self, 
               image: np.ndarray,
               detection: dict,
               nav_state: Dict[str, Any],
               path_info: Dict[str, Any]):
        """
        Update visualization with current frame.
        
        Args:
            image: Camera image
            detection: Obstacle detection result
            nav_state: Navigation state info
            path_info: Path planner info
        """
        if self._visualizer is None or image is None:
            return
        
        self._update_count += 1
        
        try:
            depth_map = detection.get('raw_depth')
            if depth_map is None:
                return
            
            # Build analysis dict for visualizer
            # Include both horizontal and vertical zone clearances
            analysis = {
                'left': detection['clearance']['left'],
                'center': detection['clearance']['center'],
                'right': detection['clearance']['right'],
                'safe_direction': detection['direction'],
                'roi_bounds': detection.get('roi_bounds', (50, 150)),
                # Vertical zone analysis
                'vertical_clearance': detection.get('vertical_clearance', {
                    'upper': 1.0, 'middle': 1.0, 'lower': 1.0
                }),
                'upper_clearance': detection.get('vertical_clearance', {}).get('upper', 1.0),
                'middle_clearance': detection.get('vertical_clearance', {}).get('middle', 1.0),
                'lower_clearance': detection.get('vertical_clearance', {}).get('lower', 1.0),
                'optimal_vertical_direction': detection.get('vertical_direction', 0),
                'vertical_escape_margin': detection.get('vertical_magnitude', 0.0),
                'horizontal_obstacle': detection.get('horizontal_obstacle', False)
            }
            
            # Build state info
            state_info = self._build_state_info(nav_state)
            
            # Build flight info
            flight_info = self._build_flight_info(nav_state, path_info, detection)
            
            self._visualizer.visualize(image, depth_map, analysis, state_info, flight_info)
            
        except Exception as e:
            if self._update_count % 100 == 0:
                log(f"[NAV] Visualization error: {e}", "WARNING")
    
    def _build_state_info(self, nav_state: Dict[str, Any]) -> dict:
        """Build state info dict for visualizer."""
        mode = nav_state.get('mode', NavigationMode.IDLE)
        phase = nav_state.get('phase', AvoidancePhase.NONE)
        
        # Map mode to visualization state
        mode_to_state = {
            NavigationMode.IDLE: 'normal',
            NavigationMode.PATH_FOLLOWING: 'normal',
            NavigationMode.AVOIDING: 'avoidance',
            NavigationMode.RETURNING: 'recovery',
            NavigationMode.EMERGENCY_STOP: 'emergency',
            NavigationMode.GOAL_REACHED: 'normal'
        }
        
        if mode == NavigationMode.AVOIDING:
            if phase == AvoidancePhase.TURNING:
                state_name = 'caution'
            else:
                state_name = 'avoidance'
        else:
            state_name = mode_to_state.get(mode, 'normal')
        
        return {
            'state': state_name,
            'emergency_count': nav_state.get('avoidance_count', 0),
            'avoidance_count': nav_state.get('avoidance_count', 0),
            'vx': nav_state.get('vx', 0.0),
            'vy': nav_state.get('vy', 0.0),
            'mode': mode.value if hasattr(mode, 'value') else str(mode),
            'phase': phase.value if hasattr(phase, 'value') else str(phase)
        }
    
    def _build_flight_info(self, nav_state: Dict[str, Any], 
                           path_info: Dict[str, Any],
                           detection: Dict[str, Any] = None) -> dict:
        """Build flight info dict for visualizer."""
        position = nav_state.get('position', (0, 0, 0))
        goal = path_info.get('goal')
        
        distance_to_goal = 0.0
        if goal is not None:
            diff = np.array(goal) - np.array(position)
            distance_to_goal = float(np.linalg.norm(diff))
        
        # Get vertical avoidance info from detection
        vertical_direction = 0
        altitude_change = 0.0
        if detection:
            vertical_direction = detection.get('vertical_direction', 0)
            altitude_change = detection.get('vertical_magnitude', 0.0) * vertical_direction * 0.15
        
        return {
            'position': tuple(position),
            'velocity': (nav_state.get('vx', 0.0), nav_state.get('vy', 0.0)),
            'goal': tuple(goal) if goal is not None else None,
            'mode': nav_state.get('mode_str', 'unknown'),
            'waypoint_idx': path_info.get('waypoint_idx', 0),
            'total_waypoints': path_info.get('total_waypoints', 0),
            'distance_to_goal': distance_to_goal,
            'flight_time': nav_state.get('flight_time', 0.0),
            'vertical_direction': vertical_direction,
            'altitude_change': altitude_change
        }
    
    def close(self):
        """Clean up visualizer resources."""
        self.disable()
