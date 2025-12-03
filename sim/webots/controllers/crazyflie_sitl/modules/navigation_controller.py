"""
Navigation Controller with Path Following and Obstacle Avoidance.

Orchestrates navigation by combining:
- Path planning (nav/path_planner)
- Obstacle detection (od/detector)
- Avoidance handling (nav/avoidance_handler)
- Velocity smoothing (nav/velocity_controller)
- Visualization (nav/visualizer)
"""

import os
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from .nav import (
    PathPlanner,
    NavigationMode,
    AvoidancePhase,
    VelocityController,
    StuckDetector,
    StuckDetectorConfig,
    AvoidanceHandler,
    AvoidanceConfig,
    NavigationVisualizer,
)
from .od import ObstacleDetector, ObstacleZone

# Import logger if available
try:
    import sys
    controller_dir = Path(__file__).parent.parent
    utils_dir = controller_dir.parent.parent / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")


class NavigationController:
    """
    Unified navigation controller combining path following and obstacle avoidance.
    
    The drone follows a preset path until an obstacle requires avoidance,
    then executes avoidance maneuver and returns to the path.
    """
    
    def __init__(self, 
                 depth_model_path: str,
                 cruise_speed: float = 0.25,
                 avoidance_speed: float = 0.18,
                 turn_rate: float = 1.0,
                 avoidance_duration: float = 1.5,
                 clear_distance: float = 0.5,
                 enable_visualization: bool = False,
                 save_viz_frames: bool = False,
                 config: dict = None):
        """
        Initialize navigation controller.
        
        Args:
            depth_model_path: Path to MiDaS ONNX model
            cruise_speed: Normal flight speed (m/s)
            avoidance_speed: Speed during avoidance (m/s)
            turn_rate: Maximum turn rate (rad/s)
            avoidance_duration: Minimum time for avoidance maneuver (s)
            clear_distance: Distance to move when clearing obstacle (m)
            enable_visualization: Enable live depth visualization
            save_viz_frames: Save visualization frames to disk
            config: Optional configuration dictionary
        """
        # Load config values
        drone_config = config.get('drone', {}) if config else {}
        nav_config = drone_config.get('navigation', {})
        sim_config = drone_config.get('simulation', {})
        path_config = drone_config.get('path_planning', {})
        
        # Speed parameters
        self.cruise_speed = cruise_speed
        self.turn_rate = turn_rate
        
        # Initialize path planner
        self.path_planner = PathPlanner(
            max_deviation=path_config.get('max_deviation', 1.5),
            return_speed=cruise_speed * path_config.get('return_speed_ratio', 0.7),
            lookahead_distance=nav_config.get('lookahead_distance', 0.5)
        )
        
        # Initialize obstacle detector
        self.obstacle_detector = ObstacleDetector(
            model_path=depth_model_path,
            critical_distance=0.8,
            close_distance=1.5,
            caution_distance=2.5,
            config=config
        )
        
        # Initialize velocity controller
        self.velocity_controller = VelocityController(
            smoothing_alpha=sim_config.get('smoothing_alpha', 0.10),
            avoidance_alpha=0.20,
            emergency_alpha=0.25
        )
        
        # Initialize avoidance handler
        # Get default altitude from config or environment variable (set by launch.py)
        default_altitude = nav_config.get('default_altitude', 
                                          float(os.environ.get('NAV_DEFAULT_ALTITUDE', '0.8')))
        avoidance_config = AvoidanceConfig(
            speed=avoidance_speed,
            turn_rate=turn_rate,
            duration=avoidance_duration,
            clear_distance=clear_distance,
            emergency_backup_speed=sim_config.get('emergency_backup_speed', -0.10),
            emergency_timeout=sim_config.get('emergency_timeout', 2.0),
            default_altitude=default_altitude,
            altitude_return_speed=nav_config.get('altitude_return_speed', 0.08),
            altitude_tolerance=nav_config.get('altitude_tolerance', 0.1)
        )
        self.avoidance_handler = AvoidanceHandler(avoidance_config, log_func=log)
        
        # Initialize stuck detector
        stuck_config = StuckDetectorConfig(
            enabled=sim_config.get('stuck_detection_enabled', True),
            history_size=sim_config.get('stuck_history_size', 20),
            threshold=sim_config.get('stuck_threshold', 0.05),
            time_window=sim_config.get('stuck_time_window', 3.0)
        )
        self.stuck_detector = StuckDetector(stuck_config)
        
        # Initialize visualizer
        self.visualizer = NavigationVisualizer(
            enabled=enable_visualization,
            save_frames=save_viz_frames
        )
        
        # Store for external access
        self._last_detection: Optional[dict] = None
        self._last_image: Optional[np.ndarray] = None
        self._last_position = np.array([0.0, 0.0, 0.0])
        self._altitude_adjustment = 0.0  # Current altitude change rate
        
        # Vision rate limiting
        self.vision_interval = 3
        self.vision_counter = 0
        self._cached_detection: Optional[dict] = None
        
        # Statistics
        self.total_flight_time = 0.0
        
        # Logging control
        self.log_interval = sim_config.get('print_interval', 15)
        self.update_count = 0
        
        log("[NAV] Navigation controller initialized", "SUCCESS")
    
    @property
    def mode(self) -> NavigationMode:
        """Current navigation mode."""
        return self.avoidance_handler.mode
    
    @property
    def avoidance_phase(self) -> AvoidancePhase:
        """Current avoidance phase."""
        return self.avoidance_handler.phase
    
    @property
    def avoidance_count(self) -> int:
        """Number of avoidance maneuvers."""
        return self.avoidance_handler.avoidance_count
    
    def set_path(self, waypoints: List[Tuple[float, float, float]], 
                 velocity: float = None):
        """Set navigation path with waypoints."""
        if velocity is not None:
            self.cruise_speed = velocity
        
        self.path_planner.set_waypoints(waypoints, default_velocity=self.cruise_speed)
        self.avoidance_handler.set_path_following()
        
        log(f"[NAV] Path set with {len(waypoints)} waypoints", "SUCCESS")
    
    def set_goal(self, goal: Tuple[float, float, float]):
        """Set single goal position."""
        self.set_path([goal])
    
    def update(self, 
               current_position: np.ndarray,
               current_yaw: float,
               camera_image: np.ndarray,
               dt: float) -> Tuple[float, float, float, float]:
        """
        Update navigation and compute velocity commands.
        
        Args:
            current_position: Current drone position [x, y, z]
            current_yaw: Current heading in radians
            camera_image: Camera image for obstacle detection
            dt: Time step in seconds
            
        Returns:
            Tuple of (forward_velocity, sideways_velocity, yaw_rate, altitude_change)
            altitude_change is the desired change in height_desired (m/s)
        """
        self.update_count += 1
        self.total_flight_time += dt
        
        current_position = np.array(current_position, dtype=np.float32)
        self._last_position = current_position
        
        # Update stuck detector
        self.stuck_detector.update(current_position, dt)
        
        # Update path planner
        path_result = self.path_planner.update(current_position, dt)
        
        # Check if goal reached
        if path_result['state'] == 'complete':
            self.avoidance_handler.set_goal_reached()
            return self._smooth_output(0.0, 0.0, 0.0)
        
        # Rate-limited obstacle detection
        detection = self._get_detection(camera_image)
        
        # Store for external access and visualization
        self._last_detection = detection
        self._last_image = camera_image
        
        # Update visualization
        self._update_visualization(camera_image, detection)
        
        # Process based on current mode
        mode = self.avoidance_handler.mode
        
        if mode == NavigationMode.IDLE:
            return self._smooth_output(0.0, 0.0, 0.0, 0.0)
        
        elif mode == NavigationMode.GOAL_REACHED:
            return self._smooth_output(0.0, 0.0, 0.0, 0.0)
        
        elif mode == NavigationMode.EMERGENCY_STOP:
            vx, vy, yaw, vz = self.avoidance_handler.handle_emergency(
                current_position, current_yaw, detection, dt
            )
            return self._smooth_output(vx, vy, yaw, vz, mode='emergency')
        
        elif mode == NavigationMode.AVOIDING:
            vx, vy, yaw, vz = self.avoidance_handler.execute_avoidance(
                current_position, current_yaw, detection, dt
            )
            return self._smooth_output(vx, vy, yaw, vz, mode='avoidance')
        
        elif mode == NavigationMode.RETURNING:
            vx, vy, yaw, vz = self._execute_return(
                current_position, current_yaw, path_result, detection, dt
            )
            return self._smooth_output(vx, vy, yaw, vz)
        
        else:  # PATH_FOLLOWING
            vx, vy, yaw, vz = self._execute_path_following(
                current_position, current_yaw, path_result, detection, dt
            )
            return self._smooth_output(vx, vy, yaw, vz)
    
    def _get_detection(self, camera_image: np.ndarray) -> dict:
        """Get obstacle detection with rate limiting."""
        self.vision_counter += 1
        if self.vision_counter >= self.vision_interval or self._cached_detection is None:
            self.vision_counter = 0
            self._cached_detection = self.obstacle_detector.detect(camera_image)
        return self._cached_detection
    
    def _execute_path_following(self, 
                                 position: np.ndarray,
                                 yaw: float,
                                 path_result: dict,
                                 detection: dict,
                                 dt: float) -> Tuple[float, float, float, float]:
        """Execute path following with obstacle checking."""
        zone = detection['zone']
        has_horizontal = detection.get('horizontal_obstacle', False)
        center_clearance = detection.get('center_band_clearance', 1.0)
        
        # Check for obstacles
        if zone == ObstacleZone.CRITICAL:
            self.avoidance_handler.trigger_emergency()
            vz = detection.get('vertical_direction', 0) * 0.1
            return self.avoidance_handler.config.emergency_backup_speed, 0.0, 0.0, vz
        
        elif zone == ObstacleZone.CLOSE:
            self.avoidance_handler.start_avoidance(position, detection)
            return self.avoidance_handler.execute_avoidance(position, yaw, detection, dt)
        
        # Compute speed reduction based on zone
        speed_reduction = self._compute_speed_reduction(zone, has_horizontal, center_clearance)
        
        # Compute path following command
        target_direction = path_result['direction']
        
        # Transform to body frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        body_x = target_direction[0] * cos_yaw + target_direction[1] * sin_yaw
        body_y = -target_direction[0] * sin_yaw + target_direction[1] * cos_yaw
        
        # Compute velocities
        forward = body_x * self.cruise_speed * speed_reduction
        sideways = body_y * self.cruise_speed * 0.3 * speed_reduction
        
        # Altitude adjustment based on vertical zone analysis
        vz = 0.0
        if has_horizontal:
            forward *= 0.7
            # Use vertical direction for proactive altitude adjustment
            vert_dir = detection.get('vertical_direction', 0)
            vert_mag = detection.get('vertical_magnitude', 0.0)
            if vert_dir != 0 and zone in [ObstacleZone.CAUTION, ObstacleZone.CLOSE]:
                vz = vert_dir * 0.1 * vert_mag  # Gentle altitude adjustment
        
        # If no obstacle-driven altitude change, check if we need to return to default altitude
        if vz == 0.0:
            current_altitude = position[2]
            vz = self.avoidance_handler.compute_altitude_return(current_altitude, detection)
        
        # Compute yaw to face target
        target_angle = np.arctan2(body_y, body_x)
        yaw_rate = np.clip(target_angle * 2.0, -self.turn_rate, self.turn_rate)
        
        # Log periodically
        if self.update_count % self.log_interval == 0:
            self._log_status(path_result, zone, has_horizontal, forward, sideways)
        
        return forward, sideways, yaw_rate, vz
    
    def _compute_speed_reduction(self, zone: ObstacleZone, 
                                  has_horizontal: bool,
                                  center_clearance: float) -> float:
        """Compute speed reduction based on obstacle zone."""
        if zone == ObstacleZone.CAUTION:
            return 0.5
        elif zone == ObstacleZone.FAR:
            return 0.7 if center_clearance < 0.5 else 0.9
        else:  # CLEAR
            return 0.8 if center_clearance < 0.6 else 1.0
    
    def _execute_return(self,
                        position: np.ndarray,
                        yaw: float,
                        path_result: dict,
                        detection: dict,
                        dt: float) -> Tuple[float, float, float, float]:
        """Return to path after avoidance."""
        zone = detection['zone']
        
        # Check if obstacle appeared during return
        if zone in [ObstacleZone.CRITICAL, ObstacleZone.CLOSE]:
            self.avoidance_handler.start_avoidance(position, detection)
            return self.avoidance_handler.execute_avoidance(position, yaw, detection, dt)
        
        # Check if back on path
        deviation = path_result['deviation']
        if deviation < 0.3 or path_result['on_path']:
            self.avoidance_handler.set_path_following()
            log("[NAV] ✓ Back on path - Resuming navigation", "SUCCESS")
        
        # Navigate toward path
        direction = path_result['direction']
        
        # Transform to body frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        body_x = direction[0] * cos_yaw + direction[1] * sin_yaw
        body_y = -direction[0] * sin_yaw + direction[1] * cos_yaw
        
        # Use return speed
        speed = self.path_planner.return_speed
        forward = body_x * speed
        sideways = body_y * speed * 0.3
        
        target_angle = np.arctan2(body_y, body_x)
        yaw_rate = np.clip(target_angle * 2.0, -self.turn_rate, self.turn_rate)
        
        # Compute altitude return to default when safe
        current_altitude = position[2]
        vz = self.avoidance_handler.compute_altitude_return(current_altitude, detection)
        
        return forward, sideways, yaw_rate, vz
    
    def _smooth_output(self, vx: float, vy: float, yaw: float, vz: float = 0.0,
                       mode: str = 'normal') -> Tuple[float, float, float, float]:
        """Apply smoothing to output velocities including altitude."""
        smoothed_vx, smoothed_vy, smoothed_yaw = self.velocity_controller.smooth(vx, vy, yaw, mode)
        
        # Smooth altitude adjustment separately with slower response
        alpha = 0.15 if mode == 'avoidance' else 0.1
        self._altitude_adjustment = alpha * vz + (1 - alpha) * self._altitude_adjustment
        
        return smoothed_vx, smoothed_vy, smoothed_yaw, self._altitude_adjustment
    
    def _log_status(self, path_result: dict, zone: ObstacleZone,
                    has_horizontal: bool, forward: float, sideways: float):
        """Log navigation status."""
        dist = path_result['distance']
        wp_idx = path_result['waypoint_idx']
        horiz_str = " [HORIZ]" if has_horizontal else ""
        log(f"[NAV] Following WP{wp_idx} | Dist:{dist:.2f}m | "
            f"Zone:{zone.value}{horiz_str} | vx:{forward:.2f} vy:{sideways:.2f}", "DEBUG")
    
    def _update_visualization(self, image: np.ndarray, detection: dict):
        """Update live visualization if enabled."""
        if not self.visualizer.is_active:
            return
        
        nav_state = {
            'mode': self.avoidance_handler.mode,
            'mode_str': self.avoidance_handler.mode.value,
            'phase': self.avoidance_handler.phase,
            'avoidance_count': self.avoidance_handler.avoidance_count,
            'vx': self.velocity_controller.state.vx,
            'vy': self.velocity_controller.state.vy,
            'position': self._last_position,
            'flight_time': self.total_flight_time,
        }
        
        current_wp = self.path_planner.get_current_waypoint()
        path_info = {
            'goal': current_wp.position if current_wp else None,
            'waypoint_idx': self.path_planner.current_waypoint_idx,
            'total_waypoints': len(self.path_planner.waypoints),
        }
        
        self.visualizer.update(image, detection, nav_state, path_info)
    
    def get_status(self) -> dict:
        """Get current navigation status."""
        path_status = self.path_planner.get_status()
        detector_stats = self.obstacle_detector.get_statistics()
        
        return {
            'mode': self.avoidance_handler.mode.value,
            'avoidance_phase': self.avoidance_handler.phase.value,
            'avoidance_count': self.avoidance_handler.avoidance_count,
            'flight_time': self.total_flight_time,
            'path': path_status,
            'detection': detector_stats
        }
    
    def get_visualization_state(self) -> dict:
        """Get state info formatted for visualization."""
        mode = self.avoidance_handler.mode
        phase = self.avoidance_handler.phase
        
        mode_to_state = {
            NavigationMode.IDLE: 'normal',
            NavigationMode.PATH_FOLLOWING: 'normal',
            NavigationMode.AVOIDING: 'avoidance',
            NavigationMode.RETURNING: 'recovery',
            NavigationMode.EMERGENCY_STOP: 'emergency',
            NavigationMode.GOAL_REACHED: 'normal'
        }
        
        if mode == NavigationMode.AVOIDING:
            state_name = 'caution' if phase == AvoidancePhase.TURNING else 'avoidance'
        else:
            state_name = mode_to_state.get(mode, 'normal')
        
        return {
            'state': state_name,
            'emergency_count': self.avoidance_handler.avoidance_count,
            'avoidance_count': self.avoidance_handler.avoidance_count,
            'vx': self.velocity_controller.state.vx,
            'vy': self.velocity_controller.state.vy,
            'mode': mode.value,
            'phase': phase.value
        }
    
    def enable_visualization(self, enable: bool = True, save_frames: bool = False):
        """Enable or disable live visualization."""
        if enable:
            self.visualizer.enable(save_frames)
        else:
            self.visualizer.disable()
    
    def reset(self):
        """Reset navigation controller."""
        self.avoidance_handler.reset()
        self.path_planner.reset()
        self.velocity_controller.reset()
        self.stuck_detector.reset()
        self._last_detection = None
        self._last_image = None
        log("[NAV] Navigation controller reset", "INFO")
    
    def close(self):
        """Clean up resources."""
        self.visualizer.close()


class SimpleNavigationController:
    """
    Simplified navigation controller for direct integration.
    
    Provides a simpler interface for basic path following with avoidance.
    """
    
    def __init__(self, depth_model_path: str, goal: Tuple[float, float, float],
                 enable_visualization: bool = False):
        """Initialize simple navigation."""
        self.controller = NavigationController(
            depth_model_path, 
            enable_visualization=enable_visualization
        )
        self.controller.set_goal(goal)
    
    def update(self, position: np.ndarray, yaw: float, 
               image: np.ndarray, dt: float) -> Tuple[float, float, float, str]:
        """Update navigation."""
        vx, vy, yaw_rate = self.controller.update(position, yaw, image, dt)
        status = self.controller.get_status()
        
        return vx, vy, yaw_rate, status['mode']
    
    def close(self):
        """Clean up resources."""
        self.controller.close()
