"""Navigation controller with depth-based obstacle avoidance."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
import cv2

from ..planning.path_planner import AStarPlanner
from ..planning.trajectory_smoother import TrajectorySmootherBezier, VelocityProfileGenerator
from ..planning.avoidance_controller import (
    StableAvoidanceController, 
    AvoidanceConfig,
    AvoidanceState
)


class DepthNavigationController:
    """
    Navigation controller using monocular depth estimation
    for intelligent obstacle avoidance.
    
    This controller uses:
    1. Depth maps to estimate distances to obstacles
    2. Zone-based analysis for directional clearance
    3. Multi-zone directional planning for smooth avoidance
    4. Path planning with trajectory smoothing
    """
    
    def __init__(self,
                 max_speed: float = 0.5,
                 safety_distance: float = 0.5,
                 avoidance_gain: float = 1.0,
                 crash_tilt_threshold: float = 45.0,
                 crash_altitude_threshold: float = 0.1,
                 max_altitude: float = 2.5,
                 min_altitude: float = 0.5,
                 planning_horizon: int = 5,
                 use_path_planning: bool = True,
                 replan_interval: int = 10,
                 path_smoothing: str = 'bezier',
                 use_stable_avoidance: bool = True,
                 config: dict = None):
        """
        Initialize depth-based navigation controller.
        
        Args:
            max_speed: Maximum velocity (m/s)
            safety_distance: Minimum distance to obstacles (m)
            avoidance_gain: Gain for avoidance maneuvers
            crash_tilt_threshold: Maximum tilt angle in degrees
            crash_altitude_threshold: Minimum altitude (m)
            max_altitude: Maximum allowed altitude (m)
            min_altitude: Minimum safe flight altitude (m)
            planning_horizon: Look-ahead frames for planning
            use_path_planning: Enable A* path planning for smooth navigation
            replan_interval: Replan path every N frames
            path_smoothing: Smoothing method ('bezier', 'spline', or 'none')
            use_stable_avoidance: Use StableAvoidanceController for improved stability
            config: Optional configuration dictionary from config/sim.yaml or config/hardware.yaml
        """
        # Load config values if provided
        drone_config = config.get('drone', {}) if config else {}
        nav_config = drone_config.get('navigation', {})
        path_config = drone_config.get('path_planning', {})
        avoid_config = drone_config.get('avoidance', {})
        
        self.max_speed = max_speed
        self.safety_distance = safety_distance
        self.avoidance_gain = avoid_config.get('avoidance_gain', avoidance_gain)
        self.crash_tilt_threshold = np.deg2rad(nav_config.get('crash_tilt_threshold', crash_tilt_threshold))
        self.crash_altitude_threshold = nav_config.get('crash_altitude_threshold', crash_altitude_threshold)
        self.max_altitude = nav_config.get('max_altitude', max_altitude)
        self.min_altitude = nav_config.get('min_altitude', min_altitude)
        self.planning_horizon = avoid_config.get('planning_horizon', planning_horizon)
        self.use_path_planning = path_config.get('enabled', use_path_planning)
        self.replan_interval = path_config.get('replan_interval', replan_interval)
        self.path_smoothing = path_config.get('path_smoothing', path_smoothing)
        self.use_stable_avoidance = use_stable_avoidance
        
        # State tracking
        self.crashed = False
        self.crash_detection_enabled = False
        self.takeoff_complete = False
        self.min_flight_altitude = nav_config.get('min_flight_altitude', 0.3)
        
        # History for temporal smoothing (from config)
        self.velocity_history = []
        self.max_history = avoid_config.get('max_history', 5)
        
        # Get avoidance config values with ratios from config
        cruise_speed_ratio = avoid_config.get('cruise_speed_ratio', 0.6)
        caution_speed_ratio = avoid_config.get('caution_speed_ratio', 0.3)
        caution_distance_ratio = avoid_config.get('caution_distance_ratio', 2.0)
        detection_distance_ratio = avoid_config.get('detection_distance_ratio', 4.0)
        emergency_distance_ratio = avoid_config.get('emergency_distance_ratio', 0.5)
        repulsive_gain_ratio = avoid_config.get('repulsive_gain_ratio', 1.5)
        velocity_smoothing = avoid_config.get('velocity_smoothing', 0.2)
        recovery_smoothing = avoid_config.get('recovery_smoothing', 0.1)
        
        # Initialize stable avoidance controller
        if self.use_stable_avoidance:
            avoidance_config = AvoidanceConfig(
                max_speed=max_speed,
                cruise_speed=max_speed * cruise_speed_ratio,
                caution_speed=max_speed * caution_speed_ratio,
                safety_distance=safety_distance,
                caution_distance=safety_distance * caution_distance_ratio,
                detection_distance=safety_distance * detection_distance_ratio,
                emergency_distance=safety_distance * emergency_distance_ratio,
                repulsive_gain=self.avoidance_gain * repulsive_gain_ratio,
                velocity_smoothing=velocity_smoothing,
                recovery_smoothing=recovery_smoothing
            )
            self.stable_avoidance = StableAvoidanceController(avoidance_config)
            logger.info("Stable avoidance controller enabled")
        
        # Path planning components
        if self.use_path_planning:
            self.path_planner = AStarPlanner(
                planning_resolution=path_config.get('planning_resolution', 0.1),
                safety_margin=path_config.get('safety_margin', 3),
                smoothness_weight=path_config.get('smoothness_weight', 0.4)
            )
            
            if self.path_smoothing == 'bezier':
                self.trajectory_smoother = TrajectorySmootherBezier(control_point_ratio=0.3)
            else:
                self.trajectory_smoother = None
            
            self.velocity_profiler = VelocityProfileGenerator(
                max_velocity=max_speed,
                max_acceleration=avoid_config.get('max_acceleration', 0.3),
                profile_type='s-curve'
            )
            
            # Path tracking state
            self.current_path = None
            self.path_index = 0
            self.frames_since_replan = 0
            
            logger.info(f"Path planning enabled with {path_smoothing} smoothing")
    
    def compute_safe_velocity(self,
                              vision_data: Dict,
                              current_position: np.ndarray,
                              target_velocity: np.ndarray,
                              target_direction: Optional[np.ndarray] = None,
                              current_heading: float = 0.0,
                              target_position: Optional[np.ndarray] = None,
                              dt: float = 0.02) -> np.ndarray:
        """
        Compute safe velocity using depth information.
        
        Args:
            vision_data: Dictionary from DepthDetector containing:
                        - depth_map: Depth estimation
                        - obstacle_regions: List of depth-based regions
                        - free_space_map: Navigable space
            current_position: Current drone position [x, y, z]
            target_velocity: Desired velocity [vx, vy, vz]
            target_direction: Optional goal direction [x, y, z]
            current_heading: Current heading in radians (for pure pursuit)
            target_position: Goal position [x, y, z] (for potential field)
            dt: Time step for state updates
            
        Returns:
            Safe velocity command [vx, vy, vz]
        """
        if not vision_data or vision_data.get('depth_map') is None:
            return self._limit_velocity(target_velocity)
        
        # Extract vision data
        depth_map = vision_data.get('depth_map')
        free_space = vision_data.get('free_space_map')
        obstacle_regions = vision_data.get('obstacle_regions', [])
        
        # Convert obstacle regions to 3D obstacles for avoidance controller
        obstacles_3d = self._convert_regions_to_3d_obstacles(
            obstacle_regions, depth_map, current_position
        )
        
        # Use stable avoidance controller if enabled
        if self.use_stable_avoidance and hasattr(self, 'stable_avoidance'):
            # Use target position if provided, otherwise estimate from target direction
            if target_position is None:
                if target_direction is not None:
                    target_position = current_position + target_direction * 5.0
                else:
                    target_position = current_position + np.array([5.0, 0.0, current_position[2]])
            
            # Get planned path if available
            path = None
            if self.use_path_planning and self.current_path is not None:
                path = self.current_path
            
            safe_velocity = self.stable_avoidance.compute_avoidance_velocity(
                current_position=current_position,
                current_heading=current_heading,
                target_position=target_position,
                target_velocity=target_velocity,
                obstacles=obstacles_3d,
                path=path,
                dt=dt
            )
            
            # Update path if in normal or caution state
            if self.stable_avoidance.state in [AvoidanceState.NORMAL, AvoidanceState.CAUTION]:
                if self.use_path_planning and free_space is not None:
                    self._update_path_if_needed(free_space, depth_map, target_direction)
            
            return safe_velocity
        
        # Fallback to reactive avoidance
        if self.use_path_planning and free_space is not None:
            safe_velocity = self._compute_path_following_velocity(
                free_space, depth_map, current_position, 
                target_velocity, target_direction
            )
        else:
            # Reactive avoidance based on depth zones
            safe_directions = self._analyze_free_space(free_space, depth_map)
            avoidance = self._compute_avoidance_from_depth(
                depth_map, obstacle_regions, current_position
            )
            
            if target_direction is not None:
                combined = self._blend_target_and_avoidance(
                    target_velocity, avoidance, target_direction, safe_directions
                )
            else:
                combined = target_velocity + avoidance
            
            safe_velocity = self._apply_temporal_smoothing(combined)
        
        # Limit velocity
        safe_velocity = self._limit_velocity(safe_velocity)
        
        logger.debug(f"Target: {target_velocity}, Safe: {safe_velocity}")
        
        return safe_velocity
    
    def _convert_regions_to_3d_obstacles(self,
                                          obstacle_regions: List[Dict],
                                          depth_map: Optional[np.ndarray],
                                          current_position: np.ndarray) -> List[Dict]:
        """
        Convert 2D obstacle regions to 3D obstacle representations.
        
        Args:
            obstacle_regions: List of depth-based obstacle regions
            depth_map: Depth map for distance estimation
            current_position: Current drone position
            
        Returns:
            List of 3D obstacle dictionaries
        """
        obstacles_3d = []
        
        if depth_map is None:
            return obstacles_3d
        
        h, w = depth_map.shape
        
        # Camera intrinsics (approximate)
        fx = w / 2
        fy = w / 2
        cx = w / 2
        cy = h / 2
        
        # Center-band for horizontal obstacle detection (35-65% of image height)
        center_band_top = 0.35 * h
        center_band_bottom = 0.65 * h
        
        for region in obstacle_regions:
            try:
                x1, y1, x2, y2 = region['bbox']
                cx_pixel, cy_pixel = region['centroid']
                
                # Get depth in region
                region_depth = depth_map[y1:y2, x1:x2]
                if region_depth.size == 0:
                    continue
                
                # Median depth for robustness
                valid_depths = region_depth[region_depth > 0]
                if valid_depths.size == 0:
                    continue
                
                # MiDaS outputs relative inverse depth (higher = closer)
                # Normalize the depth values first
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                depth_range = depth_max - depth_min + 1e-6
                
                median_raw = np.median(valid_depths)
                normalized_depth = (median_raw - depth_min) / depth_range  # 0-1, higher = closer
                
                # Convert normalized inverse depth to distance
                # normalized_depth near 1.0 = very close, near 0.0 = far away
                # Use exponential mapping for better distance estimation
                # Calibration: depth=0.9 -> ~0.5m, depth=0.5 -> ~2m, depth=0.2 -> ~5m
                if normalized_depth > 0.95:
                    distance = 0.3  # Very close
                elif normalized_depth > 0.1:
                    # Inverse relationship with floor
                    distance = 0.5 / (normalized_depth + 0.05)
                    distance = np.clip(distance, 0.5, 10.0)
                else:
                    distance = 10.0  # Far away
                
                logger.debug(f"Depth conversion: raw={median_raw:.2f}, norm={normalized_depth:.2f}, dist={distance:.2f}m")
                
                # Project to 3D (body frame: x=forward, y=left, z=up)
                horizontal_offset = (cx_pixel - cx) / fx
                vertical_offset = (cy_pixel - cy) / fy
                
                # Check if this is a horizontal obstacle at drone height
                is_horizontal_obstacle = center_band_top <= cy_pixel <= center_band_bottom
                
                # Obstacle position in body frame
                obs_x = distance  # Forward distance
                obs_y = -horizontal_offset * distance  # Left-right (negative = right)
                
                # Height adjustment
                if is_horizontal_obstacle:
                    obs_z = current_position[2] - vertical_offset * distance * 0.3
                else:
                    obs_z = current_position[2] - vertical_offset * distance * 0.5
                
                # Estimate size from bounding box
                width_3d = (x2 - x1) * distance / fx
                height_3d = (y2 - y1) * distance / fy
                
                # Convert to global position
                position_global = current_position + np.array([obs_x, obs_y, 0])
                position_global[2] = obs_z
                
                # For horizontal obstacles, increase the effective size
                size_multiplier = 1.5 if is_horizontal_obstacle else 1.0
                
                obstacles_3d.append({
                    'position': position_global,
                    'size': np.array([width_3d, width_3d, height_3d]) * size_multiplier,
                    'distance': distance,
                    'class_name': region.get('class_name', 'obstacle'),
                    'confidence': 0.8,
                    'is_horizontal': is_horizontal_obstacle
                })
                
            except Exception as e:
                logger.debug(f"Failed to convert region to 3D: {e}")
                continue
        
        return obstacles_3d
    
    def _update_path_if_needed(self,
                                free_space_map: np.ndarray,
                                depth_map: Optional[np.ndarray],
                                target_direction: Optional[np.ndarray]):
        """Update path planning if needed."""
        self.frames_since_replan += 1
        
        should_replan = (
            self.current_path is None or
            self.frames_since_replan >= self.replan_interval or
            self.path_index >= len(self.current_path) - 1
        )
        
        if should_replan:
            h, w = free_space_map.shape
            start_pos = (w // 2, h - 1)
            
            # Goal based on target direction
            if target_direction is not None:
                horizontal_angle = np.arctan2(target_direction[1], target_direction[0])
                if horizontal_angle < -np.pi/6:
                    goal_x = w // 6
                elif horizontal_angle < np.pi/6:
                    goal_x = w // 2
                else:
                    goal_x = 5 * w // 6
            else:
                goal_x = w // 2
            
            goal_pos = (goal_x, 10)
            
            # Plan path
            path = self.path_planner.plan(
                occupancy_map=1 - free_space_map.astype(np.uint8),
                start=start_pos,
                goal=goal_pos,
                depth_map=depth_map
            )
            
            # Smooth path
            if len(path) > 2 and self.trajectory_smoother is not None:
                self.current_path = self.trajectory_smoother.smooth_path(path, num_samples=50)
            elif len(path) > 0:
                self.current_path = np.array(path, dtype=np.float32)
            
            self.path_index = 0
            self.frames_since_replan = 0
            
            # Update stable avoidance controller's path
            if self.use_stable_avoidance and self.current_path is not None:
                self.stable_avoidance.set_path(self.current_path)
    
    def _compute_path_following_velocity(self,
                                        free_space_map: np.ndarray,
                                        depth_map: Optional[np.ndarray],
                                        current_position: np.ndarray,
                                        target_velocity: np.ndarray,
                                        target_direction: Optional[np.ndarray]) -> np.ndarray:
        """Compute velocity by following a planned path through free space."""
        h, w = free_space_map.shape
        
        # Determine if we need to replan
        should_replan = (
            self.current_path is None or
            self.frames_since_replan >= self.replan_interval or
            self.path_index >= len(self.current_path) - 1
        )
        
        if should_replan:
            start_pos = (w // 2, h - 1)
            
            if target_direction is not None:
                horizontal_angle = np.arctan2(target_direction[1], target_direction[0])
                if horizontal_angle < -np.pi/6:
                    goal_x = w // 6
                elif horizontal_angle < np.pi/6:
                    goal_x = w // 2
                else:
                    goal_x = 5 * w // 6
            else:
                goal_x = w // 2
            
            goal_pos = (goal_x, 10)
            
            path = self.path_planner.plan(
                occupancy_map=1 - free_space_map.astype(np.uint8),
                start=start_pos,
                goal=goal_pos,
                depth_map=depth_map
            )
            
            if len(path) > 2 and self.trajectory_smoother is not None:
                self.current_path = self.trajectory_smoother.smooth_path(path, num_samples=50)
            elif len(path) > 0:
                self.current_path = np.array(path, dtype=np.float32)
            else:
                logger.warning("No path found, using reactive avoidance")
                return target_velocity
            
            self.path_index = 0
            self.frames_since_replan = 0
            logger.debug(f"Planned new path with {len(self.current_path)} waypoints")
        
        # Follow current path
        if self.current_path is not None and len(self.current_path) > 0:
            lookahead_distance = 10
            lookahead_index = min(self.path_index + lookahead_distance, len(self.current_path) - 1)
            target_waypoint = self.current_path[lookahead_index]
            
            center_x = w // 2
            center_y = h - 1
            
            offset_x = target_waypoint[0] - center_x
            offset_y = center_y - target_waypoint[1]
            
            norm_x = offset_x / (w / 2)
            norm_y = offset_y / (h / 2)
            
            velocity = np.zeros(3)
            velocity[0] = self.max_speed * (0.5 + 0.5 * norm_y)
            velocity[1] = -norm_x * self.max_speed * 0.5
            velocity[2] = target_velocity[2]
            
            velocity = self._apply_temporal_smoothing(velocity)
            
            self.path_index = min(self.path_index + 1, len(self.current_path) - 1)
            self.frames_since_replan += 1
            
            return velocity
        
        return target_velocity
    
    def _analyze_free_space(self, 
                           free_space_map: Optional[np.ndarray],
                           depth_map: Optional[np.ndarray]) -> Dict[str, float]:
        """Analyze free space map to determine safe navigation directions."""
        if free_space_map is None:
            return {'left': 1.0, 'center': 1.0, 'right': 1.0}
        
        h, w = free_space_map.shape
        zone_width = w // 3
        
        zones = {
            'left': free_space_map[:, :zone_width],
            'center': free_space_map[:, zone_width:2*zone_width],
            'right': free_space_map[:, 2*zone_width:]
        }
        
        scores = {}
        
        for direction, zone in zones.items():
            free_ratio = np.sum(zone) / zone.size
            
            if depth_map is not None:
                zone_depth = depth_map[:, :zone_width] if direction == 'left' else \
                           depth_map[:, zone_width:2*zone_width] if direction == 'center' else \
                           depth_map[:, 2*zone_width:]
                
                avg_depth = np.mean(zone_depth[zone > 0]) if np.any(zone > 0) else 1.0
                depth_weight = np.clip(avg_depth / 5.0, 0.1, 1.0)
            else:
                depth_weight = 1.0
            
            scores[direction] = free_ratio * depth_weight
        
        return scores
    
    def _compute_avoidance_from_depth(self,
                                      depth_map: Optional[np.ndarray],
                                      obstacle_regions: List[Dict],
                                      current_position: np.ndarray) -> np.ndarray:
        """Compute avoidance vector from depth data."""
        avoidance = np.zeros(3)
        
        if depth_map is None:
            return avoidance
        
        h, w = depth_map.shape
        
        # Center-band for horizontal obstacle detection
        center_band_top = 0.35 * h
        center_band_bottom = 0.65 * h
        
        for region in obstacle_regions:
            cx, cy = region['centroid']
            area = region['area']
            depth = region.get('depth', 0.5)
            
            # Only avoid if close
            if depth > 0.4:  # Not close enough
                continue
            
            # Horizontal avoidance
            horizontal_offset = (cx - w/2) / (w/2)
            
            magnitude = self.avoidance_gain * (1.0 - depth)
            magnitude *= (area / (w * h))
            
            avoidance[0] += -horizontal_offset * magnitude
            
            # Vertical avoidance
            vertical_offset = (cy - h/2) / (h/2)
            is_horizontal_obstacle = center_band_top <= cy <= center_band_bottom
            
            if is_horizontal_obstacle:
                vertical_gain = 1.5
                x1, y1, x2, y2 = region['bbox']
                obstacle_width_ratio = (x2 - x1) / w
                
                if obstacle_width_ratio > 0.5:
                    avoidance[2] += -vertical_offset * magnitude * 2.0
                    avoidance[0] -= magnitude * 0.5
                else:
                    avoidance[1] += -horizontal_offset * magnitude * 1.5
                    avoidance[2] += -vertical_offset * magnitude * vertical_gain
            else:
                avoidance[2] += -vertical_offset * magnitude * 0.5
        
        return avoidance
    
    def _blend_target_and_avoidance(self,
                                    target_velocity: np.ndarray,
                                    avoidance: np.ndarray,
                                    target_direction: np.ndarray,
                                    safe_directions: Dict[str, float]) -> np.ndarray:
        """Intelligently blend target-seeking and avoidance behaviors."""
        target_dir_norm = target_direction / (np.linalg.norm(target_direction) + 1e-6)
        horizontal_angle = np.arctan2(target_dir_norm[1], target_dir_norm[0])
        
        if horizontal_angle < -np.pi/6:
            target_zone = 'left'
        elif horizontal_angle < np.pi/6:
            target_zone = 'center'
        else:
            target_zone = 'right'
        
        target_safety = safe_directions.get(target_zone, 0.5)
        
        if target_safety > 0.5:
            blend_weight = 0.7
        else:
            blend_weight = 0.3
        
        return blend_weight * target_velocity + (1 - blend_weight) * avoidance
    
    def _apply_temporal_smoothing(self, velocity: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce jitter."""
        self.velocity_history.append(velocity.copy())
        
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
        
        return np.mean(self.velocity_history, axis=0)
    
    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity to maximum speed."""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        return velocity
    
    def check_crash(self,
                   roll: float,
                   pitch: float,
                   altitude: float,
                   velocity: Optional[Tuple[float, float, float]] = None) -> bool:
        """Check if drone has crashed."""
        if not self.takeoff_complete:
            if altitude > self.min_flight_altitude:
                self.takeoff_complete = True
                self.crash_detection_enabled = True
                logger.info(f"Takeoff complete - Crash detection enabled")
            return False
        
        if not self.crash_detection_enabled:
            return False
        
        if abs(roll) > self.crash_tilt_threshold or abs(pitch) > self.crash_tilt_threshold:
            if not self.crashed:
                logger.error(f"Crash: Extreme tilt (roll={np.rad2deg(roll):.1f}°, "
                           f"pitch={np.rad2deg(pitch):.1f}°)")
                self.crashed = True
            return True
        
        if altitude < self.crash_altitude_threshold:
            if not self.crashed:
                logger.warning(f"Crash: Low altitude ({altitude:.3f}m)")
                self.crashed = True
            return True
        
        return False
    
    def reset_crash_state(self):
        """Reset crash detection state."""
        self.crashed = False
        if self.use_stable_avoidance and hasattr(self, 'stable_avoidance'):
            self.stable_avoidance.reset()
        logger.info("Crash state reset")
    
    def enable_crash_detection(self):
        """Manually enable crash detection."""
        self.crash_detection_enabled = True
        self.takeoff_complete = True
        logger.info("Crash detection enabled")
    
    def get_avoidance_state(self) -> Dict:
        """Get current avoidance controller state and statistics."""
        if self.use_stable_avoidance and hasattr(self, 'stable_avoidance'):
            stats = self.stable_avoidance.get_statistics()
            stats['path_index'] = getattr(self, 'path_index', 0) if getattr(self, 'current_path', None) is not None else 0
            stats['path_length'] = len(self.current_path) if getattr(self, 'current_path', None) is not None else 0
            return stats
        else:
            return {
                'state': 'legacy',
                'path_index': getattr(self, 'path_index', 0),
                'path_length': len(self.current_path) if getattr(self, 'current_path', None) is not None else 0
            }
    
    def is_in_avoidance_mode(self) -> bool:
        """Check if controller is actively avoiding obstacles."""
        if self.use_stable_avoidance and hasattr(self, 'stable_avoidance'):
            return self.stable_avoidance.state in [
                AvoidanceState.AVOIDANCE, 
                AvoidanceState.EMERGENCY,
                AvoidanceState.RECOVERY
            ]
        return False
