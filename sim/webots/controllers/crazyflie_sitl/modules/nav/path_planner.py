"""
Waypoint-based Path Planner with Deviation Recovery.

Provides path planning capabilities for drone navigation with:
- Waypoint sequence following
- Path deviation detection and correction
- Smooth return-to-path maneuvers
- Dynamic waypoint generation for obstacle avoidance
"""

import numpy as np
from typing import List, Tuple, Optional

from .state_machine import PathState

# Import logger if available
try:
    import sys
    from pathlib import Path
    controller_dir = Path(__file__).parent.parent.parent
    utils_dir = controller_dir.parent.parent / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")


class Waypoint:
    """Represents a 3D waypoint with optional metadata."""
    
    def __init__(self, position: np.ndarray, 
                 velocity: float = 0.2,
                 tolerance: float = 0.3,
                 heading: Optional[float] = None):
        """
        Initialize waypoint.
        
        Args:
            position: 3D position [x, y, z]
            velocity: Desired velocity at this waypoint
            tolerance: Distance tolerance for reaching waypoint
            heading: Optional desired heading in radians (None = auto)
        """
        self.position = np.array(position, dtype=np.float32)
        self.velocity = velocity
        self.tolerance = tolerance
        self.heading = heading
    
    def distance_to(self, pos: np.ndarray) -> float:
        """Compute 3D distance to a position."""
        return np.linalg.norm(self.position - pos)
    
    def horizontal_distance_to(self, pos: np.ndarray) -> float:
        """Compute 2D horizontal distance to a position."""
        return np.linalg.norm(self.position[:2] - pos[:2])


class PathPlanner:
    """
    Waypoint-based path planner with deviation recovery.
    
    Manages a sequence of waypoints and computes navigation commands
    to follow the path while handling deviations.
    """
    
    def __init__(self, 
                 max_deviation: float = 1.0,
                 return_speed: float = 0.15,
                 lookahead_distance: float = 0.5,
                 path_smoothing: float = 0.3):
        """
        Initialize path planner.
        
        Args:
            max_deviation: Maximum allowed deviation before returning to path (m)
            return_speed: Speed when returning to path (m/s)
            lookahead_distance: Distance to look ahead on path for smooth following
            path_smoothing: Smoothing factor for path following (0-1)
        """
        self.max_deviation = max_deviation
        self.return_speed = return_speed
        self.lookahead_distance = lookahead_distance
        self.path_smoothing = path_smoothing
        
        # Waypoint management
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        
        # State machine
        self.state = PathState.IDLE
        self.deviation_distance = 0.0
        self.return_target: Optional[np.ndarray] = None
        
        # Path tracking
        self.path_history: List[np.ndarray] = []
        self.max_history = 100
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.deviation_count = 0
    
    def set_waypoints(self, waypoints: List[Tuple[float, float, float]], 
                      default_velocity: float = 0.2,
                      default_tolerance: float = 0.3):
        """
        Set path waypoints.
        
        Args:
            waypoints: List of (x, y, z) positions
            default_velocity: Default velocity for waypoints
            default_tolerance: Default tolerance for reaching waypoints
        """
        self.waypoints = []
        for wp in waypoints:
            self.waypoints.append(Waypoint(
                position=np.array(wp),
                velocity=default_velocity,
                tolerance=default_tolerance
            ))
        
        self.current_waypoint_idx = 0
        self.state = PathState.FOLLOWING if self.waypoints else PathState.IDLE
        self.path_history.clear()
        
        log(f"[PATH] Set {len(self.waypoints)} waypoints", "INFO")
        for i, wp in enumerate(self.waypoints):
            log(f"  WP{i}: ({wp.position[0]:.2f}, {wp.position[1]:.2f}, {wp.position[2]:.2f})", "DEBUG")
    
    def set_goal(self, goal: Tuple[float, float, float], velocity: float = 0.2):
        """
        Set a single goal position (simple wrapper for single waypoint).
        
        Args:
            goal: Target position (x, y, z)
            velocity: Desired approach velocity
        """
        self.set_waypoints([goal], default_velocity=velocity)
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint."""
        if 0 <= self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def get_progress(self) -> float:
        """Get path completion progress (0-1)."""
        if not self.waypoints:
            return 0.0
        return self.current_waypoint_idx / len(self.waypoints)
    
    def update(self, current_position: np.ndarray, dt: float) -> dict:
        """
        Update path planner and compute navigation target.
        
        Args:
            current_position: Current drone position [x, y, z]
            dt: Time step in seconds
            
        Returns:
            dict with:
            - target: Target position to navigate to [x, y, z]
            - velocity: Recommended velocity
            - direction: Direction vector (normalized)
            - distance: Distance to target
            - state: Current path state
            - waypoint_idx: Current waypoint index
            - deviation: Current deviation from path
            - on_path: Whether drone is on the planned path
        """
        current_position = np.array(current_position, dtype=np.float32)
        
        # Update path history
        if len(self.path_history) > 0:
            dist_from_last = np.linalg.norm(current_position - self.path_history[-1])
            self.total_distance_traveled += dist_from_last
            
            if len(self.path_history) >= self.max_history:
                self.path_history.pop(0)
        self.path_history.append(current_position.copy())
        
        result = {
            'target': current_position.copy(),
            'velocity': 0.0,
            'direction': np.array([0.0, 0.0, 0.0]),
            'distance': 0.0,
            'state': self.state.value,
            'waypoint_idx': self.current_waypoint_idx,
            'deviation': 0.0,
            'on_path': True
        }
        
        if self.state == PathState.IDLE or self.state == PathState.PATH_COMPLETE:
            return result
        
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            self.state = PathState.PATH_COMPLETE
            result['state'] = self.state.value
            return result
        
        # Check if waypoint reached
        distance_to_waypoint = current_waypoint.distance_to(current_position)
        
        if distance_to_waypoint < current_waypoint.tolerance:
            # Waypoint reached - advance to next
            log(f"[PATH] ✓ Waypoint {self.current_waypoint_idx} reached", "SUCCESS")
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx >= len(self.waypoints):
                self.state = PathState.PATH_COMPLETE
                log("[PATH] ✓ PATH COMPLETE - All waypoints reached", "SUCCESS")
                result['state'] = self.state.value
                return result
            else:
                self.state = PathState.FOLLOWING
                current_waypoint = self.get_current_waypoint()
                distance_to_waypoint = current_waypoint.distance_to(current_position)
        
        # Compute deviation from ideal path (line between previous and current waypoint)
        deviation = self._compute_deviation(current_position)
        self.deviation_distance = deviation
        result['deviation'] = deviation
        
        # Check for path deviation
        if deviation > self.max_deviation:
            if self.state != PathState.DEVIATED and self.state != PathState.RETURNING:
                self.deviation_count += 1
                self.state = PathState.DEVIATED
                log(f"[PATH] ⚠ Deviation detected: {deviation:.2f}m (max: {self.max_deviation:.2f}m)", "WARNING")
                
                # Compute return target on path
                self.return_target = self._compute_return_point(current_position)
            result['on_path'] = False
        
        # State-based target computation
        if self.state == PathState.DEVIATED:
            # Start returning to path
            self.state = PathState.RETURNING
        
        if self.state == PathState.RETURNING:
            # Navigate to return point
            if self.return_target is not None:
                distance_to_return = np.linalg.norm(current_position[:2] - self.return_target[:2])
                
                if distance_to_return < 0.3 or deviation < self.max_deviation * 0.5:
                    # Back on path
                    self.state = PathState.FOLLOWING
                    self.return_target = None
                    log("[PATH] ✓ Returned to path", "SUCCESS")
                else:
                    result['target'] = self.return_target
                    result['velocity'] = self.return_speed
                    result['distance'] = distance_to_return
                    
                    direction = self.return_target - current_position
                    dist = np.linalg.norm(direction)
                    if dist > 0.001:
                        result['direction'] = direction / dist
                    
                    result['state'] = self.state.value
                    return result
        
        # Normal path following with lookahead
        target = self._compute_lookahead_target(current_position, current_waypoint)
        
        direction = target - current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:
            direction_normalized = direction / distance
        else:
            direction_normalized = np.array([1.0, 0.0, 0.0])
        
        # Scale velocity based on distance to waypoint
        velocity = current_waypoint.velocity
        if distance_to_waypoint < 1.0:
            # Slow down approaching waypoint
            velocity *= max(0.3, distance_to_waypoint)
        
        result['target'] = target
        result['velocity'] = velocity
        result['direction'] = direction_normalized
        result['distance'] = distance_to_waypoint
        result['state'] = self.state.value
        
        return result
    
    def _compute_deviation(self, current_position: np.ndarray) -> float:
        """
        Compute perpendicular distance from current position to path segment.
        
        Args:
            current_position: Current drone position
            
        Returns:
            Deviation distance in meters
        """
        if len(self.waypoints) < 2 or self.current_waypoint_idx == 0:
            # No previous waypoint - use distance from direct line to goal
            current_wp = self.get_current_waypoint()
            if current_wp is None:
                return 0.0
            return 0.0  # No deviation concept for first waypoint
        
        # Get previous and current waypoint
        prev_wp = self.waypoints[self.current_waypoint_idx - 1].position
        curr_wp = self.waypoints[self.current_waypoint_idx].position
        
        # Compute perpendicular distance to line segment
        line_vec = curr_wp - prev_wp
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 0.001:
            return np.linalg.norm(current_position - prev_wp)
        
        line_unit = line_vec / line_len
        
        # Project current position onto line
        pos_vec = current_position - prev_wp
        projection_length = np.dot(pos_vec, line_unit)
        projection_length = np.clip(projection_length, 0, line_len)
        
        closest_point = prev_wp + line_unit * projection_length
        
        return np.linalg.norm(current_position - closest_point)
    
    def _compute_return_point(self, current_position: np.ndarray) -> np.ndarray:
        """
        Compute optimal point to return to path.
        
        Args:
            current_position: Current drone position
            
        Returns:
            Target position to return to path
        """
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return current_position.copy()
        
        if self.current_waypoint_idx == 0:
            # Return directly to first waypoint
            return current_wp.position.copy()
        
        # Find closest point on path segment
        prev_wp = self.waypoints[self.current_waypoint_idx - 1].position
        curr_wp = current_wp.position
        
        line_vec = curr_wp - prev_wp
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 0.001:
            return curr_wp.copy()
        
        line_unit = line_vec / line_len
        
        # Project current position onto line
        pos_vec = current_position - prev_wp
        projection_length = np.dot(pos_vec, line_unit)
        
        # Clamp to segment and add lookahead
        target_distance = np.clip(projection_length + self.lookahead_distance, 0, line_len)
        
        return_point = prev_wp + line_unit * target_distance
        
        # Maintain current altitude
        return_point[2] = current_position[2]
        
        return return_point
    
    def _compute_lookahead_target(self, current_position: np.ndarray, 
                                   waypoint: Waypoint) -> np.ndarray:
        """
        Compute target with lookahead for smooth path following.
        
        Args:
            current_position: Current drone position
            waypoint: Target waypoint
            
        Returns:
            Lookahead target position
        """
        direction = waypoint.position - current_position
        distance = np.linalg.norm(direction)
        
        if distance < self.lookahead_distance:
            # Close to waypoint - target it directly
            return waypoint.position.copy()
        
        # Blend between direct path and lookahead
        direction_norm = direction / distance
        lookahead_point = current_position + direction_norm * self.lookahead_distance
        
        # Smooth blend toward waypoint
        blend = self.path_smoothing
        target = blend * waypoint.position + (1 - blend) * lookahead_point
        
        return target
    
    def force_deviation(self, avoidance_offset: np.ndarray):
        """
        Force a temporary deviation for obstacle avoidance.
        
        Args:
            avoidance_offset: Offset vector from current path [x, y, z]
        """
        if self.state == PathState.FOLLOWING:
            # Note: The path planner tracks this but actual avoidance
            # is handled by the navigation controller
            pass
    
    def reset(self):
        """Reset path planner state."""
        self.current_waypoint_idx = 0
        self.state = PathState.IDLE if not self.waypoints else PathState.FOLLOWING
        self.deviation_distance = 0.0
        self.return_target = None
        self.path_history.clear()
        self.total_distance_traveled = 0.0
    
    def get_status(self) -> dict:
        """Get current planner status."""
        return {
            'state': self.state.value,
            'waypoint_idx': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'progress': self.get_progress(),
            'deviation': self.deviation_distance,
            'deviation_count': self.deviation_count,
            'total_distance': self.total_distance_traveled
        }
