"""Navigation controller with obstacle avoidance."""

import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class NavigationController:
    """Navigation controller with obstacle avoidance capabilities."""
    
    def __init__(self, 
                 max_speed: float = 0.5,
                 safety_distance: float = 0.5,
                 avoidance_gain: float = 1.0):
        """
        Initialize navigation controller.
        
        Args:
            max_speed: Maximum velocity (m/s)
            safety_distance: Minimum distance to obstacles (m)
            avoidance_gain: Gain for avoidance maneuvers
        """
        self.max_speed = max_speed
        self.safety_distance = safety_distance
        self.avoidance_gain = avoidance_gain
    
    def compute_safe_velocity(self,
                             obstacles: List[Dict],
                             current_position: np.ndarray,
                             target_velocity: np.ndarray) -> np.ndarray:
        """
        Compute safe velocity considering obstacles.
        
        Args:
            obstacles: List of detected obstacles
            current_position: Current drone position [x, y, z]
            target_velocity: Desired velocity [vx, vy, vz]
            
        Returns:
            Safe velocity command [vx, vy, vz]
        """
        if not obstacles:
            return self._limit_velocity(target_velocity)
        
        # Compute avoidance vector
        avoidance = np.zeros(3)
        for obstacle in obstacles:
            obs_pos = obstacle.get('position', obstacle.get('centroid', np.zeros(3)))
            
            # Vector from obstacle to drone
            diff = current_position - obs_pos
            distance = np.linalg.norm(diff)
            
            if distance < self.safety_distance and distance > 0:
                # Repulsive force inversely proportional to distance
                magnitude = self.avoidance_gain * (1.0 / distance - 1.0 / self.safety_distance)
                direction = diff / distance
                avoidance += magnitude * direction
        
        # Combine target and avoidance velocities
        safe_velocity = target_velocity + avoidance
        
        # Limit velocity magnitude
        safe_velocity = self._limit_velocity(safe_velocity)
        
        logger.debug(f"Target: {target_velocity}, Safe: {safe_velocity}")
        return safe_velocity
    
    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity to maximum speed."""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        return velocity
    
    def is_path_clear(self,
                     obstacles: List[Dict],
                     current_position: np.ndarray,
                     target_position: np.ndarray) -> bool:
        """
        Check if path to target is clear of obstacles.
        
        Args:
            obstacles: List of detected obstacles
            current_position: Current position
            target_position: Target position
            
        Returns:
            True if path is clear
        """
        if not obstacles:
            return True
        
        path_vector = target_position - current_position
        path_length = np.linalg.norm(path_vector)
        
        if path_length == 0:
            return True
        
        path_direction = path_vector / path_length
        
        for obstacle in obstacles:
            obs_pos = obstacle.get('position', obstacle.get('centroid', np.zeros(3)))
            
            # Vector from current position to obstacle
            to_obstacle = obs_pos - current_position
            
            # Project obstacle onto path
            projection_length = np.dot(to_obstacle, path_direction)
            
            # Check if obstacle is along the path
            if 0 <= projection_length <= path_length:
                # Compute perpendicular distance
                projection_point = current_position + projection_length * path_direction
                perp_distance = np.linalg.norm(obs_pos - projection_point)
                
                if perp_distance < self.safety_distance:
                    return False
        
        return True
    
    def plan_avoidance_maneuver(self,
                               obstacles: List[Dict],
                               current_position: np.ndarray,
                               target_position: np.ndarray) -> List[np.ndarray]:
        """
        Plan avoidance maneuver around obstacles.
        
        Args:
            obstacles: List of obstacles
            current_position: Current position
            target_position: Target position
            
        Returns:
            List of waypoints for avoidance
        """
        if self.is_path_clear(obstacles, current_position, target_position):
            return [target_position]
        
        # Simple avoidance: move perpendicular to obstacle
        waypoints = []
        
        # Find closest obstacle on path
        closest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in obstacles:
            obs_pos = obstacle.get('position', obstacle.get('centroid', np.zeros(3)))
            distance = np.linalg.norm(obs_pos - current_position)
            
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obs_pos
        
        if closest_obstacle is not None:
            # Create waypoint offset from obstacle
            to_obstacle = closest_obstacle - current_position
            perpendicular = np.array([-to_obstacle[1], to_obstacle[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            offset_point = closest_obstacle + perpendicular * self.safety_distance * 2
            waypoints.append(offset_point)
        
        waypoints.append(target_position)
        
        return waypoints
