"""
Stable Avoidance Controller with Potential Field and Pure Pursuit Integration.

This module provides a robust obstacle avoidance system combining:
1. Artificial Potential Field (APF) for reactive obstacle avoidance
2. Pure Pursuit for smooth path following
3. Dynamic velocity scaling based on obstacle proximity
4. Recovery behavior for near-collision situations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class AvoidanceState(Enum):
    """States for the avoidance state machine."""
    NORMAL = "normal"           # Normal navigation
    CAUTION = "caution"         # Obstacle detected, slowing down
    AVOIDANCE = "avoidance"     # Active avoidance maneuver
    EMERGENCY = "emergency"     # Emergency stop/hover
    RECOVERY = "recovery"       # Recovering from emergency


@dataclass
class AvoidanceConfig:
    """
    Configuration for avoidance controller.
    
    These defaults can be overridden by config file values through
    SegmentationNavigationController. See drone.avoidance section.
    """
    # Speed limits (m/s)
    max_speed: float = 0.5
    cruise_speed: float = 0.3       # config: max_speed * cruise_speed_ratio
    caution_speed: float = 0.15     # config: max_speed * caution_speed_ratio
    min_speed: float = 0.05
    
    # Distance thresholds (meters)
    safety_distance: float = 0.5
    caution_distance: float = 1.0   # config: safety_distance * caution_distance_ratio
    detection_distance: float = 2.0 # config: safety_distance * detection_distance_ratio
    emergency_distance: float = 0.25 # config: safety_distance * emergency_distance_ratio
    
    # Potential field parameters
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.5     # config: avoidance_gain * repulsive_gain_ratio
    repulsive_distance: float = 1.2
    
    # Pure pursuit parameters
    lookahead_base: float = 0.4
    lookahead_gain: float = 0.1  # Lookahead = base + gain * speed
    min_lookahead: float = 0.2
    max_lookahead: float = 1.0
    
    # Smoothing parameters (config: drone.avoidance.velocity_smoothing/recovery_smoothing)
    velocity_smoothing: float = 0.2  # EMA alpha (lower = smoother)
    recovery_smoothing: float = 0.1  # Slower smoothing during recovery
    
    # Recovery parameters (config: drone.avoidance.recovery_time/recovery_backup_distance)
    recovery_time: float = 1.0  # Seconds to recover from emergency
    recovery_backup_distance: float = 0.1  # Distance to back up during recovery


class PotentialFieldController:
    """
    Artificial Potential Field (APF) based reactive avoidance.
    
    Creates a virtual force field where:
    - Goal creates attractive force
    - Obstacles create repulsive forces
    - Combined forces determine optimal velocity
    """
    
    def __init__(self, config: AvoidanceConfig):
        self.config = config
        
    def compute_attractive_force(self, 
                                  current_pos: np.ndarray,
                                  goal_pos: np.ndarray) -> np.ndarray:
        """
        Compute attractive force toward goal.
        
        Args:
            current_pos: Current position [x, y, z]
            goal_pos: Goal position [x, y, z]
            
        Returns:
            Attractive force vector [fx, fy, fz]
        """
        to_goal = goal_pos - current_pos
        distance = np.linalg.norm(to_goal)
        
        if distance < 1e-6:
            return np.zeros(3)
        
        # Linear attractive force (could use quadratic for stronger pull)
        direction = to_goal / distance
        magnitude = self.config.attractive_gain * min(distance, 2.0)  # Cap magnitude
        
        return magnitude * direction
    
    def compute_repulsive_force(self,
                                 current_pos: np.ndarray,
                                 obstacles: List[Dict]) -> np.ndarray:
        """
        Compute total repulsive force from all obstacles.
        
        Args:
            current_pos: Current position [x, y, z]
            obstacles: List of obstacle dictionaries with 'position' and optionally 'size'
            
        Returns:
            Total repulsive force vector [fx, fy, fz]
        """
        total_repulsion = np.zeros(3)
        
        for obstacle in obstacles:
            obs_pos = obstacle.get('position', obstacle.get('centroid', np.zeros(3)))
            if obs_pos is None:
                continue
                
            obs_pos = np.array(obs_pos)
            obs_size = obstacle.get('size', np.array([0.1, 0.1, 0.1]))
            is_horizontal = obstacle.get('is_horizontal', False)
            
            # Vector from obstacle to drone
            from_obstacle = current_pos - obs_pos
            distance = np.linalg.norm(from_obstacle)
            
            # Effective distance accounting for obstacle size
            effective_distance = max(distance - np.max(obs_size) / 2, 0.01)
            
            if effective_distance < self.config.repulsive_distance:
                # Repulsive force inversely proportional to distance squared
                direction = from_obstacle / (distance + 1e-6)
                
                # Force magnitude: k * (1/d - 1/d0) * (1/d^2)
                magnitude = self.config.repulsive_gain * (
                    (1.0 / effective_distance - 1.0 / self.config.repulsive_distance) *
                    (1.0 / (effective_distance ** 2))
                )
                
                # Increase force for very close obstacles
                if effective_distance < self.config.emergency_distance:
                    magnitude *= 3.0
                elif effective_distance < self.config.safety_distance:
                    magnitude *= 1.5
                
                # Horizontal obstacles at drone height need special handling
                if is_horizontal:
                    # Increase magnitude for horizontal obstacles
                    magnitude *= 1.5
                    
                    # For horizontal obstacles, the drone should prefer lateral movement
                    # or altitude change rather than just backing up
                    height_diff = current_pos[2] - obs_pos[2]
                    
                    # Strengthen vertical component of repulsion for horizontal obstacles
                    if abs(height_diff) < obs_size[2]:
                        # Obstacle is at similar altitude - push up or down
                        vertical_push = np.sign(height_diff) if abs(height_diff) > 0.1 else 1.0
                        direction[2] = max(abs(direction[2]), 0.5) * vertical_push
                        
                        # Also strengthen lateral component
                        lateral_magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
                        if lateral_magnitude < 0.3:
                            # If mostly forward/backward, add lateral push
                            # Push in direction away from obstacle center
                            direction[1] = np.sign(direction[1]) * 0.5 if abs(direction[1]) > 0.01 else 0.5
                        
                        # Renormalize direction
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                
                total_repulsion += magnitude * direction
        
        return total_repulsion
    
    def compute_combined_force(self,
                                current_pos: np.ndarray,
                                goal_pos: np.ndarray,
                                obstacles: List[Dict]) -> Tuple[np.ndarray, float]:
        """
        Compute combined attractive and repulsive forces.
        
        Args:
            current_pos: Current position
            goal_pos: Goal position
            obstacles: List of obstacles
            
        Returns:
            Tuple of (force_vector, min_obstacle_distance)
        """
        attractive = self.compute_attractive_force(current_pos, goal_pos)
        repulsive = self.compute_repulsive_force(current_pos, obstacles)
        
        # Find minimum obstacle distance
        min_distance = float('inf')
        for obstacle in obstacles:
            obs_pos = obstacle.get('position', obstacle.get('centroid'))
            if obs_pos is not None:
                dist = np.linalg.norm(np.array(obs_pos) - current_pos)
                min_distance = min(min_distance, dist)
        
        combined = attractive + repulsive
        
        return combined, min_distance


class PurePursuitController:
    """
    Pure Pursuit path following controller.
    
    Tracks a path by computing the curvature needed to reach a 
    look-ahead point on the path.
    """
    
    def __init__(self, config: AvoidanceConfig):
        self.config = config
        
    def compute_lookahead_distance(self, current_speed: float) -> float:
        """Compute adaptive look-ahead distance based on speed."""
        lookahead = self.config.lookahead_base + self.config.lookahead_gain * current_speed
        return np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)
    
    def find_lookahead_point(self,
                              current_pos: np.ndarray,
                              path: np.ndarray,
                              path_index: int,
                              lookahead_distance: float) -> Tuple[np.ndarray, int]:
        """
        Find the look-ahead point on the path.
        
        Args:
            current_pos: Current position [x, y]
            path: Path waypoints (N, 2)
            path_index: Current path index
            lookahead_distance: Look-ahead distance
            
        Returns:
            Tuple of (lookahead_point, updated_path_index)
        """
        if len(path) == 0:
            return current_pos[:2], path_index
        
        # Start from current index
        for i in range(path_index, len(path)):
            point = path[i]
            distance = np.linalg.norm(point - current_pos[:2])
            
            if distance >= lookahead_distance:
                return point, i
        
        # If we've passed all points, return the last one
        return path[-1], len(path) - 1
    
    def compute_steering(self,
                          current_pos: np.ndarray,
                          current_heading: float,
                          lookahead_point: np.ndarray) -> float:
        """
        Compute steering command using pure pursuit geometry.
        
        Args:
            current_pos: Current position [x, y]
            current_heading: Current heading (radians)
            lookahead_point: Target point [x, y]
            
        Returns:
            Steering angle (radians)
        """
        # Vector to lookahead point
        to_target = lookahead_point - current_pos[:2]
        distance = np.linalg.norm(to_target)
        
        if distance < 1e-6:
            return 0.0
        
        # Angle to target
        target_angle = np.arctan2(to_target[1], to_target[0])
        
        # Angle error
        angle_error = target_angle - current_heading
        
        # Normalize to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # Pure pursuit curvature: 2 * sin(alpha) / L
        curvature = 2 * np.sin(angle_error) / distance
        
        return curvature


class StableAvoidanceController:
    """
    Main controller combining potential fields, pure pursuit, and state machine.
    
    This provides stable, smooth obstacle avoidance with:
    - Reactive avoidance using potential fields
    - Smooth path following using pure pursuit
    - State machine for handling different scenarios
    - Velocity smoothing for stability
    """
    
    def __init__(self, config: Optional[AvoidanceConfig] = None):
        """
        Initialize the stable avoidance controller.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or AvoidanceConfig()
        
        # Sub-controllers
        self.potential_field = PotentialFieldController(self.config)
        self.pure_pursuit = PurePursuitController(self.config)
        
        # State machine
        self.state = AvoidanceState.NORMAL
        self.state_timer = 0.0
        
        # Velocity smoothing
        self.smooth_velocity = np.zeros(3)
        self.last_velocity = np.zeros(3)
        
        # Path tracking
        self.current_path = None
        self.path_index = 0
        
        # Recovery state
        self.recovery_start_time = None
        self.recovery_direction = None
        
        # Statistics
        self.obstacle_encounters = 0
        self.emergency_stops = 0
        
        logger.info(f"StableAvoidanceController initialized with config: "
                   f"max_speed={self.config.max_speed}, "
                   f"safety_distance={self.config.safety_distance}")
    
    def update_state(self, min_obstacle_distance: float, dt: float):
        """
        Update the state machine based on obstacle proximity.
        
        Args:
            min_obstacle_distance: Distance to nearest obstacle
            dt: Time step (seconds)
        """
        old_state = self.state
        self.state_timer += dt
        
        if self.state == AvoidanceState.EMERGENCY:
            # Stay in emergency until recovery timer expires
            if self.state_timer >= self.config.recovery_time * 0.5:
                self.state = AvoidanceState.RECOVERY
                self.state_timer = 0.0
                logger.info("Transitioning from EMERGENCY to RECOVERY")
                
        elif self.state == AvoidanceState.RECOVERY:
            # Recover gradually
            if min_obstacle_distance > self.config.safety_distance:
                self.state = AvoidanceState.NORMAL
                self.state_timer = 0.0
                logger.info("Recovered to NORMAL state")
            elif self.state_timer >= self.config.recovery_time:
                if min_obstacle_distance > self.config.emergency_distance:
                    self.state = AvoidanceState.CAUTION
                    self.state_timer = 0.0
                    
        elif min_obstacle_distance < self.config.emergency_distance:
            self.state = AvoidanceState.EMERGENCY
            self.state_timer = 0.0
            self.emergency_stops += 1
            logger.warning(f"EMERGENCY: Obstacle at {min_obstacle_distance:.2f}m")
            
        elif min_obstacle_distance < self.config.safety_distance:
            if self.state != AvoidanceState.AVOIDANCE:
                self.state = AvoidanceState.AVOIDANCE
                self.state_timer = 0.0
                self.obstacle_encounters += 1
                logger.info(f"AVOIDANCE: Obstacle at {min_obstacle_distance:.2f}m")
                
        elif min_obstacle_distance < self.config.caution_distance:
            if self.state == AvoidanceState.NORMAL:
                self.state = AvoidanceState.CAUTION
                self.state_timer = 0.0
                
        elif self.state in [AvoidanceState.CAUTION, AvoidanceState.AVOIDANCE]:
            self.state = AvoidanceState.NORMAL
            self.state_timer = 0.0
            
        if old_state != self.state:
            logger.debug(f"State transition: {old_state.value} -> {self.state.value}")
    
    def compute_velocity_scale(self, min_obstacle_distance: float) -> float:
        """
        Compute velocity scale factor based on obstacle proximity.
        
        Args:
            min_obstacle_distance: Distance to nearest obstacle
            
        Returns:
            Scale factor [0, 1]
        """
        if self.state == AvoidanceState.EMERGENCY:
            return 0.0
        elif self.state == AvoidanceState.RECOVERY:
            # Gradual recovery
            progress = min(self.state_timer / self.config.recovery_time, 1.0)
            return progress * 0.3  # Max 30% speed during recovery
        elif min_obstacle_distance < self.config.safety_distance:
            # Linear interpolation between min and safety distance
            ratio = (min_obstacle_distance - self.config.emergency_distance) / \
                   (self.config.safety_distance - self.config.emergency_distance)
            return np.clip(ratio, 0.1, 0.5)
        elif min_obstacle_distance < self.config.caution_distance:
            # Linear interpolation between safety and caution distance
            ratio = (min_obstacle_distance - self.config.safety_distance) / \
                   (self.config.caution_distance - self.config.safety_distance)
            return np.clip(ratio * 0.5 + 0.5, 0.5, 0.8)
        else:
            return 1.0
    
    def compute_avoidance_velocity(self,
                                    current_position: np.ndarray,
                                    current_heading: float,
                                    target_position: np.ndarray,
                                    target_velocity: np.ndarray,
                                    obstacles: List[Dict],
                                    path: Optional[np.ndarray] = None,
                                    dt: float = 0.02) -> np.ndarray:
        """
        Compute safe avoidance velocity command.
        
        Args:
            current_position: Current drone position [x, y, z]
            current_heading: Current heading (radians)
            target_position: Goal position [x, y, z]
            target_velocity: Desired velocity [vx, vy, vz]
            obstacles: List of obstacle dictionaries
            path: Optional planned path (N, 2)
            dt: Time step (seconds)
            
        Returns:
            Safe velocity command [vx, vy, vz]
        """
        # Compute potential field forces
        force, min_distance = self.potential_field.compute_combined_force(
            current_position, target_position, obstacles
        )
        
        # Update state machine
        self.update_state(min_distance, dt)
        
        # Compute velocity scale
        velocity_scale = self.compute_velocity_scale(min_distance)
        
        # Base velocity from potential field
        if np.linalg.norm(force) > 1e-6:
            velocity_direction = force / np.linalg.norm(force)
        else:
            velocity_direction = np.zeros(3)
        
        # Modulate by state
        if self.state == AvoidanceState.EMERGENCY:
            # Stop or back up slightly
            if self.recovery_direction is None:
                # Compute escape direction (opposite to nearest obstacle)
                for obs in obstacles:
                    obs_pos = obs.get('position', obs.get('centroid'))
                    if obs_pos is not None:
                        escape = current_position - np.array(obs_pos)
                        if np.linalg.norm(escape) > 1e-6:
                            self.recovery_direction = escape / np.linalg.norm(escape)
                            break
                if self.recovery_direction is None:
                    self.recovery_direction = np.array([-1, 0, 0])  # Default: back up
            
            base_velocity = self.recovery_direction * self.config.min_speed * 0.5
            
        elif self.state == AvoidanceState.RECOVERY:
            # Gentle movement in safe direction
            base_velocity = velocity_direction * self.config.caution_speed * velocity_scale
            self.recovery_direction = None  # Clear recovery direction
            
        elif self.state == AvoidanceState.AVOIDANCE:
            # Use potential field with reduced speed
            avoidance_speed = self.config.caution_speed * velocity_scale
            base_velocity = velocity_direction * avoidance_speed
            
        elif self.state == AvoidanceState.CAUTION:
            # Blend target velocity with potential field
            pf_weight = 0.6
            target_weight = 0.4
            
            base_velocity = (pf_weight * velocity_direction * self.config.cruise_speed +
                            target_weight * target_velocity) * velocity_scale
        else:
            # Normal navigation - use pure pursuit if path available
            if path is not None and hasattr(path, '__len__') and len(path) > 0:
                current_speed = np.linalg.norm(self.smooth_velocity)
                lookahead_dist = self.pure_pursuit.compute_lookahead_distance(current_speed)
                lookahead_point, self.path_index = self.pure_pursuit.find_lookahead_point(
                    current_position, path, self.path_index, lookahead_dist
                )
                
                # Compute heading to lookahead point
                to_lookahead = lookahead_point - current_position[:2]
                if np.linalg.norm(to_lookahead) > 1e-6:
                    lookahead_direction = to_lookahead / np.linalg.norm(to_lookahead)
                    base_velocity = np.array([
                        lookahead_direction[0] * self.config.cruise_speed,
                        lookahead_direction[1] * self.config.cruise_speed,
                        target_velocity[2]  # Maintain altitude control
                    ])
                else:
                    base_velocity = target_velocity
            else:
                # Blend with potential field for safety
                base_velocity = 0.7 * target_velocity + 0.3 * velocity_direction * self.config.cruise_speed
        
        # Apply velocity smoothing
        smoothing = (self.config.recovery_smoothing 
                    if self.state in [AvoidanceState.EMERGENCY, AvoidanceState.RECOVERY]
                    else self.config.velocity_smoothing)
        
        self.smooth_velocity = (smoothing * base_velocity + 
                               (1 - smoothing) * self.smooth_velocity)
        
        # Limit velocity magnitude
        velocity = self._limit_velocity(self.smooth_velocity, velocity_scale)
        
        # Store for next iteration
        self.last_velocity = velocity.copy()
        
        logger.debug(f"State: {self.state.value}, MinDist: {min_distance:.2f}m, "
                    f"Scale: {velocity_scale:.2f}, Vel: {np.linalg.norm(velocity):.2f}m/s")
        
        return velocity
    
    def _limit_velocity(self, velocity: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Limit velocity magnitude."""
        max_speed = self.config.max_speed * scale
        speed = np.linalg.norm(velocity)
        
        if speed > max_speed:
            return velocity * (max_speed / speed)
        return velocity
    
    def set_path(self, path: np.ndarray):
        """Set a new path to follow."""
        self.current_path = path
        self.path_index = 0
        logger.debug(f"New path set with {len(path)} waypoints")
    
    def reset(self):
        """Reset controller state."""
        self.state = AvoidanceState.NORMAL
        self.state_timer = 0.0
        self.smooth_velocity = np.zeros(3)
        self.last_velocity = np.zeros(3)
        self.current_path = None
        self.path_index = 0
        self.recovery_direction = None
        logger.info("Controller state reset")
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        return {
            'state': self.state.value,
            'obstacle_encounters': self.obstacle_encounters,
            'emergency_stops': self.emergency_stops,
            'current_speed': np.linalg.norm(self.smooth_velocity)
        }
