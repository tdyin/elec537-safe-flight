"""Shared navigation algorithms.

This module contains navigation utilities that are shared between
simulation and hardware implementations.
"""

import numpy as np
from typing import Tuple, Optional, List
from loguru import logger

from .types import Position, Velocity, VelocityCommand, Waypoint


def compute_heading_to_target(current_position: Position, 
                               target_position: Position) -> float:
    """Compute heading angle to target position.
    
    Args:
        current_position: Current drone position
        target_position: Target position
        
    Returns:
        Heading angle in radians (0 = +X, positive = CCW)
    """
    dx = target_position.x - current_position.x
    dy = target_position.y - current_position.y
    return np.arctan2(dy, dx)


def compute_velocity_to_target(current_position: Position,
                                target_position: Position,
                                max_speed: float,
                                arrival_radius: float = 0.1) -> Tuple[float, float]:
    """Compute velocity vector towards target.
    
    Args:
        current_position: Current drone position
        target_position: Target position
        max_speed: Maximum speed in m/s
        arrival_radius: Distance at which to slow down
        
    Returns:
        Tuple of (vx, vy) in m/s
    """
    dx = target_position.x - current_position.x
    dy = target_position.y - current_position.y
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 0.01:  # Already at target
        return (0.0, 0.0)
    
    # Scale speed based on distance (slow down near target)
    speed = min(max_speed, max_speed * distance / arrival_radius)
    
    # Normalize and scale
    vx = (dx / distance) * speed
    vy = (dy / distance) * speed
    
    return (vx, vy)


def compute_yaw_rate_to_heading(current_yaw: float,
                                 target_heading: float,
                                 max_yaw_rate: float) -> float:
    """Compute yaw rate to turn towards target heading.
    
    Args:
        current_yaw: Current yaw in radians
        target_heading: Target heading in radians
        max_yaw_rate: Maximum yaw rate in deg/s
        
    Returns:
        Yaw rate in deg/s
    """
    # Compute shortest angle difference
    diff = target_heading - current_yaw
    
    # Normalize to [-pi, pi]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    
    # Convert to deg/s and clamp
    yaw_rate = np.rad2deg(diff) * 2.0  # Proportional gain of 2.0
    yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)
    
    return yaw_rate


def check_waypoint_reached(current_position: Position,
                           waypoint: Waypoint,
                           horizontal_threshold: float = 0.3,
                           vertical_threshold: float = 0.2) -> bool:
    """Check if waypoint has been reached.
    
    Args:
        current_position: Current drone position
        waypoint: Target waypoint
        horizontal_threshold: Horizontal distance threshold (m)
        vertical_threshold: Vertical distance threshold (m)
        
    Returns:
        True if waypoint reached
    """
    dx = current_position.x - waypoint.position.x
    dy = current_position.y - waypoint.position.y
    dz = current_position.z - waypoint.position.z
    
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    vertical_dist = abs(dz)
    
    return horizontal_dist < horizontal_threshold and vertical_dist < vertical_threshold


class WaypointNavigator:
    """Simple waypoint navigation controller.
    
    This provides basic waypoint following that can be used by both
    simulation and hardware interfaces.
    """
    
    def __init__(self, 
                 max_speed: float = 0.5,
                 max_yaw_rate: float = 30.0,
                 arrival_radius: float = 0.3,
                 heading_tolerance: float = 0.1):
        """Initialize waypoint navigator.
        
        Args:
            max_speed: Maximum horizontal speed (m/s)
            max_yaw_rate: Maximum yaw rate (deg/s)
            arrival_radius: Distance to consider waypoint reached (m)
            heading_tolerance: Heading error tolerance (rad)
        """
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.arrival_radius = arrival_radius
        self.heading_tolerance = heading_tolerance
        
        self.waypoints: List[Waypoint] = []
        self.current_index = 0
        self.completed = False
    
    def set_waypoints(self, waypoints: List[Waypoint]) -> None:
        """Set list of waypoints to follow.
        
        Args:
            waypoints: List of waypoints
        """
        self.waypoints = waypoints
        self.current_index = 0
        self.completed = False
        logger.info(f"[NAV] Set {len(waypoints)} waypoints")
    
    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the end of the list.
        
        Args:
            waypoint: Waypoint to add
        """
        self.waypoints.append(waypoint)
    
    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint."""
        if self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None
    
    @property
    def progress(self) -> float:
        """Get navigation progress (0.0 to 1.0)."""
        if not self.waypoints:
            return 1.0
        return self.current_index / len(self.waypoints)
    
    def compute_command(self, 
                        current_position: Position,
                        current_yaw: float) -> Optional[VelocityCommand]:
        """Compute velocity command to reach next waypoint.
        
        Args:
            current_position: Current drone position
            current_yaw: Current yaw in radians
            
        Returns:
            VelocityCommand or None if navigation complete
        """
        if self.completed or not self.waypoints:
            return None
        
        waypoint = self.current_waypoint
        if waypoint is None:
            self.completed = True
            return None
        
        # Check if reached
        if check_waypoint_reached(current_position, waypoint, self.arrival_radius):
            logger.info(f"[NAV] ✓ Reached waypoint {self.current_index + 1}/{len(self.waypoints)}")
            self.current_index += 1
            
            if self.current_index >= len(self.waypoints):
                self.completed = True
                logger.info("[NAV] ✓ Navigation complete")
                return VelocityCommand()  # Hover
            
            waypoint = self.current_waypoint
        
        # Compute velocities
        target_pos = waypoint.position
        
        # Horizontal velocity
        vx, vy = compute_velocity_to_target(
            current_position, target_pos, 
            waypoint.speed or self.max_speed,
            self.arrival_radius
        )
        
        # Vertical velocity
        dz = target_pos.z - current_position.z
        vz = np.clip(dz * 0.5, -self.max_speed * 0.5, self.max_speed * 0.5)
        
        # Yaw rate (point towards target or use specified heading)
        if waypoint.heading is not None:
            target_heading = waypoint.heading
        else:
            target_heading = compute_heading_to_target(current_position, target_pos)
        
        yaw_rate = compute_yaw_rate_to_heading(
            current_yaw, target_heading, self.max_yaw_rate
        )
        
        return VelocityCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
    
    def reset(self) -> None:
        """Reset navigation state."""
        self.current_index = 0
        self.completed = False
