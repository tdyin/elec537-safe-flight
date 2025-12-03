"""
Velocity Controller for Navigation.

Handles velocity smoothing, stuck detection, and position tracking.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class VelocityState:
    """Current velocity state with smoothing."""
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0


@dataclass 
class StuckDetectorConfig:
    """Configuration for stuck detection."""
    enabled: bool = True
    history_size: int = 20
    threshold: float = 0.05  # meters
    time_window: float = 3.0  # seconds


class VelocityController:
    """
    Manages velocity smoothing and output filtering.
    
    Applies exponential smoothing to velocity commands for stable flight.
    """
    
    def __init__(self, 
                 smoothing_alpha: float = 0.10,
                 avoidance_alpha: float = 0.20,
                 emergency_alpha: float = 0.25):
        """
        Initialize velocity controller.
        
        Args:
            smoothing_alpha: Default smoothing factor (0-1, higher = more responsive)
            avoidance_alpha: Smoothing during avoidance maneuvers
            emergency_alpha: Smoothing during emergency states
        """
        self.smoothing_alpha = smoothing_alpha
        self.avoidance_alpha = avoidance_alpha
        self.emergency_alpha = emergency_alpha
        
        self.state = VelocityState()
    
    def smooth(self, vx: float, vy: float, yaw: float, 
               mode: str = 'normal') -> Tuple[float, float, float]:
        """
        Apply smoothing to velocity commands.
        
        Args:
            vx: Forward velocity command
            vy: Sideways velocity command
            yaw: Yaw rate command
            mode: 'normal', 'avoidance', or 'emergency'
            
        Returns:
            Smoothed (vx, vy, yaw) tuple
        """
        if mode == 'avoidance':
            alpha = self.avoidance_alpha
        elif mode == 'emergency':
            alpha = self.emergency_alpha
        else:
            alpha = self.smoothing_alpha
        
        self.state.vx = alpha * vx + (1 - alpha) * self.state.vx
        self.state.vy = alpha * vy + (1 - alpha) * self.state.vy
        self.state.yaw = alpha * yaw + (1 - alpha) * self.state.yaw
        
        return self.state.vx, self.state.vy, self.state.yaw
    
    def get_velocities(self) -> Tuple[float, float, float]:
        """Get current smoothed velocities."""
        return self.state.vx, self.state.vy, self.state.yaw
    
    def reset(self):
        """Reset velocity state."""
        self.state = VelocityState()


class StuckDetector:
    """
    Detects when drone is stuck (not making progress).
    
    Tracks position history and checks if drone has moved
    sufficiently over a time window.
    """
    
    def __init__(self, config: StuckDetectorConfig = None):
        """
        Initialize stuck detector.
        
        Args:
            config: Stuck detection configuration
        """
        self.config = config or StuckDetectorConfig()
        
        self.position_history: List[np.ndarray] = []
        self.timer: float = 0.0
        self.stuck_count: int = 0
        self.recovery_direction: int = 1
    
    @property
    def enabled(self) -> bool:
        """Check if stuck detection is enabled."""
        return self.config.enabled
    
    def update(self, position: np.ndarray, dt: float):
        """
        Update position history.
        
        Args:
            position: Current position [x, y, z]
            dt: Time step in seconds
        """
        if not self.config.enabled:
            return
        
        self.timer += dt
        self.position_history.append(position.copy())
        
        # Maintain history size
        while len(self.position_history) > self.config.history_size:
            self.position_history.pop(0)
    
    def is_stuck(self, allowed_modes: List[str] = None, current_mode: str = None) -> bool:
        """
        Check if drone is stuck.
        
        Args:
            allowed_modes: List of modes where stuck detection applies
            current_mode: Current navigation mode
            
        Returns:
            True if stuck, False otherwise
        """
        if not self.config.enabled:
            return False
        
        if len(self.position_history) < self.config.history_size:
            return False
        
        if self.timer < self.config.time_window:
            return False
        
        # Only check in certain modes
        if allowed_modes and current_mode:
            if current_mode not in allowed_modes:
                self.timer = 0.0
                return False
        
        # Compare oldest and newest positions (XY only)
        oldest = self.position_history[0]
        newest = self.position_history[-1]
        distance_traveled = np.linalg.norm(newest[:2] - oldest[:2])
        
        if distance_traveled < self.config.threshold:
            self.stuck_count += 1
            self.recovery_direction *= -1  # Alternate direction
            self.reset()
            return True
        
        return False
    
    def get_recovery_direction(self) -> int:
        """Get suggested recovery direction (-1=left, 1=right)."""
        return self.recovery_direction
    
    def reset(self):
        """Reset stuck detection state."""
        self.position_history.clear()
        self.timer = 0.0
