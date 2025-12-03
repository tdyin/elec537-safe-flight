"""
Avoidance State Handler for Navigation.

Manages obstacle avoidance state machine including:
- Emergency stop handling
- Avoidance phase transitions (turning, moving, scanning, clearing)
- Return-to-path logic
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .state_machine import NavigationMode, AvoidancePhase
from ..od.zones import ObstacleZone


@dataclass
class AvoidanceConfig:
    """Configuration for avoidance behavior."""
    speed: float = 0.18
    turn_rate: float = 1.0
    duration: float = 1.5
    clear_distance: float = 0.5
    emergency_backup_speed: float = -0.10
    emergency_timeout: float = 2.0
    # Vertical avoidance parameters
    vertical_speed: float = 0.15  # Altitude change speed (m/s)
    max_altitude_change: float = 0.5  # Maximum altitude adjustment per maneuver
    # Altitude return parameters
    default_altitude: float = 0.8  # Default flight altitude (m)
    altitude_return_speed: float = 0.08  # Speed to return to default altitude (m/s)
    altitude_tolerance: float = 0.1  # Tolerance for altitude return (m)


@dataclass
class AvoidanceState:
    """Current state of avoidance maneuver."""
    mode: NavigationMode = NavigationMode.IDLE
    phase: AvoidancePhase = AvoidancePhase.NONE
    timer: float = 0.0
    direction: int = 0  # -1=left, 1=right
    vertical_direction: int = 0  # -1=down, 0=none, 1=up
    vertical_magnitude: float = 0.0  # 0.0-1.0 urgency scale
    start_position: Optional[np.ndarray] = None
    count: int = 0  # Number of avoidance maneuvers
    emergency_timer: float = 0.0
    altitude_displaced: bool = False  # True if altitude was changed during avoidance
    returning_to_altitude: bool = False  # True if returning to default altitude


class AvoidanceHandler:
    """
    Handles obstacle avoidance state machine.
    
    Manages transitions between avoidance phases and generates
    appropriate velocity commands for each phase.
    """
    
    def __init__(self, config: AvoidanceConfig = None, log_func=None):
        """
        Initialize avoidance handler.
        
        Args:
            config: Avoidance configuration
            log_func: Logging function
        """
        self.config = config or AvoidanceConfig()
        self.state = AvoidanceState()
        self._log = log_func or (lambda msg, level: None)
    
    @property
    def mode(self) -> NavigationMode:
        return self.state.mode
    
    @mode.setter
    def mode(self, value: NavigationMode):
        self.state.mode = value
    
    @property
    def phase(self) -> AvoidancePhase:
        return self.state.phase
    
    @property
    def avoidance_count(self) -> int:
        return self.state.count
    
    def start_avoidance(self, position: np.ndarray, detection: dict):
        """
        Initialize avoidance maneuver.
        
        Args:
            position: Current drone position
            detection: Obstacle detection result
        """
        self.state.mode = NavigationMode.AVOIDING
        self.state.phase = AvoidancePhase.TURNING
        self.state.timer = 0.0
        self.state.direction = detection['direction']
        self.state.vertical_direction = detection.get('vertical_direction', 0)
        self.state.vertical_magnitude = detection.get('vertical_magnitude', 0.0)
        self.state.start_position = position.copy()
        self.state.count += 1
        # Mark if altitude will be displaced during this avoidance
        if self.state.vertical_direction != 0:
            self.state.altitude_displaced = True
        
        dir_str = "LEFT" if self.state.direction == -1 else "RIGHT"
        vert_str = ""
        if self.state.vertical_direction == 1:
            vert_str = " + UP"
        elif self.state.vertical_direction == -1:
            vert_str = " + DOWN"
        self._log(f"[NAV] 🔄 Avoidance #{self.state.count} - Turning {dir_str}{vert_str}", "WARNING")
    
    def handle_emergency(self, position: np.ndarray, yaw: float,
                        detection: dict, dt: float) -> Tuple[float, float, float, float]:
        """
        Handle emergency stop state.
        
        Args:
            position: Current position
            yaw: Current heading
            detection: Obstacle detection result
            dt: Time step
            
        Returns:
            Velocity command (vx, vy, yaw_rate, vz)
        """
        self.state.emergency_timer += dt
        zone = detection['zone']
        
        # Check if we can recover
        if zone in [ObstacleZone.CLEAR, ObstacleZone.FAR]:
            self.state.mode = NavigationMode.RETURNING
            self.state.emergency_timer = 0.0
            self._log("[NAV] Emergency cleared - returning to path", "SUCCESS")
            return 0.0, 0.0, 0.0, 0.0
        
        elif zone in [ObstacleZone.CAUTION, ObstacleZone.CLOSE]:
            self.start_avoidance(position, detection)
            self.state.emergency_timer = 0.0
            self._log("[NAV] Emergency → Avoidance mode", "INFO")
            return self.execute_avoidance(position, yaw, detection, dt)
        
        elif self.state.emergency_timer > self.config.emergency_timeout:
            self.start_avoidance(position, detection)
            self.state.emergency_timer = 0.0
            self._log("[NAV] Emergency timeout - forcing avoidance", "WARNING")
            return self.execute_avoidance(position, yaw, detection, dt)
        
        # Still critical - back up slowly and consider vertical escape
        backup_speed = self.config.emergency_backup_speed
        turn_direction = detection.get('direction', 1)
        yaw_rate = turn_direction * self.config.turn_rate * 0.5
        
        # Add vertical escape during emergency
        vert_dir = detection.get('vertical_direction', 0)
        vert_mag = detection.get('vertical_magnitude', 0.0)
        vz = vert_dir * self.config.vertical_speed * vert_mag
        
        return backup_speed, 0.0, yaw_rate, vz
    
    def execute_avoidance(self, position: np.ndarray, yaw: float,
                          detection: dict, dt: float) -> Tuple[float, float, float, float]:
        """
        Execute avoidance maneuver state machine.
        
        Args:
            position: Current position
            yaw: Current heading
            detection: Obstacle detection result
            dt: Time step
            
        Returns:
            Velocity command (vx, vy, yaw_rate, vz)
        """
        self.state.timer += dt
        zone = detection['zone']
        has_horizontal = detection.get('horizontal_obstacle', False)
        
        # Update direction if needed (lateral and vertical)
        self._update_direction(detection)
        self._update_vertical_direction(detection)
        
        if self.state.phase == AvoidancePhase.TURNING:
            return self._execute_turning(detection, has_horizontal)
        
        elif self.state.phase == AvoidancePhase.MOVING:
            return self._execute_moving(position, detection, has_horizontal)
        
        elif self.state.phase == AvoidancePhase.SCANNING:
            return self._execute_scanning(detection)
        
        elif self.state.phase == AvoidancePhase.CLEARING:
            return self._execute_clearing(zone)
        
        return 0.0, 0.0, 0.0, 0.0
    
    def _update_direction(self, detection: dict):
        """Update avoidance direction based on clearance."""
        if detection['direction'] != 0:
            if self.state.direction == -1 and detection['clearance']['left'] < 0.2:
                self.state.direction = 1
            elif self.state.direction == 1 and detection['clearance']['right'] < 0.2:
                self.state.direction = -1
    
    def _update_vertical_direction(self, detection: dict):
        """
        Update vertical avoidance direction based on vertical zone clearances.
        
        This allows the drone to dynamically adjust vertical escape direction
        based on current sensor readings.
        """
        vert_clearance = detection.get('vertical_clearance', {})
        upper = vert_clearance.get('upper', 1.0)
        lower = vert_clearance.get('lower', 1.0)
        
        new_vert_dir = detection.get('vertical_direction', 0)
        new_vert_mag = detection.get('vertical_magnitude', 0.0)
        
        # If currently going up but upper is blocked, switch to down
        if self.state.vertical_direction == 1 and upper < 0.25:
            if lower > 0.35:
                self.state.vertical_direction = -1
                self._log("[NAV] Vertical escape: UP blocked, switching to DOWN", "DEBUG")
        # If currently going down but lower is blocked, switch to up
        elif self.state.vertical_direction == -1 and lower < 0.25:
            if upper > 0.35:
                self.state.vertical_direction = 1
                self._log("[NAV] Vertical escape: DOWN blocked, switching to UP", "DEBUG")
        # If no vertical direction set yet, use detection's recommendation
        elif self.state.vertical_direction == 0 and new_vert_dir != 0:
            self.state.vertical_direction = new_vert_dir
            self.state.vertical_magnitude = new_vert_mag
    
    def _compute_vz(self) -> float:
        """Compute vertical velocity based on state."""
        return self.state.vertical_direction * self.config.vertical_speed * self.state.vertical_magnitude
    
    def _execute_turning(self, detection: dict, 
                         has_horizontal: bool) -> Tuple[float, float, float, float]:
        """Execute turning phase with vertical escape."""
        yaw_rate = self.state.direction * self.config.turn_rate
        lateral = self.state.direction * self.config.speed * 0.4
        
        # For horizontal obstacles, increase vertical escape
        vz = self._compute_vz()
        if has_horizontal:
            lateral *= 0.7
            # Boost vertical escape for horizontal obstacles
            vz *= 1.5
        
        # Check transition
        if self.state.timer > 0.5:
            center_clearance = detection['clearance']['center']
            if center_clearance > 0.5 or self.state.timer > 1.5:
                self.state.phase = AvoidancePhase.MOVING
                self.state.timer = 0.0
                self._log("[NAV] Avoidance: Turning → Moving", "DEBUG")
        
        return -0.05, lateral, yaw_rate, vz
    
    def _execute_moving(self, position: np.ndarray, detection: dict,
                        has_horizontal: bool) -> Tuple[float, float, float, float]:
        """Execute moving phase with vertical escape."""
        forward = self.config.speed * 0.5
        lateral = self.state.direction * self.config.speed * 0.7
        yaw_rate = self.state.direction * self.config.turn_rate * 0.3
        
        vz = self._compute_vz()
        if has_horizontal:
            lateral = self.state.direction * self.config.speed * 0.9
            forward *= 0.7
            # Continue vertical escape for horizontal obstacles
            vz *= 1.3
        
        # Check transition by distance
        if self.state.start_position is not None:
            dist_moved = np.linalg.norm(position[:2] - self.state.start_position[:2])
            if dist_moved > self.config.clear_distance:
                self.state.phase = AvoidancePhase.SCANNING
                self.state.timer = 0.0
                self._log(f"[NAV] Avoidance: Moving → Scanning (moved {dist_moved:.2f}m)", "DEBUG")
        
        # Check transition by time
        if self.state.timer > self.config.duration:
            self.state.phase = AvoidancePhase.SCANNING
            self.state.timer = 0.0
        
        return forward, lateral, yaw_rate, vz
    
    def _execute_scanning(self, detection: dict) -> Tuple[float, float, float, float]:
        """Execute scanning phase."""
        center_clearance = detection['clearance']['center']
        yaw_rate = -self.state.direction * self.config.turn_rate * 0.5
        lateral = self.state.direction * self.config.speed * 0.5
        
        # Reduce vertical velocity during scanning (stabilize altitude)
        vz = self._compute_vz() * 0.3
        
        # Check transition
        if center_clearance > 0.6 or self.state.timer > 1.0:
            self.state.phase = AvoidancePhase.CLEARING
            self.state.timer = 0.0
            self._log("[NAV] Avoidance: Scanning → Clearing", "DEBUG")
        
        return self.config.speed * 0.3, lateral, yaw_rate, vz
    
    def _execute_clearing(self, zone: ObstacleZone) -> Tuple[float, float, float, float]:
        """Execute clearing phase."""
        forward = self.config.speed
        
        # No vertical adjustment during clearing (return to normal altitude)
        vz = 0.0
        
        # Check if avoidance complete
        if self.state.timer > 0.5:
            if zone in [ObstacleZone.CLEAR, ObstacleZone.FAR, ObstacleZone.CAUTION]:
                self.state.mode = NavigationMode.RETURNING
                self.state.phase = AvoidancePhase.NONE
                # Reset vertical state but mark for altitude return
                self.state.vertical_direction = 0
                self.state.vertical_magnitude = 0.0
                if self.state.altitude_displaced:
                    self.state.returning_to_altitude = True
                    self._log("[NAV] ✓ Avoidance complete - Returning to path and default altitude", "SUCCESS")
                else:
                    self._log("[NAV] ✓ Avoidance complete - Returning to path", "SUCCESS")
            elif self.state.timer > 2.0:
                self.state.mode = NavigationMode.RETURNING
                self.state.phase = AvoidancePhase.NONE
                self.state.vertical_direction = 0
                self.state.vertical_magnitude = 0.0
                if self.state.altitude_displaced:
                    self.state.returning_to_altitude = True
                self._log("[NAV] Avoidance timeout - Attempting return", "WARNING")
        
        return forward, 0.0, 0.0, vz
    
    def trigger_emergency(self):
        """Trigger emergency stop."""
        self.state.mode = NavigationMode.EMERGENCY_STOP
        self.state.emergency_timer = 0.0
        self.state.count += 1
        self._log(f"[NAV] ⚠ EMERGENCY STOP #{self.state.count} - Critical obstacle!", "ERROR")
    
    def set_returning(self):
        """Set mode to returning to path."""
        self.state.mode = NavigationMode.RETURNING
        self.state.phase = AvoidancePhase.NONE
    
    def set_path_following(self):
        """Set mode to path following."""
        self.state.mode = NavigationMode.PATH_FOLLOWING
        self.state.phase = AvoidancePhase.NONE
    
    def set_goal_reached(self):
        """Set mode to goal reached."""
        self.state.mode = NavigationMode.GOAL_REACHED
    
    def reset(self):
        """Reset avoidance state."""
        self.state = AvoidanceState()
    
    def compute_altitude_return(self, current_altitude: float, detection: dict) -> float:
        """
        Compute vertical velocity to return to default altitude.
        
        Only returns to default altitude when it's safe (no close obstacles
        in the path of return).
        
        Args:
            current_altitude: Current drone altitude (m)
            detection: Current obstacle detection result
            
        Returns:
            Vertical velocity (m/s) for altitude return, 0 if unsafe or not needed
        """
        if not self.state.returning_to_altitude:
            return 0.0
        
        zone = detection.get('zone', ObstacleZone.CLEAR)
        
        # Don't adjust altitude if obstacles are close
        if zone in [ObstacleZone.CRITICAL, ObstacleZone.CLOSE]:
            return 0.0
        
        # Calculate altitude error
        altitude_error = self.config.default_altitude - current_altitude
        
        # Check if we've reached default altitude
        if abs(altitude_error) < self.config.altitude_tolerance:
            self.state.returning_to_altitude = False
            self.state.altitude_displaced = False
            self._log(f"[NAV] ✓ Returned to default altitude ({current_altitude:.2f}m)", "SUCCESS")
            return 0.0
        
        # Check vertical clearance before adjusting altitude
        vert_clearance = detection.get('vertical_clearance', {})
        upper_clear = vert_clearance.get('upper', 1.0)
        lower_clear = vert_clearance.get('lower', 1.0)
        
        if altitude_error > 0:  # Need to go up
            # Check if going up is safe
            if upper_clear < 0.4:
                return 0.0  # Not safe to go up
            return min(altitude_error, self.config.altitude_return_speed)
        else:  # Need to go down
            # Check if going down is safe
            if lower_clear < 0.4:
                return 0.0  # Not safe to go down
            return max(altitude_error, -self.config.altitude_return_speed)
