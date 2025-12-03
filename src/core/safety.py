"""Safety monitoring and state machine for drone flight.

This module provides a consolidated safety monitor that can be used
by both simulation and hardware interfaces.
"""

import numpy as np
from typing import Dict, Optional, Callable
from loguru import logger
from enum import Enum, auto

from .types import SafetyState, SensorData, Position


class SafetyTrigger(Enum):
    """Reasons for safety state transitions."""
    NONE = auto()
    LOW_BATTERY = auto()
    GEOFENCE_BREACH = auto()
    EXCESSIVE_TILT = auto()
    LOW_ALTITUDE = auto()
    COMMUNICATION_LOSS = auto()
    MANUAL_STOP = auto()
    STOPPED_LOW = auto()


class SafetyMonitor:
    """Monitors drone state and triggers safety responses.
    
    This class consolidates safety logic that was previously duplicated
    across webots_interface.py and depth_controller.py.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize safety monitor.
        
        Args:
            config: Configuration dictionary with safety parameters
        """
        # Extract safety config
        safety_config = {}
        if config:
            safety_config = config.get('drone', {}).get('safety', {})
        
        # Safety thresholds
        self.battery_min = safety_config.get('battery_min_voltage', 3.3)
        self.battery_warning = safety_config.get('battery_warning_voltage', 3.5)
        self.geofence_radius = safety_config.get('geofence_radius', 3.0)
        self.geofence_height = safety_config.get('geofence_height', 2.0)
        self.tilt_threshold = np.deg2rad(safety_config.get('crash_tilt_threshold', 40.0))
        self.altitude_threshold = safety_config.get('crash_altitude_threshold', 0.15)
        self.min_flight_altitude = safety_config.get('min_flight_altitude', 0.3)
        
        # State machine
        self.state = SafetyState.INITIALIZING
        self.last_trigger = SafetyTrigger.NONE
        
        # Tracking
        self.start_position: Optional[Position] = None
        self.takeoff_complete = False
        self.crash_detection_enabled = False
        self.battery_warned = False
        
        # Communication loss tracking
        self.last_data_time = 0.0
        self.comm_timeout = safety_config.get('comm_timeout', 1.0)
        
        # Callbacks
        self._on_emergency: Optional[Callable] = None
        self._on_warning: Optional[Callable] = None
        
        logger.info("[SAFETY] Monitor initialized")
        logger.debug(f"[SAFETY] Thresholds: tilt={np.rad2deg(self.tilt_threshold):.1f}° "
                    f"alt={self.altitude_threshold}m geofence={self.geofence_radius}m")
    
    def set_emergency_callback(self, callback: Callable) -> None:
        """Set callback for emergency events."""
        self._on_emergency = callback
    
    def set_warning_callback(self, callback: Callable) -> None:
        """Set callback for warning events."""
        self._on_warning = callback
    
    def set_start_position(self, position: Position) -> None:
        """Set the starting position for geofence calculations."""
        self.start_position = position
        logger.info(f"[SAFETY] Start position set: ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})")
    
    def arm(self) -> bool:
        """Transition to armed state.
        
        Returns:
            True if arming successful
        """
        if self.state in [SafetyState.READY, SafetyState.LANDED]:
            self.state = SafetyState.ARMED
            self.takeoff_complete = False
            self.crash_detection_enabled = False
            logger.info("[SAFETY] Armed - Ready for takeoff")
            return True
        else:
            logger.warning(f"[SAFETY] Cannot arm from state: {self.state}")
            return False
    
    def set_ready(self) -> None:
        """Transition to ready state (after initialization)."""
        if self.state == SafetyState.INITIALIZING:
            self.state = SafetyState.READY
            logger.info("[SAFETY] Ready")
    
    def set_flying(self) -> None:
        """Transition to flying state (after takeoff)."""
        if self.state == SafetyState.ARMED:
            self.state = SafetyState.FLYING
            logger.info("[SAFETY] Flying")
    
    def set_landing(self) -> None:
        """Transition to landing state."""
        if self.state == SafetyState.FLYING:
            self.state = SafetyState.LANDING
            self.crash_detection_enabled = False  # Disable during landing
            logger.info("[SAFETY] Landing")
    
    def set_landed(self) -> None:
        """Transition to landed state."""
        if self.state in [SafetyState.LANDING, SafetyState.FLYING]:
            self.state = SafetyState.LANDED
            self.crash_detection_enabled = False
            logger.info("[SAFETY] Landed")
    
    def check(self, sensor_data: SensorData) -> SafetyState:
        """Check all safety conditions.
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Current safety state (may have transitioned to EMERGENCY)
        """
        # Update communication tracking
        self.last_data_time = sensor_data.timestamp
        
        # Skip checks if already in emergency
        if self.state == SafetyState.EMERGENCY:
            return self.state
        
        # Check takeoff completion (enables crash detection)
        if self.state == SafetyState.FLYING and not self.takeoff_complete:
            if sensor_data.altitude > self.min_flight_altitude:
                self.takeoff_complete = True
                self.crash_detection_enabled = True
                logger.info(f"[SAFETY] ✓ Takeoff complete @ {sensor_data.altitude:.2f}m "
                          f"- Crash detection ENABLED")
        
        # Only run safety checks when flying with crash detection enabled
        if not self.crash_detection_enabled:
            return self.state
        
        # Battery check (hardware only)
        if sensor_data.battery > 0:
            if sensor_data.battery < self.battery_min:
                self._trigger_emergency(SafetyTrigger.LOW_BATTERY,
                    f"Battery critical: {sensor_data.battery:.2f}V < {self.battery_min}V")
                return self.state
            elif sensor_data.battery < self.battery_warning and not self.battery_warned:
                self.battery_warned = True
                self._warn(f"Battery low: {sensor_data.battery:.2f}V")
        
        # Geofence check
        if self.start_position is not None:
            horizontal_dist = np.sqrt(
                (sensor_data.position.x - self.start_position.x) ** 2 +
                (sensor_data.position.y - self.start_position.y) ** 2
            )
            if horizontal_dist > self.geofence_radius:
                self._trigger_emergency(SafetyTrigger.GEOFENCE_BREACH,
                    f"Geofence breach: {horizontal_dist:.2f}m > {self.geofence_radius}m")
                return self.state
            if sensor_data.position.z > self.geofence_height:
                self._trigger_emergency(SafetyTrigger.GEOFENCE_BREACH,
                    f"Height limit: {sensor_data.position.z:.2f}m > {self.geofence_height}m")
                return self.state
        
        # Tilt check
        roll = abs(sensor_data.orientation.roll)
        pitch = abs(sensor_data.orientation.pitch)
        if roll > self.tilt_threshold or pitch > self.tilt_threshold:
            self._trigger_emergency(SafetyTrigger.EXCESSIVE_TILT,
                f"Extreme tilt: roll={np.rad2deg(roll):.1f}° pitch={np.rad2deg(pitch):.1f}°")
            return self.state
        
        # Altitude check (crashed into ground)
        if sensor_data.altitude < self.altitude_threshold:
            self._trigger_emergency(SafetyTrigger.LOW_ALTITUDE,
                f"Low altitude: {sensor_data.altitude:.3f}m < {self.altitude_threshold}m")
            return self.state
        
        # Stopped at low altitude check
        speed = sensor_data.velocity.speed
        if speed < 0.01 and sensor_data.altitude < 0.5:
            self._trigger_emergency(SafetyTrigger.STOPPED_LOW,
                f"Stopped at low altitude: {sensor_data.altitude:.3f}m")
            return self.state
        
        return self.state
    
    def check_communication(self, current_time: float) -> SafetyState:
        """Check for communication loss.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Current safety state
        """
        if self.state == SafetyState.EMERGENCY:
            return self.state
        
        if self.crash_detection_enabled:
            time_since_data = current_time - self.last_data_time
            if time_since_data > self.comm_timeout:
                self._trigger_emergency(SafetyTrigger.COMMUNICATION_LOSS,
                    f"Communication loss: {time_since_data:.2f}s")
        
        return self.state
    
    def manual_emergency(self) -> None:
        """Trigger manual emergency stop."""
        self._trigger_emergency(SafetyTrigger.MANUAL_STOP, "Manual emergency stop")
    
    def reset(self) -> None:
        """Reset safety state after recovery."""
        if self.state == SafetyState.EMERGENCY:
            self.state = SafetyState.READY
            self.last_trigger = SafetyTrigger.NONE
            self.takeoff_complete = False
            self.crash_detection_enabled = False
            self.battery_warned = False
            logger.info("[SAFETY] State reset - Ready")
    
    def _trigger_emergency(self, trigger: SafetyTrigger, message: str) -> None:
        """Transition to emergency state.
        
        Args:
            trigger: Reason for emergency
            message: Log message
        """
        if self.state != SafetyState.EMERGENCY:
            self.state = SafetyState.EMERGENCY
            self.last_trigger = trigger
            logger.error(f"[SAFETY] ✗ EMERGENCY: {message}")
            
            if self._on_emergency:
                self._on_emergency(trigger, message)
    
    def _warn(self, message: str) -> None:
        """Log warning and trigger callback.
        
        Args:
            message: Warning message
        """
        logger.warning(f"[SAFETY] ⚠ {message}")
        
        if self._on_warning:
            self._on_warning(message)
    
    @property
    def is_safe(self) -> bool:
        """Check if current state is safe for flight operations."""
        return self.state not in [SafetyState.EMERGENCY, SafetyState.INITIALIZING]
    
    @property
    def can_fly(self) -> bool:
        """Check if state allows flight commands."""
        return self.state in [SafetyState.FLYING, SafetyState.ARMED]
