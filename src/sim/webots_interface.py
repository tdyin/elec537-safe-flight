"""Webots simulation interface for drone control.

This module provides the WebotsInterface class that implements the DroneInterface
ABC for controlling drones in the Webots simulator.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from loguru import logger

from ..core.base_interface import DroneInterface
from ..core.types import SensorData, Position
from ..core.safety import SafetyMonitor, SafetyState
from .bridge import SimulationBridge


class WebotsInterface(DroneInterface):
    """Interface for controlling Crazyflie in Webots simulation.
    
    Implements the DroneInterface ABC for simulation environments.
    Uses TCP socket bridge to communicate with Webots controller.
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 10020,
                 uri: str = 'webots://simulation',
                 config: Optional[dict] = None):
        """Initialize Webots interface.
        
        Args:
            host: Host address for simulation bridge
            port: Port number for simulation bridge
            uri: Identifier for the simulated drone
            config: Optional configuration dictionary
        """
        self.uri = uri
        self._is_connected = False
        
        # Load config values if provided
        drone_config = config.get('drone', {}) if config else {}
        nav_config = drone_config.get('navigation', {})
        sim_config = drone_config.get('simulation', {})
        
        # Use config values for host/port, or fall back to params/defaults
        actual_host = sim_config.get('host', host)
        actual_port = sim_config.get('port', port)
        
        # Simulation bridge
        self.bridge = SimulationBridge(host=actual_host, port=actual_port)
        
        # Safety monitor
        self.safety = SafetyMonitor(config)
        self.safety.set_emergency_callback(self._on_emergency)
        
        # Legacy compatibility attributes (to be removed)
        self.crashed = False
        self.crash_tilt_threshold = np.deg2rad(nav_config.get('crash_tilt_threshold', 45.0))
        self.crash_altitude_threshold = nav_config.get('crash_altitude_threshold', 0.1)
        self.crash_detection_enabled = False
        self.takeoff_complete = False
        self.min_flight_altitude = nav_config.get('min_flight_altitude', 0.3)
        
        # Takeoff/land state
        self._is_flying = False
        self._target_altitude = nav_config.get('default_altitude', 0.5)
        
        logger.info("[WEBOTS] Interface initialized")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to simulation."""
        return self._is_connected
    
    def connect(self) -> bool:
        """Connect to Webots simulation.
        
        Returns:
            True if connection successful
        """
        if self.bridge.connect():
            self._is_connected = True
            self.safety.set_ready()
            logger.info(f"[WEBOTS] ✓ Connected to simulation at {self.bridge.host}:{self.bridge.port}")
            return True
        else:
            logger.error("[WEBOTS] ✗ Connection failed")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Webots simulation."""
        self.bridge.disconnect()
        self._is_connected = False
        self._is_flying = False
        logger.info("[WEBOTS] Disconnected from simulation")
    
    def get_sensor_data(self) -> Dict:
        """Get sensor data from Webots simulation.
        
        Returns:
            Dictionary containing sensor readings:
            - camera: numpy array of camera image (H, W, C)
            - range_front, range_left, range_right, range_back: float distances
            - roll, pitch, yaw: orientation in radians
            - altitude: height in meters
            - velocity: (vx, vy, vz) in m/s
            - position: (x, y, z) in meters
            - timestamp: simulation time
        """
        if not self._is_connected:
            logger.warning("Not connected to simulation")
            return {}
        
        return self.bridge.get_sensor_data()
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current drone position from simulation.
        
        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        if not self._is_connected:
            return (0.0, 0.0, 0.0)
        
        return self.bridge.get_position()
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current drone orientation from simulation.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        if not self._is_connected:
            return (0.0, 0.0, 0.0)
        
        sensor_data = self.bridge.get_sensor_data()
        return (
            sensor_data.get('roll', 0.0),
            sensor_data.get('pitch', 0.0),
            sensor_data.get('yaw', 0.0)
        )
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current drone velocity from simulation.
        
        Returns:
            Tuple of (vx, vy, vz) in m/s
        """
        if not self._is_connected:
            return (0.0, 0.0, 0.0)
        
        sensor_data = self.bridge.get_sensor_data()
        velocity = sensor_data.get('velocity', (0.0, 0.0, 0.0))
        return tuple(velocity)
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Send velocity command to Webots simulation.
        
        Args:
            vx: Velocity in x direction (m/s)
            vy: Velocity in y direction (m/s)  
            vz: Velocity in z direction (m/s)
            yaw_rate: Yaw rate (deg/s) - will be converted to rad/s
        """
        if not self._is_connected or not self.bridge.connected:
            return
        
        # Block commands if crashed
        if self.crashed or self.safety.state == SafetyState.EMERGENCY:
            return
        
        # Convert yaw_rate from deg/s to rad/s for simulation
        yaw_rate_rad = np.deg2rad(yaw_rate)
        
        self.bridge.send_velocity_command(vx, vy, vz, yaw_rate_rad)
        logger.debug(f"[WEBOTS→CTRL] vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} yaw={yaw_rate:+.2f}")
    
    def takeoff(self, height: float = 0.5) -> bool:
        """Takeoff to specified height.
        
        In simulation, this initiates a vertical climb using velocity commands.
        The actual altitude control is handled by the control loop.
        
        Args:
            height: Target height in meters
            
        Returns:
            True if takeoff initiated
        """
        if not self._is_connected:
            return False
        
        self._target_altitude = height
        self._is_flying = True
        self.safety.arm()
        self.safety.set_flying()
        
        # Get current position for safety geofence
        pos = self.get_position()
        self.safety.set_start_position(Position(x=pos[0], y=pos[1], z=pos[2]))
        
        logger.info(f"[WEBOTS] Takeoff initiated → target altitude: {height}m")
        return True
    
    def land(self) -> bool:
        """Land the drone.
        
        In simulation, this initiates a descent using velocity commands.
        
        Returns:
            True if landing initiated
        """
        if not self._is_connected:
            return False
        
        self._is_flying = False
        self.safety.set_landing()
        
        logger.info("[WEBOTS] Landing initiated")
        return True
    
    def emergency_stop(self) -> None:
        """Emergency stop - send zero velocities."""
        if not self._is_connected:
            return
        
        self.bridge.send_velocity_command(0.0, 0.0, 0.0, 0.0)
        self._is_flying = False
        self.crashed = True
        logger.warning("[WEBOTS] ⚠ EMERGENCY STOP - All motors stopped")
    
    def check_crash(self) -> bool:
        """Check if drone has crashed using the safety monitor.
        
        Returns:
            True if crash detected
        """
        if not self._is_connected:
            return False
        
        sensor_data = self.get_sensor_data()
        if not sensor_data:
            return False
        
        # Convert to SensorData for safety monitor
        sd = SensorData.from_dict(sensor_data)
        
        # Run safety check
        state = self.safety.check(sd)
        
        if state == SafetyState.EMERGENCY:
            if not self.crashed:
                self.crashed = True
                self.emergency_stop()
            return True
        
        # Update legacy flags for compatibility
        self.takeoff_complete = self.safety.takeoff_complete
        self.crash_detection_enabled = self.safety.crash_detection_enabled
        
        return False
    
    def enable_crash_detection(self) -> None:
        """Manually enable crash detection (e.g., after confirming takeoff)."""
        self.crash_detection_enabled = True
        self.takeoff_complete = True
        self.safety.takeoff_complete = True
        self.safety.crash_detection_enabled = True
        logger.info("[SAFETY] Crash detection manually enabled")
    
    def reset_crash_state(self) -> None:
        """Reset crash detection state."""
        self.crashed = False
        self.safety.reset()
        logger.info("[SAFETY] Crash state reset - Ready for flight")
    
    def _on_emergency(self, trigger, message: str) -> None:
        """Handle emergency callback from safety monitor."""
        self.emergency_stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
