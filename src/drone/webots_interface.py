"""Webots simulation interface for drone control."""

import numpy as np
from typing import Optional, Dict
from loguru import logger

from .interface import CrazyflieInterface

# Handle both running as module (python -m src.main) and as script
try:
    from ..sim.bridge import SimulationBridge
except ImportError:
    from src.sim.bridge import SimulationBridge


class WebotsInterface(CrazyflieInterface):
    """Interface for controlling Crazyflie in Webots simulation."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 10020,
                 uri: str = 'webots://simulation',
                 config: dict = None):
        """
        Initialize Webots interface.
        
        Args:
            host: Host address for simulation bridge
            port: Port number for simulation bridge
            uri: Identifier for the simulated drone
            config: Optional configuration dictionary
        """
        # Don't call parent __init__ to avoid cflib initialization
        self.uri = uri
        self.is_connected = False
        self.cf = None
        self.scf = None
        
        # Load config values if provided
        drone_config = config.get('drone', {}) if config else {}
        nav_config = drone_config.get('navigation', {})
        sim_config = drone_config.get('simulation', {})
        
        # Use config values for host/port, or fall back to params/defaults
        actual_host = sim_config.get('host', host)
        actual_port = sim_config.get('port', port)
        
        # Simulation bridge
        self.bridge = SimulationBridge(host=actual_host, port=actual_port)
        
        # Crash detection (from config)
        self.crashed = False
        self.crash_tilt_threshold = np.deg2rad(nav_config.get('crash_tilt_threshold', 45.0))
        self.crash_altitude_threshold = nav_config.get('crash_altitude_threshold', 0.1)
        self.crash_detection_enabled = False  # Disabled initially
        self.takeoff_complete = False
        self.min_flight_altitude = nav_config.get('min_flight_altitude', 0.3)
        
        logger.info("[WEBOTS] Interface initialized")
    
    def connect(self) -> bool:
        """
        Connect to Webots simulation.
        
        Returns:
            True if connection successful
        """
        if self.bridge.connect():
            self.is_connected = True
            logger.info(f"[WEBOTS] ✓ Connected to simulation at {self.bridge.host}:{self.bridge.port}")
            return True
        else:
            logger.error("[WEBOTS] ✗ Connection failed")
            return False
    
    def disconnect(self):
        """Disconnect from Webots simulation."""
        self.bridge.disconnect()
        self.is_connected = False
        logger.info("[WEBOTS] Disconnected from simulation")
    
    def get_sensor_data(self) -> Dict:
        """
        Get sensor data from Webots simulation.
        
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
        if not self.is_connected:
            logger.warning("Not connected to simulation")
            return {}
        
        return self.bridge.get_sensor_data()
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Send velocity command to Webots simulation.
        
        Args:
            vx: Velocity in x direction (m/s)
            vy: Velocity in y direction (m/s)  
            vz: Velocity in z direction (m/s)
            yaw_rate: Yaw rate (deg/s) - will be converted to rad/s
        """
        if not self.is_connected or not self.bridge.connected:
            # Silently skip if not connected (avoids spam)
            return
        
        # Convert yaw_rate from deg/s to rad/s for simulation
        yaw_rate_rad = np.deg2rad(yaw_rate)
        
        self.bridge.send_velocity_command(vx, vy, vz, yaw_rate_rad)
        logger.debug(f"[WEBOTS→CTRL] vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} yaw={yaw_rate:+.2f}")
    
    def emergency_stop(self):
        """Emergency stop - send zero velocities."""
        if not self.is_connected:
            return
        
        self.bridge.send_velocity_command(0.0, 0.0, 0.0, 0.0)
        logger.warning("[WEBOTS] ⚠ EMERGENCY STOP - All motors stopped")
    
    def get_position(self) -> tuple:
        """
        Get current drone position from simulation.
        
        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        if not self.is_connected:
            return (0.0, 0.0, 0.0)
        
        return self.bridge.get_position()
    
    def get_orientation(self) -> tuple:
        """
        Get current drone orientation from simulation.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        if not self.is_connected:
            return (0.0, 0.0, 0.0)
        
        sensor_data = self.bridge.get_sensor_data()
        return (
            sensor_data.get('roll', 0.0),
            sensor_data.get('pitch', 0.0),
            sensor_data.get('yaw', 0.0)
        )
    
    def get_velocity(self) -> tuple:
        """
        Get current drone velocity from simulation.
        
        Returns:
            Tuple of (vx, vy, vz) in m/s
        """
        if not self.is_connected:
            return (0.0, 0.0, 0.0)
        
        sensor_data = self.bridge.get_sensor_data()
        velocity = sensor_data.get('velocity', (0.0, 0.0, 0.0))
        return tuple(velocity)
    
    def check_crash(self) -> bool:
        """
        Check if drone has crashed and trigger emergency stop.
        
        Returns:
            True if crash detected
        """
        if not self.is_connected:
            return False
        
        sensor_data = self.get_sensor_data()
        if not sensor_data:
            return False
        
        roll = sensor_data.get('roll', 0.0)
        pitch = sensor_data.get('pitch', 0.0)
        altitude = sensor_data.get('altitude', sensor_data.get('position', [0, 0, 0])[2])
        velocity = sensor_data.get('velocity', (0.0, 0.0, 0.0))
        
        # Enable crash detection only after takeoff
        if not self.takeoff_complete:
            if altitude > self.min_flight_altitude:
                self.takeoff_complete = True
                self.crash_detection_enabled = True
                logger.info(f"[SAFETY] ✓ Takeoff complete @ {altitude:.2f}m - Crash detection ENABLED")
            return False  # Don't check during initial takeoff
        
        # Only check if enabled
        if not self.crash_detection_enabled:
            return False
        
        # Check extreme tilt
        if abs(roll) > self.crash_tilt_threshold or abs(pitch) > self.crash_tilt_threshold:
            if not self.crashed:
                logger.error(f"[SAFETY] ✗ CRASH: Extreme tilt (roll={np.rad2deg(roll):+.1f}° pitch={np.rad2deg(pitch):+.1f}°)")
                self.emergency_stop()
                self.crashed = True
            return True
        
        # Check altitude
        if altitude < self.crash_altitude_threshold:
            if not self.crashed:
                logger.error(f"[SAFETY] ✗ CRASH: Low altitude ({altitude:.3f}m < {self.crash_altitude_threshold:.3f}m)")
                self.emergency_stop()
                self.crashed = True
            return True
        
        # Check if stopped at low altitude
        speed = np.linalg.norm(velocity)
        if speed < 0.01 and altitude < 0.5:
            if not self.crashed:
                logger.warning(f"[SAFETY] ⚠ Possible crash: Stopped @ {altitude:.3f}m")
                self.emergency_stop()
                self.crashed = True
            return True
        
        return False
    
    def enable_crash_detection(self):
        """Manually enable crash detection (e.g., after confirming takeoff)."""
        self.crash_detection_enabled = True
        self.takeoff_complete = True
        logger.info("[SAFETY] Crash detection manually enabled")
    
    def reset_crash_state(self):
        """Reset crash detection state."""
        self.crashed = False
        logger.info("[SAFETY] Crash state reset - Ready for flight")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
