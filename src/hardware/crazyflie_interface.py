"""Crazyflie hardware interface using cflib.

This module provides the CrazyflieHardwareInterface class that implements
the DroneInterface ABC for real Crazyflie drone hardware.
"""

import time
import threading
from typing import Optional, Dict, Tuple
from loguru import logger
import numpy as np

from ..core.base_interface import DroneInterface
from ..core.types import SensorData, Position
from ..core.safety import SafetyMonitor, SafetyState, SafetyTrigger, EmergencyResponse

# Try to import cflib
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.positioning.motion_commander import MotionCommander
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    logger.warning("cflib not installed. Hardware interface will not be available.")


class CrazyflieHardwareInterface(DroneInterface):
    """Interface for controlling real Crazyflie 2.1 drone hardware.
    
    Implements the DroneInterface ABC for hardware deployments.
    Uses cflib and MotionCommander for flight control.
    """
    
    def __init__(self, uri: str = 'radio://0/80/2M/E7E7E7E7E7', 
                 config: Optional[dict] = None):
        """Initialize hardware interface.
        
        Args:
            uri: Crazyflie radio URI
            config: Configuration dictionary from hardware.yaml
        """
        if not CFLIB_AVAILABLE:
            raise RuntimeError("cflib is required for hardware interface. "
                             "Install with: pip install cflib")
        
        self.uri = uri
        self._is_connected = False
        
        # cflib objects
        self.scf: Optional[SyncCrazyflie] = None
        self.mc: Optional[MotionCommander] = None
        
        # Load config
        drone_config = config.get('drone', {}) if config else {}
        self._drone_config = drone_config  # Store for sensor logger
        nav_config = drone_config.get('navigation', {})
        hw_config = drone_config.get('hardware', {})
        
        # Hardware requirements
        self.flow_deck_required = hw_config.get('flow_deck_required', True)
        self.ai_deck_required = hw_config.get('ai_deck_required', True)
        self.arming_required = hw_config.get('arming_required', True)
        
        # Navigation parameters
        self.default_altitude = nav_config.get('default_altitude', 0.5)
        self.max_speed = nav_config.get('max_speed', 0.4)
        self.forward_only = nav_config.get('forward_only', True)
        
        # Safety monitor
        self.safety = SafetyMonitor(config)
        self.safety.set_emergency_callback(self._on_emergency)
        
        # Sensor logger (will be initialized on connect)
        self.sensor_logger: Optional['SensorLogger'] = None
        
        # Initialize drivers
        cflib.crtp.init_drivers()
        
        logger.info(f"[HARDWARE] Interface initialized (URI: {uri})")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to drone."""
        return self._is_connected
    
    def connect(self) -> bool:
        """Connect to Crazyflie drone.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"[HARDWARE] Connecting to {self.uri}...")
            
            self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache'))
            self.scf.open_link()
            
            # Verify required decks
            if not self._check_decks():
                self.scf.close_link()
                return False
            
            # Initialize sensor logging with drone config
            from .sensor_logger import SensorLogger
            self.sensor_logger = SensorLogger(self.scf, self._drone_config)
            self.sensor_logger.setup_logging()
            
            # Wait for initial sensor data
            if not self.sensor_logger.wait_for_data(timeout=2.0):
                logger.warning("[HARDWARE] Timeout waiting for sensor data")
            
            self._is_connected = True
            self.safety.set_ready()
            
            logger.info(f"[HARDWARE] ✓ Connected to Crazyflie at {self.uri}")
            return True
            
        except Exception as e:
            logger.error(f"[HARDWARE] ✗ Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Crazyflie."""
        if self.mc:
            try:
                self.mc.land()
            except Exception:
                pass
            self.mc = None
        
        # Stop sensor logging
        if self.sensor_logger:
            try:
                self.sensor_logger.stop_logging()
            except Exception:
                pass
            self.sensor_logger = None
        
        if self.scf:
            self.scf.close_link()
            self.scf = None
        
        self._is_connected = False
        logger.info("[HARDWARE] Disconnected from Crazyflie")
    
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings from drone.
        
        Returns:
            Dictionary containing sensor data
        """
        if not self._is_connected or not self.sensor_logger:
            return {}
        
        return self.sensor_logger.get_sensor_data()
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position from state estimate.
        
        Returns:
            Tuple of (x, y, z) in meters
        """
        if not self._is_connected or not self.sensor_logger:
            return (0.0, 0.0, 0.0)
        
        data = self.sensor_logger.get_sensor_data()
        return data.get('position', (0.0, 0.0, 0.0))
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation from stabilizer.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        if not self._is_connected or not self.sensor_logger:
            return (0.0, 0.0, 0.0)
        
        data = self.sensor_logger.get_sensor_data()
        return (
            data.get('roll', 0.0),
            data.get('pitch', 0.0),
            data.get('yaw', 0.0)
        )
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity from state estimate.
        
        Returns:
            Tuple of (vx, vy, vz) in m/s
        """
        if not self._is_connected or not self.sensor_logger:
            return (0.0, 0.0, 0.0)
        
        data = self.sensor_logger.get_sensor_data()
        return data.get('velocity', (0.0, 0.0, 0.0))
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Send velocity command to drone.
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s) - zeroed in forward-only mode
            vz: Vertical velocity (m/s)
            yaw_rate: Yaw rate (deg/s)
        """
        if not self._is_connected or not self.mc:
            return
        
        # Block if in emergency state
        if self.safety.state == SafetyState.EMERGENCY:
            return
        
        # Forward-only mode: zero lateral velocity
        if self.forward_only:
            vy = 0.0
        
        # Clamp velocities
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
            vz *= scale
        
        try:
            self.mc.start_linear_motion(vx, vy, vz, yaw_rate)
        except Exception as e:
            logger.error(f"[HARDWARE] Velocity command failed: {e}")
    
    def takeoff(self, height: float = 0.5) -> bool:
        """Takeoff to specified height using MotionCommander.
        
        Args:
            height: Target height in meters
            
        Returns:
            True if takeoff successful
        """
        if not self._is_connected or not self.scf:
            return False
        
        try:
            self.safety.arm()
            
            # Get start position for geofence
            time.sleep(0.5)  # Wait for sensor data
            pos = self.get_position()
            self.safety.set_start_position(Position(x=pos[0], y=pos[1], z=pos[2]))
            
            # Create MotionCommander and takeoff
            self.mc = MotionCommander(self.scf, default_height=height)
            self.mc.take_off()
            
            self.safety.set_flying()
            logger.info(f"[HARDWARE] ✓ Takeoff complete → altitude: {height}m")
            return True
            
        except Exception as e:
            logger.error(f"[HARDWARE] ✗ Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        """Land the drone using MotionCommander.
        
        Returns:
            True if landing successful
        """
        if not self._is_connected or not self.mc:
            return False
        
        try:
            self.safety.set_landing()
            self.mc.land()
            self.mc = None
            self.safety.set_landed()
            logger.info("[HARDWARE] ✓ Landing complete")
            return True
            
        except Exception as e:
            logger.error(f"[HARDWARE] ✗ Landing failed: {e}")
            return False
    
    def emergency_stop(self) -> None:
        """Emergency stop - cut motors immediately."""
        if not self._is_connected or not self.scf:
            return
        
        try:
            self.scf.cf.commander.send_stop_setpoint()
            self.mc = None
        except Exception:
            pass
        
        logger.warning("[HARDWARE] ⚠ EMERGENCY STOP - Motors cut")
    
    def emergency_land(self) -> None:
        """Emergency land - controlled descent to ground.
        
        Used for battery low or communication loss situations where
        a controlled landing is still possible.
        """
        if not self._is_connected:
            return
        
        logger.warning("[HARDWARE] ⚠ EMERGENCY LAND - Initiating controlled descent")
        
        # Use MotionCommander land if available
        if self.mc:
            try:
                self.mc.land()
                self.mc = None
                logger.info("[HARDWARE] Emergency landing complete")
            except Exception as e:
                logger.error(f"[HARDWARE] Emergency land failed, cutting motors: {e}")
                self.emergency_stop()
        else:
            # Fallback: cut motors
            self.emergency_stop()
    
    def emergency_hover(self) -> None:
        """Emergency hover - stop and hold position.
        
        Used for geofence breaches where we want to stop movement
        but maintain altitude.
        """
        if not self._is_connected:
            return
        
        logger.warning("[HARDWARE] ⚠ EMERGENCY HOVER - Stopping movement")
        
        if self.mc:
            try:
                # Stop linear motion (hover in place)
                self.mc.start_linear_motion(0.0, 0.0, 0.0, 0.0)
            except Exception as e:
                logger.error(f"[HARDWARE] Emergency hover failed: {e}")
                self.emergency_stop()
    
    def _check_decks(self) -> bool:
        """Check for required decks.
        
        Returns:
            True if all required decks present
        """
        if not self.scf:
            return False
        
        # Read deck parameters
        cf = self.scf.cf
        
        # TODO: Implement proper deck detection via parameters
        # For now, log warning and assume present
        logger.warning("[HARDWARE] Deck detection not fully implemented")
        
        if self.flow_deck_required:
            logger.info("[HARDWARE] Flow deck required - assuming present")
        
        if self.ai_deck_required:
            logger.info("[HARDWARE] AI deck required - assuming present")
        
        return True
    
    def _on_emergency(self, trigger: SafetyTrigger, message: str) -> None:
        """Handle emergency callback from safety monitor.
        
        Selects appropriate response based on the trigger type:
        - LOW_BATTERY, COMMUNICATION_LOSS: Controlled landing
        - GEOFENCE_BREACH: Stop and hover
        - EXCESSIVE_TILT, LOW_ALTITUDE, MANUAL_STOP: Motor cutoff
        """
        response = self.safety.recommended_response
        
        if response == EmergencyResponse.LAND:
            self.emergency_land()
        elif response == EmergencyResponse.HOVER:
            self.emergency_hover()
        else:  # CUTOFF (default)
            self.emergency_stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
