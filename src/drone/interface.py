"""Crazyflie 2.1 drone interface."""

import time
from typing import Optional, Callable
from loguru import logger

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    logger.warning("cflib not installed. Drone interface will run in simulation mode.")


class CrazyflieInterface:
    """Interface for communicating with Crazyflie 2.1 drone."""
    
    def __init__(self, uri: str = 'radio://0/80/2M/E7E7E7E7E7'):
        """
        Initialize Crazyflie interface.
        
        Args:
            uri: Connection URI for the Crazyflie
        """
        self.uri = uri
        self.is_connected = False
        self.cf = None
        self.scf = None
        
        if CFLIB_AVAILABLE:
            cflib.crtp.init_drivers()
    
    def connect(self) -> bool:
        """
        Connect to the Crazyflie.
        
        Returns:
            True if connection successful
        """
        if not CFLIB_AVAILABLE:
            logger.warning("Running in simulation mode")
            self.is_connected = True
            return True
        
        try:
            self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache'))
            self.scf.open_link()
            self.cf = self.scf.cf
            self.is_connected = True
            logger.info(f"Connected to Crazyflie at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the Crazyflie."""
        if self.scf is not None:
            self.scf.close_link()
            self.is_connected = False
            logger.info("Disconnected from Crazyflie")
    
    def get_sensor_data(self) -> dict:
        """
        Get sensor data from the drone.
        
        Returns:
            Dictionary containing sensor readings
        """
        if not self.is_connected:
            logger.warning("Not connected to drone")
            return {}
        
        # Placeholder for actual sensor data retrieval
        # This would read from AI deck, Flow deck, Multi-ranger deck
        data = {
            'camera': None,
            'flow': None,
            'range': None,
            'battery': 0.0,
            'timestamp': time.time()
        }
        
        return data
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Send velocity command to the drone.
        
        Args:
            vx: Velocity in x direction (m/s)
            vy: Velocity in y direction (m/s)
            vz: Velocity in z direction (m/s)
            yaw_rate: Yaw rate (deg/s)
        """
        if not self.is_connected:
            logger.warning("Not connected to drone")
            return
        
        if CFLIB_AVAILABLE and self.cf is not None:
            self.cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_rate)
        else:
            logger.debug(f"Simulated velocity command: "
                        f"vx={vx}, vy={vy}, vz={vz}, yaw={yaw_rate}")
    
    def emergency_stop(self):
        """Emergency stop - immediately stop all motors."""
        if not self.is_connected:
            return
        
        if CFLIB_AVAILABLE and self.cf is not None:
            self.cf.commander.send_stop_setpoint()
        
        logger.warning("Emergency stop activated")
    
    def get_position(self) -> tuple:
        """
        Get current drone position.
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        if not self.is_connected:
            return (0.0, 0.0, 0.0)
        
        # Placeholder for position estimation
        # Would use Flow deck and/or external positioning
        return (0.0, 0.0, 0.0)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
