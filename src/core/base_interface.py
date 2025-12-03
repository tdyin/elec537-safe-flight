"""Abstract base class for drone interfaces.

This module defines the DroneInterface ABC that all drone interfaces
(simulation and hardware) must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class DroneInterface(ABC):
    """Abstract base class for drone interfaces (simulation and hardware).
    
    All drone interfaces must implement these methods to ensure consistent
    behavior across simulation and hardware deployments.
    """
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if drone is connected.
        
        Returns:
            True if connected to drone/simulation
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the drone.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the drone."""
        pass
    
    @abstractmethod
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings.
        
        Returns:
            Dictionary containing sensor data with keys:
            - camera: numpy array of camera image (H, W, C) or None
            - roll, pitch, yaw: orientation in radians
            - altitude: height in meters
            - velocity: (vx, vy, vz) in m/s
            - position: (x, y, z) in meters
            - battery: voltage (for hardware)
            - timestamp: reading timestamp
        """
        pass
    
    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z) in meters.
        
        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        pass
    
    @abstractmethod
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation (roll, pitch, yaw) in radians.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        pass
    
    @abstractmethod
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (vx, vy, vz) in m/s.
        
        Returns:
            Tuple of (vx, vy, vz) in m/s
        """
        pass
    
    @abstractmethod
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Send velocity command in body frame.
        
        Args:
            vx: Forward velocity in m/s
            vy: Lateral velocity in m/s (left positive)
            vz: Vertical velocity in m/s (up positive)
            yaw_rate: Yaw rate in deg/s
        """
        pass
    
    @abstractmethod
    def takeoff(self, height: float = 0.5) -> bool:
        """Takeoff to specified height.
        
        Args:
            height: Target height in meters
            
        Returns:
            True if takeoff successful
        """
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """Land the drone.
        
        Returns:
            True if landing successful
        """
        pass
    
    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop - cut motors immediately."""
        pass
    
    def __enter__(self):
        """Context manager entry - connect to drone."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from drone."""
        self.disconnect()
