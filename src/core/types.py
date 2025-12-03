"""Shared data types for drone interfaces.

This module defines common data structures used across simulation
and hardware interfaces.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional, List
import numpy as np
import time


class SafetyState(Enum):
    """States for the safety state machine."""
    INITIALIZING = auto()
    READY = auto()
    ARMED = auto()
    FLYING = auto()
    LANDING = auto()
    LANDED = auto()
    EMERGENCY = auto()


@dataclass
class Position:
    """3D position in meters."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)
    
    def as_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Position':
        """Create from tuple."""
        return cls(x=t[0], y=t[1], z=t[2])
    
    def distance_to(self, other: 'Position') -> float:
        """Euclidean distance to another position."""
        return np.linalg.norm(self.as_array() - other.as_array())


@dataclass
class Orientation:
    """Orientation in radians (roll, pitch, yaw)."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.roll, self.pitch, self.yaw)
    
    def as_degrees(self) -> Tuple[float, float, float]:
        """Convert to degrees."""
        return (
            np.rad2deg(self.roll),
            np.rad2deg(self.pitch),
            np.rad2deg(self.yaw)
        )
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Orientation':
        """Create from tuple (radians)."""
        return cls(roll=t[0], pitch=t[1], yaw=t[2])


@dataclass
class Velocity:
    """3D velocity in m/s."""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.vx, self.vy, self.vz)
    
    def as_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.vx, self.vy, self.vz])
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Velocity':
        """Create from tuple."""
        return cls(vx=t[0], vy=t[1], vz=t[2])
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity vector."""
        return np.linalg.norm([self.vx, self.vy, self.vz])
    
    @property
    def horizontal_speed(self) -> float:
        """Magnitude of horizontal velocity."""
        return np.linalg.norm([self.vx, self.vy])


@dataclass
class SensorData:
    """Container for sensor readings from drone."""
    # Camera
    camera: Optional[np.ndarray] = None
    
    # Position and orientation
    position: Position = field(default_factory=Position)
    orientation: Orientation = field(default_factory=Orientation)
    velocity: Velocity = field(default_factory=Velocity)
    
    # Altitude (may differ from position.z due to sensor source)
    altitude: float = 0.0
    
    # Range sensors (meters, None if not available)
    range_front: Optional[float] = None
    range_back: Optional[float] = None
    range_left: Optional[float] = None
    range_right: Optional[float] = None
    range_down: Optional[float] = None
    
    # Battery (voltage)
    battery: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for compatibility."""
        return {
            'camera': self.camera,
            'position': self.position.as_tuple(),
            'roll': self.orientation.roll,
            'pitch': self.orientation.pitch,
            'yaw': self.orientation.yaw,
            'velocity': self.velocity.as_tuple(),
            'altitude': self.altitude,
            'range_front': self.range_front,
            'range_back': self.range_back,
            'range_left': self.range_left,
            'range_right': self.range_right,
            'range_down': self.range_down,
            'battery': self.battery,
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SensorData':
        """Create from dictionary format."""
        position = data.get('position', (0, 0, 0))
        velocity = data.get('velocity', (0, 0, 0))
        
        return cls(
            camera=data.get('camera'),
            position=Position.from_tuple(position) if isinstance(position, tuple) else Position(),
            orientation=Orientation(
                roll=data.get('roll', 0.0),
                pitch=data.get('pitch', 0.0),
                yaw=data.get('yaw', 0.0),
            ),
            velocity=Velocity.from_tuple(velocity) if isinstance(velocity, tuple) else Velocity(),
            altitude=data.get('altitude', position[2] if isinstance(position, tuple) else 0.0),
            range_front=data.get('range_front'),
            range_back=data.get('range_back'),
            range_left=data.get('range_left'),
            range_right=data.get('range_right'),
            range_down=data.get('range_down', data.get('zrange')),
            battery=data.get('battery', 0.0),
            timestamp=data.get('timestamp', time.time()),
        )


@dataclass
class VelocityCommand:
    """Velocity command structure."""
    vx: float = 0.0  # Forward velocity (m/s)
    vy: float = 0.0  # Lateral velocity (m/s, left positive)
    vz: float = 0.0  # Vertical velocity (m/s, up positive)
    yaw_rate: float = 0.0  # Yaw rate (deg/s)
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple."""
        return (self.vx, self.vy, self.vz, self.yaw_rate)
    
    def clamp(self, max_speed: float, max_yaw_rate: float) -> 'VelocityCommand':
        """Return clamped velocity command."""
        # Clamp linear velocities
        speed = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        if speed > max_speed:
            scale = max_speed / speed
            vx = self.vx * scale
            vy = self.vy * scale
            vz = self.vz * scale
        else:
            vx, vy, vz = self.vx, self.vy, self.vz
        
        # Clamp yaw rate
        yaw_rate = np.clip(self.yaw_rate, -max_yaw_rate, max_yaw_rate)
        
        return VelocityCommand(vx, vy, vz, yaw_rate)


@dataclass
class Waypoint:
    """Navigation waypoint."""
    position: Position
    heading: Optional[float] = None  # Desired yaw in radians
    speed: Optional[float] = None  # Desired approach speed
    hover_time: float = 0.0  # Time to hover at waypoint (seconds)
    
    @classmethod
    def from_tuple(cls, pos: Tuple[float, float, float], 
                   heading: Optional[float] = None) -> 'Waypoint':
        """Create waypoint from position tuple."""
        return cls(position=Position.from_tuple(pos), heading=heading)
