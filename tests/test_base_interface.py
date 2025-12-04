"""Tests for DroneInterface abstract base class.

Tests the ABC contract and ensures implementations conform
to the interface requirements.
"""

import pytest
from abc import ABC
from unittest.mock import Mock
import numpy as np

from src.core.base_interface import DroneInterface


class TestDroneInterfaceABC:
    """Tests for DroneInterface abstract base class."""
    
    def test_is_abstract_class(self):
        """Test that DroneInterface is an abstract class."""
        assert issubclass(DroneInterface, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that DroneInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DroneInterface()
    
    def test_required_methods(self):
        """Test that all required abstract methods are defined."""
        required_methods = [
            'is_connected',
            'connect',
            'disconnect',
            'get_sensor_data',
            'get_position',
            'get_orientation',
            'get_velocity',
            'send_velocity_command',
            'takeoff',
            'land',
            'emergency_stop',
        ]
        
        for method in required_methods:
            assert hasattr(DroneInterface, method), f"Missing method: {method}"


class ConcreteInterface(DroneInterface):
    """Concrete implementation for testing."""
    
    def __init__(self):
        self._connected = False
        self._position = (0.0, 0.0, 0.0)
        self._orientation = (0.0, 0.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        self._connected = False
    
    def get_sensor_data(self):
        return {
            'position': self._position,
            'orientation': self._orientation,
            'velocity': self._velocity,
            'altitude': self._position[2],
            'battery': 4.0,
            'timestamp': 0.0,
        }
    
    def get_position(self):
        return self._position
    
    def get_orientation(self):
        return self._orientation
    
    def get_velocity(self):
        return self._velocity
    
    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        self._velocity = (vx, vy, vz)
    
    def takeoff(self, height=0.5):
        self._position = (0.0, 0.0, height)
        return True
    
    def land(self):
        self._position = (0.0, 0.0, 0.0)
        return True
    
    def emergency_stop(self):
        self._velocity = (0.0, 0.0, 0.0)


class TestConcreteImplementation:
    """Tests for a concrete DroneInterface implementation."""
    
    def test_can_instantiate_concrete(self):
        """Test that concrete implementation can be instantiated."""
        interface = ConcreteInterface()
        assert interface is not None
    
    def test_is_connected_property(self):
        """Test is_connected property."""
        interface = ConcreteInterface()
        assert not interface.is_connected
        
        interface.connect()
        assert interface.is_connected
        
        interface.disconnect()
        assert not interface.is_connected
    
    def test_connect_returns_bool(self):
        """Test connect returns boolean."""
        interface = ConcreteInterface()
        result = interface.connect()
        assert isinstance(result, bool)
        assert result is True
    
    def test_get_position_returns_tuple(self):
        """Test get_position returns 3-tuple."""
        interface = ConcreteInterface()
        pos = interface.get_position()
        
        assert isinstance(pos, tuple)
        assert len(pos) == 3
        assert all(isinstance(v, float) for v in pos)
    
    def test_get_orientation_returns_tuple(self):
        """Test get_orientation returns 3-tuple."""
        interface = ConcreteInterface()
        orient = interface.get_orientation()
        
        assert isinstance(orient, tuple)
        assert len(orient) == 3
        assert all(isinstance(v, float) for v in orient)
    
    def test_get_velocity_returns_tuple(self):
        """Test get_velocity returns 3-tuple."""
        interface = ConcreteInterface()
        vel = interface.get_velocity()
        
        assert isinstance(vel, tuple)
        assert len(vel) == 3
        assert all(isinstance(v, float) for v in vel)
    
    def test_get_sensor_data_returns_dict(self):
        """Test get_sensor_data returns dictionary with expected keys."""
        interface = ConcreteInterface()
        data = interface.get_sensor_data()
        
        assert isinstance(data, dict)
        assert 'position' in data
        assert 'altitude' in data
    
    def test_takeoff_changes_altitude(self):
        """Test takeoff changes altitude."""
        interface = ConcreteInterface()
        
        initial_pos = interface.get_position()
        assert initial_pos[2] == 0.0
        
        interface.takeoff(height=0.5)
        
        new_pos = interface.get_position()
        assert new_pos[2] == 0.5
    
    def test_land_returns_to_ground(self):
        """Test land returns drone to ground."""
        interface = ConcreteInterface()
        interface.takeoff(height=1.0)
        
        interface.land()
        
        pos = interface.get_position()
        assert pos[2] == 0.0
    
    def test_emergency_stop_zeros_velocity(self):
        """Test emergency_stop zeros velocity."""
        interface = ConcreteInterface()
        interface.send_velocity_command(1.0, 0.5, 0.0, 0.0)
        
        vel = interface.get_velocity()
        assert vel[0] == 1.0
        
        interface.emergency_stop()
        
        vel = interface.get_velocity()
        assert vel == (0.0, 0.0, 0.0)


class TestContextManager:
    """Tests for context manager protocol."""
    
    def test_context_manager_connects(self):
        """Test context manager calls connect on entry."""
        interface = ConcreteInterface()
        
        with interface:
            assert interface.is_connected
    
    def test_context_manager_disconnects(self):
        """Test context manager calls disconnect on exit."""
        interface = ConcreteInterface()
        
        with interface:
            pass
        
        assert not interface.is_connected
    
    def test_context_manager_disconnects_on_exception(self):
        """Test context manager disconnects even on exception."""
        interface = ConcreteInterface()
        
        try:
            with interface:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        assert not interface.is_connected
