"""Tests for CrazyflieHardwareInterface.

Tests the hardware interface implementation with mocked cflib.
Uses mock fixtures to test without real hardware.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# Mock cflib modules before importing
mock_cflib = MagicMock()
mock_crtp = MagicMock()
mock_crazyflie_module = MagicMock()
mock_syncCrazyflie = MagicMock()
mock_log = MagicMock()
mock_positioning = MagicMock()

sys.modules['cflib'] = mock_cflib
sys.modules['cflib.crtp'] = mock_crtp
sys.modules['cflib.crazyflie'] = mock_crazyflie_module
sys.modules['cflib.crazyflie.syncCrazyflie'] = mock_syncCrazyflie
sys.modules['cflib.crazyflie.log'] = mock_log
sys.modules['cflib.positioning'] = mock_positioning
sys.modules['cflib.positioning.motion_commander'] = mock_positioning

# Now import the module
from src.hardware.crazyflie_interface import CrazyflieHardwareInterface, CFLIB_AVAILABLE
from src.core.types import SafetyState


@pytest.fixture
def hardware_config():
    """Hardware configuration for testing."""
    return {
        'drone': {
            'uri': 'radio://0/80/2M/E7E7E7E7E7',
            'hardware': {
                'flow_deck_required': True,
                'ai_deck_required': True,
                'arming_required': True,
            },
            'logging': {
                'state_estimate_rate_ms': 20,
                'stabilizer_rate_ms': 20,
                'battery_rate_ms': 500,
            },
            'navigation': {
                'cruise_speed': 0.3,
                'avoidance_speed': 0.15,
                'max_speed': 0.4,
                'default_altitude': 0.5,
                'forward_only': True,
            },
            'safety': {
                'battery_min_voltage': 3.3,
                'geofence_radius': 3.0,
                'crash_tilt_threshold': 40.0,
            },
        },
    }


class TestCrazyflieInterfaceInit:
    """Tests for CrazyflieHardwareInterface initialization."""
    
    def test_init_with_config(self, hardware_config):
        """Test initialization with configuration."""
        interface = CrazyflieHardwareInterface(
            uri='radio://0/80/2M/E7E7E7E7E7',
            config=hardware_config
        )
        
        assert interface.uri == 'radio://0/80/2M/E7E7E7E7E7'
        assert interface.max_speed == 0.4
        assert interface.default_altitude == 0.5
        assert interface.forward_only is True
    
    def test_init_without_config(self):
        """Test initialization with default values."""
        interface = CrazyflieHardwareInterface()
        
        assert interface.uri == 'radio://0/80/2M/E7E7E7E7E7'  # Default URI
        assert interface.max_speed == 0.4
        assert interface.default_altitude == 0.5
    
    def test_initial_state_not_connected(self, hardware_config):
        """Test initial state is not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        assert not interface.is_connected
        assert interface.scf is None
        assert interface.mc is None


class TestConnectionMethods:
    """Tests for connection and disconnection methods."""
    
    def test_disconnect_when_not_connected(self, hardware_config):
        """Test disconnect when not connected does nothing."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        # Should not raise
        interface.disconnect()
        
        assert not interface.is_connected


class TestVelocityCommands:
    """Tests for velocity command handling."""
    
    def test_velocity_command_not_connected(self, hardware_config):
        """Test velocity command when not connected is ignored."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        # Should not raise, just return
        interface.send_velocity_command(1.0, 0.5, 0.0, 0.0)
    
    def test_forward_only_mode_zeros_vy(self, hardware_config):
        """Test forward-only mode zeros lateral velocity."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        interface._is_connected = True
        interface.mc = Mock()
        interface.safety.state = SafetyState.FLYING
        
        interface.send_velocity_command(0.3, 0.2, 0.0, 0.0)
        
        # vy should be zeroed in forward-only mode
        interface.mc.start_linear_motion.assert_called_once()
        call_args = interface.mc.start_linear_motion.call_args[0]
        assert call_args[1] == 0.0  # vy = 0
    
    def test_velocity_clamped_to_max_speed(self, hardware_config):
        """Test velocity is clamped to max speed."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        interface._is_connected = True
        interface.mc = Mock()
        interface.safety.state = SafetyState.FLYING
        interface.forward_only = False  # Allow lateral for this test
        
        # Request velocity that exceeds max_speed (0.4)
        interface.send_velocity_command(0.5, 0.5, 0.5, 0.0)
        
        interface.mc.start_linear_motion.assert_called_once()
        call_args = interface.mc.start_linear_motion.call_args[0]
        
        # Check that speed is clamped
        import numpy as np
        speed = np.sqrt(call_args[0]**2 + call_args[1]**2 + call_args[2]**2)
        assert speed <= 0.4 + 0.01  # Allow small floating point error
    
    def test_velocity_blocked_in_emergency(self, hardware_config):
        """Test velocity commands blocked in emergency state."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        interface._is_connected = True
        interface.mc = Mock()
        interface.safety.state = SafetyState.EMERGENCY
        
        interface.send_velocity_command(0.3, 0.0, 0.0, 0.0)
        
        # Should not call motion commander
        interface.mc.start_linear_motion.assert_not_called()


class TestSensorData:
    """Tests for sensor data retrieval."""
    
    def test_get_sensor_data_not_connected(self, hardware_config):
        """Test get_sensor_data returns empty when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        data = interface.get_sensor_data()
        
        assert data == {}
    
    def test_get_position_not_connected(self, hardware_config):
        """Test get_position returns zeros when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        pos = interface.get_position()
        
        assert pos == (0.0, 0.0, 0.0)
    
    def test_get_orientation_not_connected(self, hardware_config):
        """Test get_orientation returns zeros when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        orient = interface.get_orientation()
        
        assert orient == (0.0, 0.0, 0.0)
    
    def test_get_velocity_not_connected(self, hardware_config):
        """Test get_velocity returns zeros when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        vel = interface.get_velocity()
        
        assert vel == (0.0, 0.0, 0.0)


class TestFlightMethods:
    """Tests for takeoff and landing methods."""
    
    def test_takeoff_not_connected(self, hardware_config):
        """Test takeoff returns False when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        result = interface.takeoff(height=0.5)
        
        assert result is False
    
    def test_land_not_connected(self, hardware_config):
        """Test land returns False when not connected."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        result = interface.land()
        
        assert result is False


class TestEmergencyStop:
    """Tests for emergency stop functionality."""
    
    def test_emergency_stop_not_connected(self, hardware_config):
        """Test emergency stop when not connected doesn't raise."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        # Should not raise
        interface.emergency_stop()


class TestContextManager:
    """Tests for context manager protocol."""
    
    def test_context_manager_entry_exit(self, hardware_config):
        """Test context manager calls connect/disconnect."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        # Mock connect to succeed but not actually do anything
        with patch.object(interface, 'connect', return_value=False):
            with patch.object(interface, 'disconnect') as mock_disconnect:
                with interface:
                    pass
                
                mock_disconnect.assert_called_once()


class TestHardwareRequirements:
    """Tests for hardware requirement settings."""
    
    def test_flow_deck_required_setting(self, hardware_config):
        """Test flow deck required is read from config."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        assert interface.flow_deck_required is True
    
    def test_ai_deck_required_setting(self, hardware_config):
        """Test AI deck required is read from config."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        assert interface.ai_deck_required is True
    
    def test_arming_required_setting(self, hardware_config):
        """Test arming required is read from config."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        assert interface.arming_required is True


class TestSafetyIntegration:
    """Tests for safety monitor integration."""
    
    def test_safety_monitor_initialized(self, hardware_config):
        """Test safety monitor is initialized."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        assert interface.safety is not None
    
    def test_emergency_callback_registered(self, hardware_config):
        """Test emergency callback is registered with safety monitor."""
        interface = CrazyflieHardwareInterface(config=hardware_config)
        
        # The _on_emergency callback should be set
        assert interface.safety._on_emergency is not None
