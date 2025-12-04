"""Tests for SafetyMonitor state machine.

Tests the safety monitoring logic including state transitions,
safety threshold checks, and emergency handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.core.safety import SafetyMonitor, SafetyTrigger
from src.core.types import SafetyState, SensorData, Position, Orientation, Velocity


@pytest.fixture
def default_safety_config():
    """Default safety configuration."""
    return {
        'drone': {
            'safety': {
                'battery_min_voltage': 3.3,
                'battery_warning_voltage': 3.5,
                'geofence_radius': 3.0,
                'geofence_height': 2.0,
                'crash_tilt_threshold': 40.0,
                'crash_altitude_threshold': 0.15,
                'min_flight_altitude': 0.3,
                'comm_timeout': 1.0,
            }
        }
    }


@pytest.fixture
def safety_monitor(default_safety_config):
    """Create a SafetyMonitor instance."""
    return SafetyMonitor(default_safety_config)


@pytest.fixture
def normal_sensor_data():
    """Create normal (safe) sensor data."""
    return SensorData(
        position=Position(0.0, 0.0, 0.5),
        orientation=Orientation(0.0, 0.0, 0.0),
        velocity=Velocity(0.2, 0.0, 0.0),
        altitude=0.5,
        battery=3.9,
        timestamp=100.0,
    )


class TestSafetyMonitorInit:
    """Tests for SafetyMonitor initialization."""
    
    def test_init_with_config(self, default_safety_config):
        """Test initialization with configuration."""
        monitor = SafetyMonitor(default_safety_config)
        
        assert monitor.battery_min == 3.3
        assert monitor.geofence_radius == 3.0
        assert np.rad2deg(monitor.tilt_threshold) == pytest.approx(40.0)
    
    def test_init_without_config(self):
        """Test initialization with default values."""
        monitor = SafetyMonitor(None)
        
        # Should use defaults
        assert monitor.battery_min == 3.3
        assert monitor.geofence_radius == 3.0
    
    def test_initial_state(self, safety_monitor):
        """Test initial state is INITIALIZING."""
        assert safety_monitor.state == SafetyState.INITIALIZING


class TestStateTransitions:
    """Tests for safety state machine transitions."""
    
    def test_set_ready_from_init(self, safety_monitor):
        """Test transition from INITIALIZING to READY."""
        assert safety_monitor.state == SafetyState.INITIALIZING
        
        safety_monitor.set_ready()
        
        assert safety_monitor.state == SafetyState.READY
    
    def test_arm_from_ready(self, safety_monitor):
        """Test transition from READY to ARMED."""
        safety_monitor.set_ready()
        
        result = safety_monitor.arm()
        
        assert result is True
        assert safety_monitor.state == SafetyState.ARMED
    
    def test_arm_from_wrong_state(self, safety_monitor):
        """Test arming fails from wrong state."""
        # Still in INITIALIZING
        result = safety_monitor.arm()
        
        assert result is False
        assert safety_monitor.state == SafetyState.INITIALIZING
    
    def test_set_flying_from_armed(self, safety_monitor):
        """Test transition from ARMED to FLYING."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        
        safety_monitor.set_flying()
        
        assert safety_monitor.state == SafetyState.FLYING
    
    def test_set_landing_from_flying(self, safety_monitor):
        """Test transition from FLYING to LANDING."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        
        safety_monitor.set_landing()
        
        assert safety_monitor.state == SafetyState.LANDING
    
    def test_set_landed_from_landing(self, safety_monitor):
        """Test transition from LANDING to LANDED."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_landing()
        
        safety_monitor.set_landed()
        
        assert safety_monitor.state == SafetyState.LANDED
    
    def test_arm_from_landed(self, safety_monitor):
        """Test can arm again from LANDED state."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_landed()
        
        result = safety_monitor.arm()
        
        assert result is True
        assert safety_monitor.state == SafetyState.ARMED


class TestSafetyChecks:
    """Tests for safety condition checks."""
    
    def test_check_with_normal_data(self, safety_monitor, normal_sensor_data):
        """Test check passes with normal sensor data."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_start_position(Position(0, 0, 0))
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        state = safety_monitor.check(normal_sensor_data)
        
        assert state == SafetyState.FLYING
    
    def test_check_enables_crash_detection(self, safety_monitor, normal_sensor_data):
        """Test crash detection enables after reaching min altitude."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        
        assert not safety_monitor.crash_detection_enabled
        
        # First check at high altitude
        safety_monitor.check(normal_sensor_data)
        
        assert safety_monitor.takeoff_complete
        assert safety_monitor.crash_detection_enabled
    
    def test_low_battery_triggers_emergency(self, safety_monitor):
        """Test low battery triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        low_battery_data = SensorData(
            position=Position(0, 0, 0.5),
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=0.5,
            battery=3.2,  # Below 3.3V threshold
            timestamp=100.0,
        )
        
        state = safety_monitor.check(low_battery_data)
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.LOW_BATTERY
    
    def test_geofence_breach_triggers_emergency(self, safety_monitor):
        """Test geofence breach triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_start_position(Position(0, 0, 0))
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        far_position_data = SensorData(
            position=Position(4.0, 0, 0.5),  # > 3.0m geofence
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=0.5,
            battery=3.9,
            timestamp=100.0,
        )
        
        state = safety_monitor.check(far_position_data)
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.GEOFENCE_BREACH
    
    def test_height_breach_triggers_emergency(self, safety_monitor):
        """Test height limit breach triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_start_position(Position(0, 0, 0))
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        high_altitude_data = SensorData(
            position=Position(0, 0, 2.5),  # > 2.0m height limit
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=2.5,
            battery=3.9,
            timestamp=100.0,
        )
        
        state = safety_monitor.check(high_altitude_data)
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.GEOFENCE_BREACH
    
    def test_excessive_tilt_triggers_emergency(self, safety_monitor):
        """Test excessive tilt triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        tilted_data = SensorData(
            position=Position(0, 0, 0.5),
            orientation=Orientation(roll=np.deg2rad(50), pitch=0, yaw=0),  # > 40°
            velocity=Velocity(0, 0, 0),
            altitude=0.5,
            battery=3.9,
            timestamp=100.0,
        )
        
        state = safety_monitor.check(tilted_data)
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.EXCESSIVE_TILT
    
    def test_low_altitude_triggers_emergency(self, safety_monitor):
        """Test low altitude (crash) triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        crashed_data = SensorData(
            position=Position(0, 0, 0.1),
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=0.1,  # < 0.15m threshold
            battery=3.9,
            timestamp=100.0,
        )
        
        state = safety_monitor.check(crashed_data)
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.LOW_ALTITUDE


class TestCommunicationCheck:
    """Tests for communication loss detection."""
    
    def test_communication_timeout(self, safety_monitor, normal_sensor_data):
        """Test communication timeout triggers emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        # Set last data time
        safety_monitor.check(normal_sensor_data)  # timestamp = 100.0
        
        # Check with time far in the future
        state = safety_monitor.check_communication(current_time=102.0)  # 2s later
        
        assert state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.COMMUNICATION_LOSS


class TestCallbacks:
    """Tests for emergency and warning callbacks."""
    
    def test_emergency_callback_called(self, safety_monitor):
        """Test emergency callback is called on emergency."""
        callback = Mock()
        safety_monitor.set_emergency_callback(callback)
        
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        # Trigger emergency with low battery
        low_battery_data = SensorData(
            position=Position(0, 0, 0.5),
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=0.5,
            battery=3.0,
            timestamp=100.0,
        )
        safety_monitor.check(low_battery_data)
        
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == SafetyTrigger.LOW_BATTERY
    
    def test_warning_callback_for_low_battery(self, safety_monitor):
        """Test warning callback for low battery warning."""
        callback = Mock()
        safety_monitor.set_warning_callback(callback)
        
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        # Trigger warning with battery between warning and min
        warning_battery_data = SensorData(
            position=Position(0, 0, 0.5),
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0, 0, 0),
            altitude=0.5,
            battery=3.4,  # Between 3.3 (min) and 3.5 (warning)
            timestamp=100.0,
        )
        safety_monitor.check(warning_battery_data)
        
        callback.assert_called_once()


class TestManualControl:
    """Tests for manual safety controls."""
    
    def test_manual_emergency(self, safety_monitor):
        """Test manual emergency stop."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        
        safety_monitor.manual_emergency()
        
        assert safety_monitor.state == SafetyState.EMERGENCY
        assert safety_monitor.last_trigger == SafetyTrigger.MANUAL_STOP
    
    def test_reset_after_emergency(self, safety_monitor):
        """Test reset after emergency."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.manual_emergency()
        
        assert safety_monitor.state == SafetyState.EMERGENCY
        
        safety_monitor.reset()
        
        assert safety_monitor.state == SafetyState.READY
        assert safety_monitor.last_trigger == SafetyTrigger.NONE


class TestProperties:
    """Tests for safety monitor properties."""
    
    def test_is_safe_property(self, safety_monitor):
        """Test is_safe property."""
        # INITIALIZING is not safe
        assert not safety_monitor.is_safe
        
        safety_monitor.set_ready()
        assert safety_monitor.is_safe
        
        safety_monitor.arm()
        assert safety_monitor.is_safe
        
        safety_monitor.manual_emergency()
        assert not safety_monitor.is_safe
    
    def test_can_fly_property(self, safety_monitor):
        """Test can_fly property."""
        # INITIALIZING cannot fly
        assert not safety_monitor.can_fly
        
        safety_monitor.set_ready()
        assert not safety_monitor.can_fly
        
        safety_monitor.arm()
        assert safety_monitor.can_fly
        
        safety_monitor.set_flying()
        assert safety_monitor.can_fly
        
        safety_monitor.set_landing()
        assert not safety_monitor.can_fly


class TestStartPosition:
    """Tests for start position handling."""
    
    def test_set_start_position(self, safety_monitor):
        """Test setting start position."""
        pos = Position(1.0, 2.0, 0.0)
        safety_monitor.set_start_position(pos)
        
        assert safety_monitor.start_position == pos
    
    def test_geofence_relative_to_start(self, safety_monitor):
        """Test geofence is calculated relative to start position."""
        safety_monitor.set_ready()
        safety_monitor.arm()
        safety_monitor.set_flying()
        safety_monitor.set_start_position(Position(10.0, 10.0, 0.0))
        safety_monitor.takeoff_complete = True
        safety_monitor.crash_detection_enabled = True
        
        # Position within geofence of start
        near_start_data = SensorData(
            position=Position(11.0, 10.0, 0.5),  # 1m from start
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0.1, 0, 0),
            altitude=0.5,
            battery=3.9,
            timestamp=100.0,
        )
        
        state = safety_monitor.check(near_start_data)
        assert state == SafetyState.FLYING  # Should be safe
        
        # Position outside geofence of start
        far_from_start_data = SensorData(
            position=Position(15.0, 10.0, 0.5),  # 5m from start (> 3m geofence)
            orientation=Orientation(0, 0, 0),
            velocity=Velocity(0.1, 0, 0),
            altitude=0.5,
            battery=3.9,
            timestamp=101.0,
        )
        
        state = safety_monitor.check(far_from_start_data)
        assert state == SafetyState.EMERGENCY
