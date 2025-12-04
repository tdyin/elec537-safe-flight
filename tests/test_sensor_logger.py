"""Tests for SensorLogger module.

Tests the cflib LogConfig management for sensor data acquisition.
Uses mocks since tests run without actual hardware.
"""

import pytest
import threading
import time
import sys
from unittest.mock import Mock, MagicMock, patch
import numpy as np


# Create mock cflib modules before any import
mock_cflib = MagicMock()
mock_crazyflie = MagicMock()
mock_syncCrazyflie = MagicMock()
mock_log = MagicMock()

sys.modules['cflib'] = mock_cflib
sys.modules['cflib.crazyflie'] = mock_crazyflie
sys.modules['cflib.crazyflie.syncCrazyflie'] = mock_syncCrazyflie
sys.modules['cflib.crazyflie.log'] = mock_log

# Now import the module
from src.hardware.sensor_logger import SensorLogger, CFLIB_AVAILABLE


@pytest.fixture
def mock_sync_crazyflie():
    """Create a mock SyncCrazyflie instance."""
    scf = Mock()
    scf.cf = Mock()
    scf.cf.log = Mock()
    scf.cf.log.add_config = Mock()
    scf.cf.log.toc = Mock()
    scf.cf.log.toc.toc = {}  # Empty TOC by default
    return scf


@pytest.fixture
def mock_sync_crazyflie_with_multiranger():
    """Create a mock SyncCrazyflie with Multi-Ranger in TOC."""
    scf = Mock()
    scf.cf = Mock()
    scf.cf.log = Mock()
    scf.cf.log.add_config = Mock()
    scf.cf.log.toc = Mock()
    scf.cf.log.toc.toc = {
        'range': {'front': {}, 'back': {}, 'left': {}, 'right': {}, 'up': {}, 'zrange': {}}
    }
    return scf


@pytest.fixture
def default_config():
    """Default drone configuration for tests."""
    return {
        'logging': {
            'state_estimate_rate_ms': 20,
            'stabilizer_rate_ms': 20,
            'battery_rate_ms': 500,
            'range_rate_ms': 50,
        },
        'hardware': {
            'flow_deck_required': True,
            'multiranger_required': False,
        }
    }


class TestSensorLoggerInit:
    """Tests for SensorLogger initialization."""
    
    def test_init_with_default_config(self, mock_sync_crazyflie, default_config):
        """Test initialization with default configuration."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        assert logger.state_estimate_rate_ms == 20
        assert logger.stabilizer_rate_ms == 20
        assert logger.battery_rate_ms == 500
        assert logger.range_rate_ms == 50
        assert not logger.is_logging
        assert not logger.has_multiranger
    
    def test_init_with_empty_config(self, mock_sync_crazyflie):
        """Test initialization with empty config uses defaults."""
        logger = SensorLogger(mock_sync_crazyflie, {})
        
        # Should use class defaults
        assert logger.state_estimate_rate_ms == SensorLogger.DEFAULT_STATE_RATE_MS
        assert logger.stabilizer_rate_ms == SensorLogger.DEFAULT_STABILIZER_RATE_MS
        assert logger.battery_rate_ms == SensorLogger.DEFAULT_BATTERY_RATE_MS


class TestSensorLoggerData:
    """Tests for sensor data access methods."""
    
    def test_get_sensor_data_initial(self, mock_sync_crazyflie, default_config):
        """Test initial sensor data values."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        data = logger.get_sensor_data()
        
        assert data['position'] == (0.0, 0.0, 0.0)
        assert data['velocity'] == (0.0, 0.0, 0.0)
        assert data['roll'] == 0.0
        assert data['pitch'] == 0.0
        assert data['yaw'] == 0.0
        assert data['battery'] == 0.0
    
    def test_get_position(self, mock_sync_crazyflie, default_config):
        """Test position getter."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Simulate callback updating position
        logger._sensor_data['position'] = (1.5, 2.0, 0.5)
        
        pos = logger.get_position()
        assert pos == (1.5, 2.0, 0.5)
    
    def test_get_orientation(self, mock_sync_crazyflie, default_config):
        """Test orientation getter returns radians."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Set orientation in radians
        logger._sensor_data['roll'] = 0.1
        logger._sensor_data['pitch'] = -0.05
        logger._sensor_data['yaw'] = 1.57
        
        roll, pitch, yaw = logger.get_orientation()
        assert roll == pytest.approx(0.1)
        assert pitch == pytest.approx(-0.05)
        assert yaw == pytest.approx(1.57)
    
    def test_get_velocity(self, mock_sync_crazyflie, default_config):
        """Test velocity getter."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        logger._sensor_data['velocity'] = (0.3, 0.0, -0.1)
        
        vx, vy, vz = logger.get_velocity()
        assert vx == 0.3
        assert vy == 0.0
        assert vz == -0.1
    
    def test_get_battery(self, mock_sync_crazyflie, default_config):
        """Test battery voltage getter."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        logger._sensor_data['battery'] = 3.85
        
        assert logger.get_battery() == 3.85
    
    def test_get_range_readings_not_available(self, mock_sync_crazyflie, default_config):
        """Test range readings when Multi-Ranger not available."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        ranges = logger.get_range_readings()
        
        assert ranges['front'] is None
        assert ranges['back'] is None
        assert ranges['left'] is None
        assert ranges['right'] is None


class TestSensorLoggerCallbacks:
    """Tests for sensor data callback handling."""
    
    def test_state_estimate_callback(self, mock_sync_crazyflie, default_config):
        """Test StateEstimate callback updates data correctly."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Simulate callback data
        data = {
            'stateEstimate.x': 1.0,
            'stateEstimate.y': 2.0,
            'stateEstimate.z': 0.5,
            'stateEstimate.vx': 0.3,
            'stateEstimate.vy': 0.0,
            'stateEstimate.vz': -0.1,
        }
        
        logger._state_estimate_callback(12345, data, None)
        
        sensor_data = logger.get_sensor_data()
        assert sensor_data['position'] == (1.0, 2.0, 0.5)
        assert sensor_data['velocity'] == (0.3, 0.0, -0.1)
        assert sensor_data['altitude'] == 0.5
        assert sensor_data['timestamp'] == 12345
    
    def test_stabilizer_callback(self, mock_sync_crazyflie, default_config):
        """Test Stabilizer callback converts degrees to radians."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Simulate callback data in degrees
        data = {
            'stabilizer.roll': 5.0,
            'stabilizer.pitch': -3.0,
            'stabilizer.yaw': 90.0,
        }
        
        logger._stabilizer_callback(12346, data, None)
        
        sensor_data = logger.get_sensor_data()
        
        # Check degrees are stored
        assert sensor_data['roll_deg'] == 5.0
        assert sensor_data['pitch_deg'] == -3.0
        assert sensor_data['yaw_deg'] == 90.0
        
        # Check radians conversion
        assert sensor_data['roll'] == pytest.approx(np.deg2rad(5.0))
        assert sensor_data['pitch'] == pytest.approx(np.deg2rad(-3.0))
        assert sensor_data['yaw'] == pytest.approx(np.deg2rad(90.0))
    
    def test_battery_callback(self, mock_sync_crazyflie, default_config):
        """Test Battery callback updates voltage."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        data = {'pm.vbat': 3.92}
        logger._battery_callback(12347, data, None)
        
        assert logger.get_battery() == 3.92
    
    def test_range_callback_converts_mm_to_m(self, mock_sync_crazyflie, default_config):
        """Test Range callback converts millimeters to meters."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        data = {
            'range.front': 1500,  # 1.5 meters
            'range.back': 2000,   # 2.0 meters
            'range.left': 500,    # 0.5 meters
            'range.right': 3000,  # 3.0 meters
            'range.up': 1000,     # 1.0 meters
            'range.zrange': 400,  # 0.4 meters
        }
        
        logger._range_callback(12348, data, None)
        
        ranges = logger.get_range_readings()
        assert ranges['front'] == 1.5
        assert ranges['back'] == 2.0
        assert ranges['left'] == 0.5
        assert ranges['right'] == 3.0
        assert ranges['up'] == 1.0
        assert ranges['zrange'] == 0.4
    
    def test_range_callback_out_of_range_values(self, mock_sync_crazyflie, default_config):
        """Test Range callback sets out-of-range values to None."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        data = {
            'range.front': 5000,  # Out of range (>4000mm)
            'range.back': 6000,   # Out of range
            'range.left': 100,    # Valid
            'range.right': 8000,  # Out of range
            'range.up': 4001,     # Out of range
            'range.zrange': 200,  # Valid
        }
        
        logger._range_callback(12349, data, None)
        
        ranges = logger.get_range_readings()
        assert ranges['front'] is None
        assert ranges['back'] is None
        assert ranges['left'] == 0.1
        assert ranges['right'] is None
        assert ranges['up'] is None
        assert ranges['zrange'] == 0.2


class TestSensorLoggerUserCallbacks:
    """Tests for user-registered callbacks."""
    
    def test_add_and_trigger_callback(self, mock_sync_crazyflie, default_config):
        """Test registering and triggering a user callback."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        callback_data = []
        def my_callback(data):
            callback_data.append(data.copy())
        
        logger.add_callback('battery', my_callback)
        
        # Trigger battery callback
        logger._battery_callback(1000, {'pm.vbat': 4.0}, None)
        
        assert len(callback_data) == 1
        assert callback_data[0]['battery'] == 4.0
    
    def test_remove_callback(self, mock_sync_crazyflie, default_config):
        """Test removing a user callback."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        call_count = [0]
        def my_callback(data):
            call_count[0] += 1
        
        logger.add_callback('stabilizer', my_callback)
        logger._stabilizer_callback(1000, {'stabilizer.roll': 0, 'stabilizer.pitch': 0, 'stabilizer.yaw': 0}, None)
        
        assert call_count[0] == 1
        
        logger.remove_callback('stabilizer', my_callback)
        logger._stabilizer_callback(1001, {'stabilizer.roll': 0, 'stabilizer.pitch': 0, 'stabilizer.yaw': 0}, None)
        
        assert call_count[0] == 1  # Should not increment


class TestSensorLoggerThreadSafety:
    """Tests for thread-safety of sensor data access."""
    
    def test_concurrent_read_write(self, mock_sync_crazyflie, default_config):
        """Test concurrent reads and writes don't corrupt data."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        errors = []
        
        def writer():
            for i in range(100):
                data = {
                    'stateEstimate.x': float(i),
                    'stateEstimate.y': float(i * 2),
                    'stateEstimate.z': 0.5,
                    'stateEstimate.vx': 0.1,
                    'stateEstimate.vy': 0.0,
                    'stateEstimate.vz': 0.0,
                }
                logger._state_estimate_callback(i, data, None)
        
        def reader():
            for _ in range(100):
                try:
                    data = logger.get_sensor_data()
                    pos = data['position']
                    # Position should be a consistent tuple
                    assert len(pos) == 3
                    assert all(isinstance(v, float) for v in pos)
                except Exception as e:
                    errors.append(e)
        
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        
        writer_thread.start()
        reader_thread.start()
        
        writer_thread.join()
        reader_thread.join()
        
        assert len(errors) == 0


class TestSensorLoggerWaitForData:
    """Tests for wait_for_data method."""
    
    def test_wait_for_data_success(self, mock_sync_crazyflie, default_config):
        """Test wait_for_data returns True when data arrives."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Simulate data arriving in a separate thread
        def simulate_data():
            time.sleep(0.1)
            logger._sensor_data['timestamp'] = 12345
        
        thread = threading.Thread(target=simulate_data)
        thread.start()
        
        result = logger.wait_for_data(timeout=1.0)
        thread.join()
        
        assert result is True
    
    def test_wait_for_data_timeout(self, mock_sync_crazyflie, default_config):
        """Test wait_for_data returns False on timeout."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        # Don't set timestamp, should timeout
        result = logger.wait_for_data(timeout=0.2)
        
        assert result is False


class TestMultiRangerDetection:
    """Tests for Multi-Ranger deck detection."""
    
    def test_detect_multiranger_from_toc(self, mock_sync_crazyflie_with_multiranger, default_config):
        """Test Multi-Ranger detection from log TOC."""
        logger = SensorLogger(mock_sync_crazyflie_with_multiranger, default_config)
        
        result = logger._try_multiranger_detection()
        
        assert result is True
    
    def test_no_multiranger_in_toc(self, mock_sync_crazyflie, default_config):
        """Test detection returns False when Multi-Ranger not present."""
        logger = SensorLogger(mock_sync_crazyflie, default_config)
        
        result = logger._try_multiranger_detection()
        
        assert result is False
