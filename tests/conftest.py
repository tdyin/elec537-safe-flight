"""Pytest configuration file."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hardware: tests requiring real Crazyflie drone hardware"
    )
    config.addinivalue_line(
        "markers", "slow: tests taking more than 5 seconds to complete"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests requiring external dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Skip hardware tests unless --hardware flag is provided."""
    if not config.getoption("--hardware", default=False):
        skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="run tests that require real Crazyflie hardware"
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Fixture providing a sample image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_point_cloud():
    """Fixture providing a sample point cloud."""
    return np.random.randn(1000, 3) * 2.0


@pytest.fixture
def sample_obstacles():
    """Fixture providing sample obstacle detections."""
    return [
        {
            'centroid': np.array([1.0, 0.0, 0.5]),
            'size': np.array([0.1, 0.1, 0.3]),
            'num_points': 50
        },
        {
            'centroid': np.array([2.0, 1.0, 0.3]),
            'size': np.array([0.2, 0.2, 0.5]),
            'num_points': 75
        }
    ]


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def hardware_config():
    """Fixture providing hardware configuration dictionary."""
    return {
        'mode': 'hardware',
        'drone': {
            'interface': 'crazyflie',
            'uri': 'radio://0/80/2M/E7E7E7E7E7',
            'hardware': {
                'flow_deck_required': True,
                'multiranger_required': False,
                'ai_deck_required': True,
                'arming_required': True,
            },
            'logging': {
                'state_estimate_rate_ms': 20,
                'stabilizer_rate_ms': 20,
                'battery_rate_ms': 500,
                'range_rate_ms': 50,
            },
            'navigation': {
                'cruise_speed': 0.3,
                'avoidance_speed': 0.15,
                'max_speed': 0.4,
                'safety_distance': 1.0,
                'max_altitude': 1.5,
                'min_altitude': 0.3,
                'default_altitude': 0.5,
                'forward_only': True,
            },
            'safety': {
                'battery_min_voltage': 3.3,
                'battery_warning_voltage': 3.5,
                'geofence_radius': 3.0,
                'geofence_height': 2.0,
                'crash_tilt_threshold': 40.0,
                'crash_altitude_threshold': 0.15,
            },
            'flight': {
                'takeoff_height': 0.5,
                'takeoff_velocity': 0.3,
                'land_velocity': 0.2,
            },
            'ai_deck': {
                'enabled': True,
                'ip': '192.168.4.1',
                'port': 5000,
                'timeout': 5.0,
            },
        },
        'vision': {
            'depth': {
                'model_path': 'models/midas_v21_small.onnx',
                'use_gpu': False,
            },
        },
    }


@pytest.fixture
def simulation_config():
    """Fixture providing simulation configuration dictionary."""
    return {
        'mode': 'simulation',
        'drone': {
            'interface': 'webots',
            'simulation': {
                'host': 'localhost',
                'port': 10020,
                'world': 'apartment',
            },
            'navigation': {
                'cruise_speed': 0.5,
                'avoidance_speed': 0.3,
                'safety_distance': 0.8,
                'max_altitude': 2.0,
                'min_altitude': 0.5,
            },
        },
    }


# =============================================================================
# Mock cflib Fixtures
# =============================================================================

@pytest.fixture
def mock_crazyflie():
    """Mock cflib Crazyflie for unit testing.
    
    Returns a mock Crazyflie object with common attributes set up.
    """
    mock_cf = Mock()
    mock_cf.is_connected.return_value = True
    mock_cf.link_uri = 'radio://0/80/2M/E7E7E7E7E7'
    
    # Mock commander
    mock_cf.commander = Mock()
    mock_cf.commander.send_stop_setpoint = Mock()
    mock_cf.commander.send_velocity_world_setpoint = Mock()
    
    # Mock param
    mock_cf.param = Mock()
    mock_cf.param.get_value = Mock(return_value=1)
    
    # Mock log
    mock_cf.log = Mock()
    mock_cf.log.add_config = Mock()
    mock_cf.log.toc = Mock()
    mock_cf.log.toc.toc = {}
    
    return mock_cf


@pytest.fixture
def mock_sync_crazyflie(mock_crazyflie):
    """Mock SyncCrazyflie wrapper.
    
    Returns a mock SyncCrazyflie that wraps a mock Crazyflie.
    """
    mock_scf = Mock()
    mock_scf.cf = mock_crazyflie
    mock_scf.is_link_open = Mock(return_value=True)
    mock_scf.open_link = Mock()
    mock_scf.close_link = Mock()
    
    return mock_scf


@pytest.fixture
def mock_log_config():
    """Mock LogConfig for testing logging setup.
    
    Returns a mock LogConfig with callback lists.
    """
    mock_lc = Mock()
    mock_lc.name = 'TestLog'
    mock_lc.period_in_ms = 20
    mock_lc.data_received_cb = Mock()
    mock_lc.data_received_cb.add_callback = Mock()
    mock_lc.error_cb = Mock()
    mock_lc.error_cb.add_callback = Mock()
    mock_lc.add_variable = Mock()
    mock_lc.start = Mock()
    mock_lc.stop = Mock()
    mock_lc.delete = Mock()
    
    return mock_lc


@pytest.fixture
def mock_motion_commander():
    """Mock MotionCommander for flight testing.
    
    Returns a mock MotionCommander with flight control methods.
    """
    mock_mc = Mock()
    mock_mc.take_off = Mock()
    mock_mc.land = Mock()
    mock_mc.start_linear_motion = Mock()
    mock_mc.stop = Mock()
    mock_mc.get_height = Mock(return_value=0.5)
    
    return mock_mc


# =============================================================================
# Sensor Data Fixtures
# =============================================================================

@pytest.fixture
def sample_sensor_data():
    """Fixture providing sample sensor data from drone."""
    return {
        'position': (1.0, 0.5, 0.5),
        'velocity': (0.2, 0.0, 0.0),
        'altitude': 0.5,
        'roll': 0.02,
        'pitch': -0.01,
        'yaw': 0.0,
        'roll_deg': 1.15,
        'pitch_deg': -0.57,
        'yaw_deg': 0.0,
        'battery': 3.85,
        'range_front': 2.5,
        'range_back': None,
        'range_left': 1.2,
        'range_right': 3.0,
        'range_up': None,
        'range_zrange': 0.5,
        'timestamp': 12345,
        'last_update': {
            'state_estimate': 12345,
            'stabilizer': 12344,
            'battery': 12300,
        },
    }


@pytest.fixture
def critical_sensor_data():
    """Fixture providing sensor data with critical conditions."""
    return {
        'position': (3.5, 0.0, 0.5),  # Outside geofence
        'velocity': (0.0, 0.0, 0.0),
        'altitude': 0.5,
        'roll': 0.8,  # ~45 degrees - exceeds tilt threshold
        'pitch': 0.0,
        'yaw': 0.0,
        'battery': 3.2,  # Below minimum
        'timestamp': 12345,
    }


@pytest.fixture
def mock_deck_parameters():
    """Fixture providing deck detection parameters."""
    return {
        'deck.bcFlow2': 1,      # Flow deck v2 present
        'deck.bcAI': 1,         # AI deck present
        'deck.bcMultiranger': 0, # Multi-ranger not present
        'deck.bcLighthouse4': 0, # Lighthouse not present
    }
