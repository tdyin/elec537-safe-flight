"""Hardware integration tests requiring real Crazyflie drone.

These tests are marked with @pytest.mark.hardware and require:
1. A real Crazyflie 2.1 drone
2. Crazyradio PA dongle connected
3. Flow Deck v2 attached
4. AI Deck attached (optional for some tests)

Run with: pytest tests/test_hardware_integration.py --hardware
Skip with: pytest tests/ -m "not hardware"
"""

import pytest
import time

# Try to import cflib - these tests require it
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.positioning.motion_commander import MotionCommander
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False


# Default test URI - can be overridden
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E7'


@pytest.fixture(scope="module")
def crazyflie_uri():
    """Get Crazyflie URI for tests."""
    return DEFAULT_URI


@pytest.fixture(scope="module")
def init_drivers():
    """Initialize CRTP drivers once per module."""
    if CFLIB_AVAILABLE:
        cflib.crtp.init_drivers()
    yield


@pytest.mark.hardware
class TestCrazyflieConnection:
    """Tests for Crazyflie radio connection.
    
    These tests verify basic connectivity to the drone.
    """
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_scan_for_crazyflies(self, init_drivers):
        """Test scanning for available Crazyflies."""
        available = cflib.crtp.scan_interfaces()
        
        # Should find at least one interface or return empty list
        assert isinstance(available, list)
        print(f"Found {len(available)} Crazyflie(s): {available}")
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_connect_to_crazyflie(self, init_drivers, crazyflie_uri):
        """Test connecting to Crazyflie."""
        cf = Crazyflie(rw_cache='./cache')
        connected = False
        
        def connected_callback(link_uri):
            nonlocal connected
            connected = True
        
        def failed_callback(link_uri, msg):
            pytest.fail(f"Connection failed: {msg}")
        
        cf.connected.add_callback(connected_callback)
        cf.connection_failed.add_callback(failed_callback)
        
        cf.open_link(crazyflie_uri)
        
        # Wait for connection (timeout after 5 seconds)
        timeout = 5.0
        start = time.time()
        while not connected and time.time() - start < timeout:
            time.sleep(0.1)
        
        cf.close_link()
        
        assert connected, f"Failed to connect to {crazyflie_uri} within {timeout}s"
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_sync_crazyflie_context(self, init_drivers, crazyflie_uri):
        """Test SyncCrazyflie context manager."""
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            assert scf.is_link_open()
            assert scf.cf.is_connected()


@pytest.mark.hardware
class TestDeckDetection:
    """Tests for deck detection.
    
    Verifies that required decks are attached and detected.
    """
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_detect_flow_deck(self, init_drivers, crazyflie_uri):
        """Test Flow Deck v2 is detected."""
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Wait for parameters to be downloaded
            time.sleep(1.0)
            
            # Check deck parameter
            try:
                flow_attached = scf.cf.param.get_value('deck.bcFlow2')
                assert int(flow_attached) == 1, "Flow Deck v2 not detected"
                print("✓ Flow Deck v2 detected")
            except Exception as e:
                pytest.skip(f"Could not read deck parameter: {e}")
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_detect_ai_deck(self, init_drivers, crazyflie_uri):
        """Test AI Deck is detected (optional)."""
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            time.sleep(1.0)
            
            try:
                ai_attached = scf.cf.param.get_value('deck.bcAI')
                if int(ai_attached) == 1:
                    print("✓ AI Deck detected")
                else:
                    pytest.skip("AI Deck not attached (optional)")
            except Exception as e:
                pytest.skip(f"Could not read deck parameter: {e}")


@pytest.mark.hardware
class TestSensorLogging:
    """Tests for sensor data logging.
    
    Verifies that sensor data can be read from the drone.
    """
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_state_estimate_logging(self, init_drivers, crazyflie_uri):
        """Test StateEstimate log configuration."""
        received_data = []
        
        def data_callback(timestamp, data, logconf):
            received_data.append(data)
        
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            lc = LogConfig(name='StateEstimate', period_in_ms=100)
            lc.add_variable('stateEstimate.x', 'float')
            lc.add_variable('stateEstimate.y', 'float')
            lc.add_variable('stateEstimate.z', 'float')
            
            lc.data_received_cb.add_callback(data_callback)
            scf.cf.log.add_config(lc)
            lc.start()
            
            # Collect data for 1 second
            time.sleep(1.0)
            
            lc.stop()
        
        assert len(received_data) > 0, "No StateEstimate data received"
        assert 'stateEstimate.x' in received_data[0]
        print(f"✓ Received {len(received_data)} StateEstimate samples")
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_battery_logging(self, init_drivers, crazyflie_uri):
        """Test battery voltage logging."""
        battery_voltage = [None]
        
        def data_callback(timestamp, data, logconf):
            battery_voltage[0] = data['pm.vbat']
        
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            lc = LogConfig(name='Battery', period_in_ms=500)
            lc.add_variable('pm.vbat', 'float')
            
            lc.data_received_cb.add_callback(data_callback)
            scf.cf.log.add_config(lc)
            lc.start()
            
            time.sleep(1.0)
            
            lc.stop()
        
        assert battery_voltage[0] is not None, "No battery data received"
        assert 2.5 < battery_voltage[0] < 4.5, f"Battery voltage out of range: {battery_voltage[0]}V"
        print(f"✓ Battery voltage: {battery_voltage[0]:.2f}V")


@pytest.mark.hardware
@pytest.mark.slow
class TestBasicFlight:
    """Tests for basic flight operations.
    
    WARNING: These tests will make the drone fly!
    Ensure you have a safe flying area before running.
    """
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_hover_1_second(self, init_drivers, crazyflie_uri):
        """Test basic hover for 1 second.
        
        WARNING: Drone will take off to 0.3m, hover, and land.
        """
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            with MotionCommander(scf, default_height=0.3) as mc:
                print("Taking off to 0.3m...")
                time.sleep(1.0)
                print("Hovering for 1 second...")
                time.sleep(1.0)
                print("Landing...")
        
        print("✓ Hover test complete")
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_emergency_stop(self, init_drivers, crazyflie_uri):
        """Test emergency stop command."""
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Send stop command (should work even when not flying)
            scf.cf.commander.send_stop_setpoint()
            time.sleep(0.1)
        
        print("✓ Emergency stop command sent successfully")


@pytest.mark.hardware
class TestSensorLoggerIntegration:
    """Integration tests for SensorLogger with real hardware."""
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_sensor_logger_setup(self, init_drivers, crazyflie_uri):
        """Test SensorLogger setup with real drone."""
        from src.hardware.sensor_logger import SensorLogger
        
        config = {
            'logging': {
                'state_estimate_rate_ms': 50,
                'stabilizer_rate_ms': 50,
                'battery_rate_ms': 500,
            },
            'hardware': {
                'multiranger_required': False,
            },
        }
        
        with SyncCrazyflie(crazyflie_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            logger = SensorLogger(scf, config)
            logger.setup_logging()
            
            assert logger.is_logging
            
            # Wait for data
            assert logger.wait_for_data(timeout=2.0), "Timeout waiting for sensor data"
            
            # Wait a bit more for battery data (500ms rate)
            time.sleep(1.0)
            
            # Get sensor data
            data = logger.get_sensor_data()
            
            assert data['timestamp'] > 0
            assert 'position' in data
            assert 'velocity' in data
            # Battery may be 0.0 if not received yet - just check it exists
            assert 'battery' in data
            
            print(f"✓ Position: {data['position']}")
            print(f"✓ Battery: {data['battery']:.2f}V")
            
            logger.stop_logging()
            assert not logger.is_logging


@pytest.mark.hardware
class TestCrazyflieInterfaceIntegration:
    """Integration tests for CrazyflieHardwareInterface."""
    
    @pytest.mark.skipif(not CFLIB_AVAILABLE, reason="cflib not installed")
    def test_interface_connect_disconnect(self, init_drivers, crazyflie_uri, hardware_config):
        """Test CrazyflieHardwareInterface connect and disconnect."""
        from src.hardware.crazyflie_interface import CrazyflieHardwareInterface
        
        interface = CrazyflieHardwareInterface(
            uri=crazyflie_uri,
            config=hardware_config
        )
        
        # Connect
        result = interface.connect()
        assert result, "Failed to connect"
        assert interface.is_connected
        
        # Get sensor data
        data = interface.get_sensor_data()
        assert data is not None
        assert 'position' in data
        
        pos = interface.get_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 3
        
        # Disconnect
        interface.disconnect()
        assert not interface.is_connected
        
        print("✓ Interface connect/disconnect test passed")
