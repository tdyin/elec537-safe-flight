"""Sensor logging using cflib LogConfig.

This module manages cflib log subscriptions to receive sensor data
from the Crazyflie drone in real-time.

Log Variables:
    StateEstimate: x, y, z, vx, vy, vz (position/velocity from Flow Deck)
    Stabilizer: roll, pitch, yaw (orientation in degrees)
    Battery: pm.vbat (voltage)
    Range: front, back, left, right, zrange (Multi-Ranger, optional)

Usage:
    sensor_logger = SensorLogger(scf, config['drone'])
    sensor_logger.setup_logging()
    data = sensor_logger.get_sensor_data()
"""

import threading
import time
from typing import Dict, Optional, Tuple, Callable
from loguru import logger
import numpy as np

try:
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False


class SensorLogger:
    """Manages cflib LogConfig subscriptions for sensor data.
    
    Creates and manages log configurations for reading various
    sensor values from the Crazyflie drone. Thread-safe access
    to sensor data is provided via get_sensor_data().
    
    Attributes:
        is_logging: True if logging is active
        has_multiranger: True if Multi-Ranger deck detected
    """
    
    # Default logging rates (ms)
    DEFAULT_STATE_RATE_MS = 20
    DEFAULT_STABILIZER_RATE_MS = 20
    DEFAULT_BATTERY_RATE_MS = 500
    DEFAULT_RANGE_RATE_MS = 50
    
    def __init__(self, scf: 'SyncCrazyflie', config: Dict):
        """Initialize sensor logger.
        
        Args:
            scf: SyncCrazyflie instance (already connected)
            config: Drone configuration dictionary (from hardware.yaml drone section)
                   Expected keys: 'logging', 'hardware'
        """
        if not CFLIB_AVAILABLE:
            raise RuntimeError("cflib required for sensor logging")
        
        self.scf = scf
        self.cf = scf.cf
        self._config = config
        
        # Logging rates from config
        logging_config = config.get('logging', {})
        self.state_estimate_rate_ms = logging_config.get(
            'state_estimate_rate_ms', self.DEFAULT_STATE_RATE_MS
        )
        self.stabilizer_rate_ms = logging_config.get(
            'stabilizer_rate_ms', self.DEFAULT_STABILIZER_RATE_MS
        )
        self.battery_rate_ms = logging_config.get(
            'battery_rate_ms', self.DEFAULT_BATTERY_RATE_MS
        )
        self.range_rate_ms = logging_config.get(
            'range_rate_ms', self.DEFAULT_RANGE_RATE_MS
        )
        
        # Hardware configuration
        hw_config = config.get('hardware', {})
        self._multiranger_required = hw_config.get('multiranger_required', False)
        
        # Sensor data cache (thread-safe)
        self._sensor_data: Dict = {
            'position': (0.0, 0.0, 0.0),
            'velocity': (0.0, 0.0, 0.0),
            'altitude': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'roll_deg': 0.0,
            'pitch_deg': 0.0,
            'yaw_deg': 0.0,
            'battery': 0.0,
            'range_front': None,
            'range_back': None,
            'range_left': None,
            'range_right': None,
            'range_up': None,
            'range_zrange': None,
            'timestamp': 0.0,
            'last_update': {},
        }
        self._lock = threading.Lock()
        
        # Log configs
        self._log_configs: list = []
        self._is_logging = False
        self._has_multiranger = False
        
        # Error tracking
        self._error_count = 0
        self._max_errors = 10
        
        # Optional callbacks for sensor updates
        self._callbacks: Dict[str, list] = {
            'state_estimate': [],
            'stabilizer': [],
            'battery': [],
            'range': [],
        }
        
        logger.info(f"[SENSOR] Logger initialized (state: {self.state_estimate_rate_ms}ms, "
                    f"stab: {self.stabilizer_rate_ms}ms, batt: {self.battery_rate_ms}ms)")
    
    @property
    def is_logging(self) -> bool:
        """Check if logging is currently active."""
        return self._is_logging
    
    @property
    def has_multiranger(self) -> bool:
        """Check if Multi-Ranger deck is available."""
        return self._has_multiranger

    def setup_logging(self) -> None:
        """Setup all log configurations.
        
        Sets up StateEstimate, Stabilizer, Battery logs.
        Optionally sets up Range logs if Multi-Ranger is available.
        
        Raises:
            RuntimeError: If log setup fails critically
        """
        if self._is_logging:
            logger.warning("[SENSOR] Logging already active, stopping first")
            self.stop_logging()
        
        try:
            self._setup_state_estimate_log()
            self._setup_stabilizer_log()
            self._setup_battery_log()
            
            # Try to setup Multi-Ranger (optional)
            if self._multiranger_required or self._try_multiranger_detection():
                try:
                    self._setup_range_log()
                    self._has_multiranger = True
                except Exception as e:
                    if self._multiranger_required:
                        raise RuntimeError(f"Multi-Ranger required but not available: {e}")
                    logger.info(f"[SENSOR] Multi-Ranger not available: {e}")
            
            # Start all log configs
            for lc in self._log_configs:
                lc.start()
            
            self._is_logging = True
            logger.info(f"[SENSOR] ✓ Logging started ({len(self._log_configs)} configs, "
                       f"multiranger: {self._has_multiranger})")
            
        except Exception as e:
            logger.error(f"[SENSOR] ✗ Log setup failed: {e}")
            self.stop_logging()
            raise
    
    def _try_multiranger_detection(self) -> bool:
        """Try to detect Multi-Ranger deck via TOC.
        
        Returns:
            True if Multi-Ranger variables found in TOC
        """
        try:
            # Check if range variables exist in log TOC
            toc = self.cf.log.toc.toc
            return 'range' in toc and 'front' in toc.get('range', {})
        except Exception:
            return False

    def stop_logging(self) -> None:
        """Stop all log configurations."""
        for lc in self._log_configs:
            try:
                lc.stop()
                lc.delete()
            except Exception as e:
                logger.debug(f"[SENSOR] Error stopping log {lc.name}: {e}")
        
        self._log_configs.clear()
        self._is_logging = False
        self._has_multiranger = False
        logger.info("[SENSOR] Logging stopped")
    
    def add_callback(self, log_type: str, callback: Callable[[Dict], None]) -> None:
        """Register callback for sensor updates.
        
        Args:
            log_type: Type of log ('state_estimate', 'stabilizer', 'battery', 'range')
            callback: Function to call with new data
        """
        if log_type in self._callbacks:
            self._callbacks[log_type].append(callback)
    
    def remove_callback(self, log_type: str, callback: Callable[[Dict], None]) -> None:
        """Remove a registered callback.
        
        Args:
            log_type: Type of log
            callback: Callback to remove
        """
        if log_type in self._callbacks and callback in self._callbacks[log_type]:
            self._callbacks[log_type].remove(callback)

    def get_sensor_data(self) -> Dict:
        """Get latest sensor readings (thread-safe).
        
        Returns:
            Dictionary with all sensor values:
            - position: (x, y, z) in meters
            - velocity: (vx, vy, vz) in m/s
            - altitude: z in meters
            - roll, pitch, yaw: orientation in radians
            - roll_deg, pitch_deg, yaw_deg: orientation in degrees
            - battery: voltage in V
            - range_front/back/left/right/up/zrange: distances in m (or None)
            - timestamp: last data timestamp
            - last_update: dict of timestamps per log type
        """
        with self._lock:
            return self._sensor_data.copy()
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (thread-safe convenience method).
        
        Returns:
            Tuple of (x, y, z) in meters
        """
        with self._lock:
            return self._sensor_data['position']
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation in radians (thread-safe).
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        with self._lock:
            return (
                self._sensor_data['roll'],
                self._sensor_data['pitch'],
                self._sensor_data['yaw']
            )
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (thread-safe).
        
        Returns:
            Tuple of (vx, vy, vz) in m/s
        """
        with self._lock:
            return self._sensor_data['velocity']
    
    def get_battery(self) -> float:
        """Get battery voltage (thread-safe).
        
        Returns:
            Battery voltage in V
        """
        with self._lock:
            return self._sensor_data['battery']
    
    def get_range_readings(self) -> Dict[str, Optional[float]]:
        """Get Multi-Ranger distance readings (thread-safe).
        
        Returns:
            Dictionary with front/back/left/right/up/zrange in meters.
            Values are None if Multi-Ranger not available.
        """
        with self._lock:
            return {
                'front': self._sensor_data['range_front'],
                'back': self._sensor_data['range_back'],
                'left': self._sensor_data['range_left'],
                'right': self._sensor_data['range_right'],
                'up': self._sensor_data['range_up'],
                'zrange': self._sensor_data['range_zrange'],
            }
    
    def wait_for_data(self, timeout: float = 2.0) -> bool:
        """Wait for initial sensor data to arrive.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if data received within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._sensor_data['timestamp'] > 0:
                    return True
            time.sleep(0.05)
        return False

    def _setup_state_estimate_log(self) -> None:
        """Setup StateEstimate log for position and velocity.
        
        Variables logged:
        - stateEstimate.x/y/z: Position from Flow Deck
        - stateEstimate.vx/vy/vz: Velocity estimates
        """
        lc = LogConfig(name='StateEstimate', period_in_ms=self.state_estimate_rate_ms)
        
        # Position (from Flow Deck)
        lc.add_variable('stateEstimate.x', 'float')
        lc.add_variable('stateEstimate.y', 'float')
        lc.add_variable('stateEstimate.z', 'float')
        
        # Velocity
        lc.add_variable('stateEstimate.vx', 'float')
        lc.add_variable('stateEstimate.vy', 'float')
        lc.add_variable('stateEstimate.vz', 'float')
        
        lc.data_received_cb.add_callback(self._state_estimate_callback)
        lc.error_cb.add_callback(self._error_callback)
        
        self.cf.log.add_config(lc)
        self._log_configs.append(lc)
        
        logger.debug(f"[SENSOR] StateEstimate log @ {self.state_estimate_rate_ms}ms")
    
    def _setup_stabilizer_log(self) -> None:
        """Setup Stabilizer log for orientation.
        
        Variables logged:
        - stabilizer.roll/pitch/yaw: Orientation in degrees
        """
        lc = LogConfig(name='Stabilizer', period_in_ms=self.stabilizer_rate_ms)
        
        lc.add_variable('stabilizer.roll', 'float')
        lc.add_variable('stabilizer.pitch', 'float')
        lc.add_variable('stabilizer.yaw', 'float')
        
        lc.data_received_cb.add_callback(self._stabilizer_callback)
        lc.error_cb.add_callback(self._error_callback)
        
        self.cf.log.add_config(lc)
        self._log_configs.append(lc)
        
        logger.debug(f"[SENSOR] Stabilizer log @ {self.stabilizer_rate_ms}ms")
    
    def _setup_battery_log(self) -> None:
        """Setup battery voltage log.
        
        Variables logged:
        - pm.vbat: Battery voltage in V
        """
        lc = LogConfig(name='Battery', period_in_ms=self.battery_rate_ms)
        
        lc.add_variable('pm.vbat', 'float')
        
        lc.data_received_cb.add_callback(self._battery_callback)
        lc.error_cb.add_callback(self._error_callback)
        
        self.cf.log.add_config(lc)
        self._log_configs.append(lc)
        
        logger.debug(f"[SENSOR] Battery log @ {self.battery_rate_ms}ms")
    
    def _setup_range_log(self) -> None:
        """Setup Multi-Ranger distance log.
        
        Variables logged:
        - range.front/back/left/right/up: Horizontal distances (mm)
        - range.zrange: Ground distance (mm)
        
        Note: Multi-Ranger returns distances in millimeters.
        """
        lc = LogConfig(name='Range', period_in_ms=self.range_rate_ms)
        
        lc.add_variable('range.front', 'uint16_t')
        lc.add_variable('range.back', 'uint16_t')
        lc.add_variable('range.left', 'uint16_t')
        lc.add_variable('range.right', 'uint16_t')
        lc.add_variable('range.up', 'uint16_t')
        lc.add_variable('range.zrange', 'uint16_t')
        
        lc.data_received_cb.add_callback(self._range_callback)
        lc.error_cb.add_callback(self._error_callback)
        
        self.cf.log.add_config(lc)
        self._log_configs.append(lc)
        
        logger.debug(f"[SENSOR] Range log @ {self.range_rate_ms}ms")
    
    def _state_estimate_callback(self, timestamp, data, logconf) -> None:
        """Handle StateEstimate log data."""
        with self._lock:
            self._sensor_data['position'] = (
                data['stateEstimate.x'],
                data['stateEstimate.y'],
                data['stateEstimate.z']
            )
            self._sensor_data['velocity'] = (
                data['stateEstimate.vx'],
                data['stateEstimate.vy'],
                data['stateEstimate.vz']
            )
            self._sensor_data['altitude'] = data['stateEstimate.z']
            self._sensor_data['timestamp'] = timestamp
            self._sensor_data['last_update']['state_estimate'] = timestamp
        
        # Notify callbacks
        for cb in self._callbacks['state_estimate']:
            try:
                cb(self._sensor_data)
            except Exception as e:
                logger.debug(f"[SENSOR] Callback error: {e}")
    
    def _stabilizer_callback(self, timestamp, data, logconf) -> None:
        """Handle Stabilizer log data."""
        roll_deg = data['stabilizer.roll']
        pitch_deg = data['stabilizer.pitch']
        yaw_deg = data['stabilizer.yaw']
        
        with self._lock:
            # Store both degrees and radians
            self._sensor_data['roll_deg'] = roll_deg
            self._sensor_data['pitch_deg'] = pitch_deg
            self._sensor_data['yaw_deg'] = yaw_deg
            self._sensor_data['roll'] = np.deg2rad(roll_deg)
            self._sensor_data['pitch'] = np.deg2rad(pitch_deg)
            self._sensor_data['yaw'] = np.deg2rad(yaw_deg)
            self._sensor_data['last_update']['stabilizer'] = timestamp
        
        # Notify callbacks
        for cb in self._callbacks['stabilizer']:
            try:
                cb(self._sensor_data)
            except Exception as e:
                logger.debug(f"[SENSOR] Callback error: {e}")
    
    def _battery_callback(self, timestamp, data, logconf) -> None:
        """Handle Battery log data."""
        with self._lock:
            self._sensor_data['battery'] = data['pm.vbat']
            self._sensor_data['last_update']['battery'] = timestamp
        
        # Notify callbacks
        for cb in self._callbacks['battery']:
            try:
                cb(self._sensor_data)
            except Exception as e:
                logger.debug(f"[SENSOR] Callback error: {e}")
    
    def _range_callback(self, timestamp, data, logconf) -> None:
        """Handle Multi-Ranger log data.
        
        Converts millimeters to meters. Values > 4000mm are treated as
        out-of-range and set to None.
        """
        MAX_RANGE_MM = 4000  # Multi-Ranger max reliable range
        
        def mm_to_m(val_mm: int) -> Optional[float]:
            """Convert mm to m, returning None if out of range."""
            if val_mm > MAX_RANGE_MM:
                return None
            return val_mm / 1000.0
        
        with self._lock:
            self._sensor_data['range_front'] = mm_to_m(data['range.front'])
            self._sensor_data['range_back'] = mm_to_m(data['range.back'])
            self._sensor_data['range_left'] = mm_to_m(data['range.left'])
            self._sensor_data['range_right'] = mm_to_m(data['range.right'])
            self._sensor_data['range_up'] = mm_to_m(data['range.up'])
            self._sensor_data['range_zrange'] = mm_to_m(data['range.zrange'])
            self._sensor_data['last_update']['range'] = timestamp
        
        # Notify callbacks
        for cb in self._callbacks['range']:
            try:
                cb(self._sensor_data)
            except Exception as e:
                logger.debug(f"[SENSOR] Callback error: {e}")
    
    def _error_callback(self, logconf, msg) -> None:
        """Handle log errors."""
        self._error_count += 1
        logger.error(f"[SENSOR] Log error in {logconf.name}: {msg}")
        
        if self._error_count >= self._max_errors:
            logger.error(f"[SENSOR] Too many errors ({self._error_count}), stopping logging")
            self.stop_logging()
