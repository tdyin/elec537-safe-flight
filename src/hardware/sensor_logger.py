"""Sensor logging using cflib LogConfig.

This module manages cflib log subscriptions to receive sensor data
from the Crazyflie drone in real-time.
"""

import threading
from typing import Dict, Optional
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
    sensor values from the Crazyflie.
    """
    
    def __init__(self, scf: 'SyncCrazyflie', config: Dict):
        """Initialize sensor logger.
        
        Args:
            scf: SyncCrazyflie instance (already connected)
            config: Configuration dictionary with logging rates
        """
        if not CFLIB_AVAILABLE:
            raise RuntimeError("cflib required for sensor logging")
        
        self.scf = scf
        self.cf = scf.cf
        
        # Logging rates from config
        logging_config = config.get('logging', {})
        self.stabilizer_rate_ms = logging_config.get('stabilizer_rate_ms', 20)
        self.state_estimate_rate_ms = logging_config.get('state_estimate_rate_ms', 20)
        self.battery_rate_ms = logging_config.get('battery_rate_ms', 500)
        
        # Sensor data cache (thread-safe)
        self._sensor_data: Dict = {
            'position': (0.0, 0.0, 0.0),
            'velocity': (0.0, 0.0, 0.0),
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'battery': 0.0,
            'timestamp': 0.0,
        }
        self._lock = threading.Lock()
        
        # Log configs
        self._log_configs = []
        
        logger.info("[SENSOR] Logger initialized")
    
    def setup_logging(self) -> None:
        """Setup all log configurations."""
        try:
            self._setup_state_estimate_log()
            self._setup_stabilizer_log()
            self._setup_battery_log()
            
            # Start all log configs
            for lc in self._log_configs:
                lc.start()
            
            logger.info(f"[SENSOR] ✓ Logging started ({len(self._log_configs)} configs)")
            
        except Exception as e:
            logger.error(f"[SENSOR] ✗ Log setup failed: {e}")
            raise
    
    def stop_logging(self) -> None:
        """Stop all log configurations."""
        for lc in self._log_configs:
            try:
                lc.stop()
                lc.delete()
            except Exception:
                pass
        self._log_configs.clear()
        logger.info("[SENSOR] Logging stopped")
    
    def get_sensor_data(self) -> Dict:
        """Get latest sensor readings (thread-safe).
        
        Returns:
            Dictionary with sensor values
        """
        with self._lock:
            return self._sensor_data.copy()
    
    def _setup_state_estimate_log(self) -> None:
        """Setup StateEstimate log for position and velocity."""
        lc = LogConfig(name='StateEstimate', period_in_ms=self.state_estimate_rate_ms)
        
        # Position
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
        """Setup Stabilizer log for orientation."""
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
        """Setup battery voltage log."""
        lc = LogConfig(name='Battery', period_in_ms=self.battery_rate_ms)
        
        lc.add_variable('pm.vbat', 'float')
        
        lc.data_received_cb.add_callback(self._battery_callback)
        lc.error_cb.add_callback(self._error_callback)
        
        self.cf.log.add_config(lc)
        self._log_configs.append(lc)
        
        logger.debug(f"[SENSOR] Battery log @ {self.battery_rate_ms}ms")
    
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
    
    def _stabilizer_callback(self, timestamp, data, logconf) -> None:
        """Handle Stabilizer log data."""
        with self._lock:
            # Convert degrees to radians
            self._sensor_data['roll'] = np.deg2rad(data['stabilizer.roll'])
            self._sensor_data['pitch'] = np.deg2rad(data['stabilizer.pitch'])
            self._sensor_data['yaw'] = np.deg2rad(data['stabilizer.yaw'])
    
    def _battery_callback(self, timestamp, data, logconf) -> None:
        """Handle Battery log data."""
        with self._lock:
            self._sensor_data['battery'] = data['pm.vbat']
    
    def _error_callback(self, logconf, msg) -> None:
        """Handle log errors."""
        logger.error(f"[SENSOR] Log error in {logconf.name}: {msg}")
