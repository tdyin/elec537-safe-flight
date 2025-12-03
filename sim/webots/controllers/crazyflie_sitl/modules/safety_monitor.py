"""
Safety monitoring and crash detection for Crazyflie.

Monitors flight parameters and detects crash conditions.
Includes detailed crash logging for debugging.
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add utils to path for logging - need absolute path for Webots
utils_path = Path(__file__).resolve().parent.parent.parent.parent / 'utils'
utils_path_str = str(utils_path)
if utils_path_str not in sys.path:
    sys.path.insert(0, utils_path_str)

try:
    from logger import log, log_crash_report
except ImportError:
    # Fallback if logger import fails
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")
    def log_crash_report(crash_type, crash_data):
        print(f"💥 CRASH: {crash_type} - {crash_data}")


class SafetyMonitor:
    """Monitors drone safety and detects crash conditions."""
    
    def __init__(self, 
                 crash_tilt_threshold=np.deg2rad(35.0),   # Lowered from 45° to detect crashes earlier
                 warning_tilt_threshold=np.deg2rad(25.0), # Lowered from 30°
                 crash_altitude_threshold=0.50,           # Raised from 0.15m - detect crash before hitting ground
                 min_flight_altitude=1.0,
                 config: dict = None):
        """
        Initialize safety monitor.
        
        Args:
            crash_tilt_threshold: Tilt angle that indicates crash (radians)
            warning_tilt_threshold: Tilt angle to reduce aggressiveness (radians)
            crash_altitude_threshold: Minimum altitude before crash (meters)
            min_flight_altitude: Altitude to consider takeoff complete (meters)
            config: Optional configuration dictionary from config.yaml
        """
        # Load config values if provided
        drone_config = config.get('drone', {}) if config else {}
        safety_config = drone_config.get('safety', {})
        
        # Use config values or fall back to constructor defaults
        self.crash_tilt_threshold = np.deg2rad(safety_config.get('crash_tilt_threshold', np.rad2deg(crash_tilt_threshold)))
        self.warning_tilt_threshold = np.deg2rad(safety_config.get('warning_tilt_threshold', np.rad2deg(warning_tilt_threshold)))
        self.crash_altitude_threshold = safety_config.get('crash_altitude_threshold', crash_altitude_threshold)
        self.min_flight_altitude = safety_config.get('min_flight_altitude', min_flight_altitude)
        
        # State
        self.crashed = False
        self.takeoff_complete = False
        self.current_tilt_magnitude = 0.0
        
        # Crash logging
        self.crash_log = []
        self.position_history = []  # Recent positions for crash analysis
        self.max_history = safety_config.get('position_history_size', 30)  # From config
        
        # Pre-crash state tracking
        self.last_position = None
        self.last_velocity = None
        self.last_commanded_velocity = None
        
    def update_state(self, position, velocity=None, commanded_velocity=None):
        """
        Update state tracking for crash logging.
        
        Args:
            position: Current position (x, y, z)
            velocity: Current velocity (vx, vy, vz) if available
            commanded_velocity: Last commanded velocity (vx, vy) if available
        """
        self.last_position = position
        self.last_velocity = velocity
        self.last_commanded_velocity = commanded_velocity
        
        # Store in history
        self.position_history.append({
            'time': datetime.now(),
            'position': position,
            'velocity': velocity
        })
        
        # Trim history
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def get_tilt_safety_factor(self, roll, pitch):
        """
        Get safety factor based on current tilt (0.5-1.0).
        
        Returns lower values as tilt increases to reduce aggressiveness.
        
        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            
        Returns:
            Safety factor (0.5 to 1.0)
        """
        self.current_tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        
        if self.current_tilt_magnitude < self.warning_tilt_threshold:
            return 1.0  # Full speed OK
        elif self.current_tilt_magnitude < self.crash_tilt_threshold:
            # Linearly reduce from 1.0 to 0.5 as tilt approaches crash threshold
            ratio = (self.current_tilt_magnitude - self.warning_tilt_threshold) / \
                    (self.crash_tilt_threshold - self.warning_tilt_threshold)
            return 1.0 - 0.5 * ratio
        else:
            return 0.5  # Minimal movements
    
    def check_crash(self, roll, pitch, altitude):
        """
        Check for crash conditions and update crash state.
        
        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            altitude: Current altitude (meters)
            
        Returns:
            True if crashed, False otherwise
        """
        # Check if takeoff is complete
        if not self.takeoff_complete:
            if altitude > self.min_flight_altitude:
                self.takeoff_complete = True
                log(f"[SAFETY] ✓ Takeoff complete @ {altitude:.2f}m - Crash detection ENABLED", "SUCCESS")
            return False
        
        # Check tilt crash
        if abs(roll) > self.crash_tilt_threshold or abs(pitch) > self.crash_tilt_threshold:
            if not self.crashed:
                self._log_crash("TILT", roll, pitch, altitude)
                self.crashed = True
            return True
        
        # Check altitude crash
        if altitude < self.crash_altitude_threshold:
            if not self.crashed:
                self._log_crash("ALTITUDE", roll, pitch, altitude)
                self.crashed = True
            return True
        
        return False
    
    def _log_crash(self, crash_type, roll, pitch, altitude):
        """
        Log detailed crash information for debugging.
        
        Args:
            crash_type: Type of crash ("TILT" or "ALTITUDE")
            roll: Roll angle at crash (radians)
            pitch: Pitch angle at crash (radians)
            altitude: Altitude at crash (meters)
        """
        crash_info = {
            'time': datetime.now().isoformat(),
            'type': crash_type,
            'roll_deg': np.rad2deg(roll),
            'pitch_deg': np.rad2deg(pitch),
            'altitude': altitude,
            'position': self.last_position,
            'velocity': self.last_velocity,
            'commanded_velocity': self.last_commanded_velocity
        }
        self.crash_log.append(crash_info)
        
        # Use the new crash report logging
        crash_data = {
            'roll_deg': np.rad2deg(roll),
            'pitch_deg': np.rad2deg(pitch),
            'altitude': altitude,
            'position': self.last_position,
            'velocity': self.last_velocity,
            'commanded_velocity': self.last_commanded_velocity,
            'position_history': self.position_history.copy()
        }
        log_crash_report(crash_type, crash_data)
    
    def get_crash_log(self):
        """Return the list of crash events."""
        return self.crash_log
    
    def reset(self):
        """Reset safety monitor state."""
        self.crashed = False
        self.takeoff_complete = False
        self.current_tilt_magnitude = 0.0
        self.position_history = []
        self.last_position = None
        self.last_velocity = None
        self.last_commanded_velocity = None
        # Note: crash_log is NOT reset - keeps history across runs
