"""
Unified logging utility for all Webots controllers.

Provides consistent logging across all controller types with both
console output and optional file logging. Includes crash report functionality.
"""

import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class ControllerLogger:
    """Thread-safe logger for Webots controllers."""
    
    def __init__(self, controller_name: str, log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            controller_name: Name of the controller (for log prefix)
            log_file: Optional path to log file
        """
        self.controller_name = controller_name
        self.log_file = log_file
        self.file_handle = None
        self.crash_reports: List[Dict[str, Any]] = []
        
        if log_file:
            try:
                log_dir = Path(log_file).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                self.file_handle = open(log_file, 'a', buffering=1)  # Line buffered
            except Exception as e:
                print(f"⚠ Failed to open log file {log_file}: {e}")
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp and level.
        
        Args:
            message: Message to log
            level: Log level (INFO, SUCCESS, ERROR, WARNING, CRASH, DEBUG)
        """
        timestamp = time.strftime("%H:%M:%S")
        
        # Symbols for different log levels
        symbols = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "ERROR": "✗",
            "WARNING": "⚠",
            "CRASH": "💥",
            "DEBUG": "🔍",
        }
        symbol = symbols.get(level, "•")
        
        # Format log line
        log_line = f"[{timestamp}] {symbol} {self.controller_name}: {message}"
        
        # Print to console (always goes to Webots console)
        print(log_line)
        sys.stdout.flush()
        
        # Write to file if enabled
        if self.file_handle:
            try:
                self.file_handle.write(log_line + "\n")
                self.file_handle.flush()
            except Exception as e:
                print(f"⚠ Failed to write to log file: {e}")
    
    def log_crash_report(self, crash_type: str, crash_data: Dict[str, Any]):
        """
        Log a detailed crash report with full context.
        
        Args:
            crash_type: Type of crash ("TILT", "ALTITUDE", "COLLISION", etc.)
            crash_data: Dictionary containing crash details:
                - roll_deg: Roll angle in degrees
                - pitch_deg: Pitch angle in degrees  
                - altitude: Altitude in meters
                - position: (x, y, z) tuple or None
                - velocity: (vx, vy) tuple or None
                - commanded_velocity: (vx, vy) tuple or None
                - position_history: List of recent positions (optional)
                - avoidance_state: Current avoidance state (optional)
                - emergency_count: Number of emergencies (optional)
        """
        timestamp = datetime.now()
        
        # Store crash report
        crash_report = {
            'timestamp': timestamp.isoformat(),
            'type': crash_type,
            **crash_data
        }
        self.crash_reports.append(crash_report)
        
        # Format position
        pos = crash_data.get('position')
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos else "unknown"
        
        # Format velocity
        vel = crash_data.get('velocity')
        vel_str = f"({vel[0]:.2f}, {vel[1]:.2f})" if vel else "unknown"
        
        # Format commanded velocity
        cmd_vel = crash_data.get('commanded_velocity')
        cmd_str = f"({cmd_vel[0]:.2f}, {cmd_vel[1]:.2f})" if cmd_vel else "unknown"
        
        # Build crash report
        separator = "=" * 70
        report_lines = [
            "",
            separator,
            f"💥 CRASH REPORT - {crash_type}",
            separator,
            f"Time:     {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            f"Position: {pos_str}",
            f"Roll:     {crash_data.get('roll_deg', 0):.1f}°",
            f"Pitch:    {crash_data.get('pitch_deg', 0):.1f}°",
            f"Altitude: {crash_data.get('altitude', 0):.2f}m",
            f"Velocity: {vel_str}",
            f"Cmd Vel:  {cmd_str}",
        ]
        
        # Add avoidance state if available
        if 'avoidance_state' in crash_data:
            report_lines.append(f"Avoidance State: {crash_data['avoidance_state']}")
        
        # Add emergency count if available
        if 'emergency_count' in crash_data:
            report_lines.append(f"Emergency Count: {crash_data['emergency_count']}")
        
        # Add position history if available
        history = crash_data.get('position_history', [])
        if history:
            report_lines.append("")
            report_lines.append("Position History (last 5):")
            for entry in history[-5:]:
                entry_pos = entry.get('position')
                if entry_pos:
                    entry_time = entry.get('time', datetime.now())
                    if isinstance(entry_time, datetime):
                        time_str = entry_time.strftime('%H:%M:%S.%f')[:-3]
                    else:
                        time_str = str(entry_time)
                    report_lines.append(f"  [{time_str}] ({entry_pos[0]:.2f}, {entry_pos[1]:.2f}, {entry_pos[2]:.2f})")
        
        report_lines.append(separator)
        report_lines.append("")
        
        # Log each line
        for line in report_lines:
            self.log(line, "CRASH")
    
    def get_crash_reports(self) -> List[Dict[str, Any]]:
        """Return list of all crash reports."""
        return self.crash_reports.copy()
    
    def close(self):
        """Close log file handle."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global logger instance
_global_logger: Optional[ControllerLogger] = None


def setup_logger(controller_name: str, log_file: Optional[str] = None) -> ControllerLogger:
    """
    Setup global logger for a controller.
    
    Args:
        controller_name: Name of the controller
        log_file: Optional path to log file
        
    Returns:
        ControllerLogger instance
    """
    global _global_logger
    _global_logger = ControllerLogger(controller_name, log_file)
    return _global_logger


def log(message: str, level: str = "INFO"):
    """
    Log a message using the global logger.
    
    If no global logger is set up, logs directly to console.
    
    Args:
        message: Message to log
        level: Log level (INFO, SUCCESS, ERROR, WARNING, CRASH, DEBUG)
    """
    if _global_logger:
        _global_logger.log(message, level)
    else:
        # Fallback to simple console logging
        timestamp = time.strftime("%H:%M:%S")
        symbols = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "ERROR": "✗",
            "WARNING": "⚠",
            "CRASH": "💥",
            "DEBUG": "🔍",
        }
        symbol = symbols.get(level, "•")
        print(f"[{timestamp}] {symbol} {message}")
        sys.stdout.flush()


def log_crash_report(crash_type: str, crash_data: Dict[str, Any]):
    """
    Log a detailed crash report using the global logger.
    
    Args:
        crash_type: Type of crash ("TILT", "ALTITUDE", "COLLISION", etc.)
        crash_data: Dictionary containing crash details
    """
    if _global_logger:
        _global_logger.log_crash_report(crash_type, crash_data)
    else:
        # Fallback - just print the crash info
        print(f"💥 CRASH: {crash_type}")
        for key, value in crash_data.items():
            print(f"  {key}: {value}")
        sys.stdout.flush()


def get_crash_reports() -> List[Dict[str, Any]]:
    """Get all crash reports from the global logger."""
    if _global_logger:
        return _global_logger.get_crash_reports()
    return []
