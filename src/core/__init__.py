"""Core module containing shared abstractions for drone interfaces."""

from .base_interface import DroneInterface
from .types import SensorData, Position, Orientation, Velocity, SafetyState
from .safety import SafetyMonitor
from .config import (
    load_config,
    get_config_path,
    get_project_root,
    get_nested,
    is_hardware_mode,
    is_simulation_mode,
    validate_hardware_config,
    validate_simulation_config,
)

__all__ = [
    # Interface
    "DroneInterface",
    # Types
    "SensorData",
    "Position",
    "Orientation",
    "Velocity",
    "SafetyState",
    # Safety
    "SafetyMonitor",
    # Config
    "load_config",
    "get_config_path",
    "get_project_root",
    "get_nested",
    "is_hardware_mode",
    "is_simulation_mode",
    "validate_hardware_config",
    "validate_simulation_config",
]
