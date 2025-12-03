"""Configuration loader and validation for Safe Flight.

This module provides a centralized way to load and validate configuration
from YAML files for both simulation and hardware modes.

Usage:
    from src.core.config import load_config, get_config_path
    
    # Load default config based on mode
    config = load_config()  # Loads config/sim.yaml by default
    
    # Load specific config
    config = load_config('config/hardware.yaml')
    
    # Get config path
    path = get_config_path('hardware')  # Returns Path to hardware.yaml
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root (elec537-safe-flight/)
    """
    # This file is in src/core/, so go up 2 levels
    return Path(__file__).parent.parent.parent


def get_config_path(mode: str = 'sim') -> Path:
    """Get the path to a configuration file.
    
    Args:
        mode: Configuration mode ('sim' or 'hardware')
        
    Returns:
        Path to the configuration file
        
    Raises:
        ValueError: If mode is not 'sim' or 'hardware'
    """
    if mode not in ('sim', 'hardware', 'simulation'):
        raise ValueError(f"Invalid mode: {mode}. Use 'sim' or 'hardware'")
    
    if mode == 'simulation':
        mode = 'sim'
    
    config_name = f"{mode}.yaml"
    return get_project_root() / 'config' / config_name


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to config file. If None, loads config/sim.yaml.
                    Can be a string path or Path object.
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        config_path = get_config_path('sim')
    elif isinstance(config_path, str):
        # Handle relative paths
        path = Path(config_path)
        if not path.is_absolute():
            config_path = get_project_root() / path
        else:
            config_path = path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and apply defaults
    config = _apply_defaults(config)
    
    return config


def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to configuration.
    
    Ensures all expected keys exist with sensible defaults.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Configuration with defaults applied
    """
    # Ensure mode is set
    if 'mode' not in config:
        config['mode'] = 'simulation'
    
    # Vision defaults
    if 'vision' not in config:
        config['vision'] = {}
    
    vision = config['vision']
    vision.setdefault('mode', 'depth')
    vision.setdefault('depth', {})
    vision['depth'].setdefault('model_path', 'models/midas_v21_small.onnx')
    vision['depth'].setdefault('use_gpu', False)
    vision['depth'].setdefault('depth_scale', 1.0)
    
    # Obstacle detection defaults
    vision.setdefault('obstacle', {})
    vision['obstacle'].setdefault('critical_threshold', 0.15)
    vision['obstacle'].setdefault('close_threshold', 0.25)
    vision['obstacle'].setdefault('caution_threshold', 0.40)
    
    # Drone defaults
    if 'drone' not in config:
        config['drone'] = {}
    
    drone = config['drone']
    drone.setdefault('interface', 'webots')
    
    # Navigation defaults
    drone.setdefault('navigation', {})
    nav = drone['navigation']
    nav.setdefault('cruise_speed', 0.5)
    nav.setdefault('avoidance_speed', 0.3)
    nav.setdefault('max_speed', 1.0)
    nav.setdefault('safety_distance', 1.0)
    nav.setdefault('max_altitude', 2.0)
    nav.setdefault('min_altitude', 0.3)
    nav.setdefault('default_altitude', 0.5)
    
    # Safety defaults (for hardware mode)
    drone.setdefault('safety', {})
    safety = drone['safety']
    safety.setdefault('battery_min_voltage', 3.3)
    safety.setdefault('battery_warning_voltage', 3.5)
    safety.setdefault('geofence_radius', 3.0)
    safety.setdefault('geofence_height', 2.0)
    safety.setdefault('crash_tilt_threshold', 40.0)
    safety.setdefault('crash_altitude_threshold', 0.15)
    
    # Logging defaults
    if 'logging' not in config:
        config['logging'] = {}
    
    logging = config['logging']
    logging.setdefault('level', 'INFO')
    logging.setdefault('log_to_file', True)
    
    # Set appropriate log directory based on mode
    if config['mode'] == 'hardware':
        logging.setdefault('log_directory', 'logs/hardware')
    else:
        logging.setdefault('log_directory', 'sim/webots/logs')
    
    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested configuration value.
    
    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key path doesn't exist
        
    Returns:
        Value at the nested key path, or default if not found
        
    Example:
        >>> config = {'drone': {'navigation': {'cruise_speed': 0.5}}}
        >>> get_nested(config, 'drone', 'navigation', 'cruise_speed')
        0.5
        >>> get_nested(config, 'drone', 'missing', 'key', default=1.0)
        1.0
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def is_hardware_mode(config: Dict[str, Any]) -> bool:
    """Check if configuration is for hardware mode.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if hardware mode, False otherwise
    """
    return config.get('mode') == 'hardware'


def is_simulation_mode(config: Dict[str, Any]) -> bool:
    """Check if configuration is for simulation mode.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if simulation mode, False otherwise
    """
    return config.get('mode') in ('simulation', 'sim')


def validate_hardware_config(config: Dict[str, Any]) -> list[str]:
    """Validate hardware configuration and return any issues.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if config.get('mode') != 'hardware':
        issues.append("Configuration mode is not 'hardware'")
        return issues
    
    drone = config.get('drone', {})
    
    # Check required hardware settings
    if 'uri' not in drone:
        issues.append("Missing drone.uri (Crazyflie radio URI)")
    
    # Check AI deck settings if required
    hardware = drone.get('hardware', {})
    if hardware.get('ai_deck_required', False):
        ai_deck = drone.get('ai_deck', {})
        if not ai_deck.get('ip'):
            issues.append("AI Deck required but drone.ai_deck.ip not set")
    
    # Check model paths
    vision = config.get('vision', {})
    depth = vision.get('depth', {})
    model_path = depth.get('model_path')
    if model_path:
        full_path = get_project_root() / model_path
        if not full_path.exists():
            issues.append(f"Depth model not found: {model_path}")
    
    return issues


def validate_simulation_config(config: Dict[str, Any]) -> list[str]:
    """Validate simulation configuration and return any issues.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if config.get('mode') not in ('simulation', 'sim'):
        issues.append("Configuration mode is not 'simulation'")
        return issues
    
    drone = config.get('drone', {})
    
    # Check simulation settings
    sim = drone.get('simulation', {})
    if not sim.get('host'):
        issues.append("Missing drone.simulation.host")
    if not sim.get('port'):
        issues.append("Missing drone.simulation.port")
    
    # Check world file
    world_file = sim.get('world_file')
    if world_file:
        full_path = get_project_root() / world_file
        if not full_path.exists():
            issues.append(f"World file not found: {world_file}")
    
    return issues
