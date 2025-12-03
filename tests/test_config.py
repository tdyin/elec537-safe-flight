"""Tests for the configuration module."""

import pytest
from pathlib import Path

from src.core.config import (
    load_config,
    get_config_path,
    get_project_root,
    get_nested,
    is_hardware_mode,
    is_simulation_mode,
    validate_hardware_config,
    validate_simulation_config,
)


class TestConfigPaths:
    """Tests for configuration path utilities."""
    
    def test_get_project_root(self):
        """Test that project root is correctly identified."""
        root = get_project_root()
        assert root.exists()
        assert (root / 'config').exists()
        assert (root / 'src').exists()
    
    def test_get_config_path_sim(self):
        """Test getting simulation config path."""
        path = get_config_path('sim')
        assert path.name == 'sim.yaml'
        assert path.exists()
    
    def test_get_config_path_hardware(self):
        """Test getting hardware config path."""
        path = get_config_path('hardware')
        assert path.name == 'hardware.yaml'
        assert path.exists()
    
    def test_get_config_path_invalid(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            get_config_path('invalid')


class TestLoadConfig:
    """Tests for loading configuration files."""
    
    def test_load_sim_config(self):
        """Test loading simulation configuration."""
        config = load_config('config/sim.yaml')
        assert config is not None
        assert config['mode'] == 'simulation'
        assert 'vision' in config
        assert 'drone' in config
    
    def test_load_hardware_config(self):
        """Test loading hardware configuration."""
        config = load_config('config/hardware.yaml')
        assert config is not None
        assert config['mode'] == 'hardware'
        assert 'vision' in config
        assert 'drone' in config
    
    def test_load_default_config(self):
        """Test loading default configuration (sim.yaml)."""
        config = load_config()
        assert config is not None
        assert config['mode'] == 'simulation'
    
    def test_load_nonexistent_config(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent.yaml')
    
    def test_config_has_defaults_applied(self):
        """Test that defaults are applied to loaded config."""
        config = load_config()
        # Check that nested defaults exist
        assert 'depth' in config['vision']
        assert 'model_path' in config['vision']['depth']
        assert 'navigation' in config['drone']


class TestGetNested:
    """Tests for nested config value access."""
    
    def test_get_nested_existing(self):
        """Test getting existing nested value."""
        config = {'drone': {'navigation': {'cruise_speed': 0.5}}}
        result = get_nested(config, 'drone', 'navigation', 'cruise_speed')
        assert result == 0.5
    
    def test_get_nested_missing(self):
        """Test getting missing nested value returns default."""
        config = {'drone': {}}
        result = get_nested(config, 'drone', 'navigation', 'cruise_speed', default=1.0)
        assert result == 1.0
    
    def test_get_nested_partial_path(self):
        """Test getting value when only partial path exists."""
        config = {'drone': {'navigation': {}}}
        result = get_nested(config, 'drone', 'navigation', 'missing', default='default')
        assert result == 'default'


class TestModeDetection:
    """Tests for configuration mode detection."""
    
    def test_is_hardware_mode(self):
        """Test hardware mode detection."""
        config = {'mode': 'hardware'}
        assert is_hardware_mode(config)
        assert not is_simulation_mode(config)
    
    def test_is_simulation_mode(self):
        """Test simulation mode detection."""
        config = {'mode': 'simulation'}
        assert is_simulation_mode(config)
        assert not is_hardware_mode(config)
    
    def test_is_sim_mode_shorthand(self):
        """Test 'sim' shorthand is recognized as simulation."""
        config = {'mode': 'sim'}
        assert is_simulation_mode(config)


class TestValidation:
    """Tests for configuration validation."""
    
    def test_validate_hardware_config_valid(self):
        """Test validating a complete hardware config."""
        config = load_config('config/hardware.yaml')
        issues = validate_hardware_config(config)
        # URI should be present in hardware.yaml
        assert 'Missing drone.uri' not in issues
    
    def test_validate_simulation_config_valid(self):
        """Test validating a complete simulation config."""
        config = load_config('config/sim.yaml')
        issues = validate_simulation_config(config)
        # Host and port should be present
        assert 'Missing drone.simulation.host' not in issues
        assert 'Missing drone.simulation.port' not in issues
    
    def test_validate_hardware_wrong_mode(self):
        """Test that validation fails for wrong mode."""
        config = {'mode': 'simulation'}
        issues = validate_hardware_config(config)
        assert any('not' in issue and 'hardware' in issue for issue in issues)
    
    def test_validate_simulation_wrong_mode(self):
        """Test that validation fails for wrong mode."""
        config = {'mode': 'hardware'}
        issues = validate_simulation_config(config)
        assert any('not' in issue and 'simulation' in issue for issue in issues)


class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    def test_sim_config_has_expected_structure(self):
        """Test simulation config has all expected sections."""
        config = load_config('config/sim.yaml')
        
        # Top-level sections
        assert 'mode' in config
        assert 'vision' in config
        assert 'drone' in config
        assert 'fusion' in config
        assert 'logging' in config
        
        # Vision subsections
        assert 'depth' in config['vision']
        assert 'obstacle' in config['vision']
        
        # Drone subsections
        assert 'interface' in config['drone']
        assert 'navigation' in config['drone']
        assert 'simulation' in config['drone']
    
    def test_hardware_config_has_expected_structure(self):
        """Test hardware config has all expected sections."""
        config = load_config('config/hardware.yaml')
        
        # Top-level sections
        assert config['mode'] == 'hardware'
        
        # Hardware-specific sections
        assert 'uri' in config['drone']
        assert 'hardware' in config['drone']
        assert 'ai_deck' in config['drone']
        assert 'safety' in config['drone']
        assert 'flight' in config['drone']
        
        # Safety parameters
        safety = config['drone']['safety']
        assert 'battery_min_voltage' in safety
        assert 'geofence_radius' in safety
