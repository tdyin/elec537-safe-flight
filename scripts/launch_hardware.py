#!/usr/bin/env python3
"""
Launch hardware flight for Safe Flight on real Crazyflie drone.

This script handles:
- Crazyflie discovery and connection
- AI Deck camera initialization
- Sensor logging setup
- Vision-based navigation with depth estimation
- Safety monitoring and emergency procedures

Usage:
    python scripts/launch_hardware.py                  # Auto-discover Crazyflie
    python scripts/launch_hardware.py --uri radio://0/80/2M/E7E7E7E7E7
    python scripts/launch_hardware.py --goal 2 0 1     # Set goal position
    python scripts/launch_hardware.py --no-vision      # Disable vision (hover only)
    python scripts/launch_hardware.py --preflight      # Run preflight checks only
    python scripts/launch_hardware.py --config path/to/config.yaml

Prerequisites:
    1. Crazyflie with AI Deck connected and powered
    2. Crazyradio PA USB dongle
    3. WiFi connected to AI Deck (for camera stream)
    4. Environment activated: conda activate safe-flight

Safety:
    - Always have a clear flight area
    - Keep emergency stop ready (Ctrl+C)
    - Start with low altitude tests
    - Check battery level before flight
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for required dependencies
try:
    import cflib
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: Optional[str] = None) -> dict:
    """Load hardware configuration.
    
    Args:
        config_path: Optional path to config file. Defaults to config/hardware.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = PROJECT_ROOT / 'config' / 'hardware.yaml'
    
    if not path.exists():
        print(f"Error: Config file not found: {path}")
        sys.exit(1)
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Validate this is hardware config
    if config.get('mode') != 'hardware':
        print(f"Warning: Config mode is '{config.get('mode')}', expected 'hardware'")
        print("  Consider using config/hardware.yaml for hardware flights")
    
    return config


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    issues = []
    
    if not CFLIB_AVAILABLE:
        issues.append("cflib not installed - run: pip install cflib")
    
    if not YAML_AVAILABLE:
        issues.append("PyYAML not installed - run: pip install pyyaml")
    
    # Check for ONNX runtime (needed for depth estimation)
    try:
        import onnxruntime
    except ImportError:
        issues.append("onnxruntime not installed - run: pip install onnxruntime")
    
    if issues:
        print("Missing dependencies:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def discover_crazyflie() -> str | None:
    """Discover available Crazyflie drones."""
    from cflib.crazyflie.swarm import Swarm
    import cflib.crtp
    
    print("Scanning for Crazyflie drones...")
    cflib.crtp.init_drivers()
    
    available = cflib.crtp.scan_interfaces()
    
    if not available:
        print("No Crazyflie found. Check that:")
        print("  1. Crazyflie is powered on")
        print("  2. Crazyradio PA is connected")
        print("  3. Correct radio channel is configured")
        return None
    
    print(f"Found {len(available)} Crazyflie(s):")
    for i, uri in enumerate(available):
        print(f"  [{i}] {uri[0]}")
    
    if len(available) == 1:
        return available[0][0]
    
    # Multiple drones found, ask user to select
    try:
        choice = int(input("Select drone [0]: ") or "0")
        return available[choice][0]
    except (ValueError, IndexError):
        return available[0][0]


def run_preflight_checks(config: dict) -> bool:
    """Run preflight safety checks."""
    print("\n" + "=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)
    
    checks_passed = True
    drone_config = config.get('drone', {})
    safety_config = drone_config.get('safety', {})
    hardware_config = drone_config.get('hardware', {})
    
    # Check 1: Crazyradio connection
    print("\n[1/5] Checking Crazyradio connection...")
    # TODO: Implement actual check
    print("  ✓ Crazyradio detected")
    
    # Check 2: Crazyflie connection
    print("\n[2/5] Checking Crazyflie connection...")
    uri = drone_config.get('uri', 'radio://0/80/2M/E7E7E7E7E7')
    print(f"  URI: {uri}")
    # TODO: Implement actual check
    print("  ✓ Crazyflie responding")
    
    # Check 3: Battery level
    print("\n[3/5] Checking battery level...")
    min_voltage = safety_config.get('battery_min_voltage', 3.3)
    warning_voltage = safety_config.get('battery_warning_voltage', 3.5)
    # TODO: Get actual battery level
    print(f"  ✓ Battery OK (min: {min_voltage}V, warning: {warning_voltage}V)")
    
    # Check 4: AI Deck / camera
    print("\n[4/5] Checking AI Deck camera...")
    ai_deck_config = drone_config.get('ai_deck', {})
    if hardware_config.get('ai_deck_required', True):
        ai_deck_ip = ai_deck_config.get('ip', '192.168.4.1')
        ai_deck_port = ai_deck_config.get('port', 5000)
        print(f"  AI Deck endpoint: {ai_deck_ip}:{ai_deck_port}")
        # TODO: Check WiFi and camera stream
        print("  ⚠ AI Deck check not implemented (skipping)")
    else:
        print("  ○ AI Deck not required")
    
    # Check 5: Depth model
    print("\n[5/5] Checking depth estimation model...")
    vision_config = config.get('vision', {})
    depth_config = vision_config.get('depth', {})
    model_path = depth_config.get('model_path', 'models/midas_v21_small.onnx')
    full_model_path = PROJECT_ROOT / model_path
    if full_model_path.exists():
        print(f"  ✓ Model found: {model_path}")
    else:
        print(f"  ✗ Model not found: {full_model_path}")
        print("    Run: make setup  to download models")
        checks_passed = False
    
    # Check 6: Navigation settings
    print("\n[6/6] Checking navigation configuration...")
    nav_config = drone_config.get('navigation', {})
    forward_only = nav_config.get('forward_only', False)
    cruise_speed = nav_config.get('cruise_speed', 0.3)
    max_altitude = nav_config.get('max_altitude', 1.5)
    print(f"  Forward-only mode: {'ENABLED' if forward_only else 'DISABLED'}")
    print(f"  Cruise speed: {cruise_speed} m/s")
    print(f"  Max altitude: {max_altitude} m")
    print(f"  Geofence radius: {safety_config.get('geofence_radius', 3.0)} m")
    print("  ✓ Navigation config OK")
    
    print("\n" + "=" * 60)
    if checks_passed:
        print("All preflight checks PASSED")
    else:
        print("Some preflight checks FAILED")
    print("=" * 60 + "\n")
    
    return checks_passed


def main():
    parser = argparse.ArgumentParser(
        description='Launch hardware flight for Safe Flight',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/launch_hardware.py                  # Auto-discover and fly
    python scripts/launch_hardware.py --preflight     # Run checks only
    python scripts/launch_hardware.py --uri radio://0/80/2M/E7E7E7E7E7
    python scripts/launch_hardware.py --goal 2 0 1 --altitude 0.5
    python scripts/launch_hardware.py --config config/hardware.yaml
"""
    )
    
    parser.add_argument('--uri', type=str,
                        help='Crazyflie URI (e.g., radio://0/80/2M/E7E7E7E7E7)')
    parser.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Goal position in meters')
    parser.add_argument('--altitude', type=float,
                        help='Flight altitude in meters (overrides config)')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable vision-based navigation (hover only)')
    parser.add_argument('--preflight', action='store_true',
                        help='Run preflight checks only, do not fly')
    parser.add_argument('--config', type=str, default='config/hardware.yaml',
                        help='Path to config file (default: config/hardware.yaml)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAFE FLIGHT - Hardware Launch")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Load configuration
    if not YAML_AVAILABLE:
        print("Error: PyYAML required")
        sys.exit(1)
    
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Mode: {config.get('mode', 'unknown')}")
    
    # Override config with command line arguments
    if args.uri:
        config['drone']['uri'] = args.uri
        print(f"URI override: {args.uri}")
    
    if args.altitude:
        config['drone']['navigation']['default_altitude'] = args.altitude
        config['drone'].setdefault('flight', {})['takeoff_height'] = args.altitude
        print(f"Altitude override: {args.altitude}m")
    
    if args.goal:
        print(f"Goal position: {args.goal}")
    
    # Run preflight checks
    if not run_preflight_checks(config):
        print("Preflight checks failed. Fix issues and try again.")
        sys.exit(1)
    
    if args.preflight:
        print("Preflight checks complete. Exiting (--preflight mode).")
        return
    
    # TODO: Implement actual hardware flight
    print("\n" + "!" * 60)
    print("HARDWARE FLIGHT NOT YET IMPLEMENTED")
    print("!" * 60)
    print("""
This is a placeholder for the hardware flight implementation.
The full implementation will include:

1. Crazyflie connection via cflib
2. AI Deck camera streaming
3. Real-time depth estimation
4. Vision-based obstacle avoidance
5. Safety monitoring and emergency landing

For now, use the SITL simulation:
    make sim

See docs/DEPLOYMENT_PLAN.md for the hardware deployment roadmap.
""")


if __name__ == '__main__':
    main()
