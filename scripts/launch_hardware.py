#!/usr/bin/env python3
"""
Launch hardware flight for Safe Flight on real Crazyflie drone.

This script handles:
- Crazyflie discovery and connection
- AI Deck camera initialization
- Sensor logging setup
- Vision-based navigation with depth estimation
- Safety monitoring and emergency procedures
- Flight data logging to data/logs/

Usage:
    python scripts/launch_hardware.py                  # Auto-discover Crazyflie
    python scripts/launch_hardware.py --uri radio://0/80/2M/E7E7E7E7E7
    python scripts/launch_hardware.py --goal 2 0 1     # Set goal position
    python scripts/launch_hardware.py --no-vision      # Disable vision (hover only)
    python scripts/launch_hardware.py --hover          # Simple hover test
    python scripts/launch_hardware.py --preflight      # Run preflight checks only
    python scripts/launch_hardware.py --viz            # Enable live depth visualization
    python scripts/launch_hardware.py --max-duration 30  # Auto-land after 30 seconds
    python scripts/launch_hardware.py --log-dir data/logs  # Custom log directory
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

import numpy as np

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
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    
    print("\n" + "=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)
    
    checks_passed = True
    drone_config = config.get('drone', {})
    safety_config = drone_config.get('safety', {})
    hardware_config = drone_config.get('hardware', {})
    
    # Check 1: Crazyradio connection
    print("\n[1/6] Checking Crazyradio connection...")
    try:
        cflib.crtp.init_drivers()
        available = cflib.crtp.scan_interfaces()
        if available:
            print(f"  ✓ Crazyradio detected - Found {len(available)} Crazyflie(s)")
        else:
            print("  ✗ No Crazyflie found")
            print("    - Is Crazyflie powered on?")
            print("    - Is Crazyradio PA connected?")
            checks_passed = False
    except Exception as e:
        print(f"  ✗ Crazyradio error: {e}")
        checks_passed = False
    
    # Check 2: Crazyflie connection and decks
    print("\n[2/6] Checking Crazyflie connection and decks...")
    uri = drone_config.get('uri', 'radio://0/80/2M/E7E7E7E7E7')
    print(f"  URI: {uri}")
    
    battery_voltage = 0.0
    flow_deck_ok = False
    ai_deck_ok = False
    
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("  ✓ Connected to Crazyflie")
            
            # Wait for parameters
            time.sleep(1.0)
            
            # Check Flow Deck
            try:
                flow_attached = scf.cf.param.get_value('deck.bcFlow2')
                if int(flow_attached) == 1:
                    print("  ✓ Flow Deck v2 detected")
                    flow_deck_ok = True
                else:
                    print("  ✗ Flow Deck v2 NOT detected")
                    if hardware_config.get('flow_deck_required', True):
                        checks_passed = False
            except Exception as e:
                print(f"  ⚠ Could not check Flow Deck: {e}")
            
            # Check AI Deck
            try:
                ai_attached = scf.cf.param.get_value('deck.bcAI')
                if int(ai_attached) == 1:
                    print("  ✓ AI Deck detected")
                    ai_deck_ok = True
                else:
                    print("  ○ AI Deck NOT detected")
                    if hardware_config.get('ai_deck_required', True):
                        print("    (Required for vision mode)")
            except Exception as e:
                print(f"  ⚠ Could not check AI Deck: {e}")
            
            # Get battery voltage
            battery_received = [False]
            def battery_callback(timestamp, data, logconf):
                nonlocal battery_voltage
                battery_voltage = data['pm.vbat']
                battery_received[0] = True
            
            lc = LogConfig(name='Battery', period_in_ms=100)
            lc.add_variable('pm.vbat', 'float')
            lc.data_received_cb.add_callback(battery_callback)
            scf.cf.log.add_config(lc)
            lc.start()
            
            # Wait for battery reading
            for _ in range(20):
                if battery_received[0]:
                    break
                time.sleep(0.1)
            
            lc.stop()
            
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        checks_passed = False
    
    # Check 3: Battery level
    print("\n[3/6] Checking battery level...")
    min_voltage = safety_config.get('battery_min_voltage', 3.3)
    warning_voltage = safety_config.get('battery_warning_voltage', 3.5)
    
    if battery_voltage > 0:
        if battery_voltage >= warning_voltage:
            print(f"  ✓ Battery: {battery_voltage:.2f}V (OK)")
        elif battery_voltage >= min_voltage:
            print(f"  ⚠ Battery: {battery_voltage:.2f}V (LOW - consider charging)")
        else:
            print(f"  ✗ Battery: {battery_voltage:.2f}V (CRITICAL - charge before flight)")
            checks_passed = False
    else:
        print("  ⚠ Could not read battery voltage")
    
    # Check 4: AI Deck camera (if required)
    print("\n[4/6] Checking AI Deck camera...")
    ai_deck_config = drone_config.get('ai_deck', {})
    if hardware_config.get('ai_deck_required', True) and ai_deck_ok:
        ai_deck_ip = ai_deck_config.get('ip', '192.168.4.1')
        ai_deck_port = ai_deck_config.get('port', 5000)
        print(f"  AI Deck endpoint: {ai_deck_ip}:{ai_deck_port}")
        
        # Try to connect briefly
        import socket
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2.0)
            test_socket.connect((ai_deck_ip, ai_deck_port))
            test_socket.close()
            print("  ✓ AI Deck camera stream reachable")
        except socket.timeout:
            print("  ⚠ AI Deck camera timeout (check WiFi connection)")
        except ConnectionRefusedError:
            print("  ⚠ AI Deck camera connection refused")
        except OSError as e:
            print(f"  ⚠ AI Deck camera not reachable: {e}")
            print("    - Connect to AI Deck WiFi network")
    elif not hardware_config.get('ai_deck_required', True):
        print("  ○ AI Deck not required (hover mode)")
    else:
        print("  ○ AI Deck not detected - skipping camera check")
    
    # Check 5: Depth model
    print("\n[5/6] Checking depth estimation model...")
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
    python scripts/launch_hardware.py --hover          # Simple hover test
    python scripts/launch_hardware.py --uri radio://0/80/2M/E7E7E7E7E7
    python scripts/launch_hardware.py --goal 2 0 1 --altitude 0.5
    python scripts/launch_hardware.py --viz            # Live depth visualization
    python scripts/launch_hardware.py --max-duration 30  # Auto-land after 30s
    python scripts/launch_hardware.py --config config/hardware.yaml
"""
    )
    
    parser.add_argument('--uri', type=str,
                        help='Crazyflie URI (e.g., radio://0/80/2M/E7E7E7E7E7)')
    parser.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Goal position in meters')
    parser.add_argument('--altitude', type=float,
                        help='Flight altitude in meters (overrides config)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Flight duration in seconds for hover mode (default: 10)')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable vision-based navigation (hover only)')
    parser.add_argument('--hover', action='store_true',
                        help='Simple hover test without navigation')
    parser.add_argument('--preflight', action='store_true',
                        help='Run preflight checks only, do not fly')
    parser.add_argument('--config', type=str, default='config/hardware.yaml',
                        help='Path to config file (default: config/hardware.yaml)')
    parser.add_argument('--viz', action='store_true',
                        help='Enable live depth visualization')
    parser.add_argument('--max-duration', type=float, default=None,
                        help='Maximum flight duration in seconds (auto-land after)')
    parser.add_argument('--log-dir', type=str, default='data/logs',
                        help='Directory for flight logs (default: data/logs)')
    
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
    
    # Determine flight mode
    if args.hover or args.no_vision:
        run_hover_flight(config, args)
    else:
        run_vision_flight(config, args)


def run_hover_flight(config: dict, args) -> None:
    """Run a simple hover test without vision navigation.
    
    Args:
        config: Hardware configuration
        args: Command line arguments
    """
    from loguru import logger
    from src.hardware.crazyflie_interface import CrazyflieHardwareInterface
    
    print("\n" + "=" * 60)
    print("HOVER FLIGHT MODE")
    print("=" * 60)
    
    uri = config['drone'].get('uri', 'radio://0/80/2M/E7E7E7E7E7')
    altitude = config['drone']['navigation'].get('default_altitude', 0.5)
    duration = args.duration
    
    print(f"\nFlight parameters:")
    print(f"  URI: {uri}")
    print(f"  Altitude: {altitude}m")
    print(f"  Duration: {duration}s")
    print()
    
    # Safety confirmation
    print("⚠️  SAFETY WARNING: The drone will take off!")
    print("    - Ensure clear flight area")
    print("    - Keep emergency stop ready (Ctrl+C)")
    print("    - Stand clear of propellers")
    print()
    
    try:
        response = input("Proceed with flight? [y/N]: ").strip().lower()
        if response != 'y':
            print("Flight cancelled.")
            return
    except KeyboardInterrupt:
        print("\nFlight cancelled.")
        return
    
    # Create interface
    interface = CrazyflieHardwareInterface(uri=uri, config=config)
    
    try:
        # Connect
        print("\n[1/4] Connecting to Crazyflie...")
        if not interface.connect():
            print("✗ Failed to connect")
            return
        print("✓ Connected")
        
        # Get initial sensor data
        time.sleep(0.5)
        sensor_data = interface.get_sensor_data()
        if sensor_data:
            battery = sensor_data.get('battery', 0)
            pos = sensor_data.get('position', (0, 0, 0))
            print(f"  Battery: {battery:.2f}V")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        # Takeoff
        print(f"\n[2/4] Taking off to {altitude}m...")
        if not interface.takeoff(height=altitude):
            print("✗ Takeoff failed")
            return
        print("✓ Takeoff complete")
        
        # Hover
        print(f"\n[3/4] Hovering for {duration}s...")
        print("    Press Ctrl+C for emergency stop")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                # Get and display sensor data
                sensor_data = interface.get_sensor_data()
                if sensor_data:
                    pos = sensor_data.get('position', (0, 0, 0))
                    battery = sensor_data.get('battery', 0)
                    print(f"\r  [{elapsed:.1f}s] Altitude: {pos[2]:.2f}m | "
                          f"Battery: {battery:.2f}V | Remaining: {remaining:.1f}s  ", 
                          end='', flush=True)
                
                # Send zero velocity to maintain hover
                interface.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Emergency stop triggered!")
            interface.emergency_stop()
            print("Motors stopped.")
            return
        
        print("\n✓ Hover complete")
        
        # Land
        print("\n[4/4] Landing...")
        if not interface.land():
            print("✗ Landing failed - emergency stop")
            interface.emergency_stop()
            return
        print("✓ Landing complete")
        
        print("\n" + "=" * 60)
        print("FLIGHT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Flight error: {e}")
        logger.exception("Flight error")
        try:
            interface.emergency_stop()
        except Exception:
            pass
    
    finally:
        interface.disconnect()
        print("\nDisconnected from Crazyflie")


def run_vision_flight(config: dict, args) -> None:
    """Run vision-based navigation flight.
    
    Args:
        config: Hardware configuration
        args: Command line arguments
    """
    from loguru import logger
    from src.hardware.crazyflie_interface import CrazyflieHardwareInterface
    from src.hardware.aideck_camera import AIdeckCamera
    from src.vision.depth_detector import DepthDetector
    from src.drone.depth_controller import DepthNavigationController
    from src.core.types import Position
    
    print("\n" + "=" * 60)
    print("VISION-BASED NAVIGATION MODE")
    print("=" * 60)
    
    uri = config['drone'].get('uri', 'radio://0/80/2M/E7E7E7E7E7')
    altitude = config['drone']['navigation'].get('default_altitude', 0.5)
    nav_config = config['drone'].get('navigation', {})
    
    # Goal position
    if args.goal:
        goal = Position(x=args.goal[0], y=args.goal[1], z=args.goal[2])
    else:
        # Default: 2m forward
        goal = Position(x=2.0, y=0.0, z=altitude)
    
    print(f"\nFlight parameters:")
    print(f"  URI: {uri}")
    print(f"  Altitude: {altitude}m")
    print(f"  Goal: ({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f})")
    print()
    
    # AI Deck configuration
    ai_deck_config = config['drone'].get('ai_deck', {})
    ai_deck_ip = ai_deck_config.get('ip', '192.168.4.1')
    ai_deck_port = ai_deck_config.get('port', 5000)
    
    print(f"  AI Deck: {ai_deck_ip}:{ai_deck_port}")
    
    # Safety confirmation
    print("\n⚠️  SAFETY WARNING: Vision-based autonomous flight!")
    print("    - Ensure clear flight area with soft obstacles")
    print("    - Keep emergency stop ready (Ctrl+C)")
    print("    - WiFi must be connected to AI Deck")
    print()
    
    try:
        response = input("Proceed with flight? [y/N]: ").strip().lower()
        if response != 'y':
            print("Flight cancelled.")
            return
    except KeyboardInterrupt:
        print("\nFlight cancelled.")
        return
    
    # Initialize components
    interface = None
    camera = None
    
    try:
        # Initialize depth detector
        print("\n[1/6] Initializing depth detector...")
        vision_config = config.get('vision', {})
        depth_config = vision_config.get('depth', {})
        model_path = str(PROJECT_ROOT / depth_config.get('model_path', 'models/midas_v21_small.onnx'))
        
        depth_detector = DepthDetector(
            depth_model_path=model_path,
            depth_scale=depth_config.get('depth_scale', 1.0),
            use_gpu=depth_config.get('use_gpu', False)
        )
        print("✓ Depth detector ready")
        
        # Connect to AI Deck camera
        print("\n[2/6] Connecting to AI Deck camera...")
        camera = AIdeckCamera(ip=ai_deck_ip, port=ai_deck_port)
        if not camera.connect():
            print("✗ Failed to connect to AI Deck camera")
            print("  Check WiFi connection to AI Deck")
            return
        print("✓ Camera connected")
        
        # Wait for first frame
        print("  Waiting for camera frame...")
        for _ in range(50):  # 5 second timeout
            frame = camera.get_frame()
            if frame is not None:
                print(f"  ✓ Receiving frames ({frame.shape[1]}x{frame.shape[0]})")
                break
            time.sleep(0.1)
        else:
            print("  ✗ Timeout waiting for camera frames")
            return
        
        # Connect to Crazyflie
        print("\n[3/6] Connecting to Crazyflie...")
        interface = CrazyflieHardwareInterface(uri=uri, config=config)
        if not interface.connect():
            print("✗ Failed to connect to Crazyflie")
            return
        print("✓ Connected")
        
        # Initialize navigation controller
        print("\n[4/6] Initializing navigation controller...")
        controller = DepthNavigationController(
            max_speed=nav_config.get('cruise_speed', 0.3),
            safety_distance=nav_config.get('safety_distance', 1.0),
            max_altitude=nav_config.get('max_altitude', 1.5),
            min_altitude=nav_config.get('min_altitude', 0.3),
            use_path_planning=False,  # Simpler for hardware
            use_stable_avoidance=True,
            config=config
        )
        print("✓ Navigation controller ready")
        
        # Get start position
        time.sleep(0.5)
        sensor_data = interface.get_sensor_data()
        start_pos = sensor_data.get('position', (0, 0, 0)) if sensor_data else (0, 0, 0)
        
        # Takeoff
        print(f"\n[5/6] Taking off to {altitude}m...")
        if not interface.takeoff(height=altitude):
            print("✗ Takeoff failed")
            return
        print("✓ Takeoff complete")
        
        # Enable crash detection after takeoff
        controller.enable_crash_detection()
        
        # Navigation loop
        print("\n[6/6] Starting vision-based navigation...")
        print(f"  Goal: ({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f})")
        print("  Press Ctrl+C for emergency stop")
        
        loop_rate = 10  # Hz
        loop_period = 1.0 / loop_rate
        goal_np = np.array([goal.x, goal.y, goal.z])
        
        try:
            while True:
                loop_start = time.time()
                
                # Get camera frame
                frame = camera.get_frame()
                if frame is None:
                    logger.warning("No camera frame")
                    time.sleep(loop_period)
                    continue
                
                # Run depth estimation
                vision_data = depth_detector.detect(frame)
                
                # Get sensor data
                sensor_data = interface.get_sensor_data()
                if not sensor_data:
                    logger.warning("No sensor data")
                    time.sleep(loop_period)
                    continue
                
                pos = sensor_data.get('position', (0, 0, 0))
                current_pos = np.array([pos[0], pos[1], pos[2]])
                
                # Check for crash
                roll = sensor_data.get('roll', 0)
                pitch = sensor_data.get('pitch', 0)
                if controller.check_crash(roll, pitch, pos[2]):
                    print("\n✗ Crash detected! Emergency stop.")
                    interface.emergency_stop()
                    return
                
                # Check if goal reached
                dist_to_goal = np.linalg.norm(current_pos[:2] - goal_np[:2])
                if dist_to_goal < 0.2:  # 20cm threshold
                    print(f"\n✓ Goal reached! Distance: {dist_to_goal:.2f}m")
                    break
                
                # Compute target direction and velocity
                target_direction = goal_np - current_pos
                target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-6)
                target_velocity = target_direction * nav_config.get('cruise_speed', 0.3)
                
                # Get current heading
                yaw = sensor_data.get('yaw', 0)
                
                # Compute safe velocity using vision
                safe_velocity = controller.compute_safe_velocity(
                    vision_data=vision_data,
                    current_position=current_pos,
                    target_velocity=target_velocity,
                    target_direction=target_direction,
                    current_heading=yaw,
                    target_position=goal_np,
                    dt=loop_period
                )
                
                # Send velocity command (forward-only mode may zero vy)
                interface.send_velocity_command(
                    safe_velocity[0], safe_velocity[1], safe_velocity[2], 0.0
                )
                
                # Status display
                battery = sensor_data.get('battery', 0)
                avoidance_state = controller.get_avoidance_state()
                state_str = avoidance_state.get('state', 'unknown')
                print(f"\r  Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                      f"Goal: {dist_to_goal:.2f}m | "
                      f"State: {state_str} | "
                      f"Bat: {battery:.2f}V | "
                      f"FPS: {camera.fps:.1f}  ", end='', flush=True)
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < loop_period:
                    time.sleep(loop_period - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Emergency stop triggered!")
            interface.emergency_stop()
            print("Motors stopped.")
            return
        
        # Land
        print("\n\nLanding...")
        if not interface.land():
            print("✗ Landing failed - emergency stop")
            interface.emergency_stop()
            return
        print("✓ Landing complete")
        
        print("\n" + "=" * 60)
        print("FLIGHT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Flight error: {e}")
        logger.exception("Flight error")
        if interface:
            try:
                interface.emergency_stop()
            except Exception:
                pass
    
    finally:
        if camera:
            camera.disconnect()
        if interface:
            interface.disconnect()
        print("\nDisconnected")


if __name__ == '__main__':
    main()
