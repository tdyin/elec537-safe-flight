"""Main entry point for Safe Flight obstacle detection system."""

import sys
from pathlib import Path

# Add project root to path to support running as script or module
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import yaml
from loguru import logger

from src.vision.depth_detector import DepthDetector
from src.drone.depth_controller import DepthNavigationController
from src.drone import CrazyflieInterface
from src.sim.webots_interface import WebotsInterface

# Hardware interface available when cflib installed
try:
    from src.hardware import CrazyflieHardwareInterface
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict, mode: str = 'navigation'):
    """Setup logging configuration with unified naming: {timestamp}-{world}-{mode}."""
    import datetime
    import os
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_dir = config.get('logging', {}).get('log_directory', 'logs')
    
    Path(log_dir).mkdir(exist_ok=True)
    
    # Use unified log naming format
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    world_name = os.environ.get('WORLD_NAME', 'unknown')
    log_filename = f"{timestamp}-{world_name}-external-{mode}.log"
    
    logger.add(
        f"{log_dir}/{log_filename}",
        rotation="10 MB",
        level=log_level
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Safe Flight - Drone Obstacle Detection')
    parser.add_argument('--config', type=str, default='config/sim.yaml',
                       help='Path to configuration file (config/sim.yaml or config/hardware.yaml)')
    parser.add_argument('--mode', type=str, default='detection',
                       choices=['detection', 'navigation', 'data_collection'],
                       help='Operating mode')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (no hardware)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config, args.mode)
    
    # Unified startup banner
    logger.info("="*70)
    logger.info("SAFE FLIGHT - Vision-Based Obstacle Avoidance System")
    logger.info("="*70)
    logger.info(f"Mode:       {args.mode.upper()}")
    logger.info(f"Simulation: {'YES' if args.simulation else 'NO (Hardware)'}")
    logger.info(f"Config:     {args.config}")
    import datetime
    logger.info(f"Started:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Initialize modules
    # Initialize modules
    vision_config = config['vision']
    drone_config = config['drone']
    if args.mode == 'detection':
        logger.info("[MODE] Detection mode - Sensor data processing only")
        # Implement detection loop
        # This would continuously process sensor data and detect obstacles
        
    elif args.mode == 'navigation':
        logger.info("[MODE] Navigation mode - Active flight control with depth-based avoidance")
        # Initialize drone interface
        if args.simulation:
            sim_config = drone_config.get('simulation', {})
            logger.info(f"[INTERFACE] Webots simulation ({sim_config.get('host', 'localhost')}:{sim_config.get('port', 10020)})")
            drone = WebotsInterface(
                host=sim_config.get('host', 'localhost'),
                port=sim_config.get('port', 10020)
            )
        else:
            logger.info(f"[INTERFACE] Crazyflie hardware (URI: {drone_config.get('uri')})")
            drone = CrazyflieInterface(uri=drone_config.get('uri'))
        
        # Use depth-based controller with stable avoidance
        nav_config = drone_config.get('navigation', {})
        controller = DepthNavigationController(
            max_speed=nav_config.get('max_speed', 0.5),
            safety_distance=nav_config.get('safety_distance', 0.5),
            avoidance_gain=nav_config.get('avoidance_gain', 1.0),
            max_altitude=nav_config.get('max_altitude', 2.5),
            min_altitude=nav_config.get('min_altitude', 0.5),
            use_path_planning=drone_config.get('path_planning', {}).get('enabled', True),
            replan_interval=drone_config.get('path_planning', {}).get('replan_interval', 10),
            path_smoothing=drone_config.get('path_planning', {}).get('path_smoothing', 'bezier'),
            use_stable_avoidance=True  # Enable stable avoidance
        )
        
        # Initialize depth detector
        depth_config = vision_config.get('depth', {})
        detector = DepthDetector(
            depth_model_path=depth_config.get('model_path'),
            depth_scale=depth_config.get('depth_scale', 1.0),
            use_gpu=depth_config.get('use_gpu', False)
        )
        
        logger.info(f"[CONTROLLER] Depth-based navigation with stable avoidance")
        logger.info(f"             max_speed={nav_config.get('max_speed', 0.5)}m/s, "
                   f"safety={nav_config.get('safety_distance', 0.5)}m")
        
        # Connect to drone
        logger.info("[CONNECTION] Attempting to connect...")
        if not drone.connect():
            logger.error("[CONNECTION] ✗ Failed to establish connection")
            return
        logger.info("[CONNECTION] ✓ Connected successfully")
        
        try:
            # Navigation loop
            control_freq = drone_config.get('control_frequency', 50)
            logger.info(f"[NAVIGATION] Starting control loop ({control_freq} Hz)")
            control_rate = drone_config.get('control_frequency', 50)
            dt = 1.0 / control_rate
            
            import time
            import numpy as np
            
            # Goal position for navigation
            goal_position = np.array([5.0, 0.0, 1.0])
            frame_count = 0
            status_interval = 50  # Print status every N frames
            
            while True:
                loop_start = time.time()
                frame_count += 1
                
                # Check for crash first
                if args.simulation and hasattr(drone, 'check_crash'):
                    if drone.check_crash():
                        logger.error("[SAFETY] ✗ CRASH DETECTED - Emergency stop initiated")
                        break
                
                # Get sensor data
                sensor_data = drone.get_sensor_data()
                current_position = np.array(drone.get_position())
                current_heading = sensor_data.get('yaw', 0.0)
                
                # Compute target velocity toward goal
                to_goal = goal_position - current_position
                distance_to_goal = np.linalg.norm(to_goal)
                
                if distance_to_goal < 0.3:
                    logger.info("[NAVIGATION] ✓ Goal reached!")
                    break
                
                target_direction = to_goal / distance_to_goal
                target_velocity = target_direction * 0.3  # Cruise speed
                
                # Process camera image with depth detector
                vision_data = {}
                if 'camera' in sensor_data and sensor_data['camera'] is not None:
                    vision_data = detector.detect(sensor_data['camera'])
                
                # Compute safe velocity using depth-based avoidance
                safe_velocity = controller.compute_safe_velocity(
                    vision_data=vision_data,
                    current_position=current_position,
                    target_velocity=target_velocity,
                    target_direction=target_direction,
                    current_heading=current_heading,
                    target_position=goal_position,
                    dt=dt
                )
                
                # Log avoidance state periodically
                if frame_count % status_interval == 0:
                    avoidance_state = controller.get_avoidance_state()
                    logger.info(f"[STATUS] State: {avoidance_state.get('state', 'unknown')}, "
                               f"Speed: {avoidance_state.get('current_speed', 0):.2f}m/s, "
                               f"Goal: {distance_to_goal:.1f}m")
                
                # Send velocity command
                drone.send_velocity_command(
                    safe_velocity[0],
                    safe_velocity[1],
                    safe_velocity[2],
                    0.0  # yaw rate
                )
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        
        except KeyboardInterrupt:
            logger.info("[NAVIGATION] Interrupted by user (Ctrl+C)")
        finally:
            logger.info("[CONNECTION] Disconnecting from drone...")
            drone.emergency_stop()
            drone.disconnect()
            logger.info("[CONNECTION] ✓ Disconnected")
        
    elif args.mode == 'data_collection':
        logger.info("[MODE] Data collection mode")
        # Implement data collection routine
        # This would collect and save sensor data for training
    
    logger.info("="*70)
    logger.info("SAFE FLIGHT SHUTDOWN COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
