#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Webots Crazyflie SITL Controller

This controller supports two modes:
1. External control mode: TCP bridge for SITL development with external Python code
2. Vision mode: Autonomous navigation using depth estimation and obstacle avoidance

Author: Safe Flight Team
"""

from controller import Robot
import numpy as np
import sys
import os
import time
from pathlib import Path

# Local imports
from pid_controller import pid_velocity_fixed_height_controller
from modules import (
    CommunicationBridge,
    SensorManager,
    SafetyMonitor,
    KeyboardHandler,
    SimpleDepthEstimator,
    analyze_depth_map,
    VISION_AVAILABLE,
    VisionProcessor,
    DETECTOR_AVAILABLE,
    # Navigation modules
    NavigationController,
    NavigationMode,
    # Mode handlers
    ExternalModeHandler,
    VisionModeHandler,
)

# Add project root to path for scripts
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
scripts_path = project_root / 'scripts'
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

# Add utils to path for shared logging - use absolute path for Webots
utils_path = Path(__file__).resolve().parent.parent.parent / 'utils'
utils_path_str = str(utils_path)
if utils_path_str not in sys.path:
    sys.path.insert(0, utils_path_str)

try:
    from logger import log, setup_logger
except ImportError as e:
    print(f"WARNING: Failed to import logger: {e}", file=sys.stderr)
    # Fallback logging functions
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")
    def setup_logger(name, path):
        pass

# Optional live depth visualization
LIVE_VIZ_AVAILABLE = False
LiveDepthVisualizer = None
try:
    # Import from utils folder (relative to controller)
    viz_utils_path = Path(__file__).resolve().parent.parent.parent / 'utils'
    if str(viz_utils_path) not in sys.path:
        sys.path.insert(0, str(viz_utils_path))
    from viz_depth_live import LiveDepthVisualizer
    LIVE_VIZ_AVAILABLE = True
except ImportError:
    pass


class WebotsController:
    """Unified controller for Webots-based SITL development and vision-based navigation."""
    
    def __init__(self, robot, mode='external', host='localhost', port=10020, 
                 depth_model_path=None, goal_position=None, 
                 detection_model_path=None, detection_model_type='yolo'):
        """
        Initialize Webots controller.
        
        Args:
            robot: Webots robot instance
            mode: 'external' for TCP control, 'vision' for autonomous navigation
            host: Host for socket server (external mode)
            port: Port for socket server (external mode)
            depth_model_path: Path to MiDaS ONNX model (vision mode)
            goal_position: Target position [x, y, z] (vision mode)
            detection_model_path: Path to object detection model (optional, vision mode)
            detection_model_type: Type of detection model ('yolo', 'ssd', 'dronet')
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.mode = mode
        
        # Initialize motor devices
        self._setup_motors()
        
        # Initialize sensor manager
        self.sensor_manager = SensorManager(robot)
        
        # Initialize PID controller
        self.pid_controller = pid_velocity_fixed_height_controller()
        # Default altitude from config via env var (fallback to 1.0m)
        self.height_desired = float(os.environ.get('NAV_DEFAULT_ALTITUDE', '1.0'))
        
        # Mode-specific initialization
        if mode == 'external':
            self._init_external_mode(host, port)
        elif mode == 'vision':
            self._init_vision_mode(depth_model_path, detection_model_path, 
                                  detection_model_type, goal_position)
    
    def _setup_motors(self):
        """Initialize motor devices."""
        self.m1_motor = self.robot.getDevice("m1_motor")
        self.m1_motor.setPosition(float('inf'))
        self.m1_motor.setVelocity(-1)
        
        self.m2_motor = self.robot.getDevice("m2_motor")
        self.m2_motor.setPosition(float('inf'))
        self.m2_motor.setVelocity(1)
        
        self.m3_motor = self.robot.getDevice("m3_motor")
        self.m3_motor.setPosition(float('inf'))
        self.m3_motor.setVelocity(-1)
        
        self.m4_motor = self.robot.getDevice("m4_motor")
        self.m4_motor.setPosition(float('inf'))
        self.m4_motor.setVelocity(1)
    
    def _init_external_mode(self, host, port):
        """Initialize external control mode components."""
        self.comm_bridge = CommunicationBridge(host, port)
    
    def _init_vision_mode(self, depth_model_path, detection_model_path, 
                         detection_model_type, goal_position, waypoints=None):
        """Initialize vision mode components."""
        if not VISION_AVAILABLE:
            raise RuntimeError("Vision mode requires cv2 and onnxruntime. "
                             "Install with 'pip install opencv-python onnxruntime'")
        
        # Store depth model path for navigation controller
        self.depth_model_path = depth_model_path
        
        # Initialize depth estimator (used for status reporting)
        self.depth_estimator = None
        if depth_model_path and os.path.exists(depth_model_path):
            try:
                self.depth_estimator = SimpleDepthEstimator(depth_model_path)
                log(f"Depth model loaded: {Path(depth_model_path).name}", "SUCCESS")
            except Exception as e:
                log(f"Depth model failed: {e}", "ERROR")
        
        # Initialize object detector (optional)
        object_detector = None
        if detection_model_path and os.path.exists(detection_model_path) and DETECTOR_AVAILABLE:
            try:
                from vision.detector import ObstacleDetector as VisionObstacleDetector
                object_detector = VisionObstacleDetector(
                    model_path=detection_model_path,
                    model_type=detection_model_type,
                    confidence_threshold=0.4,
                    use_gpu=False
                )
                log(f"Object detector loaded: {Path(detection_model_path).name}", "SUCCESS")
            except Exception as e:
                log(f"Object detector failed: {e}", "WARNING")
        
        # Initialize vision processor (for object detection)
        self.vision_processor = VisionProcessor(object_detector)
        
        # Initialize keyboard handler
        self.keyboard_handler = KeyboardHandler(self.robot, self.timestep)
        
        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor()
        
        # Navigation parameters
        self.goal_position = np.array(goal_position if goal_position else [4.0, 0.0, 1.0])
        
        # Check if visualization is enabled via environment variable
        enable_nav_viz = os.environ.get('ENABLE_NAV_VIZ', '').lower() in ('1', 'true', 'yes')
        save_nav_frames = os.environ.get('SAVE_NAV_FRAMES', '').lower() in ('1', 'true', 'yes')
        
        # Read navigation parameters from environment (set by launch.py from config.yaml)
        cruise_speed = float(os.environ.get('NAV_CRUISE_SPEED', '0.25'))
        avoidance_speed = float(os.environ.get('NAV_AVOIDANCE_SPEED', '0.18'))
        turn_rate = float(os.environ.get('NAV_TURN_RATE', '1.0'))
        avoidance_duration = float(os.environ.get('NAV_AVOIDANCE_DURATION', '1.5'))
        
        # Initialize unified navigation controller
        log(f"[NAV] Initializing navigation (speed={cruise_speed}m/s, avoid={avoidance_speed}m/s)", "SUCCESS")
        try:
            self.nav_controller = NavigationController(
                depth_model_path=depth_model_path,
                cruise_speed=cruise_speed,
                avoidance_speed=avoidance_speed,
                turn_rate=turn_rate,
                avoidance_duration=avoidance_duration,
                clear_distance=0.5,
                enable_visualization=enable_nav_viz,
                save_viz_frames=save_nav_frames
            )
            
            # Set waypoints or single goal
            if waypoints and len(waypoints) > 0:
                self.nav_controller.set_path(waypoints)
            else:
                # Create default path: takeoff position -> goal
                default_waypoints = [
                    (self.goal_position[0], self.goal_position[1], self.goal_position[2])
                ]
                self.nav_controller.set_path(default_waypoints)
            
            log(f"[NAV] Goal: ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f}, {self.goal_position[2]:.1f})", "INFO")
        except Exception as e:
            log(f"[NAV] Failed to initialize navigation: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Navigation initialization failed: {e}")
        
        # Initialize vision mode handler
        self.vision_mode_handler = VisionModeHandler(
            robot=self.robot,
            sensor_manager=self.sensor_manager,
            pid_controller=self.pid_controller,
            safety_monitor=self.safety_monitor,
            keyboard_handler=self.keyboard_handler,
            nav_controller=self.nav_controller,
            vision_processor=self.vision_processor,
            depth_estimator=self.depth_estimator,
            height_desired=self.height_desired,
            goal_position=self.goal_position
        )
        self.vision_mode_handler.set_motors(self.m1_motor, self.m2_motor, self.m3_motor, self.m4_motor)
    
    def run(self):
        """Main control loop - supports both external and vision modes."""
        if self.mode == 'vision':
            self._run_vision_mode()
        else:
            self._run_external_mode()
    
    def _run_external_mode(self):
        """Run in external TCP control mode using ExternalModeHandler."""
        # Create external mode handler
        handler = ExternalModeHandler(
            robot=self.robot,
            comm_bridge=self.comm_bridge,
            sensor_manager=self.sensor_manager,
            pid_controller=self.pid_controller,
            height_desired=self.height_desired
        )
        handler.set_motors(self.m1_motor, self.m2_motor, self.m3_motor, self.m4_motor)
        handler.run()
    
    def _run_vision_mode(self):
        """Run in autonomous vision-based navigation mode using VisionModeHandler."""
        self.vision_mode_handler.run()
    
    def _set_motor_velocities(self, motor_power):
        """Set motor velocities from PID output."""
        self.m1_motor.setVelocity(-motor_power[0])
        self.m2_motor.setVelocity(motor_power[1])
        self.m3_motor.setVelocity(-motor_power[2])
        self.m4_motor.setVelocity(motor_power[3])


def main():
    """Main entry point."""
    import argparse
    import json
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Webots Crazyflie Controller')
    parser.add_argument('--mode', type=str, default=None, choices=['external', 'vision'],
                       help='Control mode: external (TCP) or vision (autonomous)')
    parser.add_argument('--depth-model', type=str, default=None,
                       help='Path to MiDaS depth model (vision mode)')
    parser.add_argument('--detection-model', type=str, default=None,
                       help='Path to object detection model (optional, vision mode)')
    parser.add_argument('--detection-type', type=str, default='yolo',
                       choices=['yolo', 'ssd', 'dronet'],
                       help='Type of detection model (vision mode)')
    parser.add_argument('--goal', type=float, nargs=3, default=[4.0, 0.0, 1.0],
                       help='Goal position [x y z] (vision mode)')
    parser.add_argument('--waypoints', type=str, default=None,
                       help='JSON array of waypoints [[x,y,z], ...] (vision mode)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host for TCP server (external mode)')
    parser.add_argument('--port', type=int, default=10020,
                       help='Port for TCP server (external mode)')
    
    args = parser.parse_args()
    
    # Parse waypoints if provided
    waypoints = None
    if args.waypoints:
        try:
            waypoints = json.loads(args.waypoints)
            log(f"Parsed {len(waypoints)} waypoints from command line", "INFO")
        except json.JSONDecodeError as e:
            log(f"Failed to parse waypoints JSON: {e}", "WARNING")
    
    # Check environment variables for waypoints override
    env_waypoints = os.environ.get('NAV_WAYPOINTS', None)
    if env_waypoints and not waypoints:
        try:
            waypoints = json.loads(env_waypoints)
            log(f"Parsed {len(waypoints)} waypoints from NAV_WAYPOINTS env", "INFO")
        except json.JSONDecodeError:
            pass
    
    # Check environment variables for mode override
    mode = args.mode if args.mode else os.environ.get('SITL_MODE', 'external')
    depth_model_path = args.depth_model if args.depth_model else os.environ.get('DEPTH_MODEL_PATH', None)
    detection_model_path = args.detection_model if args.detection_model else os.environ.get('VISION_MODEL_PATH', None)
    detection_model_type = os.environ.get('VISION_MODEL_TYPE', args.detection_type)
    
    # Setup unified logger with format: {timestamp}-{world}-{mode}
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    world_name = os.environ.get('WORLD_NAME', 'unknown')
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    log_file = log_dir / f'{timestamp}-{world_name}-{mode}.log'
    setup_logger('crazyflie_sitl', str(log_file))
    
    # Create robot instance
    robot = Robot()
    
    # Auto-detect depth model if not specified and in vision mode
    if mode == 'vision' and depth_model_path is None:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        depth_model = project_root / 'models' / 'midas_v21_small.onnx'
        if depth_model.exists():
            depth_model_path = str(depth_model)
            log(f"Auto-detected depth model: {depth_model.name}", "SUCCESS")
    
    # Auto-detect object detection model if not specified and in vision mode
    if mode == 'vision' and detection_model_path is None and DETECTOR_AVAILABLE:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        models_dir = project_root / 'models'
        for model_file in ["yolov5n.onnx", "ssd_mobilenet_v1.onnx", "pulp_dronet_id_4dory.onnx"]:
            test_path = models_dir / model_file
            if test_path.exists():
                detection_model_path = str(test_path)
                if "yolo" in model_file:
                    detection_model_type = "yolo"
                elif "ssd" in model_file:
                    detection_model_type = "ssd"
                elif "dronet" in model_file:
                    detection_model_type = "dronet"
                log(f"Auto-detected detection model: {model_file}", "SUCCESS")
                break
    
    log("="*60, "INFO")
    log(f"Mode: {mode.upper()}", "INFO")
    log(f"Navigation: Path-Based (with obstacle avoidance)", "INFO")
    if mode == 'vision':
        log(f"Depth model: {depth_model_path if depth_model_path else 'None'}", "INFO")
        log(f"Detection model: {detection_model_path if detection_model_path else 'None (depth-only)'}", "INFO")
        if waypoints:
            log(f"Waypoints: {len(waypoints)} points", "INFO")
    log("="*60, "INFO")
    
    # Create and run controller
    controller = WebotsController(
        robot,
        mode=mode,
        host=args.host,
        port=args.port,
        depth_model_path=depth_model_path,
        detection_model_path=detection_model_path,
        detection_model_type=detection_model_type,
        goal_position=args.goal
    )
    
    # Set waypoints if provided (after controller creation)
    if waypoints and mode == 'vision' and hasattr(controller, 'nav_controller'):
        controller.nav_controller.set_path(waypoints)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
