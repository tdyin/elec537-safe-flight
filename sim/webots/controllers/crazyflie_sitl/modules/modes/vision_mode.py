"""
Vision-based autonomous navigation mode handler.

Handles autonomous navigation using depth estimation and obstacle avoidance.
"""

import numpy as np
import os
import time
from pathlib import Path

# Import logger if available
try:
    import sys
    controller_dir = Path(__file__).parent.parent.parent
    utils_dir = controller_dir.parent.parent / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")


class VisionModeHandler:
    """Handles vision-based autonomous navigation mode."""
    
    def __init__(self, robot, sensor_manager, pid_controller, safety_monitor,
                 keyboard_handler, nav_controller, vision_processor=None,
                 depth_estimator=None, height_desired=1.0, goal_position=None):
        """
        Initialize vision mode handler.
        
        Args:
            robot: Webots robot instance
            sensor_manager: SensorManager for sensor data
            pid_controller: PID controller for motor control
            safety_monitor: SafetyMonitor for crash detection
            keyboard_handler: KeyboardHandler for manual control
            nav_controller: NavigationController for autonomous navigation
            vision_processor: Optional VisionProcessor for object detection
            depth_estimator: Optional depth estimator for status display
            height_desired: Initial desired altitude
            goal_position: Target position [x, y, z]
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.sensor_manager = sensor_manager
        self.pid_controller = pid_controller
        self.safety_monitor = safety_monitor
        self.keyboard_handler = keyboard_handler
        self.nav_controller = nav_controller
        self.vision_processor = vision_processor
        self.depth_estimator = depth_estimator
        self.height_desired = height_desired
        self.goal_position = np.array(goal_position) if goal_position is not None else np.array([4.0, 0.0, 1.0])
        
        # Motor references
        self.motors = None
        
        # Navigation state
        self.goal_reached = False
        self.autonomous_enabled = False
        self.vision_failure_logged = False
        self.takeoff_time = None
        self.current_avoidance_state = 'normal'
        
        # Configuration from environment
        self.autonomous_delay_time = 2.0
        self.cruise_speed = float(os.environ.get('NAV_CRUISE_SPEED', '0.25'))
        self.safety_distance = float(os.environ.get('NAV_SAFETY_DISTANCE', '0.8'))
        self.avoidance_speed = float(os.environ.get('NAV_AVOIDANCE_SPEED', '0.18'))
        
        # Altitude limits
        self.min_altitude = float(os.environ.get('NAV_MIN_ALTITUDE', '0.5'))
        self.max_altitude = float(os.environ.get('NAV_MAX_ALTITUDE', '2.5'))
        
        # Velocity smoothing
        self.smoothing_alpha = float(os.environ.get('SIM_SMOOTHING_ALPHA', '0.08'))
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.yaw_rate_smooth = 0.0
        self.max_velocity = float(os.environ.get('SIM_MAX_VELOCITY', '0.30'))
        self.max_yaw_rate = float(os.environ.get('SIM_MAX_YAW_RATE', '0.6'))
        
        # Tilt safety
        self.warning_tilt = np.deg2rad(float(os.environ.get('SIM_WARNING_TILT', '18.0')))
        self.critical_tilt = np.deg2rad(float(os.environ.get('SIM_CRITICAL_TILT', '28.0')))
        
        # Display
        self.print_counter = 0
        self.print_interval = int(os.environ.get('SIM_PRINT_INTERVAL', '10'))
    
    def set_motors(self, m1, m2, m3, m4):
        """Set motor device references."""
        self.motors = (m1, m2, m3, m4)
    
    def run(self):
        """Run vision-based navigation mode loop."""
        self._print_startup_info()
        self._initialize_sensors()
        
        try:
            while self.robot.step(self.timestep) != -1:
                self._run_cycle()
        except KeyboardInterrupt:
            log("[NAVIGATION] Interrupted by user (Ctrl+C)", "INFO")
    
    def _print_startup_info(self):
        """Print startup information."""
        log("="*70, "INFO")
        log("VISION-BASED AUTONOMOUS NAVIGATION", "INFO")
        log("="*70, "INFO")
        log("NavMode:   Path-Based Navigation with Obstacle Avoidance", "INFO")
        log(f"Depth:     {'ENABLED' if self.depth_estimator else 'DISABLED'}", "INFO")
        log(f"Detection: {'ENABLED' if self.vision_processor and self.vision_processor.object_detector else 'DISABLED'}", "INFO")
        
        if self.nav_controller:
            viz_status = "ENABLED" if self.nav_controller.visualizer else "DISABLED"
            log(f"LiveViz:   {viz_status} (set ENABLE_NAV_VIZ=1 to enable)", "INFO")
        
        log(f"Speed:     {self.cruise_speed}m/s | Safety: {self.safety_distance}m", "INFO")
        log(f"Goal:      ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f}, {self.goal_position[2]:.1f})", "INFO")
        log("Controls:  Space=Toggle | WASD=Move | QE=Yaw | ↑↓=Alt | R=Auto", "INFO")
        log("="*70, "INFO")
    
    def _initialize_sensors(self):
        """Initialize sensors and wait for camera warm-up."""
        log("[INIT] Initializing sensors and camera...", "INFO")
        for i in range(20):
            self.robot.step(self.timestep)
            if i == 10:
                test_img = self.sensor_manager.get_camera_image()
                if test_img is not None:
                    log(f"[INIT] ✓ Camera ready ({test_img.shape[0]}x{test_img.shape[1]})", "SUCCESS")
                else:
                    log("[INIT] ⚠ Camera still initializing...", "WARNING")
        
        final_img = self.sensor_manager.get_camera_image()
        if final_img is not None:
            log("[INIT] ✓ All sensors ready", "SUCCESS")
        else:
            log("[INIT] ✗ WARNING: Camera may not be functioning", "ERROR")
    
    def _run_cycle(self):
        """Run a single control cycle."""
        # Read sensors
        sensor_data, state = self.sensor_manager.get_sensor_data()
        roll, pitch, yaw, yaw_rate, altitude, v_x, v_y = state
        x_global, y_global = sensor_data['position'][0], sensor_data['position'][1]
        
        # Update safety monitor
        self.safety_monitor.update_state(
            position=(x_global, y_global, altitude),
            velocity=(v_x, v_y),
            commanded_velocity=(self.vx_smooth, self.vy_smooth)
        )
        
        # Check crash
        if self.safety_monitor.check_crash(roll, pitch, altitude):
            self._set_motor_velocities([0, 0, 0, 0])
            self.robot.step(self.timestep)
            return
        
        # Process keyboard
        manual_cmd = self.keyboard_handler.process_input()
        
        if manual_cmd:
            forward_desired, sideways_desired, height_change, yaw_desired = manual_cmd
            self.height_desired = np.clip(
                self.height_desired + height_change * self.timestep / 1000, 
                self.min_altitude, self.max_altitude
            )
        else:
            # Enable autonomous control after reaching stable altitude
            if not self.autonomous_enabled and self.safety_monitor.takeoff_complete:
                if self.takeoff_time is None:
                    self.takeoff_time = time.time()
                    log(f"[NAVIGATION] Takeoff complete, waiting {self.autonomous_delay_time}s for stabilization...", "INFO")
                elif time.time() - self.takeoff_time >= self.autonomous_delay_time:
                    log("[NAVIGATION] ✓ Autonomous navigation ENABLED", "SUCCESS")
                    self.autonomous_enabled = True
            
            # Autonomous control mode
            if self.autonomous_enabled:
                nav_result = self._compute_autonomous_navigation(
                    x_global, y_global, altitude, yaw
                )
                # Handle 4-tuple return with altitude change
                if len(nav_result) == 4:
                    forward_desired, sideways_desired, yaw_desired, altitude_change = nav_result
                    # Apply altitude change from vertical avoidance
                    dt = self.timestep / 1000.0
                    self.height_desired = np.clip(
                        self.height_desired + altitude_change * dt,
                        self.min_altitude, self.max_altitude
                    )
                else:
                    forward_desired, sideways_desired, yaw_desired = nav_result
            else:
                forward_desired, sideways_desired, yaw_desired = 0.0, 0.0, 0.0
        
        # Apply tilt safety
        forward_desired, sideways_desired = self._apply_tilt_safety(
            forward_desired, sideways_desired, roll, pitch
        )
        
        # Apply smoothing
        self._apply_velocity_smoothing(forward_desired, sideways_desired, yaw_desired)
        
        # Run PID controller
        dt = self.timestep / 1000.0
        motor_power = self.pid_controller.pid(
            dt, self.vx_smooth, self.vy_smooth, self.yaw_rate_smooth, 
            self.height_desired, roll, pitch, yaw_rate, altitude, v_x, v_y
        )
        
        # Set motor velocities
        self._set_motor_velocities(motor_power)
        
        # Print status periodically
        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self._print_status(altitude, roll, pitch, x_global, y_global, 
                             forward_desired, sideways_desired, manual_cmd is not None)
            self.print_counter = 0
    
    def _compute_autonomous_navigation(self, x_global, y_global, altitude, yaw):
        """Compute autonomous navigation commands including altitude adjustment."""
        current_position = np.array([x_global, y_global, altitude])
        dt = self.timestep / 1000.0
        
        image = self.sensor_manager.get_camera_image()
        
        if image is not None and self.nav_controller:
            # Get navigation command including altitude change
            nav_result = self.nav_controller.update(
                current_position, yaw, image, dt
            )
            
            # Handle both 3-tuple (old) and 4-tuple (new) returns
            if len(nav_result) == 4:
                forward_desired, sideways_desired, yaw_desired, altitude_change = nav_result
            else:
                forward_desired, sideways_desired, yaw_desired = nav_result
                altitude_change = 0.0
            
            status = self.nav_controller.get_status()
            self.current_avoidance_state = status['mode']
            
            if status['mode'] == 'goal':
                if not self.goal_reached:
                    log("[NAVIGATION] ✓ GOAL REACHED", "SUCCESS")
                    self.goal_reached = True
                return 0.0, 0.0, 0.0, 0.0
            
            if self.print_counter == 0:
                path_info = status['path']
                vert_str = ""
                if abs(altitude_change) > 0.01:
                    vert_str = f" | Alt:{'+' if altitude_change > 0 else ''}{altitude_change:.2f}"
                log(f"[NAV] Mode:{status['mode'].upper()} | "
                    f"WP:{path_info['waypoint_idx']}/{path_info['total_waypoints']} | "
                    f"Dev:{path_info['deviation']:.2f}m | "
                    f"Avoid:{status['avoidance_count']}{vert_str}", "INFO")
            
            return forward_desired, sideways_desired, yaw_desired, altitude_change
        
        elif image is None and not self.vision_failure_logged:
            log("[VISION] ✗ Camera unavailable - hovering in place", "ERROR")
            self.vision_failure_logged = True
        
        return 0.0, 0.0, 0.0, 0.0
    
    def _apply_tilt_safety(self, forward_desired, sideways_desired, roll, pitch):
        """Reduce commanded velocities based on current tilt."""
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        
        if tilt_magnitude >= self.critical_tilt:
            if self.print_counter == 0:
                log(f"[SAFETY] ⚠ Critical tilt ({np.rad2deg(tilt_magnitude):.0f}°)", "WARNING")
            return forward_desired * 0.3, 0.0
        elif tilt_magnitude >= self.warning_tilt:
            scale = 1.0 - (tilt_magnitude - self.warning_tilt) / (self.critical_tilt - self.warning_tilt)
            scale = max(0.3, scale)
            return forward_desired * scale, sideways_desired * scale * 0.5
        
        return forward_desired, sideways_desired
    
    def _apply_velocity_smoothing(self, forward_desired, sideways_desired, yaw_desired):
        """Apply exponential smoothing and velocity limits."""
        # State-based smoothing: more damping during risky maneuvers
        if self.current_avoidance_state == 'emergency':
            alpha = 0.2
        elif self.current_avoidance_state in ['avoidance', 'recovery']:
            alpha = 0.1
        elif self.current_avoidance_state == 'caution':
            alpha = 0.08
        else:
            alpha = self.smoothing_alpha
        
        self.vx_smooth = alpha * forward_desired + (1 - alpha) * self.vx_smooth
        self.vy_smooth = alpha * sideways_desired + (1 - alpha) * self.vy_smooth
        self.yaw_rate_smooth = alpha * yaw_desired + (1 - alpha) * self.yaw_rate_smooth
        
        # Limit velocities
        vel_magnitude = np.sqrt(self.vx_smooth**2 + self.vy_smooth**2)
        if vel_magnitude > self.max_velocity:
            scale = self.max_velocity / vel_magnitude
            self.vx_smooth *= scale
            self.vy_smooth *= scale
        
        self.yaw_rate_smooth = np.clip(self.yaw_rate_smooth, -self.max_yaw_rate, self.max_yaw_rate)
    
    def _set_motor_velocities(self, motor_power):
        """Set motor velocities from PID output."""
        if self.motors:
            self.motors[0].setVelocity(-motor_power[0])
            self.motors[1].setVelocity(motor_power[1])
            self.motors[2].setVelocity(-motor_power[2])
            self.motors[3].setVelocity(motor_power[3])
    
    def _print_status(self, altitude, roll, pitch, x, y, vx_cmd=None, vy_cmd=None, manual_mode=False):
        """Print flight status."""
        mode_str = "MANUAL" if manual_mode else "AUTO"
        status_msg = f"Pos3D: ({x:.2f}, {y:.2f}, {altitude:.2f}) | Tilt: {np.rad2deg(roll):.0f}°/{np.rad2deg(pitch):.0f}°"
        
        if vx_cmd is not None and vy_cmd is not None:
            status_msg += f" | Cmd: vx={vx_cmd:.2f} vy={vy_cmd:.2f}"
        
        status_msg += f" | Mode: {mode_str}"
        
        if self.depth_estimator:
            status_msg += " | Vision: ✓"
        else:
            status_msg += " | Vision: ✗"
        
        log(status_msg, "INFO")
