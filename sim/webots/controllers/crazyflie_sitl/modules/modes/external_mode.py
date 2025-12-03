"""
External control mode handler.

Handles TCP-based external control mode for SITL development.
"""

import numpy as np

# Import logger if available
try:
    import sys
    from pathlib import Path
    controller_dir = Path(__file__).parent.parent.parent
    utils_dir = controller_dir.parent.parent / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from logger import log
except ImportError:
    def log(msg, level="INFO"):
        print(f"[{level}] {msg}")


class ExternalModeHandler:
    """Handles external TCP control mode."""
    
    def __init__(self, robot, comm_bridge, sensor_manager, pid_controller, height_desired=1.0):
        """
        Initialize external mode handler.
        
        Args:
            robot: Webots robot instance
            comm_bridge: CommunicationBridge for TCP control
            sensor_manager: SensorManager for sensor data
            pid_controller: PID controller for motor control
            height_desired: Initial desired altitude
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.comm_bridge = comm_bridge
        self.sensor_manager = sensor_manager
        self.pid_controller = pid_controller
        self.height_desired = height_desired
        
        # Motor references (to be set by controller)
        self.motors = None
    
    def set_motors(self, m1, m2, m3, m4):
        """Set motor device references."""
        self.motors = (m1, m2, m3, m4)
    
    def run(self):
        """Run external control mode loop."""
        log("="*70, "INFO")
        log("WEBOTS SITL CONTROLLER - EXTERNAL MODE", "INFO")
        log("="*70, "INFO")
        log("Waiting for external control connection...", "INFO")
        
        self.comm_bridge.setup()
        
        while self.robot.step(self.timestep) != -1:
            # Check for new connections
            if not self.comm_bridge.connected:
                self.comm_bridge.accept_connection()
            
            # Receive commands from external control
            if self.comm_bridge.connected:
                self.comm_bridge.receive_commands()
            
            # Get sensor data
            sensor_data, state = self.sensor_manager.get_sensor_data()
            roll, pitch, yaw_rate, altitude = state[0], state[1], state[3], state[4]
            v_x, v_y = state[5], state[6]
            
            # Send sensor data to external control
            if self.comm_bridge.connected:
                self.comm_bridge.send_sensor_data(sensor_data)
                self.comm_bridge.send_position_data(*sensor_data['position'])
            
            # Determine velocity commands
            if self.comm_bridge.external_control:
                forward_desired = self.comm_bridge.cmd_vx
                sideways_desired = self.comm_bridge.cmd_vy
                yaw_desired = self.comm_bridge.cmd_yaw_rate
                height_diff = self.comm_bridge.cmd_vz * (self.timestep / 1000.0)
                self.height_desired += height_diff
            else:
                forward_desired = 0.0
                sideways_desired = 0.0
                yaw_desired = 0.0
            
            # Run PID controller
            dt = self.timestep / 1000.0
            motor_power = self.pid_controller.pid(
                dt, forward_desired, sideways_desired,
                yaw_desired, self.height_desired,
                roll, pitch, yaw_rate,
                altitude, v_x, v_y
            )
            
            # Set motor velocities
            self._set_motor_velocities(motor_power)
    
    def _set_motor_velocities(self, motor_power):
        """Set motor velocities from PID output."""
        if self.motors:
            self.motors[0].setVelocity(-motor_power[0])
            self.motors[1].setVelocity(motor_power[1])
            self.motors[2].setVelocity(-motor_power[2])
            self.motors[3].setVelocity(motor_power[3])
