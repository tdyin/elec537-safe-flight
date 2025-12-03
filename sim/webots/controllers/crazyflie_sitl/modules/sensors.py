"""
Sensor data collection and processing.

Handles reading and processing data from Webots sensors.
"""

import numpy as np
from math import cos, sin
import cv2


class SensorManager:
    """Manages sensor data collection from Webots robot."""
    
    def __init__(self, robot):
        """
        Initialize sensor manager.
        
        Args:
            robot: Webots robot instance
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        
        # Initialize sensors
        self.imu = robot.getDevice("inertial_unit")
        self.imu.enable(self.timestep)
        
        self.gps = robot.getDevice("gps")
        self.gps.enable(self.timestep)
        
        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        
        self.camera = robot.getDevice("camera")
        self.camera.enable(self.timestep)
        
        self.range_front = robot.getDevice("range_front")
        self.range_front.enable(self.timestep)
        
        self.range_left = robot.getDevice("range_left")
        self.range_left.enable(self.timestep)
        
        self.range_back = robot.getDevice("range_back")
        self.range_back.enable(self.timestep)
        
        self.range_right = robot.getDevice("range_right")
        self.range_right.enable(self.timestep)
        
        # Velocity estimation
        self.first_time = True
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        self.past_time = 0.0
    
    def get_sensor_data(self):
        """
        Collect all sensor data.
        
        Returns:
            tuple: (sensor_dict, state_tuple)
            - sensor_dict: Dictionary with all sensor readings
            - state_tuple: (roll, pitch, yaw, yaw_rate, altitude, v_x, v_y)
        """
        # Get IMU data
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        
        # Get GPS data
        x_global = self.gps.getValues()[0]
        y_global = self.gps.getValues()[1]
        altitude = self.gps.getValues()[2]
        
        # Calculate velocities
        dt = self.robot.getTime() - self.past_time if not self.first_time else 0.032
        
        if self.first_time:
            v_x_global = 0.0
            v_y_global = 0.0
            self.past_x_global = x_global
            self.past_y_global = y_global
            self.past_time = self.robot.getTime()
            self.first_time = False
        else:
            v_x_global = (x_global - self.past_x_global) / dt
            v_y_global = (y_global - self.past_y_global) / dt
        
        # Body-fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw
        
        # Get range sensor data (convert to meters)
        range_front_value = self.range_front.getValue() / 1000.0
        range_left_value = self.range_left.getValue() / 1000.0
        range_back_value = self.range_back.getValue() / 1000.0
        range_right_value = self.range_right.getValue() / 1000.0
        
        # Update past values
        self.past_x_global = x_global
        self.past_y_global = y_global
        self.past_time = self.robot.getTime()
        
        sensor_dict = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'yaw_rate': yaw_rate,
            'altitude': altitude,
            'velocity': [v_x, v_y, 0.0],
            'position': [x_global, y_global, altitude],
            'range_front': range_front_value,
            'range_left': range_left_value,
            'range_right': range_right_value,
            'range_back': range_back_value,
            'timestamp': self.robot.getTime(),
            'camera_width': self.camera.getWidth(),
            'camera_height': self.camera.getHeight(),
        }
        
        state_tuple = (roll, pitch, yaw, yaw_rate, altitude, v_x, v_y)
        
        return sensor_dict, state_tuple
    
    def get_camera_image(self):
        """
        Get camera image as grayscale numpy array.
        
        Returns:
            Grayscale image (H, W) or None if unavailable
        """
        camera_data = self.camera.getImage()
        if camera_data is None:
            return None
        
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
