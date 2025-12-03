"""Communication bridge between main code and Webots simulator."""

import socket
import struct
import json
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import threading
import time


class SimulationBridge:
    """Bridge for communicating with Webots simulation via TCP sockets."""
    
    # Message types
    MSG_SENSOR_DATA = 1
    MSG_VELOCITY_CMD = 2
    MSG_POSITION_DATA = 3
    MSG_HEARTBEAT = 4
    
    def __init__(self, host: str = 'localhost', port: int = 10020):
        """
        Initialize simulation bridge.
        
        Args:
            host: Host address for socket server
            port: Port number for socket communication
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = False
        
        # Data buffers
        self.latest_sensor_data = {}
        self.latest_position = (0.0, 0.0, 0.0)
        self.data_lock = threading.Lock()
        
        # Receiver thread
        self.receiver_thread = None
    
    def connect(self, timeout: float = 10.0, retry_count: int = 3) -> bool:
        """
        Connect to Webots controller.
        
        Args:
            timeout: Connection timeout in seconds
            retry_count: Number of connection retry attempts
            
        Returns:
            True if connection successful
        """
        for attempt in range(retry_count):
            try:
                logger.info(f"[BRIDGE] Connecting to Webots at {self.host}:{self.port} (attempt {attempt + 1}/{retry_count})")
                
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(timeout)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)  # Remove timeout after connection
                self.connected = True
                
                # Start receiver thread
                self.running = True
                self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receiver_thread.start()
                
                logger.success(f"✓ Connected to Webots simulation at {self.host}:{self.port}")
                return True
                
            except socket.timeout:
                logger.warning(f"[BRIDGE] Timeout at {self.host}:{self.port} (attempt {attempt + 1}/{retry_count})")
                if self.socket:
                    try:
                        self.socket.close()
                    except OSError:
                        pass
                    self.socket = None
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
            except ConnectionRefusedError:
                logger.warning(f"[BRIDGE] Connection refused at {self.host}:{self.port} (attempt {attempt + 1}/{retry_count})")
                logger.info("[BRIDGE] → Ensure Webots is running with crazyflie_sitl controller")
                if self.socket:
                    try:
                        self.socket.close()
                    except OSError:
                        pass
                    self.socket = None
                if attempt < retry_count - 1:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to connect to simulation: {e}")
                if self.socket:
                    try:
                        self.socket.close()
                    except OSError:
                        pass
                    self.socket = None
                if attempt < retry_count - 1:
                    time.sleep(1)
        
        logger.error(f"[BRIDGE] ✗ Failed to connect after {retry_count} attempts")
        return False
    
    def disconnect(self):
        """Disconnect from Webots controller."""
        self.running = False
        
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2.0)
        
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None
        
        self.connected = False
        logger.info("[BRIDGE] Disconnected from Webots simulation")
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Send velocity command to simulator.
        
        Args:
            vx: Velocity in x direction (m/s)
            vy: Velocity in y direction (m/s)
            vz: Velocity in z direction (m/s)
            yaw_rate: Yaw rate (rad/s)
        """
        if not self.connected or not self.socket:
            logger.debug("Not connected to simulator - skipping velocity command")
            return
        
        try:
            # Pack message: [msg_type, vx, vy, vz, yaw_rate]
            data = struct.pack('!Bffff', self.MSG_VELOCITY_CMD, vx, vy, vz, yaw_rate)
            self.socket.sendall(data)
            logger.debug(f"[BRIDGE→SIM] CMD: vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} yaw={yaw_rate:+.3f}")
            
        except (socket.error, BrokenPipeError) as e:
            logger.warning(f"Lost connection to simulator: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Failed to send velocity command: {e}")
            self.connected = False
    
    def get_sensor_data(self) -> Dict:
        """
        Get latest sensor data from simulator.
        
        Returns:
            Dictionary containing sensor readings
        """
        with self.data_lock:
            return self.latest_sensor_data.copy()
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current position from simulator.
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        with self.data_lock:
            return self.latest_position
    
    def _receive_loop(self):
        """Background thread for receiving data from simulator."""
        buffer = b''
        
        while self.running and self.connected:
            try:
                # Receive data
                chunk = self.socket.recv(4096)
                if not chunk:
                    logger.warning("Simulator connection closed")
                    self.connected = False
                    break
                
                buffer += chunk
                
                # Process complete messages
                while len(buffer) >= 1:
                    msg_type = buffer[0]
                    
                    if msg_type == self.MSG_SENSOR_DATA:
                        # Parse sensor data message
                        if len(buffer) < 1 + 4:  # msg_type + json_length
                            break
                        
                        json_length = struct.unpack('!I', buffer[1:5])[0]
                        
                        if len(buffer) < 1 + 4 + json_length:
                            break
                        
                        json_data = buffer[5:5 + json_length].decode('utf-8')
                        sensor_data = json.loads(json_data)
                        
                        with self.data_lock:
                            self.latest_sensor_data = sensor_data
                        
                        buffer = buffer[5 + json_length:]
                        
                    elif msg_type == self.MSG_POSITION_DATA:
                        # Parse position data: [msg_type, x, y, z]
                        if len(buffer) < 1 + 12:
                            break
                        
                        x, y, z = struct.unpack('!fff', buffer[1:13])
                        
                        with self.data_lock:
                            self.latest_position = (x, y, z)
                        
                        buffer = buffer[13:]
                        
                    elif msg_type == self.MSG_HEARTBEAT:
                        # Heartbeat message
                        buffer = buffer[1:]
                        
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                        buffer = buffer[1:]
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self.connected = False
                break
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
