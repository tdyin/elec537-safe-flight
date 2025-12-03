"""
TCP Communication module for external SITL control.

Provides socket-based communication between Webots controller and external Python code.
"""

import socket
import struct
import json


class CommunicationBridge:
    """Handles TCP socket communication for external control mode."""
    
    # Message types
    MSG_SENSOR_DATA = 1
    MSG_VELOCITY_CMD = 2
    MSG_POSITION_DATA = 3
    MSG_HEARTBEAT = 4
    
    def __init__(self, host='localhost', port=10020):
        """
        Initialize communication bridge.
        
        Args:
            host: Host address for socket server
            port: Port number for socket server
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.connected = False
        
        # Received commands
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0
        self.external_control = False
    
    def setup(self):
        """Setup socket server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.setblocking(False)
            print(f"[TCP] ✓ Server listening on {self.host}:{self.port}")
        except Exception as e:
            print(f"[TCP] ✗ Server setup failed: {e}")
    
    def accept_connection(self):
        """Accept incoming client connection."""
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.client_socket.setblocking(False)
            self.connected = True
            print(f"[TCP] ✓ Client connected from {addr[0]}:{addr[1]}")
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[TCP] Connection error: {e}")
    
    def receive_commands(self):
        """Receive velocity commands from external control."""
        if not self.connected or not self.client_socket:
            return
        
        try:
            # Read message header (1 byte type + 4 bytes length)
            header = self.client_socket.recv(5)
            if not header:
                self.disconnect()
                return
            
            msg_type, msg_len = struct.unpack('!BI', header)
            
            if msg_type == self.MSG_VELOCITY_CMD:
                # Read velocity command (4 floats)
                data = self.client_socket.recv(16)
                if len(data) == 16:
                    self.cmd_vx, self.cmd_vy, self.cmd_vz, self.cmd_yaw_rate = struct.unpack('!ffff', data)
                    self.external_control = True
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[TCP] Receive error: {e}")
            self.disconnect()
    
    def send_sensor_data(self, sensor_data):
        """Send sensor data to external control."""
        if not self.connected or not self.client_socket:
            return
        
        try:
            data_json = json.dumps(sensor_data).encode('utf-8')
            message = struct.pack('!BI', self.MSG_SENSOR_DATA, len(data_json)) + data_json
            self.client_socket.sendall(message)
        except Exception as e:
            print(f"[TCP] Send sensor error: {e}")
            self.disconnect()
    
    def send_position_data(self, x, y, z):
        """Send position data to external control."""
        if not self.connected or not self.client_socket:
            return
        
        try:
            message = struct.pack('!Bfff', self.MSG_POSITION_DATA, x, y, z)
            self.client_socket.sendall(message)
        except Exception as e:
            print(f"[TCP] Send position error: {e}")
    
    def disconnect(self):
        """Disconnect client."""
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        self.client_socket = None
        self.connected = False
        self.external_control = False
        print("[TCP] ✗ Client disconnected")
    
    def cleanup(self):
        """Cleanup sockets."""
        self.disconnect()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
