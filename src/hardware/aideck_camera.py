"""AI Deck camera streaming over WiFi.

This module provides camera image streaming from the Crazyflie AI Deck
over its WiFi connection.
"""

import socket
import struct
import numpy as np
from typing import Optional
from loguru import logger
import threading
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available for image decoding")


class AIdeckCamera:
    """WiFi camera streaming from AI Deck.
    
    Connects to the AI Deck's WiFi access point and receives
    JPEG-encoded camera frames over TCP.
    """
    
    def __init__(self, 
                 ip: str = "192.168.4.1", 
                 port: int = 5000,
                 timeout: float = 5.0):
        """Initialize AI Deck camera interface.
        
        Args:
            ip: AI Deck IP address (default when in AP mode)
            port: Camera streaming port
            timeout: Socket timeout in seconds
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Frame buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        self._receiver_thread: Optional[threading.Thread] = None
        
        # Stats
        self._start_time = 0.0
        self._last_frame_time = 0.0
        
        logger.info(f"[AIDECK] Camera initialized (target: {ip}:{port})")
    
    def connect(self) -> bool:
        """Connect to AI Deck WiFi stream.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"[AIDECK] Connecting to {self.ip}:{self.port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.ip, self.port))
            
            self.connected = True
            self._start_time = time.time()
            
            # Start receiver thread
            self.running = True
            self._receiver_thread = threading.Thread(
                target=self._receive_loop, 
                daemon=True,
                name="AIdeckCamera"
            )
            self._receiver_thread.start()
            
            logger.info(f"[AIDECK] ✓ Connected to camera stream")
            return True
            
        except socket.timeout:
            logger.error(f"[AIDECK] ✗ Connection timeout")
            return False
        except ConnectionRefusedError:
            logger.error(f"[AIDECK] ✗ Connection refused - is AI deck streaming?")
            return False
        except Exception as e:
            logger.error(f"[AIDECK] ✗ Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close connection to AI Deck."""
        self.running = False
        
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1.0)
        
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        
        self.connected = False
        logger.info("[AIDECK] Disconnected from camera")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame.
        
        Returns:
            numpy array (H, W, C) in BGR format, or None if no frame available
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    @property
    def fps(self) -> float:
        """Get current frame rate."""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0
    
    def _receive_loop(self) -> None:
        """Background thread for receiving camera frames."""
        if not CV2_AVAILABLE:
            logger.error("[AIDECK] OpenCV required for frame decoding")
            return
        
        while self.running and self.connected:
            try:
                frame = self._receive_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"[AIDECK] Receive error: {e}")
                break
        
        logger.debug("[AIDECK] Receiver thread exiting")
    
    def _receive_frame(self) -> Optional[np.ndarray]:
        """Receive and decode a single frame.
        
        Returns:
            Decoded frame as numpy array, or None on error
        """
        if not self.socket:
            return None
        
        # Read frame size header (4 bytes, big-endian uint32)
        header = self._recv_exact(4)
        if not header:
            return None
        
        frame_size = struct.unpack('>I', header)[0]
        
        if frame_size <= 0 or frame_size > 1000000:  # Sanity check
            logger.warning(f"[AIDECK] Invalid frame size: {frame_size}")
            return None
        
        # Read JPEG data
        jpeg_data = self._recv_exact(frame_size)
        if not jpeg_data:
            return None
        
        # Decode JPEG
        try:
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.warning(f"[AIDECK] Decode error: {e}")
            return None
    
    def _recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly size bytes from socket.
        
        Args:
            size: Number of bytes to receive
            
        Returns:
            Received bytes, or None on error
        """
        data = b''
        while len(data) < size:
            remaining = size - len(data)
            try:
                chunk = self.socket.recv(remaining)
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                return None
            except Exception:
                return None
        return data
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
