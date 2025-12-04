"""AI Deck camera streaming over WiFi.

This module provides camera image streaming from the Crazyflie AI Deck
over its WiFi connection using the CPX (Crazyflie Packet eXchange) protocol.

Protocol details (from Bitcraze wifi-img-streamer):
- Packet header: 4 bytes <HBB (length, routing, function)
- Image header: magic(0xBC), width, height, depth, format, size
- Image data: chunked with packet headers
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


# Image format constants
FORMAT_RAW = 0
FORMAT_JPEG = 1
MAGIC_BYTE = 0xBC


class AIdeckCamera:
    """WiFi camera streaming from AI Deck.
    
    Connects to the AI Deck's WiFi access point and receives
    camera frames over TCP using the CPX protocol.
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
        
        # Image info from last frame
        self._width = 0
        self._height = 0
        self._format = FORMAT_JPEG
        
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
        if elapsed > 0 and self._frame_count > 0:
            return self._frame_count / elapsed
        return 0.0
    
    @property
    def resolution(self) -> tuple:
        """Get image resolution (width, height)."""
        return (self._width, self._height)
    
    def _receive_loop(self) -> None:
        """Background thread for receiving camera frames."""
        if not CV2_AVAILABLE:
            logger.error("[AIDECK] OpenCV required for frame decoding")
            return
        
        logger.debug("[AIDECK] Receiver thread started")
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running and self.connected:
            try:
                frame = self._receive_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    consecutive_errors = 0  # Reset on success
                    
                    if self._frame_count == 1:
                        logger.info(f"[AIDECK] First frame received: {self._width}x{self._height}")
                else:
                    consecutive_errors += 1
                    if consecutive_errors == 1:
                        logger.debug("[AIDECK] No frame received, waiting...")
                    if consecutive_errors >= max_consecutive_errors:
                        # Try to resync by reconnecting
                        logger.warning(f"[AIDECK] {consecutive_errors} consecutive errors, attempting resync...")
                        self._resync_stream()
                        consecutive_errors = 0
                    
            except socket.timeout:
                logger.debug("[AIDECK] Socket timeout in receive loop")
                consecutive_errors += 1
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"[AIDECK] Receive error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                break
        
        logger.debug("[AIDECK] Receiver thread exiting")
    
    def _resync_stream(self) -> None:
        """Attempt to resync the stream by finding next valid frame header.
        
        Scans for a valid packet: 4-byte header with length=13 followed by magic 0xBC.
        """
        if not self.socket:
            return
        
        logger.debug("[AIDECK] Attempting stream resync...")
        
        try:
            buffer = bytearray()
            max_scan = 50000  # Max bytes to scan
            scanned = 0
            
            while scanned < max_scan:
                # Read more data
                chunk = self.socket.recv(1024)
                if not chunk:
                    break
                buffer.extend(chunk)
                scanned += len(chunk)
                
                # Look for valid packet header pattern
                i = 0
                while i < len(buffer) - 5:
                    length = struct.unpack('<H', buffer[i:i+2])[0]
                    # Image header packet is 13 bytes, magic is at offset 4
                    if length == 13 and len(buffer) > i + 4:
                        if buffer[i+4] == MAGIC_BYTE:
                            # Found! Discard everything before
                            del buffer[:i]
                            logger.debug(f"[AIDECK] Resync found valid packet after {scanned} bytes")
                            return
                    i += 1
                
                # Keep last few bytes
                if len(buffer) > 10:
                    del buffer[:-10]
                    
        except Exception as e:
            logger.debug(f"[AIDECK] Resync error: {e}")
    
    def _rx_bytes(self, size: int) -> bytearray:
        """Receive exactly size bytes from socket.
        
        Args:
            size: Number of bytes to receive
            
        Returns:
            Received bytes as bytearray
            
        Raises:
            Exception if socket error or disconnect
        """
        data = bytearray()
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Socket closed")
            data.extend(chunk)
        return data
    
    def _receive_frame(self) -> Optional[np.ndarray]:
        """Receive and decode a single frame using CPX protocol.
        
        Returns:
            Decoded frame as numpy array, or None on error
        """
        if not self.socket:
            return None
        
        try:
            # Read packet info header (4 bytes)
            # Format: <HBB = length (uint16), routing (uint8), function (uint8)
            logger.debug("[AIDECK] Waiting for packet header...")
            packet_info = self._rx_bytes(4)
            length, routing, function = struct.unpack('<HBB', packet_info)
            
            logger.debug(f"[AIDECK] Packet: len={length}, routing={routing}, func={function}")
            
            # Read image header (length - 2 bytes, since length includes routing+function)
            img_header = self._rx_bytes(length - 2)
            
            # Parse image header (11 bytes, packed struct)
            # Format: <BHHBBI = magic(1), width(2), height(2), depth(1), format(1), size(4)
            if len(img_header) < 11:
                logger.warning(f"[AIDECK] Image header too short: {len(img_header)}")
                return None
            
            magic, width, height, depth, img_format, img_size = struct.unpack('<BHHBBI', img_header[:11])
            
            logger.debug(f"[AIDECK] Header: magic=0x{magic:02X}, {width}x{height}, depth={depth}, fmt={img_format}, size={img_size}")
            
            # Verify magic byte
            if magic != MAGIC_BYTE:
                logger.warning(f"[AIDECK] Invalid magic byte: 0x{magic:02X} (expected 0x{MAGIC_BYTE:02X})")
                return None
            
            self._width = width
            self._height = height
            self._format = img_format
            
            # Receive image data in chunks
            img_stream = bytearray()
            while len(img_stream) < img_size:
                # Read chunk header
                chunk_info = self._rx_bytes(4)
                chunk_length, chunk_dst, chunk_src = struct.unpack('<HBB', chunk_info)
                
                # Read chunk data
                chunk_data = self._rx_bytes(chunk_length - 2)
                img_stream.extend(chunk_data)
            
            # Decode image based on format
            if img_format == FORMAT_RAW:
                # Raw Bayer image
                bayer_img = np.frombuffer(img_stream, dtype=np.uint8)
                bayer_img = bayer_img.reshape((height, width))
                # Convert Bayer to BGR
                frame = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
                return frame
                
            elif img_format == FORMAT_JPEG:
                # JPEG image
                nparr = np.frombuffer(img_stream, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("[AIDECK] Failed to decode JPEG")
                    return None
                return frame
                
            else:
                logger.warning(f"[AIDECK] Unknown image format: {img_format}")
                return None
                
        except socket.timeout:
            return None
        except ConnectionError:
            logger.warning("[AIDECK] Connection closed")
            self.connected = False
            return None
        except Exception as e:
            logger.warning(f"[AIDECK] Frame decode error: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
