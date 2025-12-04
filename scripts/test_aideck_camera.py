#!/usr/bin/env python3
"""Test AI Deck camera streaming.

This script tests the WiFi camera streaming from the Crazyflie AI Deck.

Prerequisites:
1. AI Deck flashed with wifi-img-streamer (JPEG streaming)
2. Computer connected to AI Deck WiFi AP (typically "Bitcraze AI-deck")
3. Default IP: 192.168.4.1, Port: 5000

Usage:
    python scripts/test_aideck_camera.py              # Basic test
    python scripts/test_aideck_camera.py --display    # Show live video
    python scripts/test_aideck_camera.py --save       # Save frames to disk
    python scripts/test_aideck_camera.py --ip 192.168.4.1 --port 5000
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - display mode disabled")

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


def test_connection(ip: str, port: int, timeout: float = 5.0) -> bool:
    """Test basic TCP connection to AI Deck.
    
    Args:
        ip: AI Deck IP address
        port: Streaming port
        timeout: Connection timeout
        
    Returns:
        True if connection successful
    """
    import socket
    
    print(f"\n{'='*60}")
    print("AI DECK CAMERA CONNECTION TEST")
    print(f"{'='*60}")
    print(f"Target: {ip}:{port}")
    print(f"Timeout: {timeout}s")
    print()
    
    try:
        print("[1/3] Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        print(f"[2/3] Connecting to {ip}:{port}...")
        sock.connect((ip, port))
        
        print("[3/3] Connection established!")
        sock.close()
        
        print(f"\n✓ SUCCESS: AI Deck camera is reachable at {ip}:{port}")
        return True
        
    except socket.timeout:
        print(f"\n✗ TIMEOUT: Could not connect within {timeout}s")
        print("\nTroubleshooting:")
        print("  1. Ensure AI Deck is powered on")
        print("  2. Connect to 'Bitcraze AI-deck' WiFi network")
        print("  3. Verify wifi-img-streamer is running on AI Deck")
        return False
        
    except ConnectionRefusedError:
        print(f"\n✗ REFUSED: Connection to {ip}:{port} refused")
        print("\nTroubleshooting:")
        print("  1. Check wifi-img-streamer is running on GAP8")
        print("  2. Verify correct port number")
        return False
        
    except OSError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you're connected to AI Deck WiFi")
        print("  2. Check IP address is correct (default: 192.168.4.1)")
        return False


def test_streaming(ip: str, port: int, 
                   duration: float = 10.0,
                   display: bool = False,
                   save: bool = False,
                   save_dir: str = "data/aideck_test") -> bool:
    """Test camera frame streaming.
    
    Args:
        ip: AI Deck IP address
        port: Streaming port
        duration: Test duration in seconds
        display: Show frames in window
        save: Save frames to disk
        save_dir: Directory for saved frames
        
    Returns:
        True if frames received successfully
    """
    from src.hardware.aideck_camera import AIdeckCamera
    
    print(f"\n{'='*60}")
    print("AI DECK CAMERA STREAMING TEST")
    print(f"{'='*60}")
    print(f"Target: {ip}:{port}")
    print(f"Duration: {duration}s")
    print(f"Display: {display}")
    print(f"Save frames: {save}")
    print()
    
    if save:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {save_path.absolute()}")
    
    camera = AIdeckCamera(ip=ip, port=port)
    
    print("\n[1/4] Connecting to camera...")
    if not camera.connect():
        print("\n✗ FAILED: Could not connect to camera stream")
        return False
    
    print("[2/4] Waiting for first frame...")
    
    start_time = time.time()
    frames_received = 0
    first_frame_time = None
    last_fps_report = start_time
    
    try:
        while time.time() - start_time < duration:
            frame = camera.get_frame()
            
            if frame is not None:
                frames_received += 1
                
                if first_frame_time is None:
                    first_frame_time = time.time() - start_time
                    print(f"[3/4] First frame received in {first_frame_time:.2f}s")
                    print(f"      Resolution: {camera.resolution[0]}x{camera.resolution[1]}")
                    print(f"      Frame shape: {frame.shape}")
                
                # Display frame
                if display and CV2_AVAILABLE:
                    # Add FPS overlay
                    fps_text = f"FPS: {camera.fps:.1f} | Frames: {frames_received}"
                    cv2.putText(frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("AI Deck Camera", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        print("\nUser requested exit")
                        break
                
                # Save frame
                if save and CV2_AVAILABLE:
                    frame_path = save_path / f"frame_{frames_received:05d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
            
            # Report FPS periodically
            now = time.time()
            if now - last_fps_report >= 2.0:
                print(f"[4/4] Streaming... FPS: {camera.fps:.1f}, Frames: {frames_received}")
                last_fps_report = now
            
            time.sleep(0.01)  # Small delay to avoid busy-waiting
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.disconnect()
        if display and CV2_AVAILABLE:
            cv2.destroyAllWindows()
    
    # Report results
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("STREAMING TEST RESULTS")
    print(f"{'='*60}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Frames received: {frames_received}")
    print(f"Average FPS: {frames_received / elapsed:.1f}" if elapsed > 0 else "N/A")
    print(f"Resolution: {camera.resolution[0]}x{camera.resolution[1]}")
    
    if frames_received > 0:
        print(f"\n✓ SUCCESS: Camera streaming working")
        if save:
            print(f"  Frames saved to: {save_path.absolute()}")
        return True
    else:
        print(f"\n✗ FAILED: No frames received")
        print("\nTroubleshooting:")
        print("  1. Check wifi-img-streamer is streaming JPEG")
        print("  2. Verify camera is connected to AI Deck")
        return False


def test_depth_integration(ip: str, port: int, duration: float = 5.0) -> bool:
    """Test integration with depth detector.
    
    Args:
        ip: AI Deck IP address
        port: Streaming port
        duration: Test duration
        
    Returns:
        True if depth estimation works
    """
    from src.hardware.aideck_camera import AIdeckCamera
    
    print(f"\n{'='*60}")
    print("AI DECK + DEPTH ESTIMATION TEST")
    print(f"{'='*60}")
    
    # Check for ONNX
    try:
        from src.vision.depth_detector import DepthDetector, ONNX_AVAILABLE
        if not ONNX_AVAILABLE:
            print("⚠ ONNX runtime not available - skipping depth test")
            return False
    except ImportError as e:
        print(f"⚠ Could not import depth detector: {e}")
        return False
    
    # Check for model
    model_path = Path("models/midas_v21_small.onnx")
    if not model_path.exists():
        print(f"⚠ Depth model not found at {model_path}")
        print("  Run: make setup")
        return False
    
    print(f"Model: {model_path}")
    
    # Initialize components
    camera = AIdeckCamera(ip=ip, port=port)
    detector = DepthDetector(depth_model_path=str(model_path), use_gpu=False)
    
    if not camera.connect():
        print("✗ Could not connect to camera")
        return False
    
    print("Waiting for frames...")
    
    # Wait for first frame (up to 2 seconds)
    wait_start = time.time()
    while camera.get_frame() is None and time.time() - wait_start < 2.0:
        time.sleep(0.1)
    
    start_time = time.time()
    depth_frames = 0
    inference_times = []
    
    try:
        while time.time() - start_time < duration:
            frame = camera.get_frame()
            
            if frame is not None:
                # Run depth estimation
                t0 = time.time()
                result = detector.detect(frame)
                depth_map = result.get('depth_map')
                inference_time = (time.time() - t0) * 1000
                
                if depth_map is not None:
                    depth_frames += 1
                    inference_times.append(inference_time)
                    
                    # Analyze depth
                    obstacles = result.get('obstacle_regions', [])
                    
                    if depth_frames == 1:
                        print(f"\nFirst depth frame:")
                        print(f"  Depth shape: {depth_map.shape}")
                        print(f"  Inference: {inference_time:.1f}ms")
                        print(f"  Obstacles detected: {len(obstacles)}")
                    
                    # Display if available
                    if CV2_AVAILABLE:
                        # Normalize depth for visualization
                        depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                        depth_viz = depth_viz.astype(np.uint8)
                        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
                        
                        # Show side by side
                        combined = np.hstack([frame, depth_viz])
                        cv2.imshow("Camera | Depth", combined)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                            break
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.disconnect()
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
    
    # Results
    print(f"\n{'='*60}")
    print("DEPTH INTEGRATION RESULTS")
    print(f"{'='*60}")
    print(f"Depth frames processed: {depth_frames}")
    
    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Estimated throughput: {1000/avg_inference:.1f} FPS")
        print("\n✓ SUCCESS: Depth estimation working with AI Deck camera")
        return True
    else:
        print("\n✗ FAILED: No depth frames processed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test AI Deck camera")
    parser.add_argument("--ip", default="192.168.4.1", help="AI Deck IP address")
    parser.add_argument("--port", type=int, default=5000, help="Streaming port")
    parser.add_argument("--timeout", type=float, default=5.0, help="Connection timeout")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration (seconds)")
    parser.add_argument("--display", action="store_true", help="Display live video")
    parser.add_argument("--save", action="store_true", help="Save frames to disk")
    parser.add_argument("--save-dir", default="data/aideck_test", help="Directory for saved frames")
    parser.add_argument("--depth", action="store_true", help="Test depth estimation integration")
    parser.add_argument("--connection-only", action="store_true", help="Only test connection")
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")
    
    print("\n" + "="*60)
    print("AI DECK CAMERA TEST SUITE")
    print("="*60)
    print(f"\nTarget: {args.ip}:{args.port}")
    print(f"OpenCV available: {CV2_AVAILABLE}")
    
    # Test 1: Connection
    if not test_connection(args.ip, args.port, args.timeout):
        print("\n" + "="*60)
        print("CONNECTION FAILED - Aborting further tests")
        print("="*60)
        sys.exit(1)
    
    if args.connection_only:
        print("\n✓ Connection test passed")
        sys.exit(0)
    
    # Test 2: Streaming
    if not test_streaming(args.ip, args.port, args.duration, 
                          args.display, args.save, args.save_dir):
        print("\n" + "="*60)
        print("STREAMING FAILED")
        print("="*60)
        sys.exit(1)
    
    # Test 3: Depth integration (optional)
    if args.depth:
        if not test_depth_integration(args.ip, args.port, args.duration):
            print("\nDepth integration test failed (non-critical)")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
    sys.exit(0)


if __name__ == "__main__":
    main()
