#!/usr/bin/env python3
"""
Real-time Depth Visualization for Webots Simulation

Displays live depth map analysis during simulation including:
- Horizontal zone clearances (left/center/right)
- Vertical zone clearances (upper/middle/lower) for altitude-based avoidance
- Avoidance direction arrows (lateral and vertical)
- Flight status information

Can be enabled via config or command line.

Usage:
    # Add to config/sim.yaml:
    visualization:
      enabled: true
      depth_display: true
      save_frames: false
      
    # Or run standalone for testing:
    python scripts/live_depth_viz.py --test-image path/to/image.jpg
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add project root (utils is in sim/webots/utils, so go up 3 levels)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available")


class LiveDepthVisualizer:
    """Real-time depth visualization overlay."""
    
    def __init__(self, window_name='Depth Analysis', 
                 display_size=None,
                 save_dir=None):
        """
        Initialize live visualizer.
        
        Args:
            window_name: Name of display window
            display_size: (width, height) of display, or None to use input image size
            save_dir: Optional directory to save frames
        """
        self.window_name = window_name
        self.display_size = display_size  # None means use input size
        self.save_dir = Path(save_dir) if save_dir else None
        self.frame_count = 0
        self.enabled = CV2_AVAILABLE
        self._window_created = False
        self._current_size = None  # Track current window size
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def _ensure_window(self, width=640, height=480):
        """Create or resize window (deferred for Webots compatibility).
        
        Args:
            width: Window width
            height: Window height
        """
        target_size = (width, height)
        
        if self.enabled and not self._window_created:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(self.window_name, width, height)
                cv2.moveWindow(self.window_name, 50, 50)  # Position window
                self._window_created = True
                self._current_size = target_size
            except Exception as e:
                print(f"Warning: Could not create window: {e}")
                self.enabled = False
        elif self._window_created and self._current_size != target_size:
            # Resize if size changed
            try:
                cv2.resizeWindow(self.window_name, width, height)
                self._current_size = target_size
            except Exception:
                pass
            
    def visualize(self, image, depth_map, analysis, state_info=None, flight_info=None):
        """
        Create and display visualization.
        
        Args:
            image: Original camera image (BGR)
            depth_map: Depth map from MiDaS
            analysis: Result from depth analyzer containing:
                - left, center, right: Horizontal clearances (0-1)
                - upper_clearance, middle_clearance, lower_clearance: Vertical clearances
                - vertical_clearance: Dict with upper/middle/lower
                - optimal_vertical_direction: -1 (down), 0 (none), 1 (up)
                - vertical_escape_margin: How much better the escape direction is
                - horizontal_obstacle: Whether horizontal obstacle detected
                - safe_direction or direction: Lateral avoidance direction
            state_info: Optional avoidance state info dict
            flight_info: Optional flight status dict with:
                - position: (x, y, z) current position
                - velocity: (vx, vy) current velocity
                - goal: (x, y, z) goal position
                - mode: navigation mode string
                - waypoint_idx: current waypoint index
                - total_waypoints: total number of waypoints
                - distance_to_goal: distance remaining
                - flight_time: total flight time in seconds
                - vertical_direction: current vertical avoidance direction
                - altitude_change: current altitude change rate
            
        Returns:
            Visualization image (BGR)
        """
        if not self.enabled:
            return None
        
        # Handle grayscale images - convert to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        h, w = image.shape[:2]
        
        # Determine visualization size - use input image size if not specified
        if self.display_size is not None:
            viz_w, viz_h = self.display_size
        else:
            # Use 2x width (for side-by-side layout) and same height as input
            viz_w = w * 2
            viz_h = h
        
        # Create window on first call with appropriate size
        self._ensure_window(viz_w, viz_h)
        if not self._window_created:
            return None
        
        canvas = np.zeros((viz_h, viz_w, 3), dtype=np.uint8)
        
        # Layout: 
        # Left half: original image with overlays
        # Right half: depth map with zone analysis
        half_w = viz_w // 2
        
        # Resize and place original image
        img_resized = cv2.resize(image, (half_w, viz_h))
        canvas[:, :half_w] = img_resized
        
        # Normalize and colorize depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), 
                                          cv2.COLORMAP_VIRIDIS)
        depth_resized = cv2.resize(depth_colored, (half_w, viz_h))
        canvas[:, half_w:] = depth_resized
        
        # Draw zone divisions on right half (depth map) - 3x3 grid
        zone_w = half_w // 3
        upper_line = int(viz_h * 0.30)
        lower_line = int(viz_h * 0.70)
        
        # Vertical lines (left/center/right divisions) on right half
        cv2.line(canvas, (half_w + zone_w, 0), (half_w + zone_w, viz_h), (0, 255, 255), 1)
        cv2.line(canvas, (half_w + 2*zone_w, 0), (half_w + 2*zone_w, viz_h), (0, 255, 255), 1)
        
        # Horizontal lines (upper/middle/lower divisions) on right half
        cv2.line(canvas, (half_w, upper_line), (viz_w, upper_line), (0, 255, 255), 1)
        cv2.line(canvas, (half_w, lower_line), (viz_w, lower_line), (0, 255, 255), 1)
        
        # Horizontal clearance bars at bottom of right half (matching vertical bar style)
        horiz_bar_y = viz_h - 35  # Bottom of canvas
        horiz_bar_h = 25
        horiz_zones = [
            ('L', analysis.get('left', 0), half_w + zone_w // 2),           # Left zone center
            ('C', analysis.get('center', 0), half_w + zone_w + zone_w // 2),  # Center zone
            ('R', analysis.get('right', 0), half_w + 2 * zone_w + zone_w // 2)  # Right zone
        ]
        
        # Draw state info at top-left of left half
        state = 'unknown'
        state_color = (255, 255, 255)
        emg = 0
        avoid = 0
        if state_info:
            state = state_info.get('state', 'unknown')
            state_colors = {
                'normal': (0, 255, 0),
                'caution': (0, 255, 255),
                'avoidance': (0, 128, 255),
                'emergency': (0, 0, 255),
                'recovery': (255, 128, 0),
                'path': (0, 255, 0),
                'avoiding': (0, 128, 255),
                'returning': (255, 128, 0),
                'goal': (0, 255, 128),
                'idle': (128, 128, 128)
            }
            state_color = state_colors.get(state, (255, 255, 255))
            emg = state_info.get('emergency_count', 0)
            avoid = state_info.get('avoidance_count', 0)
            
            # State banner at top
            cv2.rectangle(canvas, (0, 0), (200, 35), state_color, -1)
            cv2.putText(canvas, state.upper(), (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw horizontal clearance bars (L/C/R) at bottom of right half - matching vertical bar style
        for label, clearance, x_center in horiz_zones:
            # Bar dimensions matching U/M/L style
            bar_w = 40
            bar_x1 = x_center - bar_w // 2
            bar_x2 = x_center + bar_w // 2
            
            # Background
            cv2.rectangle(canvas, (bar_x1, horiz_bar_y), (bar_x2, horiz_bar_y + horiz_bar_h), 
                         (50, 50, 50), -1)
            
            # Fill based on clearance (vertical fill from bottom)
            fill_h = int(horiz_bar_h * clearance)
            
            # Color based on clearance
            if clearance < 0.15:
                color = (0, 0, 255)  # Red - critical
            elif clearance < 0.30:
                color = (0, 128, 255)  # Orange - close
            elif clearance < 0.45:
                color = (0, 255, 255)  # Yellow - caution
            else:
                color = (0, 255, 0)  # Green - clear
            
            cv2.rectangle(canvas, (bar_x1 + 2, horiz_bar_y + horiz_bar_h - fill_h), 
                         (bar_x2 - 2, horiz_bar_y + horiz_bar_h - 2), color, -1)
            
            # Label (above the bar)
            cv2.putText(canvas, f'{label}:{clearance:.2f}', 
                       (bar_x1 - 5, horiz_bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
        # Draw four directional arrows originating from center point on left half
        safe_dir = analysis.get('safe_direction', analysis.get('direction', 0))
        vert_dir = analysis.get('optimal_vertical_direction', 0)
        arrow_y = viz_h // 2
        arrow_x = half_w // 2  # Center of left half
        arrow_len = 25  # Reduced arrow length
        arrow_thickness = 2
        
        # Define colors: green for active direction, dim gray for inactive
        active_color = (0, 255, 0)    # Green
        inactive_color = (80, 80, 80)  # Dim gray
        
        # Left arrow (originates from center, points left)
        left_color = active_color if safe_dir == -1 else inactive_color
        cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x - arrow_len, arrow_y),
                       left_color, arrow_thickness, tipLength=0.4)
        
        # Right arrow (originates from center, points right)
        right_color = active_color if safe_dir == 1 else inactive_color
        cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x + arrow_len, arrow_y),
                       right_color, arrow_thickness, tipLength=0.4)
        
        # Up arrow (originates from center, points up)
        up_color = active_color if vert_dir == 1 else inactive_color
        cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x, arrow_y - arrow_len),
                       up_color, arrow_thickness, tipLength=0.4)
        
        # Down arrow (originates from center, points down)
        down_color = active_color if vert_dir == -1 else inactive_color
        cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x, arrow_y + arrow_len),
                       down_color, arrow_thickness, tipLength=0.4)
        
        # Draw vertical clearance bars on the right side of depth panel
        vert_clearance = analysis.get('vertical_clearance', {})
        upper_cl = vert_clearance.get('upper', analysis.get('upper_clearance', 1.0))
        middle_cl = vert_clearance.get('middle', analysis.get('middle_clearance', 1.0))
        lower_cl = vert_clearance.get('lower', analysis.get('lower_clearance', 1.0))
        
        vert_bar_x = viz_w - 35  # Right edge
        vert_bar_w = 25
        vert_zones = [
            ('U', upper_cl, int(viz_h * 0.15)),   # Upper zone center
            ('M', middle_cl, int(viz_h * 0.50)),  # Middle zone center  
            ('L', lower_cl, int(viz_h * 0.85))    # Lower zone center
        ]
        
        for label, clearance, y_center in vert_zones:
            # Vertical bar (horizontal orientation showing clearance)
            bar_h = 40
            bar_y1 = y_center - bar_h // 2
            bar_y2 = y_center + bar_h // 2
            
            # Background
            cv2.rectangle(canvas, (vert_bar_x, bar_y1), (vert_bar_x + vert_bar_w, bar_y2), 
                         (50, 50, 50), -1)
            
            # Fill based on clearance
            fill_w = int(vert_bar_w * clearance)
            
            # Color based on clearance
            if clearance < 0.15:
                color = (0, 0, 255)  # Red - critical
            elif clearance < 0.30:
                color = (0, 128, 255)  # Orange - close
            elif clearance < 0.45:
                color = (0, 255, 255)  # Yellow - caution
            else:
                color = (0, 255, 0)  # Green - clear
            
            cv2.rectangle(canvas, (vert_bar_x + 2, bar_y1 + 2), 
                         (vert_bar_x + fill_w - 2, bar_y2 - 2), color, -1)
            
            # Label
            cv2.putText(canvas, f'{label}:{clearance:.2f}', 
                       (vert_bar_x - 60, y_center + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw flight info panel (under status banner on left half)
        panel_x = 5
        panel_y = 40  # Just below the status banner
        line_height = 20
        panel_height = 130
        panel_width = 190
        
        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        
        current_y = panel_y + line_height

        # EMG and AVD stats
        cv2.putText(canvas, f'EMG: {emg}  AVD: {avoid}', 
                   (panel_x + 5, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        current_y += line_height

        # Position
        if flight_info:
            # Flight time
            flight_time = flight_info.get('flight_time', 0)
            mins = int(flight_time // 60)
            secs = flight_time % 60
            cv2.putText(canvas, f'Flying: {mins}:{secs:05.2f}', 
                       (panel_x + 5, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            current_y += line_height

            # Next waypoint
            next_wp = flight_info.get('next_waypoint', flight_info.get('goal', None))
            if next_wp:
                cv2.putText(canvas, f'Next WP: ({next_wp[0]:.1f}, {next_wp[1]:.1f}, {next_wp[2]:.1f})', 
                           (panel_x + 5, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
            current_y += line_height
            
            # Distance to next waypoint
            dist = flight_info.get('distance_to_waypoint', flight_info.get('distance_to_goal', 0))
            cv2.putText(canvas, f'Dist to WP: {dist:.2f}m', 
                       (panel_x + 5, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
            current_y += line_height

            # Position
            pos = flight_info.get('position', (0, 0, 0))
            cv2.putText(canvas, f'Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})', 
                       (panel_x + 5, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            current_y += line_height
            
            # Velocity
            vel = flight_info.get('velocity', (0, 0))
            vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            cv2.putText(canvas, f'Vel: {vel_mag:.2f} m/s', 
                       (panel_x + 5, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            current_y += line_height
                   
        # Display
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
        
        # Save if enabled
        if self.save_dir:
            save_path = self.save_dir / f'frame_{self.frame_count:06d}.png'
            cv2.imwrite(str(save_path), canvas)
            
        return canvas
        
    def close(self):
        """Close visualization window."""
        if self.enabled and self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass
            self._window_created = False


def test_with_image(image_path):
    """Test visualizer with a static image."""
    if not CV2_AVAILABLE:
        print("OpenCV not available")
        return
        
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
        
    # Try to load depth model
    try:
        import onnxruntime as ort
        model_path = project_root / 'models' / 'midas_v21_small.onnx'
        if not model_path.exists():
            print(f"Depth model not found: {model_path}")
            return
            
        # Load model
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        # Preprocess
        img_resized = cv2.resize(image, (256, 256))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)
        
        # Run inference
        depth = session.run(None, {input_name: img_input})[0]
        depth = cv2.resize(depth[0], (image.shape[1], image.shape[0]))
        
        # Analyze
        sys.path.insert(0, str(project_root / 'sim' / 'webots' / 'controllers' / 'crazyflie_sitl' / 'modules'))
        from vision_estimator import analyze_depth_map
        analysis = analyze_depth_map(depth)
        
        # Visualize
        viz = LiveDepthVisualizer()
        viz.visualize(image, depth, analysis, {'state': 'normal'})
        
        print("Press any key to close...")
        cv2.waitKey(0)
        viz.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-image', type=str, help='Test with static image')
    args = parser.parse_args()
    
    if args.test_image:
        test_with_image(args.test_image)
    else:
        print("Use --test-image to test with a static image")
        print("Or import LiveDepthVisualizer for use in simulation")
