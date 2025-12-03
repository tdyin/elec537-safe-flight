"""
Vision-based Obstacle Detection and Ranging.

Provides obstacle detection using monocular depth estimation:
- Region-based obstacle detection (left, center, right)
- Distance estimation with zone classification
- Clear decision outputs for when to maneuver
"""

import numpy as np
from typing import Optional
from pathlib import Path
import cv2

from .zones import ObstacleZone
from .depth_analyzer import DepthAnalyzer

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

# Vision dependencies
VISION_AVAILABLE = False
try:
    import onnxruntime as ort
    VISION_AVAILABLE = True
except ImportError:
    pass


class ObstacleDetector:
    """
    Vision-based obstacle detection using monocular depth estimation.
    
    Uses MiDaS or similar depth model to estimate relative distances
    and determine when/whether the drone should maneuver.
    """
    
    def __init__(self, model_path: str,
                 critical_distance: float = 0.8,
                 close_distance: float = 1.5,
                 caution_distance: float = 2.5,
                 far_distance: float = 4.0,
                 depth_scale: float = 5.0,
                 config: dict = None):
        """
        Initialize obstacle detector.
        
        Args:
            model_path: Path to MiDaS ONNX model
            critical_distance: Distance for critical zone (meters, estimated)
            close_distance: Distance for close zone (meters)
            caution_distance: Distance for caution zone (meters)
            far_distance: Distance for far zone (meters)
            depth_scale: Scale factor to convert depth to meters
            config: Optional configuration dictionary
        """
        if not VISION_AVAILABLE:
            raise RuntimeError("ONNX runtime not available")
        
        # Load config values if provided
        vision_config = config.get('vision', {}) if config else {}
        depth_config = vision_config.get('depth', {})
        obstacle_config = vision_config.get('obstacle', {})
        
        # Load depth model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Model dimensions (from config or defaults)
        self.model_height = depth_config.get('model_height', 256)
        self.model_width = depth_config.get('model_width', 256)
        
        # Distance thresholds
        self.critical_distance = critical_distance
        self.close_distance = close_distance
        self.caution_distance = caution_distance
        self.far_distance = far_distance
        self.depth_scale = depth_config.get('depth_scale', depth_scale)
        
        # Initialize depth analyzer with config
        self.depth_analyzer = DepthAnalyzer(
            roi_top_ratio=obstacle_config.get('roi_top_ratio', 0.20),
            roi_bottom_ratio=obstacle_config.get('roi_bottom_ratio', 0.65),
            critical_threshold=obstacle_config.get('critical_threshold', 0.10),
            close_threshold=obstacle_config.get('close_threshold', 0.20),
            caution_threshold=obstacle_config.get('caution_threshold', 0.35),
            far_threshold=obstacle_config.get('far_threshold', 0.50),
            horizontal_obstacle_threshold=obstacle_config.get('horizontal_obstacle_threshold', 0.35)
        )
        
        # Statistics
        self.frame_count = 0
        self.obstacle_detections = 0
        
        log(f"[OBSTACLE] Detector initialized with model: {Path(model_path).name}", "SUCCESS")
        log(f"[OBSTACLE] Thresholds - Critical:{self.depth_analyzer.critical_threshold:.2f} "
            f"Close:{self.depth_analyzer.close_threshold:.2f} "
            f"Caution:{self.depth_analyzer.caution_threshold:.2f}", "DEBUG")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from camera image.
        
        Args:
            image: BGR image from camera (H, W, 3)
            
        Returns:
            Depth map (H, W) - higher values = closer objects (MiDaS convention)
        """
        # Resize and normalize for model
        img_resized = cv2.resize(image, (self.model_width, self.model_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
        img_input = np.expand_dims(img_input, axis=0)  # Add batch
        
        # Run inference
        depth = self.session.run(None, {self.input_name: img_input})[0]
        
        # Resize to original image dimensions
        depth = cv2.resize(depth[0], (image.shape[1], image.shape[0]))
        
        return depth
    
    def detect(self, image: np.ndarray) -> dict:
        """
        Detect obstacles and determine if maneuver is required.
        
        This is the main interface for the navigation controller.
        
        Args:
            image: BGR camera image
            
        Returns:
            dict with:
            - should_maneuver: bool, whether drone should start avoiding
            - zone: ObstacleZone enum indicating urgency
            - direction: Recommended avoidance direction (-1=left, 0=none, 1=right)
            - clearance: dict with left/center/right clearance values (0-1)
            - distance_estimate: Estimated distance to closest obstacle (meters)
            - confidence: Detection confidence (0-1)
        """
        self.frame_count += 1
        
        # Default result for when detection fails
        default_result = {
            'should_maneuver': False,
            'zone': ObstacleZone.CLEAR,
            'direction': 0,
            'vertical_direction': 0,
            'vertical_magnitude': 0.0,
            'clearance': {'left': 1.0, 'center': 1.0, 'right': 1.0},
            'vertical_clearance': {'upper': 1.0, 'middle': 1.0, 'lower': 1.0},
            'distance_estimate': float('inf'),
            'confidence': 0.0,
            'horizontal_obstacle': False,
            'center_band_clearance': 1.0
        }
        
        if image is None:
            return default_result
        
        try:
            # Get depth map
            depth_map = self.estimate_depth(image)
            
            # Analyze depth to get obstacle info
            analysis = self.depth_analyzer.analyze(depth_map)
            
            # Determine zone and whether to maneuver
            center_clearance = analysis['center']
            zone = self.depth_analyzer.classify_zone(center_clearance)
            
            # Check for horizontal obstacles at drone height
            has_horizontal_obstacle = analysis.get('horizontal_obstacle', False)
            center_band_clearance = analysis.get('center_band_clearance', 1.0)
            
            # Horizontal obstacles are especially dangerous - upgrade zone if needed
            if has_horizontal_obstacle:
                if zone == ObstacleZone.CLEAR or zone == ObstacleZone.FAR:
                    zone = ObstacleZone.CAUTION
                elif zone == ObstacleZone.CAUTION:
                    zone = ObstacleZone.CLOSE
            
            # Should maneuver if in CLOSE or CRITICAL zone
            should_maneuver = zone in [ObstacleZone.CLOSE, ObstacleZone.CRITICAL]
            
            if should_maneuver:
                self.obstacle_detections += 1
            
            # Determine avoidance direction
            direction = self.depth_analyzer.compute_avoidance_direction(analysis)
            
            # Compute vertical avoidance using vertical zone analysis
            vertical_direction, vertical_magnitude = self.depth_analyzer.compute_vertical_avoidance(analysis)
            
            # For horizontal obstacles, ensure vertical escape is considered
            if has_horizontal_obstacle and vertical_direction == 0:
                # Fallback: if horizontal obstacle detected but no clear vertical escape,
                # default to going up if upper zone has any clearance advantage
                upper_cl = analysis.get('upper_clearance', 1.0)
                lower_cl = analysis.get('lower_clearance', 1.0)
                if upper_cl > lower_cl and upper_cl > 0.3:
                    vertical_direction = 1
                    vertical_magnitude = 0.5
                elif lower_cl > upper_cl and lower_cl > 0.3:
                    vertical_direction = -1
                    vertical_magnitude = 0.5
            
            # Estimate distance
            distance_estimate = self._clearance_to_distance(center_clearance)
            
            result = {
                'should_maneuver': should_maneuver,
                'zone': zone,
                'direction': direction,
                'vertical_direction': vertical_direction,
                'vertical_magnitude': vertical_magnitude,
                'clearance': {
                    'left': analysis['left'],
                    'center': analysis['center'],
                    'right': analysis['right']
                },
                'vertical_clearance': analysis.get('vertical_clearance', {
                    'upper': 1.0, 'middle': 1.0, 'lower': 1.0
                }),
                'distance_estimate': distance_estimate,
                'confidence': analysis['confidence'],
                'raw_depth': depth_map,  # For visualization
                'roi_bounds': analysis['roi_bounds'],
                'horizontal_obstacle': has_horizontal_obstacle,
                'center_band_clearance': center_band_clearance
            }
            
            return result
            
        except Exception as e:
            log(f"[OBSTACLE] Detection error: {e}", "ERROR")
            return default_result
    
    def _clearance_to_distance(self, clearance: float) -> float:
        """
        Convert clearance value to estimated distance in meters.
        
        This is a rough estimate since monocular depth is relative.
        
        Args:
            clearance: Clearance value (0-1)
            
        Returns:
            Estimated distance in meters
        """
        if clearance >= 1.0:
            return float('inf')
        
        # Exponential mapping: clearance 0 -> 0.5m, clearance 1 -> inf
        base_distance = 0.5
        clamped = max(0.01, min(0.99, clearance))
        distance = base_distance / (1.0 - clamped)
        
        return min(distance * self.depth_scale, 100.0)  # Cap at 100m
    
    def get_maneuver_recommendation(self, detection_result: dict) -> tuple:
        """
        Get clear maneuver recommendation from detection result.
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Tuple of (should_maneuver, reason, direction)
        """
        zone = detection_result['zone']
        direction = detection_result['direction']
        distance = detection_result['distance_estimate']
        
        if zone == ObstacleZone.CRITICAL:
            return True, f"CRITICAL: Obstacle at {distance:.1f}m - STOP", direction
        elif zone == ObstacleZone.CLOSE:
            dir_str = "LEFT" if direction == -1 else "RIGHT" if direction == 1 else "AWAY"
            return True, f"CLOSE: Obstacle at {distance:.1f}m - Turn {dir_str}", direction
        elif zone == ObstacleZone.CAUTION:
            return False, f"CAUTION: Obstacle at {distance:.1f}m - Prepare", direction
        elif zone == ObstacleZone.FAR:
            return False, f"FAR: Obstacle detected at {distance:.1f}m", 0
        else:
            return False, "CLEAR: No obstacles", 0
    
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        return {
            'frame_count': self.frame_count,
            'obstacle_detections': self.obstacle_detections,
            'detection_rate': self.obstacle_detections / max(1, self.frame_count)
        }


class SimpleObstacleDetector:
    """
    Simplified obstacle detector wrapper for compatibility.
    
    Wraps ObstacleDetector with a simpler interface matching
    the existing analyze_depth_map function signature.
    """
    
    def __init__(self, model_path: str):
        """Initialize with depth model path."""
        self.detector = ObstacleDetector(model_path)
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyze image for obstacles.
        
        Args:
            image: Camera image
            
        Returns:
            dict compatible with existing stable_avoidance interface
        """
        result = self.detector.detect(image)
        
        # Convert to format expected by existing code
        return {
            'left': result['clearance']['left'],
            'center': result['clearance']['center'],
            'right': result['clearance']['right'],
            'min_depth': min(result['clearance'].values()),
            'safe_direction': result['direction'],
            'should_maneuver': result['should_maneuver'],
            'zone': result['zone'].value,
            'distance_estimate': result['distance_estimate']
        }
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Get raw depth map."""
        return self.detector.estimate_depth(image)
