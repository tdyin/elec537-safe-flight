"""
Vision processing utilities for obstacle detection.

Provides helper functions for processing vision data and detecting obstacles.
"""

import sys
from pathlib import Path

# Try to import object detector from src/vision
DETECTOR_AVAILABLE = False
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))
    from vision.detector import ObstacleDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    pass


class VisionProcessor:
    """Processes vision data for obstacle detection."""
    
    def __init__(self, object_detector=None):
        """
        Initialize vision processor.
        
        Args:
            object_detector: Optional ObstacleDetector instance
        """
        self.object_detector = object_detector
        self.last_detections = []
        self.last_wire_zones = None
        self.last_depth_zones = None
        self.vision_inference_counter = 0
        self.vision_interval = 2  # Process vision every N timesteps (reduced from 5 for better obstacle avoidance)
    
    def detect_obstacles(self, image):
        """
        Detect obstacles using object detector (if available).
        
        Args:
            image: Grayscale or RGB image
            
        Returns:
            List of detections or empty list
        """
        if not self.object_detector or image is None:
            return []
        
        try:
            # Use wire detection if available
            if hasattr(self.object_detector, 'detect_with_wire_detection'):
                detections = self.object_detector.detect_with_wire_detection(image)
            else:
                detections = self.object_detector.detect(image)
            
            self.last_detections = detections
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_wire_zones(self, image):
        """
        Detect wire obstacles in zones (left/center/right).
        
        Args:
            image: Grayscale or RGB image
            
        Returns:
            dict with 'left', 'center', 'right' zone scores (0-1)
        """
        default_zones = {'left': 0.0, 'center': 0.0, 'right': 0.0}
        
        if not self.object_detector or image is None:
            return default_zones
        
        try:
            if hasattr(self.object_detector, 'get_wire_obstacle_zones'):
                zones = self.object_detector.get_wire_obstacle_zones(image)
                self.last_wire_zones = zones
                return zones
        except Exception as e:
            print(f"Wire detection error: {e}")
        
        return default_zones
    
    def should_process_vision(self):
        """
        Determine if vision should be processed this timestep.
        
        Returns:
            True if vision should be processed
        """
        self.vision_inference_counter += 1
        if self.vision_inference_counter >= self.vision_interval:
            self.vision_inference_counter = 0
            return True
        return False
