"""
Vision estimation components for depth and object detection.

This module provides wrappers for vision models used in autonomous navigation.
"""

import numpy as np
import cv2
from pathlib import Path

# Vision dependencies (optional)
VISION_AVAILABLE = False
try:
    import onnxruntime as ort
    VISION_AVAILABLE = True
except ImportError:
    pass


class SimpleDepthEstimator:
    """Lightweight MiDaS depth estimation wrapper."""
    
    def __init__(self, model_path):
        """
        Initialize depth estimator.
        
        Args:
            model_path: Path to MiDaS ONNX model
        """
        if not VISION_AVAILABLE:
            raise RuntimeError("Vision dependencies not available")
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # MiDaS input dimensions
        self.model_height = 256
        self.model_width = 256
        
        print(f"✓ Depth model loaded: {Path(model_path).name}")
    
    def estimate_depth(self, image):
        """
        Estimate depth map from RGB image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Depth map (H, W) - higher values = closer objects
        """
        # Resize and normalize
        img_resized = cv2.resize(image, (self.model_width, self.model_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
        img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
        
        # Run inference
        depth = self.session.run(None, {self.input_name: img_input})[0]
        
        # Resize to original dimensions
        depth = cv2.resize(depth[0], (image.shape[1], image.shape[0]))
        
        return depth


def analyze_depth_map(depth_map):
    """
    Analyze depth map to determine obstacle zones.
    
    MiDaS outputs INVERSE DEPTH (higher value = closer to camera).
    We convert to a clearance metric where:
    - 0.0 = very close obstacle (danger)
    - 1.0 = far/clear (safe)
    
    Args:
        depth_map: Depth map from estimator (H, W)
    
    Returns:
        dict with:
        - left, center, right: clearance in each zone (0=close, 1=far)
        - min_depth: minimum clearance value
        - safe_direction: -1 (left), 0 (center), 1 (right)
        - raw_center: raw MiDaS value for center zone (for debugging)
        - roi_bounds: (top, bottom) of ROI for visualization
    """
    h, w = depth_map.shape
    
    # Focus on MIDDLE band of image (20% to 70% from top)
    # This includes:
    # - Horizontal obstacles at drone height (center band 40-60%)
    # - Objects slightly above/below flight level
    # Excludes:
    # - Top 20%: sky/ceiling (usually far, not relevant)
    # - Bottom 30%: floor (appears close but not an obstacle for forward flight)
    roi_top = int(h * 0.20)
    roi_bottom = int(h * 0.70)
    roi = depth_map[roi_top:roi_bottom, :]
    
    # Also analyze center-band specifically for horizontal obstacles at drone height
    # This band (35-65%) is where obstacles at the same altitude appear
    center_band_top = int(h * 0.35)
    center_band_bottom = int(h * 0.65)
    center_band = depth_map[center_band_top:center_band_bottom, :]
    
    roi_h, roi_w = roi.shape
    
    # Divide ROI into three vertical zones (left/center/right)
    left_zone = roi[:, :roi_w//3]
    center_zone = roi[:, roi_w//3:2*roi_w//3]
    right_zone = roi[:, 2*roi_w//3:]
    
    # Get max values in each zone (MiDaS: higher = closer = more dangerous)
    # Using 90th percentile to be robust to noise while still catching obstacles
    left_max = np.percentile(left_zone, 90)
    center_max = np.percentile(center_zone, 90)
    right_max = np.percentile(right_zone, 90)
    
    # Analyze center-band for HORIZONTAL obstacles at drone height
    # These obstacles are dangerous because the drone can't go over/under easily
    cb_h, cb_w = center_band.shape
    cb_left = center_band[:, :cb_w//3]
    cb_center = center_band[:, cb_w//3:2*cb_w//3]
    cb_right = center_band[:, 2*cb_w//3:]
    
    # Use 98th percentile for center-band - thin wires may only be a few pixels
    # Also use max() as backup for very thin obstacles
    cb_left_max = max(np.percentile(cb_left, 98), cb_left.max() * 0.8)
    cb_center_max = max(np.percentile(cb_center, 98), cb_center.max() * 0.8)
    cb_right_max = max(np.percentile(cb_right, 98), cb_right.max() * 0.8)
    
    # Combine: take the maximum (closest obstacle) from both ROI and center-band
    # This ensures horizontal obstacles at drone height are detected
    left_max = max(left_max, cb_left_max)
    center_max = max(center_max, cb_center_max)
    right_max = max(right_max, cb_right_max)
    
    # Use ROI stats for normalization (not global - floor would skew this)
    roi_max = roi.max()
    roi_min = roi.min()
    
    if roi_max - roi_min < 1e-6:
        return {
            'left': 1.0,
            'center': 1.0,
            'right': 1.0,
            'min_depth': 1.0,
            'safe_direction': 0,
            'raw_center': 0.0,
            'roi_bounds': (roi_top, roi_bottom)
        }
    
    # Convert to clearance: 0 = close (high MiDaS value), 1 = far (low MiDaS value)
    def to_clearance(val):
        # Normalize within ROI range then invert
        normalized = (val - roi_min) / (roi_max - roi_min)
        return 1.0 - normalized
    
    left_clearance = to_clearance(left_max)
    center_clearance = to_clearance(center_max)
    right_clearance = to_clearance(right_max)
    
    # Min clearance overall (within ROI)
    min_clearance = to_clearance(roi_max)
    
    # Determine safest direction (highest clearance)
    if center_clearance > 0.45:  # Center is reasonably safe (lowered threshold)
        safe_dir = 0
    elif left_clearance > right_clearance + 0.1:  # Significant difference
        safe_dir = -1  # Go left
    elif right_clearance > left_clearance + 0.1:
        safe_dir = 1  # Go right
    else:
        # Roughly equal - pick based on which has more room
        safe_dir = -1 if left_clearance >= right_clearance else 1
    
    # Detect if there's a horizontal obstacle (high inverse depth in center-band)
    # A horizontal obstacle at drone height is detected when:
    # - The CENTER zone has low clearance (obstacle directly ahead), OR
    # - Multiple zones have low clearance (wide obstacle)
    horizontal_obstacle_threshold = 0.35  # If clearance is below this in center-band
    cb_clearances = [to_clearance(cb_left_max), to_clearance(cb_center_max), to_clearance(cb_right_max)]
    center_band_has_obstacle = cb_clearances[1] < horizontal_obstacle_threshold  # Center zone
    multiple_zones_blocked = sum(1 for c in cb_clearances if c < horizontal_obstacle_threshold) >= 2
    has_horizontal_obstacle = center_band_has_obstacle or multiple_zones_blocked
    
    return {
        'left': left_clearance,
        'center': center_clearance,
        'right': right_clearance,
        'min_depth': min_clearance,
        'safe_direction': safe_dir,
        'raw_center': center_max,
        'roi_bounds': (roi_top, roi_bottom),
        'horizontal_obstacle': has_horizontal_obstacle,
        'center_band_clearance': min(cb_clearances)  # Worst case in center-band
    }
