"""Depth map analysis for obstacle detection.

Provides analysis of depth maps to extract obstacle information,
including zone clearances, horizontal obstacle detection, and
vertical zone analysis for altitude-based avoidance.
"""

import numpy as np
from typing import Tuple, Dict

from .zones import ObstacleZone


class DepthAnalyzer:
    """
    Analyzes depth maps to detect obstacles and compute clearances.
    
    Converts MiDaS depth output (higher = closer) to clearance metrics
    (0 = close/danger, 1 = far/safe).
    """
    
    def __init__(self, 
                 roi_top_ratio: float = 0.20,
                 roi_bottom_ratio: float = 0.65,
                 critical_threshold: float = 0.10,
                 close_threshold: float = 0.20,
                 caution_threshold: float = 0.35,
                 far_threshold: float = 0.50,
                 horizontal_obstacle_threshold: float = 0.35):
        """
        Initialize depth analyzer.
        
        Args:
            roi_top_ratio: Top boundary of ROI as ratio of image height
            roi_bottom_ratio: Bottom boundary of ROI as ratio of image height
            critical_threshold: Clearance threshold for critical zone
            close_threshold: Clearance threshold for close zone
            caution_threshold: Clearance threshold for caution zone
            far_threshold: Clearance threshold for far zone
            horizontal_obstacle_threshold: Threshold for horizontal obstacle detection
        """
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        
        # Clearance thresholds (0 = close, 1 = far)
        self.critical_threshold = critical_threshold
        self.close_threshold = close_threshold
        self.caution_threshold = caution_threshold
        self.far_threshold = far_threshold
        
        # Horizontal obstacle detection
        self.horizontal_obstacle_threshold = horizontal_obstacle_threshold
        
        # Vertical zone boundaries (as ratios of image height)
        # Upper zone: obstacles above drone (going up would hit them)
        # Middle zone: obstacles at drone height (horizontal obstacles)
        # Lower zone: obstacles below drone (going down would hit them)
        self.vertical_upper_ratio = 0.30    # Top 30% of image
        self.vertical_middle_top = 0.30     # Middle zone: 30-70%
        self.vertical_middle_bottom = 0.70
        self.vertical_lower_ratio = 0.70    # Bottom 30% of image
    
    def analyze(self, depth_map: np.ndarray) -> dict:
        """
        Analyze depth map to extract obstacle information.
        
        Args:
            depth_map: Raw depth map from MiDaS model (H, W)
            
        Returns:
            Analysis dict with clearance values for each zone:
            - left: Left zone clearance (0-1)
            - center: Center zone clearance (0-1)
            - right: Right zone clearance (0-1)
            - confidence: Detection confidence
            - roi_bounds: (top, bottom) pixel boundaries of ROI
            - horizontal_obstacle: Whether a horizontal obstacle is detected
            - center_band_clearance: Minimum clearance in center band
        """
        h, w = depth_map.shape
        
        # Extract ROI - middle band focusing on flight path
        roi_top = int(h * self.roi_top_ratio)
        roi_bottom = int(h * self.roi_bottom_ratio)
        roi = depth_map[roi_top:roi_bottom, :]
        
        # Also analyze center-band specifically for horizontal obstacles at drone height
        # This band (35-65%) is where obstacles at the same altitude appear
        center_band_top = int(h * 0.35)
        center_band_bottom = int(h * 0.65)
        center_band = depth_map[center_band_top:center_band_bottom, :]
        
        roi_h, roi_w = roi.shape
        cb_h, cb_w = center_band.shape
        
        # Divide into three vertical zones (left/center/right)
        third = roi_w // 3
        left_zone = roi[:, :third]
        center_zone = roi[:, third:2*third]
        right_zone = roi[:, 2*third:]
        
        # Get obstacle indicators (high depth = close in MiDaS)
        # Use 90th percentile to be robust to noise
        left_max = np.percentile(left_zone, 90)
        center_max = np.percentile(center_zone, 90)
        right_max = np.percentile(right_zone, 90)
        
        # Analyze center-band for HORIZONTAL obstacles at drone height
        cb_third = cb_w // 3
        cb_left_zone = center_band[:, :cb_third]
        cb_center_zone = center_band[:, cb_third:2*cb_third]
        cb_right_zone = center_band[:, 2*cb_third:]
        
        # Use 98th percentile for center-band - thin wires may only be a few pixels
        cb_left_max = max(np.percentile(cb_left_zone, 98), cb_left_zone.max() * 0.8)
        cb_center_max = max(np.percentile(cb_center_zone, 98), cb_center_zone.max() * 0.8)
        cb_right_max = max(np.percentile(cb_right_zone, 98), cb_right_zone.max() * 0.8)
        
        # Combine: take the maximum (closest obstacle) from both ROI and center-band
        left_max = max(left_max, cb_left_max)
        center_max = max(center_max, cb_center_max)
        right_max = max(right_max, cb_right_max)
        
        # Convert to clearance (0=close, 1=far)
        left_clearance, center_clearance, right_clearance = self._to_clearances(
            left_max, center_max, right_max
        )
        
        # Detect horizontal obstacles
        cb_clearances = [
            self._to_clearance_single(cb_left_max),
            self._to_clearance_single(cb_center_max),
            self._to_clearance_single(cb_right_max)
        ]
        
        center_band_has_obstacle = cb_clearances[1] < self.horizontal_obstacle_threshold
        multiple_zones_blocked = sum(1 for c in cb_clearances if c < self.horizontal_obstacle_threshold) >= 2
        has_horizontal_obstacle = center_band_has_obstacle or multiple_zones_blocked
        
        # Compute confidence based on depth range in ROI
        roi_max = roi.max()
        roi_min = roi.min()
        depth_range = roi_max - roi_min
        confidence = min(1.0, depth_range / 200.0)
        
        # Analyze vertical zones for altitude-based avoidance
        vertical_analysis = self.analyze_vertical_zones(depth_map)
        
        return {
            'left': left_clearance,
            'center': center_clearance,
            'right': right_clearance,
            'confidence': confidence,
            'roi_bounds': (roi_top, roi_bottom),
            'horizontal_obstacle': has_horizontal_obstacle,
            'center_band_clearance': min(cb_clearances),
            # Vertical zone analysis
            'vertical_clearance': vertical_analysis['vertical_clearance'],
            'upper_clearance': vertical_analysis['upper_clearance'],
            'middle_clearance': vertical_analysis['middle_clearance'],
            'lower_clearance': vertical_analysis['lower_clearance'],
            'optimal_vertical_direction': vertical_analysis['optimal_direction'],
            'vertical_escape_margin': vertical_analysis['escape_margin']
        }
    
    def _to_clearances(self, left_max: float, center_max: float, right_max: float) -> Tuple[float, float, float]:
        """
        Convert raw depth values to clearance metrics.
        
        Uses GLOBAL normalization instead of per-frame normalization.
        MiDaS depth values are relative but consistent within a scene.
        
        Args:
            left_max: Maximum depth value in left zone
            center_max: Maximum depth value in center zone
            right_max: Maximum depth value in right zone
            
        Returns:
            Tuple of (left_clearance, center_clearance, right_clearance)
        """
        global_max = 1000.0  # MiDaS typical max for indoor scenes
        
        def to_clearance(val):
            val = np.clip(val, 0.0, global_max)
            normalized = val / global_max
            return 1.0 - normalized
        
        return (
            to_clearance(left_max),
            to_clearance(center_max),
            to_clearance(right_max)
        )
    
    def _to_clearance_single(self, value: float) -> float:
        """Convert a single depth value to clearance."""
        global_max = 1000.0
        value = np.clip(value, 0.0, global_max)
        normalized = value / global_max
        return 1.0 - normalized
    
    def classify_zone(self, clearance: float) -> ObstacleZone:
        """
        Classify obstacle zone based on clearance value.
        
        Args:
            clearance: Clearance value (0=close, 1=far)
            
        Returns:
            ObstacleZone classification
        """
        if clearance < self.critical_threshold:
            return ObstacleZone.CRITICAL
        elif clearance < self.close_threshold:
            return ObstacleZone.CLOSE
        elif clearance < self.caution_threshold:
            return ObstacleZone.CAUTION
        elif clearance < self.far_threshold:
            return ObstacleZone.FAR
        else:
            return ObstacleZone.CLEAR
    
    def analyze_vertical_zones(self, depth_map: np.ndarray) -> Dict:
        """
        Analyze depth map split into vertical zones (upper/middle/lower).
        
        This provides guidance for altitude-based obstacle avoidance:
        - Upper zone: Obstacles that would be hit if drone goes UP
        - Middle zone: Obstacles at current drone altitude (horizontal obstacles)
        - Lower zone: Obstacles that would be hit if drone goes DOWN
        
        Args:
            depth_map: Raw depth map from MiDaS model (H, W)
            
        Returns:
            Dict with:
            - upper_clearance: Clearance in upper zone (0-1)
            - middle_clearance: Clearance in middle zone (0-1)
            - lower_clearance: Clearance in lower zone (0-1)
            - vertical_clearance: Dict with all three clearances
            - optimal_direction: -1 (down), 0 (none), 1 (up)
            - escape_margin: How much better the optimal direction is
        """
        h, w = depth_map.shape
        
        # Define vertical zone boundaries
        upper_bottom = int(h * self.vertical_upper_ratio)
        middle_top = int(h * self.vertical_middle_top)
        middle_bottom = int(h * self.vertical_middle_bottom)
        lower_top = int(h * self.vertical_lower_ratio)
        
        # Extract zones - focus on center third horizontally for forward path
        third = w // 3
        center_left = third
        center_right = 2 * third
        
        upper_zone = depth_map[:upper_bottom, center_left:center_right]
        middle_zone = depth_map[middle_top:middle_bottom, center_left:center_right]
        lower_zone = depth_map[lower_top:, center_left:center_right]
        
        # Get obstacle indicators (high depth = close in MiDaS)
        # Use 95th percentile to catch obstacles while being robust to noise
        upper_max = np.percentile(upper_zone, 95) if upper_zone.size > 0 else 0
        middle_max = np.percentile(middle_zone, 95) if middle_zone.size > 0 else 0
        lower_max = np.percentile(lower_zone, 95) if lower_zone.size > 0 else 0
        
        # Convert to clearance (0=close, 1=far)
        upper_clearance = self._to_clearance_single(upper_max)
        middle_clearance = self._to_clearance_single(middle_max)
        lower_clearance = self._to_clearance_single(lower_max)
        
        # Determine optimal vertical direction
        # Only suggest vertical escape if middle zone is blocked
        optimal_direction = 0
        escape_margin = 0.0
        
        if middle_clearance < self.caution_threshold:
            # Middle zone is blocked - need to escape vertically or laterally
            margin = 0.15  # Require significant difference to suggest direction
            
            # Compare upper vs lower clearance
            if upper_clearance > lower_clearance + margin:
                # Upper is clearer - go UP (but only if upper is actually safe)
                if upper_clearance > self.close_threshold:
                    optimal_direction = 1  # Go up
                    escape_margin = upper_clearance - middle_clearance
            elif lower_clearance > upper_clearance + margin:
                # Lower is clearer - go DOWN (but only if lower is actually safe)
                if lower_clearance > self.close_threshold:
                    optimal_direction = -1  # Go down
                    escape_margin = lower_clearance - middle_clearance
            else:
                # Both directions similar - prefer UP (safer default)
                if upper_clearance > self.close_threshold:
                    optimal_direction = 1
                    escape_margin = upper_clearance - middle_clearance
        
        return {
            'upper_clearance': upper_clearance,
            'middle_clearance': middle_clearance,
            'lower_clearance': lower_clearance,
            'vertical_clearance': {
                'upper': upper_clearance,
                'middle': middle_clearance,
                'lower': lower_clearance
            },
            'optimal_direction': optimal_direction,
            'escape_margin': escape_margin
        }
    
    def compute_avoidance_direction(self, analysis: dict) -> int:
        """
        Compute recommended avoidance direction.
        
        Args:
            analysis: Depth analysis results from analyze()
            
        Returns:
            -1 for left, 0 for straight, 1 for right
        """
        left = analysis['left']
        center = analysis['center']
        right = analysis['right']
        
        # If center is clear, no direction needed
        if center > self.caution_threshold:
            return 0
        
        # Choose direction with more clearance
        margin = 0.1  # Require significant difference
        
        if left > right + margin:
            return -1  # Go left
        elif right > left + margin:
            return 1   # Go right
        else:
            # Similar clearance - prefer right (convention)
            return 1 if right >= left else -1
    
    def compute_vertical_avoidance(self, analysis: dict) -> Tuple[int, float]:
        """
        Compute recommended vertical avoidance direction and magnitude.
        
        Args:
            analysis: Depth analysis results from analyze()
            
        Returns:
            Tuple of (direction, magnitude):
            - direction: -1 (down), 0 (none), 1 (up)
            - magnitude: Suggested altitude change rate (0.0-1.0)
        """
        optimal_dir = analysis.get('optimal_vertical_direction', 0)
        escape_margin = analysis.get('vertical_escape_margin', 0.0)
        middle_clearance = analysis.get('middle_clearance', 1.0)
        
        if optimal_dir == 0:
            return 0, 0.0
        
        # Scale magnitude based on urgency
        # Lower middle clearance = more urgent = higher magnitude
        if middle_clearance < self.critical_threshold:
            magnitude = 1.0  # Maximum urgency
        elif middle_clearance < self.close_threshold:
            magnitude = 0.7
        elif middle_clearance < self.caution_threshold:
            magnitude = 0.4
        else:
            magnitude = 0.2
        
        # Also consider escape margin - larger margin = more confident
        magnitude *= min(1.0, escape_margin / 0.3 + 0.5)
        
        return optimal_dir, min(1.0, magnitude)
