"""Multimodal sensor fusion for robust obstacle detection."""

import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class SensorFusion:
    """Fuses vision and LiDAR detections for robust obstacle detection."""
    
    def __init__(self, 
                 vision_weight: float = 0.5,
                 lidar_weight: float = 0.5,
                 confidence_threshold: float = 0.5):
        """
        Initialize sensor fusion module.
        
        Args:
            vision_weight: Weight for vision detections
            lidar_weight: Weight for LiDAR detections
            confidence_threshold: Minimum confidence for fused detections
        """
        self.vision_weight = vision_weight
        self.lidar_weight = lidar_weight
        self.confidence_threshold = confidence_threshold
        
        # Normalize weights
        total = vision_weight + lidar_weight
        self.vision_weight /= total
        self.lidar_weight /= total
    
    def fuse(self, 
             vision_detections: List[Dict],
             lidar_detections: List[Dict]) -> List[Dict]:
        """
        Fuse vision and LiDAR detections.
        
        Args:
            vision_detections: List of vision-based detections
            lidar_detections: List of LiDAR-based detections
            
        Returns:
            List of fused detections with combined confidence
        """
        if not vision_detections and not lidar_detections:
            return []
        
        if not vision_detections:
            return self._format_lidar_only(lidar_detections)
        
        if not lidar_detections:
            return self._format_vision_only(vision_detections)
        
        # Match detections across modalities
        matches = self._match_detections(vision_detections, lidar_detections)
        
        # Combine matched detections
        fused = []
        for vision_det, lidar_det, confidence in matches:
            if confidence >= self.confidence_threshold:
                fused_det = self._combine_detections(
                    vision_det, lidar_det, confidence
                )
                fused.append(fused_det)
        
        logger.info(f"Fused {len(fused)} obstacles from "
                   f"{len(vision_detections)} vision and "
                   f"{len(lidar_detections)} LiDAR detections")
        
        return fused
    
    def _match_detections(self,
                         vision_detections: List[Dict],
                         lidar_detections: List[Dict]) -> List[Tuple]:
        """
        Match detections between vision and LiDAR.
        
        Args:
            vision_detections: Vision detections
            lidar_detections: LiDAR detections
            
        Returns:
            List of (vision_det, lidar_det, confidence) tuples
        """
        matches = []
        
        # Simple matching based on spatial proximity
        for v_det in vision_detections:
            best_match = None
            best_distance = float('inf')
            
            for l_det in lidar_detections:
                distance = self._compute_distance(v_det, l_det)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = l_det
            
            # Compute confidence based on distance
            if best_match is not None:
                confidence = self._compute_match_confidence(best_distance)
                matches.append((v_det, best_match, confidence))
        
        return matches
    
    def _compute_distance(self, vision_det: Dict, lidar_det: Dict) -> float:
        """
        Compute distance between vision and LiDAR detections.
        
        Args:
            vision_det: Vision detection
            lidar_det: LiDAR detection
            
        Returns:
            Distance metric
        """
        # Placeholder: assumes both have 'position' field
        # In practice, this requires proper calibration and projection
        v_pos = vision_det.get('position', np.zeros(3))
        l_pos = lidar_det.get('centroid', np.zeros(3))
        
        return np.linalg.norm(v_pos - l_pos)
    
    def _compute_match_confidence(self, distance: float) -> float:
        """
        Compute matching confidence based on distance.
        
        Args:
            distance: Distance between detections
            
        Returns:
            Confidence score [0, 1]
        """
        # Exponential decay function
        sigma = 1.0
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    def _combine_detections(self,
                           vision_det: Dict,
                           lidar_det: Dict,
                           confidence: float) -> Dict:
        """
        Combine matched vision and LiDAR detections.
        
        Args:
            vision_det: Vision detection
            lidar_det: LiDAR detection
            confidence: Match confidence
            
        Returns:
            Fused detection
        """
        return {
            'position': lidar_det.get('centroid', np.zeros(3)),
            'size': lidar_det.get('size', np.zeros(3)),
            'confidence': confidence,
            'vision_data': vision_det,
            'lidar_data': lidar_det,
            'type': 'fused'
        }
    
    def _format_vision_only(self, detections: List[Dict]) -> List[Dict]:
        """Format vision-only detections."""
        return [{
            **det,
            'confidence': det.get('confidence', 0.5) * self.vision_weight,
            'type': 'vision_only'
        } for det in detections]
    
    def _format_lidar_only(self, detections: List[Dict]) -> List[Dict]:
        """Format LiDAR-only detections."""
        return [{
            **det,
            'confidence': 1.0 * self.lidar_weight,
            'type': 'lidar_only'
        } for det in detections]
    
    def update_weights(self, vision_weight: float, lidar_weight: float):
        """
        Update fusion weights dynamically.
        
        Args:
            vision_weight: New vision weight
            lidar_weight: New LiDAR weight
        """
        total = vision_weight + lidar_weight
        self.vision_weight = vision_weight / total
        self.lidar_weight = lidar_weight / total
        logger.info(f"Updated weights: vision={self.vision_weight:.2f}, "
                   f"lidar={self.lidar_weight:.2f}")
