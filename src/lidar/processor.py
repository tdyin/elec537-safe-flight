"""LiDAR data processing utilities."""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class LidarProcessor:
    """Processes LiDAR point cloud data."""
    
    def __init__(self, max_range: float = 10.0, min_range: float = 0.1):
        """
        Initialize LiDAR processor.
        
        Args:
            max_range: Maximum detection range in meters
            min_range: Minimum detection range in meters
        """
        self.max_range = max_range
        self.min_range = min_range
    
    def filter_range(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points by distance range.
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            Filtered point cloud
        """
        distances = np.linalg.norm(points, axis=1)
        mask = (distances >= self.min_range) & (distances <= self.max_range)
        return points[mask]
    
    def remove_outliers(self, points: np.ndarray, 
                       nb_neighbors: int = 20, 
                       std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from point cloud.
        
        Args:
            points: Nx3 array of point coordinates
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation threshold
            
        Returns:
            Filtered point cloud
        """
        if len(points) < nb_neighbors:
            return points
        
        # Compute distances to k-nearest neighbors
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=nb_neighbors + 1)
        
        # Remove first column (distance to self)
        distances = distances[:, 1:]
        
        # Compute mean distances
        mean_distances = np.mean(distances, axis=1)
        
        # Filter outliers
        threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
        mask = mean_distances < threshold
        
        return points[mask]
    
    def voxel_downsample(self, points: np.ndarray, 
                        voxel_size: float = 0.05) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Nx3 array of point coordinates
            voxel_size: Size of voxel grid cells
            
        Returns:
            Downsampled point cloud
        """
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        
        # Compute centroid for each voxel
        downsampled = np.zeros((len(unique_voxels), 3))
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled[i] = np.mean(points[mask], axis=0)
        
        return downsampled
    
    def segment_ground(self, points: np.ndarray, 
                      threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ground plane from point cloud.
        
        Args:
            points: Nx3 array of point coordinates
            threshold: Distance threshold for ground plane
            
        Returns:
            Tuple of (ground_points, non_ground_points)
        """
        # Simple height-based segmentation
        # Assumes z-axis points up and drone is roughly level
        z_values = points[:, 2]
        min_z = np.min(z_values)
        
        ground_mask = np.abs(z_values - min_z) < threshold
        
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]
        
        return ground_points, non_ground_points
    
    def process(self, points: np.ndarray, 
                downsample: bool = True,
                remove_outliers: bool = True,
                segment_ground: bool = True) -> np.ndarray:
        """
        Complete LiDAR processing pipeline.
        
        Args:
            points: Raw point cloud data
            downsample: Whether to downsample the cloud
            remove_outliers: Whether to remove outliers
            segment_ground: Whether to remove ground plane
            
        Returns:
            Processed point cloud
        """
        processed = self.filter_range(points)
        
        if remove_outliers and len(processed) > 0:
            processed = self.remove_outliers(processed)
        
        if downsample and len(processed) > 0:
            processed = self.voxel_downsample(processed)
        
        if segment_ground and len(processed) > 0:
            _, processed = self.segment_ground(processed)
        
        logger.info(f"Processed {len(points)} -> {len(processed)} points")
        return processed
