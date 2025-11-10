"""LiDAR-based obstacle detection."""

import numpy as np
from typing import List, Tuple
from loguru import logger


class LidarObstacleDetector:
    """Detects obstacles using LiDAR point cloud data."""
    
    def __init__(self, 
                 cluster_tolerance: float = 0.1,
                 min_cluster_size: int = 10,
                 max_cluster_size: int = 10000):
        """
        Initialize LiDAR obstacle detector.
        
        Args:
            cluster_tolerance: Distance threshold for clustering
            min_cluster_size: Minimum points in a cluster
            max_cluster_size: Maximum points in a cluster
        """
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
    
    def detect(self, points: np.ndarray) -> List[dict]:
        """
        Detect obstacles in point cloud.
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            List of detected obstacles with bounding boxes
        """
        if len(points) == 0:
            return []
        
        # Cluster points
        clusters = self._cluster_points(points)
        
        # Extract obstacle information
        obstacles = []
        for cluster in clusters:
            obstacle = self._extract_obstacle_info(cluster)
            obstacles.append(obstacle)
        
        logger.info(f"Detected {len(obstacles)} obstacles")
        return obstacles
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using Euclidean clustering.
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            List of point clusters
        """
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        visited = np.zeros(len(points), dtype=bool)
        clusters = []
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # Find neighbors
            cluster_indices = tree.query_ball_point(
                points[i], 
                self.cluster_tolerance
            )
            
            if len(cluster_indices) < self.min_cluster_size:
                visited[i] = True
                continue
            
            # Grow cluster
            cluster = self._grow_cluster(
                points, tree, cluster_indices, visited
            )
            
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(points[cluster])
        
        return clusters
    
    def _grow_cluster(self, 
                     points: np.ndarray,
                     tree: 'KDTree',
                     seed_indices: List[int],
                     visited: np.ndarray) -> List[int]:
        """
        Grow cluster using region growing.
        
        Args:
            points: Point cloud
            tree: KD-tree for neighbor search
            seed_indices: Initial seed points
            visited: Visited flags array
            
        Returns:
            List of point indices in cluster
        """
        cluster = []
        queue = list(seed_indices)
        
        while queue:
            idx = queue.pop(0)
            
            if visited[idx]:
                continue
            
            visited[idx] = True
            cluster.append(idx)
            
            # Find new neighbors
            neighbors = tree.query_ball_point(
                points[idx],
                self.cluster_tolerance
            )
            
            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    queue.append(neighbor_idx)
        
        return cluster
    
    def _extract_obstacle_info(self, cluster: np.ndarray) -> dict:
        """
        Extract obstacle information from cluster.
        
        Args:
            cluster: Point cluster
            
        Returns:
            Dictionary with obstacle properties
        """
        # Compute bounding box
        min_bounds = np.min(cluster, axis=0)
        max_bounds = np.max(cluster, axis=0)
        
        # Compute centroid
        centroid = np.mean(cluster, axis=0)
        
        # Compute size
        size = max_bounds - min_bounds
        
        return {
            'centroid': centroid,
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'size': size,
            'num_points': len(cluster)
        }
    
    def detect_thin_obstacles(self, points: np.ndarray, 
                             thickness_threshold: float = 0.05) -> List[dict]:
        """
        Specifically detect thin obstacles like wires.
        
        Args:
            points: Point cloud
            thickness_threshold: Maximum thickness for thin obstacles
            
        Returns:
            List of detected thin obstacles
        """
        obstacles = self.detect(points)
        
        # Filter by thickness
        thin_obstacles = []
        for obs in obstacles:
            min_dimension = np.min(obs['size'])
            if min_dimension < thickness_threshold:
                thin_obstacles.append(obs)
        
        return thin_obstacles
