"""Depth estimation utilities for vision-based ranging."""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from loguru import logger


class DepthEstimator:
    """
    Utilities for monocular depth estimation and depth map processing.
    
    Supports various depth estimation models (MiDaS, DPT, etc.) and
    provides tools for depth map refinement and metric conversion.
    """
    
    def __init__(self, 
                 depth_scale: float = 1.0,
                 max_depth: float = 10.0,
                 min_depth: float = 0.1):
        """
        Initialize depth estimator.
        
        Args:
            depth_scale: Scale factor for depth values
            max_depth: Maximum depth value in meters
            min_depth: Minimum depth value in meters
        """
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.min_depth = min_depth
    
    def normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0, 1] range.
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Normalized depth map
        """
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max - depth_min < 1e-6:
            return np.zeros_like(depth_map)
        
        normalized = (depth_map - depth_min) / (depth_max - depth_min)
        return normalized
    
    def convert_to_metric(self, 
                         depth_map: np.ndarray,
                         camera_params: Dict) -> np.ndarray:
        """
        Convert relative depth to metric depth using camera parameters.
        
        Args:
            depth_map: Relative depth map
            camera_params: Camera intrinsics (focal_length, baseline, etc.)
            
        Returns:
            Metric depth map in meters
        """
        # For monocular depth, we typically need additional calibration
        # This is a simplified conversion
        focal_length = camera_params.get('focal_length', 1.0)
        baseline = camera_params.get('baseline', 0.1)
        
        # Avoid division by zero
        depth_map_safe = np.maximum(depth_map, 1e-6)
        
        # Convert using stereo-like formula (approximation)
        metric_depth = (focal_length * baseline) / depth_map_safe
        
        # Clip to valid range
        metric_depth = np.clip(metric_depth, self.min_depth, self.max_depth)
        
        return metric_depth
    
    def refine_depth_edges(self, 
                          depth_map: np.ndarray,
                          image: np.ndarray,
                          sigma_space: float = 5.0,
                          sigma_color: float = 0.1) -> np.ndarray:
        """
        Refine depth map edges using bilateral filtering guided by image.
        
        Args:
            depth_map: Input depth map
            image: Corresponding RGB image
            sigma_space: Spatial sigma for bilateral filter
            sigma_color: Color sigma for bilateral filter
            
        Returns:
            Refined depth map
        """
        # Convert image to grayscale for edge guidance
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize depth map for filtering
        depth_normalized = self.normalize_depth(depth_map)
        
        # Apply joint bilateral filter
        refined = cv2.ximgproc.jointBilateralFilter(
            gray.astype(np.uint8),
            (depth_normalized * 255).astype(np.uint8),
            d=9,
            sigmaColor=sigma_color * 255,
            sigmaSpace=sigma_space
        )
        
        # Restore original scale
        refined = refined.astype(np.float32) / 255.0
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        refined = refined * (depth_max - depth_min) + depth_min
        
        return refined
    
    def compute_depth_gradients(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute depth gradients for edge detection.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        return grad_x, grad_y
    
    def detect_depth_edges(self, 
                          depth_map: np.ndarray,
                          threshold: float = 0.1) -> np.ndarray:
        """
        Detect edges in depth map (depth discontinuities).
        
        Args:
            depth_map: Input depth map
            threshold: Edge detection threshold
            
        Returns:
            Binary edge map
        """
        grad_x, grad_y = self.compute_depth_gradients(depth_map)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        gradient_normalized = self.normalize_depth(gradient_magnitude)
        edges = (gradient_normalized > threshold).astype(np.uint8)
        
        return edges
    
    def project_depth_to_3d(self,
                           depth_map: np.ndarray,
                           camera_matrix: np.ndarray) -> np.ndarray:
        """
        Project depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map (H x W)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            
        Returns:
            Point cloud (H x W x 3) in camera coordinates
        """
        h, w = depth_map.shape
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Get camera intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Compute 3D coordinates
        x = (u - cx) * depth_map / fx
        y = (v - cy) * depth_map / fy
        z = depth_map
        
        # Stack to create point cloud
        point_cloud = np.stack([x, y, z], axis=-1)
        
        return point_cloud
    
    def estimate_distance_to_region(self,
                                    depth_map: np.ndarray,
                                    mask: np.ndarray,
                                    method: str = 'median') -> float:
        """
        Estimate distance to a specific region in the depth map.
        
        Args:
            depth_map: Depth map
            mask: Binary mask indicating region of interest
            method: Estimation method ('median', 'mean', 'min', 'percentile')
            
        Returns:
            Estimated distance in meters
        """
        # Extract depth values in masked region
        region_depth = depth_map[mask > 0]
        
        if region_depth.size == 0:
            return float('inf')
        
        # Filter out invalid depths
        valid_depths = region_depth[region_depth > self.min_depth]
        if valid_depths.size == 0:
            return float('inf')
        
        # Compute distance based on method
        if method == 'median':
            distance = np.median(valid_depths)
        elif method == 'mean':
            distance = np.mean(valid_depths)
        elif method == 'min':
            distance = np.min(valid_depths)
        elif method == 'percentile':
            # Use 25th percentile (closer objects)
            distance = np.percentile(valid_depths, 25)
        else:
            distance = np.median(valid_depths)
        
        return float(distance)
    
    def create_occupancy_grid(self,
                             depth_map: np.ndarray,
                             segmentation_mask: Optional[np.ndarray] = None,
                             grid_size: Tuple[int, int] = (10, 10),
                             distance_threshold: float = 2.0) -> np.ndarray:
        """
        Create a top-down occupancy grid from depth map.
        
        Args:
            depth_map: Depth map
            segmentation_mask: Optional obstacle segmentation mask
            grid_size: Size of occupancy grid (rows, cols)
            distance_threshold: Maximum distance to consider (meters)
            
        Returns:
            Occupancy grid (grid_size) with values [0, 1]
        """
        h, w = depth_map.shape
        grid_h, grid_w = grid_size
        
        # Initialize occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.float32)
        counts = np.zeros(grid_size, dtype=np.int32)
        
        # Compute cell size
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract cell region
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < grid_h - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < grid_w - 1 else w
                
                cell_depth = depth_map[y1:y2, x1:x2]
                
                # Apply segmentation mask if available
                if segmentation_mask is not None:
                    cell_mask = segmentation_mask[y1:y2, x1:x2]
                    cell_depth = cell_depth[cell_mask > 0]
                
                # Compute occupancy
                if cell_depth.size > 0:
                    valid_depths = cell_depth[cell_depth > 0]
                    if valid_depths.size > 0:
                        avg_depth = np.mean(valid_depths)
                        # Closer objects have higher occupancy
                        if avg_depth < distance_threshold:
                            occupancy[i, j] = 1.0 - (avg_depth / distance_threshold)
        
        return occupancy
    
    def visualize_depth(self, 
                       depth_map: np.ndarray,
                       colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
        """
        Create visualization of depth map with colormap.
        
        Args:
            depth_map: Input depth map
            colormap: OpenCV colormap constant
            
        Returns:
            Colorized depth map (H x W x 3)
        """
        # Normalize to [0, 255]
        depth_normalized = self.normalize_depth(depth_map)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)
        
        return depth_colored
    
    def fuse_depth_with_lidar(self,
                             depth_map: np.ndarray,
                             lidar_points: np.ndarray,
                             camera_matrix: np.ndarray,
                             camera_to_lidar: np.ndarray) -> np.ndarray:
        """
        Fuse monocular depth with sparse LiDAR measurements.
        
        Args:
            depth_map: Dense depth map from vision
            lidar_points: Sparse LiDAR points (N x 3) in LiDAR frame
            camera_matrix: Camera intrinsic matrix
            camera_to_lidar: Transformation from camera to LiDAR frame (4 x 4)
            
        Returns:
            Fused depth map
        """
        # Transform LiDAR points to camera frame
        # Invert transformation
        lidar_to_camera = np.linalg.inv(camera_to_lidar)
        
        # Convert to homogeneous coordinates
        lidar_homogeneous = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        
        # Transform to camera frame
        camera_points = (lidar_to_camera @ lidar_homogeneous.T).T[:, :3]
        
        # Project to image plane
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        u = (camera_points[:, 0] * fx / camera_points[:, 2] + cx).astype(int)
        v = (camera_points[:, 1] * fy / camera_points[:, 2] + cy).astype(int)
        
        # Filter valid projections
        h, w = depth_map.shape
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (camera_points[:, 2] > 0)
        
        # Create fused depth map starting from vision depth
        fused_depth = depth_map.copy()
        
        # Update with LiDAR measurements (more accurate)
        for i in np.where(valid)[0]:
            # Use LiDAR depth in a small region around projection
            lidar_depth = camera_points[i, 2]
            fused_depth[max(0, v[i]-2):min(h, v[i]+3), 
                       max(0, u[i]-2):min(w, u[i]+3)] = lidar_depth
        
        return fused_depth
