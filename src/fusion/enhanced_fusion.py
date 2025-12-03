"""Enhanced sensor fusion with depth map and segmentation integration."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger


class EnhancedSensorFusion:
    """
    Advanced multimodal sensor fusion combining:
    1. Vision-based segmentation masks
    2. Monocular depth maps
    3. LiDAR point clouds
    
    This produces robust 3D obstacle representations for navigation.
    """
    
    def __init__(self,
                 vision_weight: float = 0.4,
                 depth_weight: float = 0.3,
                 lidar_weight: float = 0.3,
                 confidence_threshold: float = 0.4):
        """
        Initialize enhanced sensor fusion.
        
        Args:
            vision_weight: Weight for vision segmentation
            depth_weight: Weight for depth estimation
            lidar_weight: Weight for LiDAR measurements
            confidence_threshold: Minimum confidence for fused detections
        """
        self.vision_weight = vision_weight
        self.depth_weight = depth_weight
        self.lidar_weight = lidar_weight
        self.confidence_threshold = confidence_threshold
        
        # Normalize weights
        total = vision_weight + depth_weight + lidar_weight
        self.vision_weight /= total
        self.depth_weight /= total
        self.lidar_weight /= total
        
        logger.info(f"Fusion weights - Vision: {self.vision_weight:.2f}, "
                   f"Depth: {self.depth_weight:.2f}, LiDAR: {self.lidar_weight:.2f}")
    
    def fuse_multimodal(self,
                       vision_data: Optional[Dict],
                       lidar_detections: Optional[List[Dict]],
                       camera_params: Optional[Dict] = None) -> Dict:
        """
        Fuse vision (segmentation + depth) with LiDAR data.
        
        Args:
            vision_data: Dictionary containing:
                        - segmentation_mask: Binary obstacle mask
                        - depth_map: Depth estimation
                        - obstacle_regions: Segmented regions
                        - free_space_map: Navigable space
            lidar_detections: List of LiDAR obstacle detections
            camera_params: Optional camera parameters for projection
            
        Returns:
            Fused representation with 3D obstacle information
        """
        result = {
            'obstacles_3d': [],
            'free_space_3d': None,
            'confidence_map': None,
            'occupancy_grid': None
        }
        
        # Case 1: Both vision and LiDAR available - full fusion
        if vision_data is not None and lidar_detections:
            result = self._fuse_vision_and_lidar(
                vision_data, lidar_detections, camera_params
            )
        
        # Case 2: Vision only
        elif vision_data is not None:
            result = self._process_vision_only(vision_data, camera_params)
        
        # Case 3: LiDAR only
        elif lidar_detections:
            result = self._process_lidar_only(lidar_detections)
        
        return result
    
    def _fuse_vision_and_lidar(self,
                              vision_data: Dict,
                              lidar_detections: List[Dict],
                              camera_params: Optional[Dict]) -> Dict:
        """
        Fuse vision segmentation/depth with LiDAR point cloud.
        
        Args:
            vision_data: Vision detection data
            lidar_detections: LiDAR detections
            camera_params: Camera parameters
            
        Returns:
            Fused 3D obstacle representation
        """
        seg_mask = vision_data.get('segmentation_mask')
        depth_map = vision_data.get('depth_map')
        obstacle_regions = vision_data.get('obstacle_regions', [])
        
        fused_obstacles = []
        
        # Convert vision obstacles to 3D using depth map
        vision_obstacles_3d = self._project_obstacles_to_3d(
            obstacle_regions, depth_map, camera_params
        )
        
        # Match vision and LiDAR obstacles
        matches = self._match_vision_lidar_3d(
            vision_obstacles_3d, lidar_detections
        )
        
        # Combine matched obstacles
        for vision_obs, lidar_obs, confidence in matches:
            if confidence >= self.confidence_threshold:
                fused_obs = self._combine_3d_obstacles(
                    vision_obs, lidar_obs, confidence
                )
                fused_obstacles.append(fused_obs)
        
        # Create confidence map
        confidence_map = self._create_confidence_map(
            seg_mask, depth_map, lidar_detections, camera_params
        )
        
        # Create 3D occupancy grid
        occupancy_grid = self._create_3d_occupancy_grid(
            vision_obstacles_3d, lidar_detections
        )
        
        return {
            'obstacles_3d': fused_obstacles,
            'confidence_map': confidence_map,
            'occupancy_grid': occupancy_grid,
            'fusion_type': 'full'
        }
    
    def _process_vision_only(self,
                            vision_data: Dict,
                            camera_params: Optional[Dict]) -> Dict:
        """
        Process vision data without LiDAR.
        
        Args:
            vision_data: Vision detection data
            camera_params: Camera parameters
            
        Returns:
            Vision-based 3D representation
        """
        depth_map = vision_data.get('depth_map')
        obstacle_regions = vision_data.get('obstacle_regions', [])
        
        # Project to 3D
        obstacles_3d = self._project_obstacles_to_3d(
            obstacle_regions, depth_map, camera_params
        )
        
        # Apply vision weight to confidence
        for obs in obstacles_3d:
            obs['confidence'] *= self.vision_weight
        
        return {
            'obstacles_3d': obstacles_3d,
            'confidence_map': None,
            'occupancy_grid': None,
            'fusion_type': 'vision_only'
        }
    
    def _process_lidar_only(self, lidar_detections: List[Dict]) -> Dict:
        """
        Process LiDAR data without vision.
        
        Args:
            lidar_detections: LiDAR detections
            
        Returns:
            LiDAR-based 3D representation
        """
        # Format LiDAR detections
        obstacles_3d = []
        
        for det in lidar_detections:
            obstacles_3d.append({
                'position': det.get('centroid', np.zeros(3)),
                'size': det.get('size', np.array([0.1, 0.1, 0.1])),
                'confidence': self.lidar_weight,
                'source': 'lidar'
            })
        
        return {
            'obstacles_3d': obstacles_3d,
            'confidence_map': None,
            'occupancy_grid': None,
            'fusion_type': 'lidar_only'
        }
    
    def _project_obstacles_to_3d(self,
                                obstacle_regions: List[Dict],
                                depth_map: Optional[np.ndarray],
                                camera_params: Optional[Dict]) -> List[Dict]:
        """
        Project 2D obstacle regions to 3D using depth map.
        
        Args:
            obstacle_regions: 2D segmented regions
            depth_map: Depth map
            camera_params: Camera intrinsics
            
        Returns:
            List of 3D obstacle representations
        """
        if depth_map is None:
            return []
        
        obstacles_3d = []
        
        # Default camera parameters if not provided
        if camera_params is None:
            h, w = depth_map.shape
            camera_params = {
                'fx': w / 2,
                'fy': w / 2,
                'cx': w / 2,
                'cy': h / 2
            }
        
        fx = camera_params.get('fx', 320)
        fy = camera_params.get('fy', 320)
        cx = camera_params.get('cx', 320)
        cy = camera_params.get('cy', 240)
        
        for region in obstacle_regions:
            # Get bounding box
            x1, y1, x2, y2 = region['bbox']
            
            # Extract depth in region
            region_depth = depth_map[y1:y2, x1:x2]
            
            if region_depth.size == 0:
                continue
            
            # Estimate distance (median depth)
            valid_depths = region_depth[region_depth > 0]
            if valid_depths.size == 0:
                continue
            
            median_depth = np.median(valid_depths)
            
            # Project center to 3D
            cx_pixel, cy_pixel = region['centroid']
            
            x_3d = (cx_pixel - cx) * median_depth / fx
            y_3d = (cy_pixel - cy) * median_depth / fy
            z_3d = median_depth
            
            # Estimate 3D size from bounding box
            width_3d = (x2 - x1) * median_depth / fx
            height_3d = (y2 - y1) * median_depth / fy
            depth_3d = 0.5  # Assume some depth
            
            obstacles_3d.append({
                'position': np.array([x_3d, y_3d, z_3d]),
                'size': np.array([width_3d, height_3d, depth_3d]),
                'confidence': self.vision_weight,
                'class_id': region.get('class_id', 0),
                'class_name': region.get('class_name', 'obstacle'),
                'source': 'vision',
                'area_2d': region.get('area', 0)
            })
        
        return obstacles_3d
    
    def _match_vision_lidar_3d(self,
                              vision_obstacles: List[Dict],
                              lidar_obstacles: List[Dict]) -> List[Tuple]:
        """
        Match 3D obstacles from vision and LiDAR.
        
        Args:
            vision_obstacles: Vision-projected 3D obstacles
            lidar_obstacles: LiDAR 3D obstacles
            
        Returns:
            List of (vision_obs, lidar_obs, confidence) tuples
        """
        matches = []
        
        for v_obs in vision_obstacles:
            best_match = None
            best_distance = float('inf')
            
            v_pos = v_obs['position']
            
            for l_obs in lidar_obstacles:
                l_pos = l_obs.get('centroid', l_obs.get('position', np.zeros(3)))
                
                # 3D Euclidean distance
                distance = np.linalg.norm(v_pos - l_pos)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = l_obs
            
            if best_match is not None:
                # Confidence based on distance
                # Closer matches have higher confidence
                sigma = 0.5  # meters
                confidence = np.exp(-(best_distance ** 2) / (2 * sigma ** 2))
                
                matches.append((v_obs, best_match, confidence))
        
        return matches
    
    def _combine_3d_obstacles(self,
                             vision_obs: Dict,
                             lidar_obs: Dict,
                             match_confidence: float) -> Dict:
        """
        Combine matched vision and LiDAR 3D obstacles.
        
        Args:
            vision_obs: Vision obstacle
            lidar_obs: LiDAR obstacle
            match_confidence: Matching confidence
            
        Returns:
            Fused 3D obstacle
        """
        # Weighted average of positions (LiDAR more accurate for position)
        v_pos = vision_obs['position']
        l_pos = lidar_obs.get('centroid', lidar_obs.get('position', np.zeros(3)))
        
        # Weight LiDAR position more heavily
        fused_pos = 0.3 * v_pos + 0.7 * l_pos
        
        # Use LiDAR size if available, otherwise vision size
        fused_size = lidar_obs.get('size', vision_obs['size'])
        
        # Combined confidence
        fused_confidence = (
            self.vision_weight * vision_obs['confidence'] +
            self.lidar_weight +
            self.depth_weight * match_confidence
        )
        
        return {
            'position': fused_pos,
            'size': fused_size,
            'confidence': fused_confidence,
            'class_id': vision_obs.get('class_id', 0),
            'class_name': vision_obs.get('class_name', 'obstacle'),
            'source': 'fused',
            'vision_data': vision_obs,
            'lidar_data': lidar_obs
        }
    
    def _create_confidence_map(self,
                              seg_mask: Optional[np.ndarray],
                              depth_map: Optional[np.ndarray],
                              lidar_detections: List[Dict],
                              camera_params: Optional[Dict]) -> Optional[np.ndarray]:
        """
        Create a 2D confidence map for obstacle presence.
        
        Args:
            seg_mask: Segmentation mask
            depth_map: Depth map
            lidar_detections: LiDAR detections
            camera_params: Camera parameters
            
        Returns:
            Confidence map (H x W)
        """
        if seg_mask is None or depth_map is None:
            return None
        
        h, w = seg_mask.shape
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # Vision contribution
        confidence_map += seg_mask.astype(np.float32) * self.vision_weight
        
        # Depth contribution (higher confidence for closer valid depths)
        if depth_map is not None:
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            depth_confidence = (1.0 - depth_normalized) * self.depth_weight
            confidence_map += depth_confidence * seg_mask
        
        # LiDAR contribution (project to image)
        # This is simplified - would need proper projection in production
        
        # Normalize to [0, 1]
        if confidence_map.max() > 0:
            confidence_map /= confidence_map.max()
        
        return confidence_map
    
    def _create_3d_occupancy_grid(self,
                                 vision_obstacles: List[Dict],
                                 lidar_obstacles: List[Dict],
                                 grid_size: Tuple[int, int, int] = (20, 20, 10),
                                 grid_bounds: Tuple[float, float, float] = (5.0, 5.0, 3.0)) -> np.ndarray:
        """
        Create 3D occupancy grid from obstacles.
        
        Args:
            vision_obstacles: Vision 3D obstacles
            lidar_obstacles: LiDAR obstacles
            grid_size: Grid dimensions (x, y, z)
            grid_bounds: Physical bounds in meters (x, y, z)
            
        Returns:
            3D occupancy grid
        """
        occupancy = np.zeros(grid_size, dtype=np.float32)
        
        gx, gy, gz = grid_size
        bx, by, bz = grid_bounds
        
        # Process all obstacles
        all_obstacles = vision_obstacles + [
            {
                'position': obs.get('centroid', obs.get('position', np.zeros(3))),
                'size': obs.get('size', np.array([0.1, 0.1, 0.1])),
                'confidence': self.lidar_weight
            }
            for obs in lidar_obstacles
        ]
        
        for obs in all_obstacles:
            pos = obs['position']
            size = obs['size']
            confidence = obs.get('confidence', 0.5)
            
            # Convert to grid indices
            # Assuming drone is at origin, positive z is forward
            ix = int((pos[0] + bx/2) / bx * gx)
            iy = int((pos[1] + by/2) / by * gy)
            iz = int(pos[2] / bz * gz)
            
            # Mark occupied cells
            if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
                occupancy[ix, iy, iz] = max(occupancy[ix, iy, iz], confidence)
        
        return occupancy
    
    def update_weights(self, vision: float, depth: float, lidar: float):
        """Update fusion weights dynamically."""
        total = vision + depth + lidar
        self.vision_weight = vision / total
        self.depth_weight = depth / total
        self.lidar_weight = lidar / total
        
        logger.info(f"Updated weights - Vision: {self.vision_weight:.2f}, "
                   f"Depth: {self.depth_weight:.2f}, LiDAR: {self.lidar_weight:.2f}")
