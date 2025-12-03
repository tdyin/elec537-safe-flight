"""Depth-based obstacle detector using monocular depth estimation."""

import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")


class DepthDetector:
    """
    Vision-based obstacle detector using monocular depth estimation.
    
    This approach uses MiDaS or similar depth estimation models to detect
    obstacles based on proximity, without semantic segmentation.
    """
    
    def __init__(self,
                 depth_model_path: Optional[str] = None,
                 depth_scale: float = 1.0,
                 use_gpu: bool = False):
        """
        Initialize depth-based detector.
        
        Args:
            depth_model_path: Path to depth estimation ONNX model
            depth_scale: Scale factor for depth values
            use_gpu: Whether to use GPU acceleration
        """
        self.depth_session = None
        self.depth_scale = depth_scale
        self.use_gpu = use_gpu
        
        # Model input/output info
        self.depth_input_name = None
        self.depth_output_names = None
        self.depth_input_shape = None
        
        # Load model if provided
        if depth_model_path:
            self.load_depth_model(depth_model_path)
    
    def load_depth_model(self, model_path: str):
        """Load monocular depth estimation model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not installed")
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Depth model not found: {model_path}")
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.depth_session = ort.InferenceSession(
                str(model_path),
                session_options,
                providers=providers
            )
            
            self.depth_input_name = self.depth_session.get_inputs()[0].name
            self.depth_output_names = [output.name for output in self.depth_session.get_outputs()]
            self.depth_input_shape = self.depth_session.get_inputs()[0].shape
            
            logger.info(f"Depth model loaded: {model_path.name}")
            logger.info(f"Input shape: {self.depth_input_shape}")
            logger.info(f"Providers: {self.depth_session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect obstacles using depth estimation.
        
        Args:
            image: Input image (numpy array, HWC, BGR format)
            
        Returns:
            Dictionary containing:
            {
                'depth_map': np.ndarray,          # H x W depth map
                'obstacle_regions': List[Dict],   # List of detected obstacle regions
                'free_space_map': np.ndarray      # H x W binary map of navigable space
            }
        """
        result = {
            'depth_map': None,
            'obstacle_regions': [],
            'free_space_map': None
        }
        
        # Run depth estimation
        if self.depth_session is not None:
            depth_map = self._run_depth_estimation(image)
            result['depth_map'] = depth_map
            
            # Extract obstacle regions from depth
            result['obstacle_regions'] = self._extract_obstacle_regions_from_depth(depth_map)
            
            # Compute free space map
            result['free_space_map'] = self._compute_free_space(depth_map)
        
        return result
    
    def _run_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Run monocular depth estimation on image.
        
        Returns:
            Depth map (H x W) with values in meters
        """
        # Preprocess image for depth estimation
        depth_input = self._preprocess_depth(image)
        
        # Run inference
        outputs = self.depth_session.run(self.depth_output_names, {self.depth_input_name: depth_input})
        
        # Get depth map
        depth_map = outputs[0]
        
        # Remove batch dimension if present
        if len(depth_map.shape) == 4:
            depth_map = depth_map[0, 0]
        elif len(depth_map.shape) == 3:
            depth_map = depth_map[0]
        
        # Resize to original image size
        h, w = image.shape[:2]
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply depth scale and inversion (some models output inverse depth)
        # MiDaS outputs relative depth, normalize and scale
        depth_map = depth_map * self.depth_scale
        
        return depth_map
    
    def _preprocess_depth(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for depth estimation model."""
        # Get target size from model
        if len(self.depth_input_shape) == 4:
            _, channels, height, width = [dim if isinstance(dim, int) else 384 for dim in self.depth_input_shape]
        else:
            height, width, channels = 384, 384, 3
        
        # Resize
        img_resized = cv2.resize(image, (width, height))
        
        # Convert to RGB
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW and add batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        
        return img_batch
    
    def _extract_obstacle_regions_from_depth(self, depth_map: np.ndarray) -> List[Dict]:
        """
        Extract obstacle regions from depth map based on proximity.
        
        Args:
            depth_map: Depth map (higher values = closer for MiDaS)
            
        Returns:
            List of obstacle region dictionaries
        """
        h, w = depth_map.shape
        
        # Normalize depth map
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # Threshold for close obstacles (MiDaS: higher values = closer)
        close_threshold = 0.6  # Top 40% of depth values are considered obstacles
        obstacle_mask = (depth_normalized > close_threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_mask, connectivity=8
        )
        
        regions = []
        
        # Skip label 0 (background)
        for label_id in range(1, num_labels):
            # Get region stats
            x, y, w_box, h_box, area = stats[label_id]
            cx, cy = centroids[label_id]
            
            # Filter small regions (noise)
            if area < 100:  # Minimum area threshold
                continue
            
            # Get depth statistics for this region
            region_mask = (labels == label_id)
            region_depths = depth_normalized[region_mask]
            
            regions.append({
                'bbox': [int(x), int(y), int(x + w_box), int(y + h_box)],
                'centroid': [float(cx), float(cy)],
                'area': int(area),
                'depth': float(np.median(region_depths)),
                'class_name': 'obstacle',
                'mask': region_mask
            })
        
        return regions
    
    def _compute_free_space(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Compute navigable free space map from depth.
        
        Args:
            depth_map: Depth map
            
        Returns:
            Binary free space map (1 = navigable, 0 = obstacle)
        """
        # Normalize depth map
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # Free space where depth is low (far from camera in MiDaS output)
        close_threshold = 0.7  # Areas with normalized depth > 0.7 are too close
        free_space = (depth_normalized < close_threshold).astype(np.uint8)
        
        return free_space
    
    def get_obstacle_direction_from_depth(self, 
                                          depth_map: np.ndarray,
                                          image_shape: Tuple) -> Dict[str, float]:
        """
        Analyze depth distribution to determine avoidance direction.
        
        Args:
            depth_map: Depth map for distance estimation
            image_shape: Shape of the image (H, W, C)
            
        Returns:
            Dictionary with directional scores and recommended action
        """
        h, w = image_shape[:2]
        
        # Normalize depth
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # ROI: focus on middle portion of image (20-70% height)
        roi_top = int(0.2 * h)
        roi_bottom = int(0.7 * h)
        roi = depth_normalized[roi_top:roi_bottom, :]
        
        # Divide image into zones: left, center, right
        zone_width = w // 3
        
        zones = {
            'left': roi[:, :zone_width],
            'center': roi[:, zone_width:2*zone_width],
            'right': roi[:, 2*zone_width:]
        }
        
        # Calculate clearance for each zone (lower depth = more clearance)
        # Use 90th percentile to catch thin obstacles
        clearances = {}
        for direction, zone in zones.items():
            # Clearance is inverse of obstacle proximity
            max_depth = np.percentile(zone, 90)
            clearances[direction] = 1.0 - max_depth
        
        # Determine best avoidance direction (highest clearance)
        best_direction = max(clearances, key=clearances.get)
        
        # Calculate urgency based on center zone
        center_max = np.percentile(zones['center'], 95)
        urgency = center_max  # Higher = more urgent (obstacle closer)
        
        return {
            'left_clearance': clearances['left'],
            'center_clearance': clearances['center'],
            'right_clearance': clearances['right'],
            'recommended_direction': best_direction,
            'urgency': urgency
        }
    
    def estimate_obstacle_distance(self, 
                                   obstacle_region: Dict,
                                   depth_map: np.ndarray,
                                   camera_params: Optional[Dict] = None) -> float:
        """
        Estimate distance to obstacle using depth map.
        
        Args:
            obstacle_region: Obstacle region from detection
            depth_map: Depth map
            camera_params: Optional camera intrinsics for metric depth
            
        Returns:
            Estimated distance in meters
        """
        # Get obstacle bounding box
        x1, y1, x2, y2 = obstacle_region['bbox']
        
        # Extract depth values in obstacle region
        region_depth = depth_map[y1:y2, x1:x2]
        
        if region_depth.size == 0:
            return float('inf')
        
        # Use median depth as robust estimate
        valid_depths = region_depth[region_depth > 0]
        if valid_depths.size == 0:
            return float('inf')
        
        estimated_depth = np.median(valid_depths)
        
        # Apply camera calibration if available
        if camera_params:
            # Convert relative depth to metric depth
            focal_length = camera_params.get('focal_length', 1.0)
            baseline = camera_params.get('baseline', 0.1)
            estimated_depth = (focal_length * baseline) / (estimated_depth + 1e-6)
        
        return float(estimated_depth)
