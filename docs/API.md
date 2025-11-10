# API Documentation

## Vision Module

### ObstacleDetector
Detects obstacles using deep learning and classical computer vision.

**Methods:**
- `__init__(model_path=None, use_gpu=True)`: Initialize detector
- `load_model(model_path)`: Load pretrained model
- `detect(image)`: Detect obstacles in image
- `detect_wires_classical(image)`: Classical wire detection using Canny + Hough

### ImageProcessor
Preprocesses and enhances images.

**Methods:**
- `__init__(target_size=(224, 224))`: Initialize processor
- `resize(image, size=None)`: Resize image
- `normalize(image)`: Normalize to [0, 1]
- `enhance_contrast(image)`: Apply CLAHE enhancement
- `denoise(image)`: Remove noise
- `preprocess(image, enhance=True, denoise=False)`: Full pipeline

## LiDAR Module

### LidarProcessor
Processes LiDAR point cloud data.

**Methods:**
- `__init__(max_range=10.0, min_range=0.1)`: Initialize processor
- `filter_range(points)`: Filter by distance
- `remove_outliers(points, nb_neighbors=20, std_ratio=2.0)`: Remove outliers
- `voxel_downsample(points, voxel_size=0.05)`: Downsample point cloud
- `segment_ground(points, threshold=0.1)`: Segment ground plane
- `process(points, ...)`: Full processing pipeline

### LidarObstacleDetector
Detects obstacles in point clouds.

**Methods:**
- `__init__(cluster_tolerance=0.1, min_cluster_size=10, max_cluster_size=10000)`: Initialize
- `detect(points)`: Detect obstacles
- `detect_thin_obstacles(points, thickness_threshold=0.05)`: Detect thin obstacles like wires

## Fusion Module

### SensorFusion
Fuses vision and LiDAR detections.

**Methods:**
- `__init__(vision_weight=0.5, lidar_weight=0.5, confidence_threshold=0.5)`: Initialize
- `fuse(vision_detections, lidar_detections)`: Fuse detections from both sensors
- `update_weights(vision_weight, lidar_weight)`: Dynamically adjust fusion weights

## Drone Module

### CrazyflieInterface
Interface for Crazyflie 2.1 communication.

**Methods:**
- `__init__(uri='radio://0/80/2M/E7E7E7E7E7')`: Initialize interface
- `connect()`: Connect to drone
- `disconnect()`: Disconnect from drone
- `get_sensor_data()`: Get sensor readings
- `send_velocity_command(vx, vy, vz, yaw_rate)`: Send velocity command
- `emergency_stop()`: Emergency stop
- `get_position()`: Get current position

### NavigationController
Navigation with obstacle avoidance.

**Methods:**
- `__init__(max_speed=0.5, safety_distance=0.5, avoidance_gain=1.0)`: Initialize
- `compute_safe_velocity(obstacles, current_position, target_velocity)`: Compute safe velocity
- `is_path_clear(obstacles, current_position, target_position)`: Check if path is clear
- `plan_avoidance_maneuver(obstacles, current_position, target_position)`: Plan avoidance path
