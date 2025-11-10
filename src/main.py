"""Main entry point for Safe Path obstacle detection system."""

import argparse
import yaml
from pathlib import Path
from loguru import logger

from vision import ObstacleDetector, ImageProcessor
from lidar import LidarProcessor, LidarObstacleDetector
from fusion import SensorFusion
from drone import CrazyflieInterface, NavigationController


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_dir = config.get('logging', {}).get('log_directory', 'logs')
    
    Path(log_dir).mkdir(exist_ok=True)
    
    logger.add(
        f"{log_dir}/safe_path.log",
        rotation="10 MB",
        level=log_level
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Safe Path - Drone Obstacle Detection')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='detection',
                       choices=['detection', 'navigation', 'data_collection'],
                       help='Operating mode')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (no hardware)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    logger.info("Starting Safe Path system")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Simulation: {args.simulation}")
    
    # Initialize modules
    vision_config = config['vision']
    lidar_config = config['lidar']
    fusion_config = config['fusion']
    drone_config = config['drone']
    
    # Vision module
    vision_detector = ObstacleDetector(
        model_path=vision_config.get('model_path'),
        use_gpu=vision_config.get('use_gpu', True)
    )
    image_processor = ImageProcessor(
        target_size=tuple(vision_config.get('target_size', [224, 224]))
    )
    
    # LiDAR module
    lidar_processor = LidarProcessor(
        max_range=lidar_config.get('max_range', 10.0),
        min_range=lidar_config.get('min_range', 0.1)
    )
    lidar_detector = LidarObstacleDetector(
        cluster_tolerance=lidar_config.get('cluster_tolerance', 0.1),
        min_cluster_size=lidar_config.get('min_cluster_size', 10),
        max_cluster_size=lidar_config.get('max_cluster_size', 10000)
    )
    
    # Sensor fusion
    sensor_fusion = SensorFusion(
        vision_weight=fusion_config.get('vision_weight', 0.5),
        lidar_weight=fusion_config.get('lidar_weight', 0.5),
        confidence_threshold=fusion_config.get('confidence_threshold', 0.5)
    )
    
    if args.mode == 'detection':
        logger.info("Running in detection mode")
        # Implement detection loop
        # This would continuously process sensor data and detect obstacles
        
    elif args.mode == 'navigation':
        logger.info("Running in navigation mode")
        # Initialize drone interface
        drone = CrazyflieInterface(uri=drone_config.get('uri'))
        controller = NavigationController(
            max_speed=drone_config.get('max_speed', 0.5),
            safety_distance=drone_config.get('safety_distance', 0.5),
            avoidance_gain=drone_config.get('avoidance_gain', 1.0)
        )
        
        # Implement navigation loop
        # This would control the drone with obstacle avoidance
        
    elif args.mode == 'data_collection':
        logger.info("Running in data collection mode")
        # Implement data collection routine
        # This would collect and save sensor data for training
    
    logger.info("Safe Path system shutdown")


if __name__ == '__main__':
    main()
