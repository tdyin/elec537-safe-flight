# Safe Flight - Vision-Based Obstacle Avoidance

Real-time obstacle detection and avoidance for Crazyflie drones using ONNX-optimized deep learning models.

## Overview

Safe Flight enables autonomous navigation with vision-based obstacle avoidance. The system uses monocular depth estimation (MiDaS) for robust obstacle detection and stable path planning.

**Status**: ✅ Fully functional in Webots SITL  
**Models**: MiDaS v2.1 Small (depth estimation)  
**Performance**: Real-time inference on CPU  
**Environment**: Conda (safe-flight)

## Key Features

✅ **Depth-Based Obstacle Detection**
- MiDaS monocular depth estimation
- ONNX-optimized for edge deployment
- Zone-based obstacle proximity analysis

✅ **Stable Avoidance**
- State-machine based navigation
- Depth-aware obstacle mapping
- Smooth velocity transitions
- A* path planning with Bezier smoothing

✅ **Intelligent Navigation**
- Zone-based obstacle classification
- Graduated response (emergency/avoidance/caution/clear)
- Path following with lookahead
- Automatic crash detection & recovery

✅ **Safety Features**
- Crash detection (tilt angle & altitude monitoring)
- Automatic emergency stop on crash
- Recoverable crash state
- Real-time safety monitoring

✅ **SITL Development**
- Webots simulation integration
- External Python control via TCP
- All sensors accessible
- Rapid prototyping environment

## Hardware

**Target Platform**:
- Crazyflie 2.1 drone
- AI deck (GAP8 processor) - for deployment
- Flow deck (optical flow positioning)
- Multi-ranger deck (range sensing)

**Current Development**:
- Webots simulation (SITL)
- CPU-based inference
- Ready for hardware deployment

## Project Structure

```
elec537-safe-flight/
├── launch.py                # 🚀 Unified SITL launcher (main entry point)
├── src/                     # Source code
│   ├── vision/              # Depth estimation
│   │   ├── depth_detector.py         # MiDaS depth detector
│   │   └── depth_utils.py            # Depth processing
│   ├── fusion/              # Depth-aware sensor fusion
│   │   └── enhanced_fusion.py        # Depth-based fusion
│   ├── drone/               # Drone interfaces & controllers
│   │   ├── interface.py                   # Hardware interface
│   │   ├── webots_interface.py            # SITL interface
│   │   └── depth_controller.py            # Navigation controller
│   ├── planning/            # Path planning & avoidance
│   │   ├── path_planner.py           # A* planning
│   │   ├── trajectory_smoother.py    # Bezier/spline smoothing
│   │   └── avoidance_controller.py   # Stable avoidance logic
│   └── sim/                 # Simulation bridge & utilities
├── sim/webots/              # Webots simulation files
│   ├── controllers/         # Webots robot controllers
│   │   ├── crazyflie_sitl/              # Basic SITL controller
│   │   ├── crazyflie_vision_avoidance/  # Vision-based controller
│   │   └── crazyflie_py_wallfollowing/  # Wall-following demo
│   └── worlds/              # Simulation world files
│       ├── crazyflie_apartment.wbt      # Complex environment
│       └── crazyflie_open.wbt          # Open world
├── examples/                # Demo applications & examples
│   ├── vision_avoidance_demo.py     # Vision-based navigation
│   ├── line_obstacle_avoidance.py   # Wire obstacle demo
│   └── simple_sitl_test.py          # Basic SITL test
├── scripts/                 # Utility scripts
│   ├── download_models.py           # Model downloader
│   ├── test_vision_detector.py      # Vision testing
│   └── test_modules.py              # Module testing
├── tests/                   # Unit & integration tests
├── models/                  # ONNX model files
│   └── midas_v21_small.onnx             # Depth estimation (63.7 MB)
├── docs/                    # Documentation
│   ├── DEVELOPMENT_LOG.md   # ⭐ Chronological dev history
│   ├── VISION_AVOIDANCE.md  # Vision system guide
│   ├── SITL.md              # SITL architecture & usage
│   ├── CONDA_SETUP.md       # Environment management
│   ├── SETUP.md             # Initial setup guide
│   ├── API.md               # API reference
│   └── archive/             # Historical documentation
├── config.yaml              # Main configuration
├── environment.yaml         # Conda environment spec
└── README.md                # This file
```

## Documentation

### 📖 Main Guides

- **[Development Log](docs/DEVELOPMENT_LOG.md)** - Complete chronological development history
- **[Vision Avoidance](docs/VISION_AVOIDANCE.md)** - Vision system architecture and usage
- **[SITL Guide](docs/SITL.md)** - Software-in-the-loop simulation
- **[Conda Setup](docs/CONDA_SETUP.md)** - Environment management
- **[Setup Guide](docs/SETUP.md)** - Initial setup instructions
- **[API Reference](docs/API.md)** - Code API documentation

### 🔧 Quick References

- **[SITL Quick Ref](docs/SITL_QUICKREF.md)** - SITL command reference
- **[SITL Implementation](docs/SITL_IMPLEMENTATION.md)** - SITL technical details

### 📦 Configuration Files

- `config.yaml` - Main configuration (models, parameters, etc.)
- `environment.yaml` - Conda environment definition
- `requirements.txt` - Python dependencies

## Usage Examples

### Vision Detection

```python
from vision.depth_detector import DepthDetector
import cv2

# Initialize detector
detector = DepthDetector(
    depth_model_path="models/midas_v21_small.onnx",
    depth_scale=1.0
)

# Detect obstacles
image = cv2.imread("frame.jpg")
vision_data = detector.detect(image)

# Analyze obstacles
print(f"Depth map shape: {vision_data['depth_map'].shape}")
print(f"Obstacles detected: {len(vision_data['obstacle_regions'])}")
```

### SITL Control

```python
from sim.bridge import SimulationBridge

# Connect to Webots
bridge = SimulationBridge(host='localhost', port=10020)
bridge.connect()

# Get sensor data
sensor_data = bridge.get_sensor_data()

# Send velocity command
bridge.send_velocity(vx=0.3, vy=0.0, vz=0.0, yaw_rate=0.0)
```

### Webots Integration

The project includes a unified Webots controller:

- **crazyflie_sitl** - SITL with external control and autonomous vision mode

Model paths are configured in `config.yaml`.

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/tdyin/elec537-safe-flight.git
cd elec537-safe-flight

# Create conda environment (recommended)
conda env create -f environment.yaml
conda activate safe-flight
```

### 2. Download Vision Models

```bash
# Download all models
python scripts/download_models.py
```

### 3. Test System

```bash
# Run pytest tests
pytest tests/ -v

# Test specific modules
pytest tests/test_avoidance.py  # Avoidance controller tests
pytest tests/test_planning.py   # Path planning tests
```

### 4. Run Simulation

```bash
# Launch SITL with vision-based navigation
python launch.py                          # Default: apartment world with GUI
python launch.py --world open             # Different world
python launch.py --goal 6 0 1             # Set goal position
python launch.py --no-gui                 # Headless mode (faster)
python launch.py --viz                    # Enable visualization
python launch.py --analyze                # Analyze latest log file

# See all options
python launch.py --help
```

**Manual Launch (Alternative)**
```bash
# 1. Open Webots and load world file
#    File → Open World → sim/webots/worlds/crazyflie_apartment.wbt
#    Set controller to 'crazyflie_vision_avoidance' and click Play

# 2. Or run example demos
python examples/vision_avoidance_demo.py
python examples/simple_sitl_test.py
```

## Performance

**Vision System**: MiDaS depth estimation, real-time CPU inference  
**Flight**: 0.3 m/s cruise, 0.8m safety distance, stable avoidance with path planning  
**Detection**: Depth-based obstacle mapping with zone analysis

## Troubleshooting

**Model not loading**: `python scripts/download_models.py`  
**List downloaded models**: `python scripts/download_models.py --list`  
**Webots controller issues**: Verify path in `sim/webots/controllers/`  
**Environment issues**: `conda env update -n safe-flight -f environment.yaml --prune`

## Testing

```bash
# Activate environment
conda activate safe-flight

# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_avoidance.py   # Avoidance controllers
pytest tests/test_planning.py    # Path planning

# Run with coverage
pytest --cov=src tests/
```

## Development

### Daily Workflow

```bash
# 1. Activate environment
conda activate safe-flight

# 2. Make changes to code
# ... edit files ...

# 3. Test changes
pytest tests/ -v

# 4. Run in simulation
python launch.py

# 5. Commit changes
git add .
git commit -m "Description of changes"
git push
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of ELEC 537 coursework at Rice University.

## Acknowledgments

- **Bitcraze** - Crazyflie platform
- **Cyberbotics** - Webots simulation
- **Intel ISL** - MiDaS depth estimation
- **ONNX Runtime** - Edge inference optimization

---

**Last Updated**: December 1, 2025  
**Status**: ✅ Fully Functional in SITL with Path-Based Navigation  
**Next Milestone**: Hardware Deployment
