# Safe Path

## Description
This project aims to enhance drone navigation safety by enabling reliable detection of small, hard-to-see obstacles such as wires and cables. We are developing a multimodal sensing pipeline that integrates computer vision (deep learning + classical image processing) and LiDAR-based sensing for robust detection under varying lighting and environmental conditions. The final goal is a lightweight, real-time system deployable on embedded drone hardware.

## Features
- **Vision-based Detection**: Deep learning + classical CV for wire/cable detection
- **LiDAR Processing**: Point cloud-based obstacle detection and clustering
- **Sensor Fusion**: Multi-modal detection combining vision and LiDAR
- **Drone Interface**: Crazyflie 2.1 integration with real-time control
- **Navigation Controller**: Obstacle avoidance and path planning

## Hardware
- Crazyflie 2.1 drone
- AI deck (GAP8 processor)
- Flow deck (optical flow positioning)
- Multi-ranger deck (LiDAR sensing)

## Project Structure
```
elec537-safe-path/
├── src/
│   ├── vision/          # Computer vision module
│   ├── lidar/           # LiDAR processing module
│   ├── fusion/          # Sensor fusion module
│   ├── drone/           # Drone interface and control
│   └── main.py          # Main entry point
├── tests/               # Unit tests
├── config/              # Configuration files
├── data/                # Data storage
│   ├── raw/             # Raw sensor data
│   └── processed/       # Processed datasets
├── models/              # Trained models
├── scripts/             # Utility scripts
└── docs/                # Documentation
```

## Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_vision.py

# Run with coverage
pytest --cov=src tests/
```

### Run System
```bash
# Detection mode (no hardware)
python src/main.py --mode detection --simulation

# Navigation mode (with drone)
python src/main.py --mode navigation

# Data collection mode
python src/main.py --mode data_collection
```

## Documentation
- [Setup Guide](docs/SETUP.md) - Detailed installation and configuration
- [API Documentation](docs/API.md) - Module and function reference

## Development
- Code formatting: `black src/ tests/`
- Linting: `flake8 src/ tests/`
- Import sorting: `isort src/ tests/`

## References
- [PULP-Dronet](https://github.com/pulp-platform/pulp-dronet)
- [AI Deck GAP8 Examples](https://github.com/bitcraze/aideck-gap8-examples)
- [Crazyflie Python Library](https://github.com/bitcraze/crazyflie-lib-python)

## License
See [LICENSE](LICENSE) file for details.