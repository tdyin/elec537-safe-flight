# Safe Flight - Documentation

**Vision-Based Obstacle Avoidance for Crazyflie Drone**

Last Updated: December 3, 2025  
Project Status: ✅ SITL Functional | 🔄 Hardware Deployment In Progress

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Navigation System](#navigation-system)
4. [Vision System](#vision-system)
5. [SITL Development](#sitl-development)
6. [Hardware Deployment](#hardware-deployment)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Environment Setup
```bash
# Setup conda environment
conda env create -f environment.yaml
conda activate safe-flight

# Setup project (download models, verify dependencies)
make setup
```

### Run SITL Simulation
```bash
# Launch with vision-based navigation (recommended)
make sim

# Or use the script directly:
python scripts/launch_sim.py

# With specific goal
python scripts/launch_sim.py --goal 6 0 1

# With custom waypoints
python scripts/launch_sim.py --waypoints "[[2,0,1],[4,1,1],[6,0,1]]"

# With visualization
make sim-viz

# Different worlds
make sim-open
make sim-headless  # No GUI (faster)
```

### Run Tests
```bash
make test
# or
pytest tests/ -v
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Main Control System (src/)                  │
├─────────────────────────────────────────────────────────┤
│  vision/                 │  drone/                       │
│  ├── depth_detector.py   │  ├── depth_controller.py       │
│  └── depth_utils.py      │  ├── webots_interface.py       │
│                          │  └── interface.py              │
│  planning/               │  fusion/                      │
│  ├── path_planner.py     │  └── enhanced_fusion.py        │
│  ├── avoidance_          │                               │
│  │   controller.py       │  sim/                         │
│  └── trajectory_         │  └── bridge.py                 │
│      smoother.py         │                               │
└──────────────────────────┼───────────────────────────────┘
                           │
                  TCP Socket (10020)
                           │
┌──────────────────────────┼───────────────────────────────┐
│         Webots Simulation (sim/webots/)                  │
│                                                          │
│  controllers/crazyflie_sitl/                             │
│  ├── crazyflie_sitl.py      # Main controller            │
│  ├── pid_controller.py      # Motor control              │
│  └── modules/                                            │
│      ├── navigation_controller.py                        │
│      ├── depth_analyzer.py                               │
│      ├── path_planner.py                                 │
│      ├── safety_monitor.py                               │
│      └── sensors.py                                      │
└──────────────────────────────────────────────────────────┘
```

### Supported Models

| Model | Size | Purpose |
|-------|------|---------|
| MiDaS v2.1 Small | ~64 MB | Monocular depth estimation |

---

## Navigation System

### State Machine

```
PATH_FOLLOWING ──(obstacle)──> AVOIDING
      ↑                            │
      └────(clear path)──── RETURNING
                    │
                    ↓
              GOAL_REACHED
```

### Obstacle Zones

| Zone | Depth Value | Behavior |
|------|-------------|----------|
| CRITICAL | < 0.2 | Emergency stop |
| CLOSE | 0.2 - 0.35 | Active avoidance |
| CAUTION | 0.35 - 0.5 | Slow down |
| FAR | 0.5 - 0.6 | Monitor |
| CLEAR | > 0.6 | Cruise speed |

*Depth values are normalized (0=close, 1=far)*

### Path Planning

The system uses A* path planning with Bezier smoothing:
- Grid-based obstacle mapping from depth/segmentation
- Safety margin around obstacles
- Smooth trajectory generation
- Periodic replanning for dynamic environments

---

## Vision System

### Pipeline

1. **Image Acquisition** - Camera image (BGRA)
2. **Depth Estimation** - MiDaS generates depth map
3. **Zone Analysis** - Classify left/center/right zones by depth
4. **Obstacle Detection** - Identify close obstacles from depth
5. **Navigation Decision** - Determine safe velocity
6. **Smoothing** - Apply velocity smoothing

### Performance

| Metric | Value |
|--------|-------|
| Depth Inference | ~30 ms |
| Navigation Update | 50 Hz |
| Obstacle Response | <100 ms |

---

## SITL Development

### Available Worlds

| World | Description |
|-------|-------------|
| `apartment` | Indoor with furniture (default) |
| `open` | Open world |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Toggle Manual/Autonomous |
| `W/S` | Forward/Backward |
| `A/D` | Strafe Left/Right |
| `Q/E` | Yaw Left/Right |
| `↑/↓` | Ascend/Descend |
| `R` | Reset to Autonomous |

### Communication

- **Protocol**: TCP socket on port 10020
- **Data Rate**: 50 Hz
- **Messages**: Sensor data (JSON), Velocity commands (binary)

---

## Hardware Deployment

> **Note:** Hardware flight support is under development. See `docs/DEPLOYMENT_PLAN.md` for full details.

### Target Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Drone | Crazyflie 2.1 | Flight platform |
| Positioning | Flow Deck v2 | Relative position estimation |
| Camera | AI Deck | Vision input for depth estimation |
| Radio | Crazyradio PA | Communication link |

### Hardware vs Simulation

| Feature | Simulation | Hardware |
|---------|------------|----------|
| Config | `config/sim.yaml` | `config/hardware.yaml` |
| Launcher | `scripts/launch_sim.py` | `scripts/launch_hardware.py` |
| Make target | `make sim` | `make hardware` |
| Interface | `WebotsInterface` | `CrazyflieInterface` |
| Camera | Webots camera | AI Deck WiFi stream |
| Position | Ground truth | Flow Deck estimation |
| Control | Velocity commands | MotionCommander |

### Hardware Safety Features

- Battery monitoring (auto-land at 3.3V)
- Geofence (3m radius from start)
- Tilt limit (40° emergency stop)
- Communication timeout handling

---

## Configuration

Configuration is split by environment:
- `config/sim.yaml` - Simulation settings
- `config/hardware.yaml` - Hardware settings (conservative)

Key sections:

### Vision
```yaml
vision:
  depth:
    model_path: "models/midas_v21_small.onnx"
  obstacle:
    critical_threshold: 0.15
    close_threshold: 0.25
    caution_threshold: 0.40
```

### Navigation
```yaml
drone:
  navigation:
    cruise_speed: 0.5
    avoidance_speed: 0.3
    safety_distance: 0.8
    zone_critical: 0.2
    zone_close: 0.35
```

### Path Planning
```yaml
drone:
  path_planning:
    enabled: true
    planning_resolution: 0.1
    safety_margin: 3
    path_smoothing: "bezier"
```

### Simulation
```yaml
drone:
  simulation:
    host: "localhost"
    port: 10020
```

---

## Troubleshooting

### Connection Failed
```bash
# Check Webots is running with crazyflie_sitl controller
# Verify port 10020 is not blocked
pkill -f webots  # Kill stale processes
```

### Import Errors
```bash
conda activate safe-flight
pip install -e .
```

### Model Not Found
```bash
make setup-models
ls -lh models/
```

### Low FPS
- Increase `vision_interval` in controller
- Use MobileNetV3 instead of ResNet50
- Enable headless mode: `make sim-headless`

### Environment Issues
```bash
# Recreate environment
conda env remove -n safe-flight
conda env create -f environment.yaml
conda activate safe-flight
```

---

## Project Structure

```
elec537-safe-flight/
├── Makefile                 # Primary entry point for all commands
├── config/                  # Configuration files
│   ├── sim.yaml            # Simulation config
│   └── hardware.yaml       # Hardware config
├── environment.yaml         # Conda environment
├── models/                  # ONNX models
├── src/                     # Main Python code
│   ├── core/               # Shared library (base interface, types, safety)
│   ├── hardware/           # Hardware interfaces (cflib, AI Deck)
│   ├── sim/                # Simulation code (bridge, webots_interface)
│   ├── drone/              # Controllers (depth_controller)
│   ├── vision/             # Detection & depth
│   ├── planning/           # Path planning
│   └── fusion/             # Sensor fusion
├── scripts/                 # Launch scripts & utilities
│   ├── launch_sim.py       # SITL launcher
│   ├── launch_hardware.py  # Hardware launcher
│   ├── setup.py            # Environment & model setup
│   └── cleanup.py          # Storage cleanup
├── sim/webots/             # Webots simulation
│   ├── controllers/        # Webots controllers
│   ├── worlds/             # Environment files
│   └── logs/               # Flight logs
├── tests/                   # Unit tests
├── data/                    # Generated data
│   ├── raw/                # Raw flight data
│   ├── processed/          # Processed data
│   └── visualization/      # Visualization outputs
└── docs/                    # Documentation
    ├── DOCUMENTATION.md    # This file
    ├── DEVELOPMENT_LOG.md  # Development history
    └── DEPLOYMENT_PLAN.md  # Hardware deployment plan
```

---

## References

- **Webots**: https://cyberbotics.com/doc/guide/index
- **Crazyflie**: https://www.bitcraze.io/documentation/
- **cflib**: https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/
- **AI Deck**: https://www.bitcraze.io/documentation/repository/AIdeck_examples/master/
- **MiDaS**: https://github.com/isl-org/MiDaS
- **ONNX Runtime**: https://onnxruntime.ai/docs/

---

**Project:** ELEC 537 Safe Flight  
**Repository:** tdyin/elec537-safe-flight
