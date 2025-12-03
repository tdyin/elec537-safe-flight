# Safe Flight - AI Agent Instructions

## Project Overview
Vision-based obstacle avoidance system for Crazyflie drones using ONNX-optimized deep learning models. The system uses monocular depth estimation for real-time autonomous navigation in Software-In-The-Loop (SITL) simulation.

## Architecture Patterns

### Vision-Based Navigation
The system uses **depth-based vision** for obstacle detection:
- Depth: `vision/depth_detector.py` → monocular depth estimation
- Control: `drone/depth_controller.py` → stable avoidance and path planning
- Fusion: `fusion/enhanced_fusion.py` → depth-aware obstacle mapping

Vision mode is configured in `config.yaml` under `vision.mode: "depth"`.

### SITL Communication Architecture
The system uses a **TCP socket bridge** between external Python control and Webots simulation:
- `sim/bridge.py`: Client-side bridge (connects to simulator)
- `sim/webots/controllers/crazyflie_sitl/crazyflie_sitl.py`: Server-side Webots controller
- Communication protocol: Binary messages with type codes (MSG_SENSOR_DATA=1, MSG_VELOCITY_CMD=2, etc.)
- Port: 10020 (default, configurable)

**Critical**: External control code (`src/`) communicates with Webots via `WebotsInterface` → `SimulationBridge` → TCP → Webots controller. Never attempt direct Webots API calls from `src/`.

### Sensor Fusion
Depth-based fusion for obstacle mapping:
- **Vision**: Depth → projected 3D obstacles
- Creates occupancy grid for path planning

See `fusion/enhanced_fusion.py::fuse_multimodal()` for implementation.

## Development Workflows

### Environment Setup
**Always use conda** - environment managed by `environment.yaml`:
```bash
conda env create -f environment.yaml
conda activate safe-flight
```

Key dependencies: `onnxruntime`, `opencv`, `open3d`, `cflib`, `loguru`

### Running SITL Tests
Standard workflow for testing navigation:
```bash
# Launch SITL with vision-based navigation (default)
python launch.py

# Different worlds and options
python launch.py --world open
python launch.py --no-gui             # Headless mode (faster)
python launch.py --goal 6 0 1         # Set goal position
python launch.py --viz                # Enable visualization
python launch.py --analyze            # Analyze latest log
```

**Important**: `launch.py` is the canonical entry point for SITL. It handles Webots path detection, world file loading, and coordinated startup.

### Model Management
Models live in `models/` and are downloaded via `scripts/download_models.py`:
- Depth: MiDaS v2.1 Small (63.7 MB)

Model paths configured in `config.yaml` under `vision.depth`.
### Testing
```bash
pytest tests/ -v                    # Full test suite
python scripts/test_vision_detector.py  # Vision system only
python scripts/test_modules.py      # Module verification
```

Test structure follows pytest conventions (`tests/test_*.py`, classes `Test*`, functions `test_*`). Coverage config in `pytest.ini`.

## Code Conventions

### Configuration-Driven Behavior
**Never hardcode paths or parameters** - use `config.yaml`:
- Vision model paths: `vision.depth.model_path`
- Navigation parameters: `drone.navigation.*`
- Fusion weights: `fusion.depth_weight`

Load config via `yaml.safe_load()` in main scripts.
### Interface Abstraction Pattern
Drone interfaces inherit from base `CrazyflieInterface`:
- `drone/interface.py`: Hardware interface (cflib)
- `drone/webots_interface.py`: SITL interface (TCP bridge)

Both expose identical API: `connect()`, `get_sensor_data()`, `send_velocity_command()`, `disconnect()`.

**When adding features**: Implement in both interfaces or abstract to base class.

### Crash Detection State Machine
Navigation controllers track crash state with careful sequencing:
1. `crash_detection_enabled = False` initially
2. After takeoff reaches `min_flight_altitude` (0.3m) → `takeoff_complete = True`
3. Only then enable crash detection
4. Once `crashed = True`, all commands rejected until recovery

See `drone/depth_controller.py::check_crash()` and `drone/webots_interface.py::_check_crash()`.
### ONNX Model Loading Pattern
Standard pattern for vision models:
```python
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
self.session = ort.InferenceSession(model_path, session_options, providers=providers)
```

Always check `ONNX_AVAILABLE` flag before loading (see `vision/depth_detector.py`).
## Critical Implementation Details

### Coordinate Systems
- **Webots**: X-forward, Y-left, Z-up (right-handed)
- **Crazyflie cflib**: X-forward, Y-left, Z-up (matches Webots)
- **Velocity commands**: Body-frame (vx=forward, vy=left, vz=up)
- **Camera frame**: Origin at top-left, Y-down (OpenCV convention)

### PID Controller Integration
Webots controllers use embedded PID (`sim/webots/controllers/*/pid_controller.py`):
- Converts high-level velocity commands → motor PWM
- Runs at Webots timestep (32ms default)
- External control sends velocity setpoints, PID handles low-level stabilization

### Depth Map Interpretation
MiDaS outputs **inverse depth** (disparity):
- Larger values = closer objects
- Normalize and invert: `depth = 1.0 / (disparity + epsilon)`
- Apply `depth_scale` from config for calibration
- See `vision/depth_utils.py` for conversion utilities

## Common Pitfalls

1. **Mixed control modes**: Don't combine Webots keyboard control with external TCP commands - use one or the other
2. **Model path errors**: Models must be in `models/` directory relative to project root, not `scripts/`
3. **Conda vs system Python**: Always activate `safe-flight` environment - system Python lacks ONNX
4. **Webots socket binding**: If port 10020 in use, kill old Webots processes: `pkill -f webots`

## Reference Files
- Main documentation: `docs/DOCUMENTATION.md` (comprehensive guide)
- Development history: `docs/DEVELOPMENT_LOG.md` (chronological decisions)
- Project structure: README.md lines 60-100
