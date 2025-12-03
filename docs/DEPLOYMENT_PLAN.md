# Hardware Deployment Plan

**Project:** Safe Flight - Vision-Based Obstacle Avoidance  
**Target:** Crazyflie 2.1 with Flow Deck + AI Deck  
**Created:** December 3, 2025  
**Status:** ✅ Phase 1-2 Complete | 🔄 Phase 3 Next

---

## Overview

This document outlines the implementation plan to enable real Crazyflie drone flight, transitioning from the current SITL-only system to a unified codebase supporting both simulation and hardware.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vision Processing | Offboard (laptop) | Simpler integration, full MiDaS model capability |
| Positioning | Flow Deck (relative) | No external infrastructure needed |
| Motion Control | `MotionCommander` | Built-in safety, simpler API |
| Initial Flight Mode | Forward-only | Reduced complexity for first tests |

---

## Phase 1: Codebase Refactoring

**Goal:** Clean separation between simulation and hardware with shared core logic.

### 1.1 New Directory Structure

```
elec537-safe-flight/
├── Makefile                  # Primary entry point for all commands
├── config/
│   ├── sim.yaml              # Simulation configuration
│   └── hardware.yaml         # Hardware configuration
├── scripts/
│   ├── launch_sim.py         # SITL launcher (moved from root)
│   ├── launch_hardware.py    # Hardware launcher (NEW)
│   ├── setup.py              # Environment setup & model download
│   └── cleanup.py            # Storage cleanup utility
├── src/
│   ├── core/                 # Shared library (NEW)
│   │   ├── __init__.py
│   │   ├── base_interface.py # Abstract drone interface
│   │   ├── navigation.py     # Navigation algorithms
│   │   ├── safety.py         # Safety state machine
│   │   └── types.py          # Shared data types
│   ├── sim/                  # Simulation-specific
│   │   ├── __init__.py
│   │   ├── bridge.py         # TCP bridge (existing)
│   │   └── webots_interface.py  # Moved from drone/
│   ├── hardware/             # Hardware-specific (NEW)
│   │   ├── __init__.py
│   │   ├── crazyflie_interface.py  # cflib integration
│   │   ├── aideck_camera.py  # AI deck streaming
│   │   └── sensor_logger.py  # cflib LogConfig wrapper
│   ├── vision/               # Shared vision (existing)
│   ├── planning/             # Shared planning (existing)
│   ├── fusion/               # Shared fusion (existing)
│   └── drone/                # Shared controllers
│       ├── __init__.py
│       └── depth_controller.py  # Navigation controller
└── tests/
    ├── unit/                 # Unit tests
    ├── integration/          # Integration tests
    └── hardware/             # Hardware-in-the-loop tests
```

### 1.2 Refactoring Tasks

| Task | Current Location | New Location | Changes |
|------|------------------|--------------|---------|
| Abstract interface | `drone/interface.py` | `core/base_interface.py` | Extract ABC |
| Webots interface | `drone/webots_interface.py` | `sim/webots_interface.py` | Inherit from ABC |
| Safety logic | Duplicated | `core/safety.py` | Consolidate |
| TCP bridge | `sim/bridge.py` | `sim/bridge.py` | No change |
| Move launcher | `launch.py` | `scripts/launch_sim.py` | Move to scripts/ |
| Setup script | `scripts/download_models.py` | `scripts/setup.py` | Enhanced setup |
| Cleanup script | `scripts/cleanup.sh` | `scripts/cleanup.py` | Rewrite in Python |
| Makefile | N/A | `Makefile` | New entry point |

### 1.3 Abstract Base Interface

```python
# src/core/base_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np

class DroneInterface(ABC):
    """Abstract base class for drone interfaces (sim and hardware)."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to drone. Returns True on success."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from drone."""
        pass
    
    @abstractmethod
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings."""
        pass
    
    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z) in meters."""
        pass
    
    @abstractmethod
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation (roll, pitch, yaw) in radians."""
        pass
    
    @abstractmethod
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Send velocity command in body frame."""
        pass
    
    @abstractmethod
    def takeoff(self, height: float = 0.5) -> bool:
        """Takeoff to specified height. Returns True on success."""
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """Land the drone. Returns True on success."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop - cut motors immediately."""
        pass
```

---

## Phase 2: Configuration System

**Goal:** Separate configuration files for simulation and hardware.

### 2.1 Simulation Config (`config/sim.yaml`)

```yaml
# Simulation-specific configuration
mode: "simulation"

vision:
  depth:
    model_path: "models/midas_v21_small.onnx"
    use_gpu: false
  obstacle:
    critical_threshold: 0.15
    close_threshold: 0.25
    caution_threshold: 0.40

drone:
  interface: "webots"
  simulation:
    host: "localhost"
    port: 10020
    world: "apartment"
  
  navigation:
    cruise_speed: 0.5
    avoidance_speed: 0.3
    safety_distance: 0.8
    max_altitude: 2.0
    min_altitude: 0.5

  path_planning:
    enabled: true
    planning_resolution: 0.1
    safety_margin: 3
    path_smoothing: "bezier"

logging:
  level: "INFO"
  log_directory: "sim/webots/logs"
```

### 2.2 Hardware Config (`config/hardware.yaml`)

```yaml
# Hardware-specific configuration
mode: "hardware"

vision:
  depth:
    model_path: "models/midas_v21_small.onnx"
    use_gpu: false  # Laptop CPU inference
  obstacle:
    # More conservative thresholds for real flight
    critical_threshold: 0.20
    close_threshold: 0.30
    caution_threshold: 0.45

drone:
  interface: "crazyflie"
  uri: "radio://0/80/2M/E7E7E7E7E7"
  
  # Hardware requirements
  hardware:
    flow_deck_required: true
    multiranger_required: false  # Optional
    ai_deck_required: true
    arming_required: true
  
  # AI Deck camera
  ai_deck:
    enabled: true
    ip: "192.168.4.1"
    port: 5000
    encoding: "jpeg"
    timeout: 5.0
    frame_rate: 15  # Target FPS
  
  # cflib logging rates
  logging:
    stabilizer_rate_ms: 20
    state_estimate_rate_ms: 20
    battery_rate_ms: 500
  
  navigation:
    # Conservative speeds for real hardware
    cruise_speed: 0.3
    avoidance_speed: 0.15
    max_speed: 0.4
    safety_distance: 1.0
    max_altitude: 1.5
    min_altitude: 0.3
    default_altitude: 0.5
    
    # Forward-only mode for initial testing
    forward_only: true
    
  safety:
    battery_min_voltage: 3.3  # Land if below
    battery_warning_voltage: 3.5
    geofence_radius: 3.0  # meters from start
    geofence_height: 2.0
    crash_tilt_threshold: 40.0  # degrees
    crash_altitude_threshold: 0.15

  flight:
    takeoff_height: 0.5
    takeoff_velocity: 0.3
    land_velocity: 0.2

logging:
  level: "DEBUG"  # More verbose for hardware
  log_directory: "logs/hardware"
```

---

## Phase 3: cflib Logging System

**Goal:** Implement sensor data acquisition via cflib LogConfig.

### 3.1 Sensor Logger Module (`src/hardware/sensor_logger.py`)

```python
# Key components:
class SensorLogger:
    """Manages cflib LogConfig subscriptions for sensor data."""
    
    def __init__(self, scf: SyncCrazyflie, config: dict):
        self.scf = scf
        self._sensor_data = {}  # Thread-safe sensor cache
        self._lock = threading.Lock()
        
    def setup_logging(self):
        """Setup all log configurations."""
        # StateEstimate: position and velocity
        # Stabilizer: roll, pitch, yaw
        # Battery: voltage
        # Range: multi-ranger (if available)
        
    def get_sensor_data(self) -> Dict:
        """Get latest sensor readings (thread-safe)."""
```

### 3.2 Log Variables Required

| LogConfig | Variables | Rate | Purpose |
|-----------|-----------|------|---------|
| StateEstimate | x, y, z, vx, vy, vz | 20ms | Position/velocity |
| Stabilizer | roll, pitch, yaw | 20ms | Orientation |
| Battery | pm.vbat | 500ms | Safety monitoring |
| Range | front, back, left, right, zrange | 50ms | Multi-ranger (optional) |

---

## Phase 4: Hardware Tests

**Goal:** Establish testing framework before flight implementation.

### 4.1 Test Structure

```
tests/
├── unit/
│   ├── test_base_interface.py    # ABC contract tests
│   ├── test_safety_machine.py    # Safety state machine
│   └── test_sensor_logger.py     # LogConfig setup
├── integration/
│   ├── test_cflib_connection.py  # Radio connection
│   └── test_deck_detection.py    # Deck availability
└── hardware/
    ├── test_hover.py             # Basic hover test
    ├── test_forward_flight.py    # Forward motion
    └── test_emergency_stop.py    # Safety systems
```

### 4.2 Test Markers

```python
# conftest.py additions
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: tests requiring real drone")
    config.addinivalue_line("markers", "slow: tests taking >5 seconds")

# Usage
@pytest.mark.hardware
def test_hover():
    """Test basic hover (requires drone)."""
    pass
```

### 4.3 Mock Fixtures

```python
@pytest.fixture
def mock_crazyflie():
    """Mock cflib Crazyflie for unit testing."""
    with patch('cflib.crazyflie.Crazyflie') as mock:
        mock.return_value.is_connected.return_value = True
        yield mock

@pytest.fixture
def mock_log_config():
    """Mock LogConfig for testing logging setup."""
    with patch('cflib.crazyflie.log.LogConfig') as mock:
        yield mock
```

---

## Phase 5: Flight Sequence Methods

**Goal:** Implement takeoff, hover, land using MotionCommander.

### 5.1 CrazyflieInterface Implementation

```python
# src/hardware/crazyflie_interface.py
class CrazyflieInterface(DroneInterface):
    
    def __init__(self, config: dict):
        self.config = config
        self.uri = config['drone']['uri']
        self.scf = None
        self.mc = None  # MotionCommander
        self.sensor_logger = None
        
    def connect(self) -> bool:
        """Connect and verify deck presence."""
        cflib.crtp.init_drivers()
        self.scf = SyncCrazyflie(self.uri)
        self.scf.open_link()
        
        # Verify required decks
        if not self._check_decks():
            return False
        
        # Setup logging
        self.sensor_logger = SensorLogger(self.scf, self.config)
        self.sensor_logger.setup_logging()
        
        return True
    
    def takeoff(self, height: float = 0.5) -> bool:
        """Takeoff using MotionCommander."""
        self.mc = MotionCommander(self.scf, default_height=height)
        self.mc.take_off()
        return True
    
    def land(self) -> bool:
        """Land using MotionCommander."""
        if self.mc:
            self.mc.land()
        return True
    
    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """Send velocity using MotionCommander."""
        if self.mc:
            # Forward-only mode: ignore vy
            if self.config['drone']['navigation'].get('forward_only', False):
                vy = 0.0
            self.mc.start_linear_motion(vx, vy, vz, yaw_rate)
```

### 5.2 Forward-Only Navigation

For initial testing, the drone will only fly forward:
- `vy` (lateral) commands are zeroed
- Obstacle avoidance triggers yaw rotation instead of lateral movement
- Simpler dynamics, easier to predict behavior

---

## Phase 6: Safety State Machine

**Goal:** Robust safety monitoring with emergency handling.

### 6.1 Safety States

```
INITIALIZING ──> READY ──> ARMED ──> FLYING ──> LANDING ──> LANDED
                   │                    │
                   │                    ▼
                   └──────────────> EMERGENCY
```

### 6.2 Safety Monitor (`src/core/safety.py`)

```python
class SafetyMonitor:
    """Monitors drone state and triggers safety responses."""
    
    def __init__(self, config: dict):
        self.state = SafetyState.INITIALIZING
        self.battery_min = config['drone']['safety']['battery_min_voltage']
        self.geofence_radius = config['drone']['safety']['geofence_radius']
        self.tilt_threshold = config['drone']['safety']['crash_tilt_threshold']
        
    def check(self, sensor_data: Dict) -> SafetyState:
        """Check all safety conditions. Returns new state."""
        
        # Battery check
        if sensor_data['battery'] < self.battery_min:
            return SafetyState.EMERGENCY
        
        # Geofence check
        pos = sensor_data['position']
        if np.linalg.norm(pos[:2]) > self.geofence_radius:
            return SafetyState.EMERGENCY
        
        # Tilt check
        if abs(sensor_data['roll']) > self.tilt_threshold:
            return SafetyState.EMERGENCY
        if abs(sensor_data['pitch']) > self.tilt_threshold:
            return SafetyState.EMERGENCY
        
        return self.state
```

### 6.3 Emergency Responses

| Trigger | Response |
|---------|----------|
| Low battery (<3.3V) | Immediate land |
| Geofence breach | Stop and hover |
| Excessive tilt (>40°) | Motor cutoff |
| Communication loss | Hover 3s, then land |

---

## Phase 7: AI Deck Camera Module

**Goal:** Stream camera images from AI Deck over WiFi.

### 7.1 Camera Interface (`src/hardware/aideck_camera.py`)

```python
class AIdeckCamera:
    """WiFi camera streaming from AI Deck."""
    
    def __init__(self, ip: str = "192.168.4.1", port: int = 5000):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to AI Deck WiFi stream."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5.0)
        self.socket.connect((self.ip, self.port))
        self.connected = True
        return True
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Receive and decode JPEG frame."""
        # Read frame size header
        # Read JPEG data
        # Decode to numpy array
        pass
        
    def disconnect(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
        self.connected = False
```

### 7.2 Integration with Vision Pipeline

```python
# In main control loop
camera = AIdeckCamera(config['drone']['ai_deck']['ip'])
camera.connect()

while running:
    frame = camera.get_frame()
    if frame is not None:
        vision_data = depth_detector.detect(frame)
        safe_velocity = controller.compute_safe_velocity(vision_data, ...)
        drone.send_velocity_command(*safe_velocity)
```

---

## Phase 8: Simulation Equivalent Features

**Goal:** Ensure simulation mirrors hardware capabilities for testing.

### 8.1 Feature Parity Matrix

| Feature | Hardware | Simulation | Notes |
|---------|----------|------------|-------|
| Takeoff/Land | MotionCommander | Velocity ramp | Equiv. behavior |
| Forward-only mode | Config flag | Config flag | Same logic |
| Safety monitor | Real sensors | Sim sensors | Same code |
| Geofence | GPS-denied | Perfect position | Same limits |
| Camera | AI Deck WiFi | Webots camera | Same resolution |

### 8.2 Simulation Additions

```python
# src/sim/webots_interface.py additions

def takeoff(self, height: float = 0.5) -> bool:
    """Simulate takeoff by ramping altitude."""
    # Send increasing vz until height reached
    pass

def land(self) -> bool:
    """Simulate landing by decreasing altitude."""
    # Send decreasing vz until ground
    pass
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Status | Deliverables |
|-------|----------|--------------|--------|--------------|
| 1. Refactoring | 2 days | None | ✅ Complete | New structure, ABC |
| 2. Configuration | 1 day | Phase 1 | ✅ Complete | Split config files, config module |
| 3. Logging System | 2 days | Phase 1, 2 | 🔄 Next | SensorLogger class |
| 4. Hardware Tests | 1 day | Phase 1-3 | ⏳ Pending | Test framework |
| 5. Flight Sequence | 2 days | Phase 3, 4 | ⏳ Pending | Takeoff/land |
| 6. Safety Machine | 1 day | Phase 3, 5 | ⏳ Pending | SafetyMonitor |
| 7. AI Deck Camera | 2 days | Phase 1 | ⏳ Pending | AIdeckCamera |
| 8. Sim Equivalence | 1 day | Phase 5-7 | ⏳ Pending | Updated sim |

**Total Estimated Time:** 12 days

---

## Testing Checklist

### Pre-Flight (No Props)
- [ ] Radio connection successful
- [ ] Flow deck detected
- [ ] AI deck detected
- [ ] Battery voltage reading
- [ ] Sensor logging active
- [ ] Camera stream working
- [ ] Arm/disarm commands
- [ ] Emergency stop tested

### Tethered Flight
- [ ] Takeoff to 0.5m
- [ ] Stable hover (30s)
- [ ] Forward motion (0.1 m/s)
- [ ] Controlled land
- [ ] Emergency stop mid-flight

### Free Flight (Controlled Environment)
- [ ] Takeoff and hover
- [ ] Forward navigation (2m)
- [ ] Obstacle detection triggers
- [ ] Avoidance maneuver
- [ ] Return to start
- [ ] Land on command

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Radio interference | Test in RF-quiet environment |
| Camera latency | Reduce resolution, increase compression |
| Position drift | Short flights, frequent recalibration |
| Battery depletion | Conservative 3.3V cutoff |
| Collision | Foam props, soft obstacles, tether |

---

## References

- [Crazyflie Python Library (cflib)](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/)
- [AI Deck Documentation](https://www.bitcraze.io/documentation/repository/AIdeck_examples/master/)
- [MotionCommander API](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/positioning/motion_commander/)
- [Flow Deck Documentation](https://www.bitcraze.io/products/flow-deck-v2/)

---

**Next Steps:** Begin Phase 1 refactoring after plan approval.
