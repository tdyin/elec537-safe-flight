# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for deep learning acceleration

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install in Development Mode
```bash
pip install -e .
```

## Crazyflie Setup

### Hardware Setup
1. Attach the AI deck to the Crazyflie 2.1
2. Attach the Flow deck for optical flow positioning
3. Attach the Multi-ranger deck for LiDAR sensing
4. Charge the battery

### Firmware
Follow the Bitcraze documentation to flash the latest firmware:
- [Crazyflie Firmware](https://github.com/bitcraze/crazyflie-firmware)
- [AI Deck GAP8 Examples](https://github.com/bitcraze/aideck-gap8-examples)

### Connection
1. Power on the Crazyflie
2. Configure the radio URI in `config/config.yaml`
3. Test connection: `python scripts/test_connection.py`

## Running the System

### Detection Mode (No Drone)
```bash
python src/main.py --mode detection --simulation
```

### Navigation Mode (With Drone)
```bash
python src/main.py --mode navigation
```

### Data Collection Mode
```bash
python src/main.py --mode data_collection
```

## Testing

Run module tests:
```bash
python scripts/test_modules.py
```

Run unit tests:
```bash
pytest tests/
```

## Configuration

Edit `config/config.yaml` to adjust:
- Detection parameters
- Sensor fusion weights
- Navigation settings
- Logging options

## Troubleshooting

### Crazyflie Connection Issues
- Check battery charge
- Verify radio URI matches your Crazyflie
- Ensure cflib is properly installed

### GPU Issues
- Set `use_gpu: false` in config.yaml to use CPU
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
