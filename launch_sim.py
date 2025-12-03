#!/usr/bin/env python3
"""
SITL launcher for Safe Flight.

Usage:
    python launch.py                          # Default: apartment world with GUI
    python launch.py --world open             # Different world
    python launch.py --goal 6 0 1             # Set goal position
    python launch.py --no-gui                 # Headless mode (faster)
    python launch.py --viz                    # Enable visualization
    python launch.py --analyze                # Analyze latest log file
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import os
import json
from typing import Optional, List
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Available worlds
WORLDS = {
    'apartment': 'sim/webots/worlds/crazyflie_apartment.wbt',
    'open': 'sim/webots/worlds/crazyflie_open.wbt',
}


def find_latest_log() -> Optional[Path]:
    """Find the most recent log file."""
    log_dir = Path(__file__).parent / 'sim' / 'webots' / 'logs'
    if not log_dir.exists():
        return None
    log_files = sorted(log_dir.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


def run_visualization_analysis(log_path: Path, show: bool = True):
    """Run post-flight visualization analysis."""
    print(f"\n📊 Analyzing log: {log_path.name}")
    
    viz_script = Path(__file__).parent / 'sim' / 'webots' / 'utils' / 'viz_nav_graph.py'
    if not viz_script.exists():
        print("Warning: Visualization script not found")
        return
    
    output_dir = Path(__file__).parent / 'data' / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [sys.executable, str(viz_script), '--log', str(log_path), '--output', str(output_dir)]
    if show:
        cmd.append('--show')
        
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ Visualizations saved to: {output_dir}")
        else:
            print(f"Warning: Visualization failed: {result.stderr}")
    except Exception as e:
        print(f"Warning: Visualization error: {e}")


def find_webots_executable() -> Optional[str]:
    """Find Webots executable on the system."""
    # Check WEBOTS_HOME environment variable first
    webots_home = os.environ.get('WEBOTS_HOME')
    if webots_home:
        webots_bin = os.path.join(webots_home, 'webots')
        if os.path.exists(webots_bin):
            return webots_bin
    
    # Try finding in PATH
    try:
        result = subprocess.run(['which', 'webots'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'webots'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Common installation locations
    paths = [
        '/Applications/Webots.app/Contents/MacOS/webots',  # macOS
        '~/Applications/Webots.app/Contents/MacOS/webots',
        '/usr/local/webots/webots',  # Linux
        '/snap/webots/current/usr/bin/webots',
    ]
    
    for path in paths:
        expanded = Path(path).expanduser()
        if expanded.exists():
            return str(expanded)
    
    return None


def launch_webots(world_file: str, no_gui: bool = False,
                  waypoints: Optional[List] = None,
                  goal: Optional[List[float]] = None,
                  enable_viz: bool = False,
                  config: Optional[dict] = None) -> subprocess.Popen:
    """
    Launch Webots with vision-based autonomous navigation.
    
    Args:
        world_file: Path to Webots world file
        no_gui: Run in headless mode
        waypoints: List of waypoints [[x,y,z], ...] for path navigation
        goal: Single goal position [x, y, z]
        enable_viz: Enable real-time visualization
        config: Configuration dict from config.yaml
        
    Returns:
        Subprocess handle for Webots
    """
    world_path = Path(world_file).resolve()
    
    if not world_path.exists():
        print(f"Error: World file not found: {world_path}")
        sys.exit(1)
    
    webots_exe = find_webots_executable()
    if not webots_exe:
        print("\nError: Webots not found!")
        print("Install from: https://github.com/cyberbotics/webots/releases")
        sys.exit(1)
    
    # Build command
    cmd = [webots_exe]
    cmd.append('--mode=fast' if no_gui else '--mode=realtime')
    if no_gui:
        cmd.append('--no-rendering')  # True headless - disables 3D rendering
        cmd.append('--minimize')      # Also minimize window
        cmd.append('--batch')          # Prevent blocking pop-ups
    cmd.append(str(world_path))
    
    # Set environment
    env = os.environ.copy()
    env['SITL_MODE'] = 'vision'
    env['WORLD_NAME'] = Path(world_file).stem.replace('crazyflie_', '')
    
    # Set depth model path
    depth_model = Path(__file__).parent / 'models' / 'midas_v21_small.onnx'
    if depth_model.exists():
        env['DEPTH_MODEL_PATH'] = str(depth_model)
    
    if waypoints:
        env['NAV_WAYPOINTS'] = json.dumps(waypoints)
    if goal:
        env['NAV_GOAL'] = json.dumps(goal)
    if enable_viz:
        env['ENABLE_NAV_VIZ'] = '1'
    
    # Pass navigation config from config.yaml
    if config:
        nav_config = config.get('drone', {}).get('navigation', {})
        sim_config = config.get('drone', {}).get('simulation', {})
        
        # Navigation parameters
        if 'cruise_speed' in nav_config:
            env['NAV_CRUISE_SPEED'] = str(nav_config['cruise_speed'])
        if 'avoidance_speed' in nav_config:
            env['NAV_AVOIDANCE_SPEED'] = str(nav_config['avoidance_speed'])
        if 'safety_distance' in nav_config:
            env['NAV_SAFETY_DISTANCE'] = str(nav_config['safety_distance'])
        if 'turn_rate' in nav_config:
            env['NAV_TURN_RATE'] = str(nav_config['turn_rate'])
        if 'avoidance_duration' in nav_config:
            env['NAV_AVOIDANCE_DURATION'] = str(nav_config['avoidance_duration'])
        if 'min_altitude' in nav_config:
            env['NAV_MIN_ALTITUDE'] = str(nav_config['min_altitude'])
        if 'max_altitude' in nav_config:
            env['NAV_MAX_ALTITUDE'] = str(nav_config['max_altitude'])
        if 'default_altitude' in nav_config:
            env['NAV_DEFAULT_ALTITUDE'] = str(nav_config['default_altitude'])
        
        # Simulation parameters
        if 'smoothing_alpha' in sim_config:
            env['SIM_SMOOTHING_ALPHA'] = str(sim_config['smoothing_alpha'])
        if 'max_velocity' in sim_config:
            env['SIM_MAX_VELOCITY'] = str(sim_config['max_velocity'])
        if 'max_yaw_rate' in sim_config:
            env['SIM_MAX_YAW_RATE'] = str(sim_config['max_yaw_rate'])
        if 'warning_tilt' in sim_config:
            env['SIM_WARNING_TILT'] = str(sim_config['warning_tilt'])
        if 'critical_tilt' in sim_config:
            env['SIM_CRITICAL_TILT'] = str(sim_config['critical_tilt'])
        if 'print_interval' in sim_config:
            env['SIM_PRINT_INTERVAL'] = str(sim_config['print_interval'])
    
    print(f"\nLaunching Webots...")
    print(f"  World: {world_path.name}")
    print(f"  Mode:  {'Headless' if no_gui else 'GUI'}")
    if config:
        nav_config = config.get('drone', {}).get('navigation', {})
        print(f"  Speed: {nav_config.get('cruise_speed', 0.25)}m/s")
        print(f"  Altitude: {nav_config.get('default_altitude', 1.0)}m (min: {nav_config.get('min_altitude', 0.5)}m, max: {nav_config.get('max_altitude', 2.0)}m)")
    if goal:
        print(f"  Goal:  {goal}")
    if waypoints:
        print(f"  Waypoints: {len(waypoints)} points")
    
    process = subprocess.Popen(cmd, env=env, stdout=None, stderr=None)
    time.sleep(3)
    
    return process


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='SITL launcher for Safe Flight drone simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python launch.py                            # Default: apartment world
  python launch.py --world open               # Open world
  python launch.py --goal 6 0 1               # Set goal position
  python launch.py --waypoints "[[2,0,1],[4,1,1]]"  # Set waypoints
  python launch.py --no-gui                   # Headless mode (faster)
  python launch.py --viz                      # Enable visualization
  python launch.py --analyze                  # Analyze latest log

Available worlds: {', '.join(WORLDS.keys())}
        """
    )
    
    parser.add_argument('--world', type=str, default='apartment',
                       choices=list(WORLDS.keys()),
                       help='World to load (default: apartment)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run in headless mode')
    parser.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Goal position [x y z]')
    parser.add_argument('--waypoints', type=str,
                       help='JSON array of waypoints [[x,y,z], ...]')
    parser.add_argument('--viz', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze latest log file and exit')
    
    args = parser.parse_args()
    
    # Handle --analyze option
    if args.analyze:
        latest_log = find_latest_log()
        if latest_log:
            run_visualization_analysis(latest_log)
            return 0
        print("No log files found in sim/webots/logs/")
        return 1
    
    # Parse waypoints
    waypoints = None
    if args.waypoints:
        try:
            waypoints = json.loads(args.waypoints)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid waypoints JSON: {e}")
            return 1
    
    # Print header
    print("\n" + "="*50)
    print("Safe Flight - SITL Launcher")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    webots_process = None
    
    try:
        webots_process = launch_webots(
            WORLDS[args.world],
            no_gui=args.no_gui,
            waypoints=waypoints,
            goal=args.goal,
            enable_viz=args.viz,
            config=config
        )
        
        print("\n" + "="*50)
        print(f"Webots running (PID: {webots_process.pid})")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        # Wait for process
        while webots_process.poll() is None:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        if webots_process:
            webots_process.terminate()
            try:
                webots_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                webots_process.kill()
        
        # Run visualization if enabled
        if args.viz:
            time.sleep(1)
            latest_log = find_latest_log()
            if latest_log:
                run_visualization_analysis(latest_log)
        
        print("Done.")


if __name__ == '__main__':
    sys.exit(main())
