#!/usr/bin/env python3
"""
Navigation Visualization Tool

Provides separate visualizations for analyzing:
1. Path planning - A* grid, waypoints, planned paths
2. Obstacle detection - Depth maps, zone clearances, detected obstacles

Usage:
    # Analyze a log file
    python scripts/visualize_navigation.py --log sim/webots/logs/LOGFILE.log
    
    # Live visualization during simulation
    python scripts/visualize_navigation.py --live
    
    # Analyze saved depth images
    python scripts/visualize_navigation.py --depth-dir data/visualization/depth
"""

import argparse
import sys
from pathlib import Path
import re
from datetime import datetime
import yaml

# Add project root to path (utils is in sim/webots/utils, so go up 3 levels)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Optional imports
try:
    import matplotlib
    # Use non-interactive backend if no display available
    import os
    if not os.environ.get('DISPLAY') and os.environ.get('MPLBACKEND') is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    from matplotlib.collections import PatchCollection
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
    # Check if we can show interactive plots
    MATPLOTLIB_INTERACTIVE = matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps', 'cairo')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_INTERACTIVE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def load_config():
    """Load configuration from config/sim.yaml."""
    config_path = project_root / 'config' / 'sim.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


class PathPlanningVisualizer:
    """Visualizer for path planning analysis with 3D support."""
    
    def __init__(self, world_bounds=(-10, 10, -5, 5, 0, 3), target_altitude=1.0):
        """
        Initialize path planning visualizer.
        
        Args:
            world_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) world boundaries
            target_altitude: Target flight altitude for visualization
        """
        self.world_bounds = world_bounds
        self.target_altitude = target_altitude
        self.positions = []
        self.waypoints = []
        self.obstacles = []
        self.obstacle_encounters = []  # Track obstacle encounter events
        self.avoidance_events = []  # Track avoidance maneuver events
        self.goal = None
        self.start = None
        
    def add_position(self, x, y, z=0.0, timestamp=None):
        """Add a 3D position to the trajectory."""
        self.positions.append({
            'x': x, 'y': y, 'z': z, 'time': timestamp
        })
        
    def add_waypoint(self, x, y, label=None):
        """Add a waypoint."""
        self.waypoints.append({'x': x, 'y': y, 'label': label})
        
    def add_obstacle(self, x, y, radius=0.5, obstacle_type='unknown'):
        """Add an obstacle."""
        self.obstacles.append({
            'x': x, 'y': y, 'radius': radius, 'type': obstacle_type
        })
        
    def set_goal(self, x, y, z=1.0):
        """Set goal position."""
        self.goal = (x, y, z)
        
    def set_start(self, x, y, z=0.0):
        """Set start position."""
        self.start = (x, y, z)
        
    def add_obstacle_encounter(self, x, y, z=0.0, timestamp=None, severity='normal'):
        """Add an obstacle encounter event."""
        self.obstacle_encounters.append({
            'x': x, 'y': y, 'z': z, 'time': timestamp, 'severity': severity
        })
        
    def add_avoidance_event(self, x, y, z=0.0, timestamp=None, direction='unknown'):
        """Add an avoidance maneuver event."""
        self.avoidance_events.append({
            'x': x, 'y': y, 'z': z, 'time': timestamp, 'direction': direction
        })
        
    def parse_log_file(self, log_path):
        """Parse a log file to extract 3D navigation data."""
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            return
            
        # Regex patterns - support both old 2D and new 3D formats
        pos3d_pattern = r'Pos3D: \((-?\d+\.?\d*), (-?\d+\.?\d*), (-?\d+\.?\d*)\)'
        pos2d_pattern = r'Pos: \((-?\d+\.?\d*), (-?\d+\.?\d*)\)'
        alt_pattern = r'Alt: (\d+\.?\d*)m'
        goal_pattern = r'Goal:\s+\((-?\d+\.?\d*), (-?\d+\.?\d*), (-?\d+\.?\d*)\)'
        time_pattern = r'\[(\d{2}:\d{2}:\d{2})\]'
        
        # Patterns for obstacle encounters and avoidance events
        obstacle_pattern = r'\[OBSTACLE\]|\[EMERGENCY\]|\[CAUTION\]|obstacle detected|emergency'
        avoidance_pattern = r'\[AVOIDANCE\]|avoiding|avoidance|evasive'
        direction_pattern = r'direction[:\s]+(\w+)|moving\s+(\w+)|turn\s+(\w+)'
        
        current_pos = None
        
        with open(log_path, 'r') as f:
            for line in f:
                # Extract time
                time_match = re.search(time_pattern, line)
                timestamp = time_match.group(1) if time_match else None
                
                # Try 3D position first (new format)
                pos3d_match = re.search(pos3d_pattern, line)
                if pos3d_match:
                    x = float(pos3d_match.group(1))
                    y = float(pos3d_match.group(2))
                    z = float(pos3d_match.group(3))
                    self.add_position(x, y, z, timestamp)
                    current_pos = (x, y, z)
                else:
                    # Fall back to 2D position + altitude (old format)
                    pos2d_match = re.search(pos2d_pattern, line)
                    alt_match = re.search(alt_pattern, line)
                    if pos2d_match:
                        x = float(pos2d_match.group(1))
                        y = float(pos2d_match.group(2))
                        z = float(alt_match.group(1)) if alt_match else 0.0
                        self.add_position(x, y, z, timestamp)
                        current_pos = (x, y, z)
                    
                # Extract goal
                goal_match = re.search(goal_pattern, line)
                if goal_match and self.goal is None:
                    self.set_goal(
                        float(goal_match.group(1)),
                        float(goal_match.group(2)),
                        float(goal_match.group(3))
                    )
                
                # Extract obstacle encounters
                if current_pos and re.search(obstacle_pattern, line, re.IGNORECASE):
                    severity = 'emergency' if 'emergency' in line.lower() else 'normal'
                    self.add_obstacle_encounter(
                        current_pos[0], current_pos[1], current_pos[2],
                        timestamp, severity
                    )
                
                # Extract avoidance events
                if current_pos and re.search(avoidance_pattern, line, re.IGNORECASE):
                    dir_match = re.search(direction_pattern, line, re.IGNORECASE)
                    direction = 'unknown'
                    if dir_match:
                        direction = dir_match.group(1) or dir_match.group(2) or dir_match.group(3) or 'unknown'
                    self.add_avoidance_event(
                        current_pos[0], current_pos[1], current_pos[2],
                        timestamp, direction
                    )
                    
        # Set start from first position
        if self.positions and self.start is None:
            p = self.positions[0]
            self.set_start(p['x'], p['y'], p.get('z', 0.0))
            
        print(f"Parsed {len(self.positions)} 3D positions from log")
        print(f"Found {len(self.obstacle_encounters)} obstacle encounters, {len(self.avoidance_events)} avoidance events")
        
    def plot(self, save_path=None, show=True, plot_3d=True):
        """Generate path planning visualization with optional 3D view."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return
        
        if plot_3d and self.positions and any(p.get('z', 0) != 0 for p in self.positions):
            return self._plot_3d(save_path, show)
        else:
            return self._plot_2d(save_path, show)
    
    def _plot_2d(self, save_path=None, show=True):
        """Generate 2D path planning visualization (top-down view)."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Set world bounds (use first 4 values)
        bounds = self.world_bounds
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Path Planning - Top View (2D)')
        
        # Plot obstacles
        for obs in self.obstacles:
            circle = Circle((obs['x'], obs['y']), obs['radius'], 
                           color='red', alpha=0.5, label='Obstacle')
            ax.add_patch(circle)
            
        # Plot trajectory
        if self.positions:
            xs = [p['x'] for p in self.positions]
            ys = [p['y'] for p in self.positions]
            ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.scatter(xs, ys, c=range(len(xs)), cmap='Blues', s=20, zorder=5)
        
        # Plot avoidance events (orange circles, semi-transparent)
        if self.avoidance_events:
            avd_xs = [e['x'] for e in self.avoidance_events]
            avd_ys = [e['y'] for e in self.avoidance_events]
            ax.scatter(avd_xs, avd_ys, c='orange', s=80, marker='o', 
                      alpha=0.5, zorder=15, label='Avoidance')
            
        # Plot start
        if self.start:
            ax.scatter(self.start[0], self.start[1], c='green', s=200, 
                      marker='o', zorder=10, label='Start')
            ax.annotate('START', (self.start[0], self.start[1]), 
                       textcoords="offset points", xytext=(10, 10),
                       fontsize=12, color='green', fontweight='bold')
            
        # Plot goal
        if self.goal:
            ax.scatter(self.goal[0], self.goal[1], c='gold', s=200, 
                      marker='*', zorder=10, label='Goal')
            ax.annotate('GOAL', (self.goal[0], self.goal[1]), 
                       textcoords="offset points", xytext=(10, 10),
                       fontsize=12, color='goldenrod', fontweight='bold')
            
        ax.legend(loc='upper right', fontsize=8, labelspacing=0.8, handletextpad=0.5, markerscale=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 2D path visualization to {save_path}")
            
        if show and MATPLOTLIB_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def _plot_3d(self, save_path=None, show=True):
        """Generate 3D path planning visualization."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create layout: 3D flight path and Top-down view side by side at top,
        # Altitude profile at the bottom (slightly narrower, centered)
        gs = gridspec.GridSpec(2, 6, height_ratios=[2, 1], hspace=0.3, wspace=0.3,
                              left=0.06, right=0.94, top=0.94, bottom=0.08)
        
        ax3d = fig.add_subplot(gs[0, 0:3], projection='3d')  # Left half of top row
        ax_top = fig.add_subplot(gs[0, 3:6])  # Right half of top row
        ax_alt = fig.add_subplot(gs[1, 1:5])  # Center 4 columns of bottom row (narrower)
        
        # Extract data
        if self.positions:
            xs = [p['x'] for p in self.positions]
            ys = [p['y'] for p in self.positions]
            zs = [p.get('z', 0) for p in self.positions]
            times = list(range(len(xs)))
        else:
            xs, ys, zs, times = [], [], [], []
        
        # --- 3D Plot ---
        bounds = self.world_bounds
        ax3d.set_xlim(bounds[0], bounds[1])
        ax3d.set_ylim(bounds[2], bounds[3])
        z_min = bounds[4] if len(bounds) > 4 else 0
        z_max = bounds[5] if len(bounds) > 5 else 3
        ax3d.set_zlim(z_min, z_max)
        
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z / Altitude (m)')
        ax3d.set_title('3D Flight Path')
        
        # Plot 3D trajectory
        if xs:
            # Color by time
            ax3d.plot(xs, ys, zs, 'b-', linewidth=1.5, alpha=0.7, label='Flight Path')
            ax3d.scatter(xs, ys, zs, c=times, cmap='viridis', s=15, alpha=0.8)
            
            # Project trajectory onto ground plane
            ax3d.plot(xs, ys, [z_min]*len(xs), 'k--', linewidth=0.5, alpha=0.3)
        
        # Plot start
        if self.start:
            ax3d.scatter([self.start[0]], [self.start[1]], [self.start[2]], 
                        c='green', s=200, marker='o', label='Start', zorder=10)
            
        # Plot goal
        if self.goal:
            ax3d.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], 
                        c='gold', s=300, marker='*', label='Goal', zorder=10)
            
        ax3d.legend(loc='upper left', fontsize=8, labelspacing=0.8, handletextpad=0.5, markerscale=0.6)
        
        # --- Top-down view ---
        ax_top.set_xlim(bounds[0], bounds[1])
        ax_top.set_ylim(bounds[2], bounds[3])
        ax_top.set_aspect('equal')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xlabel('X (m)')
        ax_top.set_ylabel('Y (m)')
        ax_top.set_title('Top-Down View')
        
        if xs:
            ax_top.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
            scatter = ax_top.scatter(xs, ys, c=zs, cmap='viridis', s=20, zorder=5)
            plt.colorbar(scatter, ax=ax_top, label='Altitude (m)')
            
        if self.start:
            ax_top.scatter(self.start[0], self.start[1], c='green', s=150, 
                          marker='o', zorder=10, label='Start')
        if self.goal:
            ax_top.scatter(self.goal[0], self.goal[1], c='gold', s=200, 
                          marker='*', zorder=10, label='Goal')
        
        # Plot avoidance events on top-down (orange circles, semi-transparent)
        if self.avoidance_events:
            avd_xs = [e['x'] for e in self.avoidance_events]
            avd_ys = [e['y'] for e in self.avoidance_events]
            ax_top.scatter(avd_xs, avd_ys, c='orange', s=60, marker='o', 
                          alpha=0.5, zorder=15, label='Avoidance')
        
        ax_top.legend(loc='upper right', fontsize=8, labelspacing=0.8, handletextpad=0.5, markerscale=0.6)
        
        # --- Altitude Profile (bottom, full width) ---
        if zs:
            ax_alt.plot(times, zs, 'b-', linewidth=2, label='Altitude')
            ax_alt.fill_between(times, 0, zs, alpha=0.3)
            ax_alt.axhline(y=self.target_altitude, color='g', linestyle='--', alpha=0.5, label='Target Alt')
            
            ax_alt.set_xlabel('Time Step')
            ax_alt.set_ylabel('Altitude (m)')
            ax_alt.set_title('Altitude Profile')
            ax_alt.grid(True, alpha=0.3)
            ax_alt.legend(loc='upper right', fontsize=8, labelspacing=0.8, handletextpad=0.5)
            ax_alt.set_ylim(0, max(zs) * 1.2 if zs else 2)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D path visualization to {save_path}")
            
        if show and MATPLOTLIB_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_2d(self, save_path=None, show=True):
        """Generate 2D path planning visualization (legacy method)."""
        return self._plot_2d(save_path, show)


class ObstacleDetectionVisualizer:
    """Visualizer for obstacle detection and depth analysis."""
    
    def __init__(self, thresholds=None):
        """Initialize obstacle detection visualizer."""
        self.depth_frames = []
        self.clearance_history = []
        self.state_history = []
        self.thresholds = thresholds or {
            'critical': 0.15,
            'close': 0.30,
            'caution': 0.45,
            'clear': 0.55
        }
        
    def add_depth_frame(self, depth_map, analysis=None, timestamp=None):
        """Add a depth frame with optional analysis."""
        self.depth_frames.append({
            'depth': depth_map,
            'analysis': analysis,
            'time': timestamp
        })
        
    def add_clearance_reading(self, left, center, right, state='normal', timestamp=None):
        """Add clearance values from log."""
        self.clearance_history.append({
            'left': left,
            'center': center,
            'right': right,
            'state': state,
            'time': timestamp
        })
        
    def parse_log_file(self, log_path):
        """Parse log file for depth and avoidance data."""
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            return
            
        # Patterns
        depth_pattern = r'\[DEPTH\] L:(\d+\.\d+) C:(\d+\.\d+) R:(\d+\.\d+)'
        state_pattern = r'\[AVOIDANCE\].*State: (\w+)'
        time_pattern = r'\[(\d{2}:\d{2}:\d{2})\]'
        
        current_state = 'normal'
        
        with open(log_path, 'r') as f:
            for line in f:
                time_match = re.search(time_pattern, line)
                timestamp = time_match.group(1) if time_match else None
                
                # Check for state change
                state_match = re.search(state_pattern, line, re.IGNORECASE)
                if state_match:
                    current_state = state_match.group(1).lower()
                    self.state_history.append({
                        'state': current_state,
                        'time': timestamp
                    })
                    
                # Check for depth readings
                depth_match = re.search(depth_pattern, line)
                if depth_match:
                    self.add_clearance_reading(
                        float(depth_match.group(1)),
                        float(depth_match.group(2)),
                        float(depth_match.group(3)),
                        current_state,
                        timestamp
                    )
                    
        print(f"Parsed {len(self.clearance_history)} depth readings, {len(self.state_history)} state changes")
        
    def plot_clearance_timeline(self, save_path=None, show=True):
        """Plot clearance values over time."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return
            
        if not self.clearance_history:
            print("No clearance data to plot")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Extract data
        times = list(range(len(self.clearance_history)))
        lefts = [c['left'] for c in self.clearance_history]
        centers = [c['center'] for c in self.clearance_history]
        rights = [c['right'] for c in self.clearance_history]
        states = [c['state'] for c in self.clearance_history]
        
        # Plot clearances
        ax1 = axes[0]
        ax1.plot(times, lefts, 'b-', label='Left', linewidth=1.5)
        ax1.plot(times, centers, 'g-', label='Center', linewidth=2)
        ax1.plot(times, rights, 'r-', label='Right', linewidth=1.5)
        
        # Add threshold lines
        ax1.axhline(y=self.thresholds['critical'], color='red', linestyle='--', alpha=0.5, label='Emergency')
        ax1.axhline(y=self.thresholds['close'], color='orange', linestyle='--', alpha=0.5, label='Close')
        ax1.axhline(y=self.thresholds['caution'], color='yellow', linestyle='--', alpha=0.5, label='Caution')
        ax1.axhline(y=self.thresholds['clear'], color='green', linestyle='--', alpha=0.5, label='Clear')
        
        ax1.set_ylabel('Clearance (0=close, 1=far)')
        ax1.set_title('Zone Clearance Over Time')
        ax1.legend(loc='upper right', ncol=4)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot states as colored regions
        ax2 = axes[1]
        state_colors = {
            'normal': 'green',
            'caution': 'yellow', 
            'avoidance': 'orange',
            'emergency': 'red',
            'recovery': 'blue'
        }
        
        # Create state timeline
        for i, state in enumerate(states):
            color = state_colors.get(state, 'gray')
            ax2.axvspan(i, i+1, alpha=0.5, color=color)
            
        ax2.set_ylabel('State')
        ax2.set_xlabel('Time Step')
        ax2.set_title('Avoidance State Timeline')
        
        # Add legend for states
        handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.5) 
                   for c in state_colors.values()]
        ax2.legend(handles, state_colors.keys(), loc='upper right', ncol=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved clearance timeline to {save_path}")
            
        if show and MATPLOTLIB_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        return fig
        
    def plot_depth_frame(self, depth_map, analysis=None, save_path=None, show=True):
        """Visualize a single depth frame with zone analysis."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return
            
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])
        
        # Depth map
        ax1 = fig.add_subplot(gs[0])
        im = ax1.imshow(depth_map, cmap='viridis')
        ax1.set_title('Depth Map (brighter = closer)')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # Draw ROI bounds if available
        if analysis and 'roi_bounds' in analysis:
            roi_top, roi_bottom = analysis['roi_bounds']
            h, w = depth_map.shape
            ax1.axhline(y=roi_top, color='red', linestyle='--', alpha=0.7)
            ax1.axhline(y=roi_bottom, color='red', linestyle='--', alpha=0.7)
            ax1.axvline(x=w//3, color='white', linestyle=':', alpha=0.5)
            ax1.axvline(x=2*w//3, color='white', linestyle=':', alpha=0.5)
            
        # Zone clearances bar chart
        if analysis:
            ax2 = fig.add_subplot(gs[1])
            zones = ['Left', 'Center', 'Right']
            clearances = [analysis.get('left', 0), 
                         analysis.get('center', 0), 
                         analysis.get('right', 0)]
            colors = []
            for c in clearances:
                if c < self.thresholds['critical']:
                    colors.append('red')
                elif c < self.thresholds['close']:
                    colors.append('orange')
                elif c < self.thresholds['caution']:
                    colors.append('yellow')
                else:
                    colors.append('green')
                    
            bars = ax2.bar(zones, clearances, color=colors, edgecolor='black')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Clearance')
            ax2.set_title('Zone Clearances')
            ax2.axhline(y=self.thresholds['critical'], color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.thresholds['close'], color='orange', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.thresholds['caution'], color='yellow', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, val in zip(bars, clearances):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
                        
            # Safe direction indicator
            ax3 = fig.add_subplot(gs[2])
            safe_dir = analysis.get('safe_direction', 0)
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
            ax3.set_aspect('equal')
            
            # Draw drone representation
            drone = Circle((0, 0), 0.3, color='blue', alpha=0.7)
            ax3.add_patch(drone)
            
            # Draw direction arrow
            if safe_dir == -1:
                arrow_end = (-1, 0)
                arrow_label = 'LEFT'
            elif safe_dir == 1:
                arrow_end = (1, 0)
                arrow_label = 'RIGHT'
            else:
                arrow_end = (0, 1)
                arrow_label = 'FORWARD'
                
            ax3.annotate('', xy=arrow_end, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3))
            ax3.text(0, -1.2, f'Safe Direction: {arrow_label}', 
                    ha='center', fontsize=12, fontweight='bold')
            ax3.set_title('Recommended Direction')
            ax3.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved depth visualization to {save_path}")
            
        if show and MATPLOTLIB_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        return fig


def create_combined_visualization(log_path, output_dir=None):
    """Create combined visualization from log file."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
        
    log_path = Path(log_path)
    if output_dir is None:
        output_dir = Path('data/visualization')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Default values
    world_bounds = (-10, 10, -5, 5, 0, 3)
    thresholds = None
    target_altitude = 1.0
    
    if config:
        # Update target altitude from drone config
        if 'drone' in config and 'navigation' in config['drone']:
            target_altitude = config['drone']['navigation'].get('default_altitude', 1.0)
        
        # Update thresholds from vision config
        if 'vision' in config and 'obstacle' in config['vision']:
            obs_config = config['vision']['obstacle']
            thresholds = {
                'critical': obs_config.get('critical_threshold', 0.15),
                'close': obs_config.get('close_threshold', 0.25),
                'caution': obs_config.get('caution_threshold', 0.40),
                'clear': obs_config.get('far_threshold', 0.55)
            }
    
    # Create visualizers
    path_viz = PathPlanningVisualizer(world_bounds=world_bounds, target_altitude=target_altitude)
    obstacle_viz = ObstacleDetectionVisualizer(thresholds=thresholds)
    
    # Parse log
    path_viz.parse_log_file(log_path)
    obstacle_viz.parse_log_file(log_path)
    
    # Generate base filename
    base_name = log_path.stem
    
    # Generate visualizations
    path_viz.plot(
        save_path=output_dir / f'{base_name}_path.png',
        show=False
    )
    
    if obstacle_viz.clearance_history:
        obstacle_viz.plot_clearance_timeline(
            save_path=output_dir / f'{base_name}_clearance.png',
            show=False
        )
        
    print(f"\nVisualizations saved to {output_dir}/")
    return path_viz, obstacle_viz


def main():
    parser = argparse.ArgumentParser(description='Navigation Visualization Tool')
    parser.add_argument('--log', type=str, help='Path to log file to analyze')
    parser.add_argument('--output', type=str, default='data/visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--latest', action='store_true', 
                       help='Analyze the latest log file')
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)
        
    # Find log file
    if args.latest:
        log_dir = Path('sim/webots/logs')
        log_files = sorted(log_dir.glob('*.log'), key=lambda p: p.stat().st_mtime)
        if log_files:
            args.log = str(log_files[-1])
            print(f"Using latest log: {args.log}")
        else:
            print("No log files found in sim/webots/logs/")
            sys.exit(1)
            
    if not args.log:
        print("Please specify a log file with --log or use --latest")
        parser.print_help()
        sys.exit(1)
        
    # Create visualizations
    path_viz, obstacle_viz = create_combined_visualization(
        args.log, 
        output_dir=args.output
    )
    
    if args.show:
        # Show interactive plots
        path_viz.plot(show=True)
        if obstacle_viz.clearance_history:
            obstacle_viz.plot_clearance_timeline(show=True)


if __name__ == '__main__':
    main()
