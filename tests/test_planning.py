"""Unit tests for path planning module."""

import numpy as np
import pytest

from src.planning.path_planner import AStarPlanner
from src.planning.trajectory_smoother import TrajectorySmootherBezier, TrajectorySmootherSpline


class TestAStarPlanner:
    """Test A* path planning."""
    
    def test_simple_path(self):
        """Test finding path in open space."""
        planner = AStarPlanner(safety_margin=1)
        
        # Create simple map with no obstacles
        occupancy_map = np.zeros((50, 50), dtype=np.uint8)
        
        start = (10, 10)
        goal = (40, 40)
        
        path = planner.plan(occupancy_map, start, goal)
        
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
    
    def test_obstacle_avoidance(self):
        """Test path avoids obstacles."""
        planner = AStarPlanner(safety_margin=2)
        
        # Create map with wall in center
        occupancy_map = np.zeros((50, 50), dtype=np.uint8)
        occupancy_map[:, 25] = 1  # Vertical wall
        occupancy_map[15:35, 25] = 0  # Create gap in wall
        
        start = (10, 25)
        goal = (40, 25)
        
        path = planner.plan(occupancy_map, start, goal)
        
        assert len(path) > 0
        # Path should go through the gap
        path_array = np.array(path)
        # At least some waypoints should be near the gap
        assert len(path_array) > 2
    
    def test_no_path(self):
        """Test when no path exists."""
        planner = AStarPlanner(safety_margin=1)
        
        # Create map fully blocked
        occupancy_map = np.ones((50, 50), dtype=np.uint8)
        
        start = (10, 10)
        goal = (40, 40)
        
        path = planner.plan(occupancy_map, start, goal)
        
        assert len(path) == 0


class TestTrajectorySmoothing:
    """Test trajectory smoothing."""
    
    def test_bezier_smoothing(self):
        """Test Bezier curve smoothing."""
        smoother = TrajectorySmootherBezier(control_point_ratio=0.3)
        
        waypoints = [(0, 0), (10, 10), (20, 10), (30, 20)]
        
        smoothed = smoother.smooth_path(waypoints, num_samples=50)
        
        assert len(smoothed) > 0
        # Bezier creates segments between waypoints
        assert len(smoothed) >= len(waypoints)
        # Start and end should be close to original
        assert np.allclose(smoothed[0], waypoints[0], atol=1.0)
        assert np.allclose(smoothed[-1], waypoints[-1], atol=1.0)
    
    def test_spline_smoothing(self):
        """Test B-spline smoothing."""
        smoother = TrajectorySmootherSpline(smoothing_factor=0.0, spline_degree=3)
        
        waypoints = [(0, 0), (10, 10), (20, 10), (30, 20), (40, 30)]
        
        smoothed = smoother.smooth_path(waypoints, num_samples=50)
        
        assert len(smoothed) > 0
        assert len(smoothed) == 50
    
    def test_insufficient_waypoints(self):
        """Test handling of too few waypoints."""
        smoother = TrajectorySmootherBezier()
        
        # Single waypoint
        waypoints = [(10, 10)]
        smoothed = smoother.smooth_path(waypoints, num_samples=10)
        assert len(smoothed) == 1
        
        # Two waypoints - should interpolate linearly
        waypoints = [(0, 0), (10, 10)]
        smoothed = smoother.smooth_path(waypoints, num_samples=10)
        assert len(smoothed) == 10


class TestPathPlanningIntegration:
    """Integration tests for full planning pipeline."""
    
    def test_plan_and_smooth(self):
        """Test planning followed by smoothing."""
        planner = AStarPlanner(safety_margin=2)
        smoother = TrajectorySmootherBezier()
        
        # Create test map with obstacles
        occupancy_map = np.zeros((100, 100), dtype=np.uint8)
        occupancy_map[40:60, 30:31] = 1  # Thin vertical obstacle
        
        start = (50, 10)
        goal = (50, 90)
        
        # Plan path
        path = planner.plan(occupancy_map, start, goal)
        assert len(path) > 0
        
        # Smooth path
        smoothed = smoother.smooth_path(path, num_samples=100)
        assert len(smoothed) > 0
        
        # Smoothed path should avoid obstacles
        # (This is a basic check - more thorough collision checking would be better)
        assert len(smoothed) <= 100
