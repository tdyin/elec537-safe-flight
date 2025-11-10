"""Unit tests for drone module."""

import pytest
import numpy as np

from drone import NavigationController


class TestNavigationController:
    """Test NavigationController class."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = NavigationController(
            max_speed=0.5,
            safety_distance=0.5,
            avoidance_gain=1.0
        )
        assert controller.max_speed == 0.5
        assert controller.safety_distance == 0.5
    
    def test_limit_velocity(self):
        """Test velocity limiting."""
        controller = NavigationController(max_speed=0.5)
        high_velocity = np.array([1.0, 1.0, 0.0])
        limited = controller._limit_velocity(high_velocity)
        speed = np.linalg.norm(limited)
        assert speed <= controller.max_speed + 1e-6
    
    def test_is_path_clear_no_obstacles(self):
        """Test path clear check with no obstacles."""
        controller = NavigationController()
        current = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0, 0.0])
        assert controller.is_path_clear([], current, target)
    
    def test_is_path_clear_with_obstacle(self):
        """Test path clear check with obstacle in path."""
        controller = NavigationController(safety_distance=0.5)
        current = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0, 0.0])
        obstacles = [
            {'centroid': np.array([2.5, 0.0, 0.0])}  # Right in the path
        ]
        assert not controller.is_path_clear(obstacles, current, target)
    
    def test_compute_safe_velocity_no_obstacles(self):
        """Test safe velocity computation without obstacles."""
        controller = NavigationController()
        current = np.array([0.0, 0.0, 0.0])
        target_vel = np.array([0.3, 0.0, 0.0])
        safe_vel = controller.compute_safe_velocity([], current, target_vel)
        np.testing.assert_array_almost_equal(safe_vel, target_vel)
    
    def test_compute_safe_velocity_with_obstacle(self):
        """Test safe velocity with nearby obstacle."""
        controller = NavigationController(safety_distance=1.0)
        current = np.array([0.0, 0.0, 0.0])
        target_vel = np.array([0.3, 0.0, 0.0])
        obstacles = [
            {'centroid': np.array([0.5, 0.0, 0.0])}  # Very close
        ]
        safe_vel = controller.compute_safe_velocity(obstacles, current, target_vel)
        # Safe velocity should be different from target due to avoidance
        assert not np.allclose(safe_vel, target_vel)
    
    def test_plan_avoidance_maneuver(self):
        """Test avoidance maneuver planning."""
        controller = NavigationController()
        current = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0, 0.0])
        obstacles = [
            {'centroid': np.array([2.5, 0.0, 0.0])}
        ]
        waypoints = controller.plan_avoidance_maneuver(obstacles, current, target)
        assert len(waypoints) >= 1
        # Last waypoint should be the target
        np.testing.assert_array_equal(waypoints[-1], target)
