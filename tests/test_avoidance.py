"""Tests for the stable avoidance controller."""

import pytest
import numpy as np
from src.planning.avoidance_controller import (
    StableAvoidanceController,
    AvoidanceConfig,
    AvoidanceState,
    PotentialFieldController,
    PurePursuitController
)


class TestAvoidanceConfig:
    """Test AvoidanceConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AvoidanceConfig()
        assert config.max_speed == 0.5
        assert config.safety_distance == 0.5
        assert config.emergency_distance == 0.25
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AvoidanceConfig(
            max_speed=1.0,
            safety_distance=1.0,
            repulsive_gain=2.0
        )
        assert config.max_speed == 1.0
        assert config.safety_distance == 1.0
        assert config.repulsive_gain == 2.0


class TestPotentialFieldController:
    """Test PotentialFieldController."""
    
    @pytest.fixture
    def controller(self):
        return PotentialFieldController(AvoidanceConfig())
    
    def test_attractive_force_toward_goal(self, controller):
        """Test attractive force points toward goal."""
        current = np.array([0.0, 0.0, 1.0])
        goal = np.array([5.0, 0.0, 1.0])
        
        force = controller.compute_attractive_force(current, goal)
        
        # Force should point toward goal (positive x)
        assert force[0] > 0
        assert abs(force[1]) < 1e-6
        assert abs(force[2]) < 1e-6
    
    def test_attractive_force_at_goal(self, controller):
        """Test attractive force is zero at goal."""
        pos = np.array([5.0, 0.0, 1.0])
        
        force = controller.compute_attractive_force(pos, pos)
        
        assert np.linalg.norm(force) < 1e-6
    
    def test_repulsive_force_away_from_obstacle(self, controller):
        """Test repulsive force points away from obstacle."""
        current = np.array([1.0, 0.0, 1.0])
        obstacles = [{'position': np.array([0.5, 0.0, 1.0])}]
        
        force = controller.compute_repulsive_force(current, obstacles)
        
        # Force should push away from obstacle (positive x)
        assert force[0] > 0
    
    def test_repulsive_force_zero_when_far(self, controller):
        """Test repulsive force is zero when far from obstacle."""
        current = np.array([0.0, 0.0, 1.0])
        obstacles = [{'position': np.array([10.0, 0.0, 1.0])}]
        
        force = controller.compute_repulsive_force(current, obstacles)
        
        assert np.linalg.norm(force) < 1e-6
    
    def test_multiple_obstacles(self, controller):
        """Test repulsive force with multiple obstacles."""
        current = np.array([0.0, 0.0, 1.0])
        obstacles = [
            {'position': np.array([0.5, 0.0, 1.0])},
            {'position': np.array([0.0, 0.5, 1.0])}
        ]
        
        force = controller.compute_repulsive_force(current, obstacles)
        
        # Force should point away from both obstacles (negative x and y)
        assert force[0] < 0
        assert force[1] < 0


class TestPurePursuitController:
    """Test PurePursuitController."""
    
    @pytest.fixture
    def controller(self):
        return PurePursuitController(AvoidanceConfig())
    
    def test_lookahead_distance_scales_with_speed(self, controller):
        """Test lookahead distance increases with speed."""
        slow_lookahead = controller.compute_lookahead_distance(0.1)
        fast_lookahead = controller.compute_lookahead_distance(0.5)
        
        assert fast_lookahead > slow_lookahead
    
    def test_lookahead_distance_bounded(self, controller):
        """Test lookahead distance is bounded."""
        min_look = controller.compute_lookahead_distance(0.0)
        max_look = controller.compute_lookahead_distance(10.0)
        
        assert min_look >= controller.config.min_lookahead
        assert max_look <= controller.config.max_lookahead
    
    def test_find_lookahead_point_on_path(self, controller):
        """Test finding lookahead point on path."""
        current = np.array([0.0, 0.0, 1.0])
        path = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0]
        ])
        
        point, index = controller.find_lookahead_point(current, path, 0, 0.4)
        
        assert index >= 0
        assert np.linalg.norm(point - current[:2]) >= 0.4 or index == len(path) - 1
    
    def test_steering_toward_target(self, controller):
        """Test steering computation toward target."""
        current = np.array([0.0, 0.0])
        heading = 0.0  # Facing positive x
        target = np.array([1.0, 0.5])  # Target to the left
        
        curvature = controller.compute_steering(current, heading, target)
        
        # Should steer left (positive curvature)
        assert curvature > 0


class TestStableAvoidanceController:
    """Test StableAvoidanceController."""
    
    @pytest.fixture
    def controller(self):
        return StableAvoidanceController()
    
    def test_initial_state(self, controller):
        """Test initial state is NORMAL."""
        assert controller.state == AvoidanceState.NORMAL
    
    def test_state_transition_to_emergency(self, controller):
        """Test transition to EMERGENCY state when too close."""
        controller.update_state(0.1, 0.02)  # Very close obstacle
        
        assert controller.state == AvoidanceState.EMERGENCY
    
    def test_state_transition_to_avoidance(self, controller):
        """Test transition to AVOIDANCE state."""
        controller.update_state(0.4, 0.02)  # Within safety distance
        
        assert controller.state == AvoidanceState.AVOIDANCE
    
    def test_state_transition_to_caution(self, controller):
        """Test transition to CAUTION state."""
        controller.update_state(0.8, 0.02)  # Within caution distance
        
        assert controller.state == AvoidanceState.CAUTION
    
    def test_state_stays_normal_when_clear(self, controller):
        """Test state stays NORMAL when path is clear."""
        controller.update_state(5.0, 0.02)  # Far from obstacles
        
        assert controller.state == AvoidanceState.NORMAL
    
    def test_velocity_scale_emergency(self, controller):
        """Test velocity scale is zero in emergency."""
        controller.state = AvoidanceState.EMERGENCY
        
        scale = controller.compute_velocity_scale(0.1)
        
        assert scale == 0.0
    
    def test_velocity_scale_normal(self, controller):
        """Test velocity scale is 1.0 when clear."""
        controller.state = AvoidanceState.NORMAL
        
        scale = controller.compute_velocity_scale(5.0)
        
        assert scale == 1.0
    
    def test_velocity_scale_avoidance(self, controller):
        """Test velocity scale is reduced during avoidance."""
        controller.state = AvoidanceState.AVOIDANCE
        
        scale = controller.compute_velocity_scale(0.4)
        
        assert 0 < scale < 1.0
    
    def test_compute_avoidance_velocity_no_obstacles(self, controller):
        """Test velocity computation with no obstacles."""
        current = np.array([0.0, 0.0, 1.0])
        goal = np.array([5.0, 0.0, 1.0])
        target_vel = np.array([0.3, 0.0, 0.0])
        
        velocity = controller.compute_avoidance_velocity(
            current_position=current,
            current_heading=0.0,
            target_position=goal,
            target_velocity=target_vel,
            obstacles=[],
            dt=0.02
        )
        
        # Should move toward goal
        assert velocity[0] > 0
        assert controller.state == AvoidanceState.NORMAL
    
    def test_compute_avoidance_velocity_with_obstacle(self, controller):
        """Test velocity computation with close obstacle."""
        current = np.array([0.0, 0.0, 1.0])
        goal = np.array([5.0, 0.0, 1.0])
        target_vel = np.array([0.3, 0.0, 0.0])
        obstacles = [{'position': np.array([0.3, 0.0, 1.0])}]
        
        velocity = controller.compute_avoidance_velocity(
            current_position=current,
            current_heading=0.0,
            target_position=goal,
            target_velocity=target_vel,
            obstacles=obstacles,
            dt=0.02
        )
        
        # Should be slower due to obstacle
        assert np.linalg.norm(velocity) < np.linalg.norm(target_vel)
    
    def test_velocity_smoothing(self, controller):
        """Test that velocities are smoothed over time."""
        current = np.array([0.0, 0.0, 1.0])
        goal = np.array([5.0, 0.0, 1.0])
        target_vel = np.array([0.3, 0.0, 0.0])
        
        # First call
        vel1 = controller.compute_avoidance_velocity(
            current, 0.0, goal, target_vel, [], 0.02
        )
        
        # Second call with different target
        target_vel2 = np.array([0.0, 0.3, 0.0])
        vel2 = controller.compute_avoidance_velocity(
            current, 0.0, goal, target_vel2, [], 0.02
        )
        
        # Velocity should be smoothed (not jump immediately)
        assert vel2[1] < 0.3  # Not fully toward new target yet
    
    def test_reset(self, controller):
        """Test controller reset."""
        controller.state = AvoidanceState.EMERGENCY
        controller.smooth_velocity = np.array([0.1, 0.2, 0.0])
        
        controller.reset()
        
        assert controller.state == AvoidanceState.NORMAL
        assert np.allclose(controller.smooth_velocity, np.zeros(3))
    
    def test_statistics(self, controller):
        """Test statistics gathering."""
        # Trigger some state changes
        controller.update_state(0.1, 0.02)  # Emergency
        controller.update_state(0.1, 0.02)
        
        stats = controller.get_statistics()
        
        assert 'state' in stats
        assert 'emergency_stops' in stats
        assert stats['emergency_stops'] >= 1
    
    def test_path_following(self, controller):
        """Test path following integration."""
        path = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        
        controller.set_path(path)
        
        assert controller.current_path is not None
        assert len(controller.current_path) == 3
        assert controller.path_index == 0


class TestIntegration:
    """Integration tests for complete avoidance scenarios."""
    
    def test_navigate_around_obstacle(self):
        """Test navigating around a single obstacle."""
        controller = StableAvoidanceController()
        
        current = np.array([0.0, 0.0, 1.0])
        goal = np.array([5.0, 0.0, 1.0])
        obstacle = {'position': np.array([2.0, 0.0, 1.0]), 'size': np.array([0.5, 0.5, 0.5])}
        
        # Simulate multiple steps
        for _ in range(50):
            target_vel = (goal - current) / np.linalg.norm(goal - current) * 0.3
            
            velocity = controller.compute_avoidance_velocity(
                current, 0.0, goal, target_vel, [obstacle], 0.02
            )
            
            # Update position
            current = current + velocity * 0.02
            
            # Check we don't collide
            dist_to_obstacle = np.linalg.norm(current - obstacle['position'])
            assert dist_to_obstacle > 0.2  # Should maintain some distance
    
    def test_emergency_recovery(self):
        """Test recovery from emergency state."""
        controller = StableAvoidanceController()
        
        # Force into emergency
        controller.update_state(0.1, 0.02)
        assert controller.state == AvoidanceState.EMERGENCY
        
        # Simulate time passing with clear path
        for _ in range(100):
            controller.update_state(2.0, 0.02)  # Clear path
        
        # Should recover to normal
        assert controller.state in [AvoidanceState.NORMAL, AvoidanceState.CAUTION, AvoidanceState.RECOVERY]
