"""Trajectory smoothing and interpolation for path following."""

import numpy as np
from typing import List, Tuple, Optional
from loguru import logger
from scipy.interpolate import splprep, splev, CubicSpline


class TrajectorySmootherBezier:
    """
    Smooth trajectories using Bezier curves.
    
    Bezier curves provide smooth, controllable interpolation between waypoints
    with guaranteed smoothness at junction points.
    """
    
    def __init__(self, control_point_ratio: float = 0.3):
        """
        Initialize Bezier smoother.
        
        Args:
            control_point_ratio: Ratio for control point placement (0-1)
        """
        self.control_point_ratio = control_point_ratio
    
    def smooth_path(self,
                   waypoints: List[Tuple[int, int]],
                   num_samples: int = 50) -> np.ndarray:
        """
        Smooth path using Bezier curves.
        
        Args:
            waypoints: List of (x, y) waypoints
            num_samples: Number of samples along smoothed path
            
        Returns:
            Smoothed path as (N, 2) array
        """
        if len(waypoints) < 2:
            return np.array(waypoints)
        
        if len(waypoints) == 2:
            # Linear interpolation for two points
            return self._linear_interpolate(waypoints[0], waypoints[1], num_samples)
        
        # Convert to numpy array
        waypoints = np.array(waypoints, dtype=np.float32)
        
        # Generate Bezier curve segments between waypoints
        smoothed_segments = []
        
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p3 = waypoints[i + 1]
            
            # Calculate control points
            direction = p3 - p0
            distance = np.linalg.norm(direction)
            
            if i > 0:
                # Use previous direction for smoother transitions
                prev_dir = waypoints[i] - waypoints[i - 1]
                prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-6)
            else:
                prev_dir = direction / (distance + 1e-6)
            
            if i < len(waypoints) - 2:
                # Use next direction for smoother transitions
                next_dir = waypoints[i + 2] - waypoints[i + 1]
                next_dir = next_dir / (np.linalg.norm(next_dir) + 1e-6)
            else:
                next_dir = direction / (distance + 1e-6)
            
            # Place control points
            p1 = p0 + prev_dir * distance * self.control_point_ratio
            p2 = p3 - next_dir * distance * self.control_point_ratio
            
            # Generate Bezier curve samples
            segment = self._cubic_bezier(p0, p1, p2, p3, num_samples // (len(waypoints) - 1))
            smoothed_segments.append(segment)
        
        # Concatenate all segments
        smoothed_path = np.vstack(smoothed_segments)
        
        return smoothed_path
    
    def _cubic_bezier(self,
                     p0: np.ndarray,
                     p1: np.ndarray,
                     p2: np.ndarray,
                     p3: np.ndarray,
                     num_samples: int) -> np.ndarray:
        """
        Generate cubic Bezier curve.
        
        B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        
        Args:
            p0, p1, p2, p3: Control points
            num_samples: Number of samples
            
        Returns:
            Curve points as (N, 2) array
        """
        t = np.linspace(0, 1, num_samples)
        
        curve = (
            (1 - t)[:, None]**3 * p0 +
            3 * (1 - t)[:, None]**2 * t[:, None] * p1 +
            3 * (1 - t)[:, None] * t[:, None]**2 * p2 +
            t[:, None]**3 * p3
        )
        
        return curve
    
    def _linear_interpolate(self,
                          p0: Tuple[int, int],
                          p1: Tuple[int, int],
                          num_samples: int) -> np.ndarray:
        """Linear interpolation between two points."""
        t = np.linspace(0, 1, num_samples)
        p0 = np.array(p0, dtype=np.float32)
        p1 = np.array(p1, dtype=np.float32)
        
        return p0 + t[:, None] * (p1 - p0)


class TrajectorySmootherSpline:
    """
    Smooth trajectories using B-splines.
    
    B-splines provide parametric curve fitting with automatic smoothness
    and good numerical properties.
    """
    
    def __init__(self,
                 smoothing_factor: float = 0.0,
                 spline_degree: int = 3):
        """
        Initialize spline smoother.
        
        Args:
            smoothing_factor: Smoothing factor (0=interpolate exactly, >0=smooth)
            spline_degree: Degree of spline (1=linear, 2=quadratic, 3=cubic)
        """
        self.smoothing_factor = smoothing_factor
        self.spline_degree = min(spline_degree, 5)  # Max degree is 5
    
    def smooth_path(self,
                   waypoints: List[Tuple[int, int]],
                   num_samples: int = 50) -> np.ndarray:
        """
        Smooth path using B-spline interpolation.
        
        Args:
            waypoints: List of (x, y) waypoints
            num_samples: Number of samples along smoothed path
            
        Returns:
            Smoothed path as (N, 2) array
        """
        if len(waypoints) < 2:
            return np.array(waypoints)
        
        # Need at least k+1 points for degree k spline
        if len(waypoints) <= self.spline_degree:
            logger.warning(f"Not enough waypoints for degree {self.spline_degree} spline, "
                         f"using linear interpolation")
            return self._linear_interpolate(waypoints, num_samples)
        
        # Convert to numpy array and transpose for splprep
        waypoints = np.array(waypoints, dtype=np.float32).T
        
        try:
            # Fit B-spline
            tck, u = splprep(waypoints, s=self.smoothing_factor, k=self.spline_degree)
            
            # Evaluate spline at uniform intervals
            u_new = np.linspace(0, 1, num_samples)
            smoothed = splev(u_new, tck)
            
            # Transpose back to (N, 2) format
            smoothed_path = np.column_stack(smoothed)
            
            return smoothed_path
            
        except Exception as e:
            logger.error(f"Spline smoothing failed: {e}, using linear interpolation")
            return self._linear_interpolate(waypoints.T, num_samples)
    
    def _linear_interpolate(self,
                          waypoints: np.ndarray,
                          num_samples: int) -> np.ndarray:
        """Linear interpolation through waypoints."""
        if len(waypoints) < 2:
            return waypoints
        
        # Calculate cumulative distance along path
        dists = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(dists)])
        total_dist = cumulative_dist[-1]
        
        # Sample uniformly along path
        sample_dists = np.linspace(0, total_dist, num_samples)
        
        # Interpolate x and y separately
        x_interp = np.interp(sample_dists, cumulative_dist, waypoints[:, 0])
        y_interp = np.interp(sample_dists, cumulative_dist, waypoints[:, 1])
        
        return np.column_stack([x_interp, y_interp])


class VelocityProfileGenerator:
    """
    Generate smooth velocity profiles for path following.
    
    Creates trapezoidal or S-curve velocity profiles with smooth
    acceleration and deceleration phases.
    """
    
    def __init__(self,
                 max_velocity: float = 0.5,
                 max_acceleration: float = 0.3,
                 profile_type: str = 's-curve'):
        """
        Initialize velocity profile generator.
        
        Args:
            max_velocity: Maximum velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            profile_type: 'trapezoidal' or 's-curve'
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.profile_type = profile_type
    
    def generate_profile(self,
                        path_length: float,
                        time_step: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate velocity profile for given path length.
        
        Args:
            path_length: Total length of path (meters)
            time_step: Time step for profile (seconds)
            
        Returns:
            Tuple of (time_array, velocity_array)
        """
        if self.profile_type == 's-curve':
            return self._generate_s_curve(path_length, time_step)
        else:
            return self._generate_trapezoidal(path_length, time_step)
    
    def _generate_trapezoidal(self,
                             path_length: float,
                             time_step: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trapezoidal velocity profile."""
        # Calculate acceleration and deceleration times
        t_accel = self.max_velocity / self.max_acceleration
        
        # Distance covered during acceleration/deceleration
        d_accel = 0.5 * self.max_acceleration * t_accel**2
        
        # Check if we reach max velocity
        if 2 * d_accel < path_length:
            # Three phases: accel, cruise, decel
            d_cruise = path_length - 2 * d_accel
            t_cruise = d_cruise / self.max_velocity
            total_time = 2 * t_accel + t_cruise
        else:
            # Two phases: accel, decel (no cruise)
            t_accel = np.sqrt(path_length / self.max_acceleration)
            t_cruise = 0
            total_time = 2 * t_accel
        
        # Generate time array
        time = np.arange(0, total_time, time_step)
        velocity = np.zeros_like(time)
        
        for i, t in enumerate(time):
            if t < t_accel:
                # Acceleration phase
                velocity[i] = self.max_acceleration * t
            elif t < t_accel + t_cruise:
                # Cruise phase
                velocity[i] = self.max_velocity
            else:
                # Deceleration phase
                t_decel = t - t_accel - t_cruise
                velocity[i] = self.max_velocity - self.max_acceleration * t_decel
        
        return time, velocity
    
    def _generate_s_curve(self,
                         path_length: float,
                         time_step: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate S-curve velocity profile with smooth jerk."""
        # Simplified S-curve using sigmoid functions
        # This provides smoother acceleration changes than trapezoidal
        
        t_total = 2 * path_length / self.max_velocity  # Approximate total time
        time = np.arange(0, t_total, time_step)
        
        # Smooth acceleration using sigmoid
        t_mid = t_total / 2
        steepness = 8 / t_total
        
        velocity = self.max_velocity / (1 + np.exp(-steepness * (time - t_total/4)))
        velocity -= self.max_velocity / (1 + np.exp(-steepness * (time - 3*t_total/4)))
        
        return time, velocity
