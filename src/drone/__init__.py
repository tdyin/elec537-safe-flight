"""Drone interface module for Crazyflie communication.

This module re-exports interfaces for backward compatibility.
New code should import from:
- src.sim.webots_interface for simulation
- src.hardware.crazyflie_interface for hardware
- src.core.base_interface for the ABC
"""

from .interface import CrazyflieInterface
from .depth_controller import DepthNavigationController

# Re-export WebotsInterface from sim module for backward compatibility
from ..sim.webots_interface import WebotsInterface

__all__ = ["CrazyflieInterface", "DepthNavigationController", "WebotsInterface"]
