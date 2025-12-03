"""
Navigation state machine definitions.

Defines states for path following and navigation control.
"""

from enum import Enum


class PathState(Enum):
    """States for path following state machine."""
    IDLE = "idle"                   # Not navigating
    FOLLOWING = "following"         # Following planned path
    DEVIATED = "deviated"           # Off path, computing return
    RETURNING = "returning"         # Returning to path
    WAYPOINT_REACHED = "reached"    # Current waypoint reached
    PATH_COMPLETE = "complete"      # All waypoints reached


class NavigationMode(Enum):
    """Current navigation mode for the overall controller."""
    IDLE = "idle"               # Not navigating
    PATH_FOLLOWING = "path"     # Following preset path
    AVOIDING = "avoiding"       # Executing avoidance maneuver
    RETURNING = "returning"     # Returning to path after avoidance
    EMERGENCY_STOP = "emergency" # Emergency hover
    GOAL_REACHED = "goal"       # Destination reached


class AvoidancePhase(Enum):
    """Phase of avoidance maneuver."""
    NONE = "none"           # Not avoiding
    TURNING = "turning"     # Rotating to face safe direction
    MOVING = "moving"       # Moving away from obstacle
    SCANNING = "scanning"   # Checking if path is clear
    CLEARING = "clearing"   # Final clear movement
