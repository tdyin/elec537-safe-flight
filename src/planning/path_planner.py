"""Path planning algorithms for smooth obstacle avoidance."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Node for A* search."""
    priority: float
    position: Tuple[int, int] = field(compare=False)
    g_cost: float = field(compare=False, default=0.0)
    h_cost: float = field(compare=False, default=0.0)
    parent: Optional['Node'] = field(compare=False, default=None)


class PathPlanner:
    """Base class for path planning algorithms."""
    
    def __init__(self, planning_resolution: float = 0.1):
        """
        Initialize path planner.
        
        Args:
            planning_resolution: Resolution for planning grid (meters)
        """
        self.planning_resolution = planning_resolution
        
    def plan(self,
             occupancy_map: np.ndarray,
             start: Tuple[int, int],
             goal: Tuple[int, int],
             **kwargs) -> List[Tuple[int, int]]:
        """
        Plan a path from start to goal.
        
        Args:
            occupancy_map: Binary occupancy grid (0=free, 1=occupied)
            start: Start position (x, y) in grid coordinates
            goal: Goal position (x, y) in grid coordinates
            
        Returns:
            List of waypoints from start to goal
        """
        raise NotImplementedError("Subclasses must implement plan()")


class AStarPlanner(PathPlanner):
    """
    A* path planning with configurable heuristic and cost functions.
    
    This planner finds optimal paths through free space while maintaining
    safety margins from obstacles. Parameters can be configured via 
    config.yaml drone.path_planning section.
    """
    
    def __init__(self,
                 planning_resolution: float = 0.1,
                 safety_margin: int = 2,
                 diagonal_cost: float = 1.414,
                 straight_cost: float = 1.0,
                 smoothness_weight: float = 0.3,
                 config: dict = None):
        """
        Initialize A* planner.
        
        Args:
            planning_resolution: Resolution for planning grid (meters)
            safety_margin: Minimum cells to maintain from obstacles
            diagonal_cost: Cost for diagonal movement
            straight_cost: Cost for straight movement
            smoothness_weight: Weight for favoring smooth paths (0-1)
            config: Optional configuration dictionary from config.yaml
        """
        # Load config values if provided
        path_config = {}
        if config:
            path_config = config.get('drone', {}).get('path_planning', {})
        
        super().__init__(path_config.get('planning_resolution', planning_resolution))
        self.safety_margin = path_config.get('safety_margin', safety_margin)
        self.diagonal_cost = path_config.get('diagonal_cost', diagonal_cost)
        self.straight_cost = path_config.get('straight_cost', straight_cost)
        self.smoothness_weight = path_config.get('smoothness_weight', smoothness_weight)
        
    def plan(self,
             occupancy_map: np.ndarray,
             start: Tuple[int, int],
             goal: Tuple[int, int],
             depth_map: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Plan path using A* algorithm with depth-aware costs.
        
        Args:
            occupancy_map: Binary occupancy grid (0=free, 1=occupied)
            start: Start position (x, y) in grid coordinates
            goal: Goal position (x, y) in grid coordinates
            depth_map: Optional depth map for distance-aware planning
            
        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        h, w = occupancy_map.shape
        
        # Validate start and goal
        if not self._is_valid_position(start, occupancy_map):
            logger.warning(f"Start position {start} is invalid")
            return []
        
        if not self._is_valid_position(goal, occupancy_map):
            logger.warning(f"Goal position {goal} is invalid, finding nearest free space")
            goal = self._find_nearest_free_position(goal, occupancy_map)
            if goal is None:
                logger.error("Could not find valid goal position")
                return []
        
        # Inflate obstacles for safety margin
        inflated_map = self._inflate_obstacles(occupancy_map, self.safety_margin)
        
        # Initialize A* data structures
        open_set = []
        start_node = Node(
            priority=0.0,
            position=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal)
        )
        heapq.heappush(open_set, start_node)
        
        closed_set = set()
        cost_map = {start: 0.0}
        came_from = {}
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Goal reached
            if current.position == goal:
                return self._reconstruct_path(came_from, current.position)
            
            # Skip if already processed
            if current.position in closed_set:
                continue
            
            closed_set.add(current.position)
            
            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current.position, inflated_map.shape):
                if neighbor_pos in closed_set:
                    continue
                
                # Check if traversable
                if not self._is_traversable(neighbor_pos, inflated_map):
                    continue
                
                # Calculate costs
                move_cost = self._calculate_move_cost(
                    current.position, 
                    neighbor_pos,
                    came_from,
                    depth_map
                )
                tentative_g = current.g_cost + move_cost
                
                # Update if better path found
                if neighbor_pos not in cost_map or tentative_g < cost_map[neighbor_pos]:
                    cost_map[neighbor_pos] = tentative_g
                    h_cost = self._heuristic(neighbor_pos, goal)
                    f_cost = tentative_g + h_cost
                    
                    neighbor_node = Node(
                        priority=f_cost,
                        position=neighbor_pos,
                        g_cost=tentative_g,
                        h_cost=h_cost,
                        parent=current
                    )
                    
                    heapq.heappush(open_set, neighbor_node)
                    came_from[neighbor_pos] = current.position
        
        logger.warning("A* failed to find path to goal")
        return []
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def _get_neighbors(self, pos: Tuple[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 8-connected neighbors."""
        x, y = pos
        h, w = shape
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _calculate_move_cost(self,
                            current: Tuple[int, int],
                            neighbor: Tuple[int, int],
                            came_from: Dict,
                            depth_map: Optional[np.ndarray] = None) -> float:
        """
        Calculate cost of moving from current to neighbor.
        
        Incorporates:
        - Distance cost (diagonal vs straight)
        - Smoothness cost (penalize direction changes)
        - Depth cost (prefer areas with more clearance)
        """
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        
        # Base movement cost
        if abs(dx) + abs(dy) == 2:  # Diagonal
            cost = self.diagonal_cost
        else:  # Straight
            cost = self.straight_cost
        
        # Smoothness cost - penalize direction changes
        if current in came_from:
            parent = came_from[current]
            prev_dx = current[0] - parent[0]
            prev_dy = current[1] - parent[1]
            
            # Dot product of movement directions
            dot = (dx * prev_dx + dy * prev_dy) / (
                np.sqrt(dx*dx + dy*dy) * np.sqrt(prev_dx*prev_dx + prev_dy*prev_dy) + 1e-6
            )
            # Penalize sharp turns (dot product close to -1)
            smoothness_cost = (1.0 - dot) * self.smoothness_weight
            cost += smoothness_cost
        
        # Depth cost - prefer areas with more clearance
        if depth_map is not None:
            ny, nx = neighbor[1], neighbor[0]
            if 0 <= ny < depth_map.shape[0] and 0 <= nx < depth_map.shape[1]:
                depth_value = depth_map[ny, nx]
                # Lower cost for areas with more depth (farther from obstacles)
                depth_cost = max(0.0, 1.0 - depth_value / 5.0)
                cost += depth_cost * 0.5
        
        return cost
    
    def _is_valid_position(self, pos: Tuple[int, int], occupancy_map: np.ndarray) -> bool:
        """Check if position is within bounds and not occupied."""
        x, y = pos
        h, w = occupancy_map.shape
        
        if not (0 <= x < w and 0 <= y < h):
            return False
        
        return occupancy_map[y, x] == 0
    
    def _is_traversable(self, pos: Tuple[int, int], inflated_map: np.ndarray) -> bool:
        """Check if position is traversable (not occupied in inflated map)."""
        x, y = pos
        return inflated_map[y, x] == 0
    
    def _inflate_obstacles(self, occupancy_map: np.ndarray, margin: int) -> np.ndarray:
        """
        Inflate obstacles by margin to ensure safety clearance.
        
        Args:
            occupancy_map: Binary occupancy grid
            margin: Number of cells to inflate
            
        Returns:
            Inflated occupancy map
        """
        if margin <= 0:
            return occupancy_map
        
        import cv2
        kernel = np.ones((2*margin+1, 2*margin+1), np.uint8)
        inflated = cv2.dilate(occupancy_map.astype(np.uint8), kernel, iterations=1)
        return inflated
    
    def _find_nearest_free_position(self,
                                    pos: Tuple[int, int],
                                    occupancy_map: np.ndarray,
                                    max_search_radius: int = 50) -> Optional[Tuple[int, int]]:
        """Find nearest free position using BFS."""
        x, y = pos
        h, w = occupancy_map.shape
        
        visited = set()
        queue = [(x, y, 0)]  # (x, y, distance)
        
        while queue:
            cx, cy, dist = queue.pop(0)
            
            if (cx, cy) in visited or dist > max_search_radius:
                continue
            
            visited.add((cx, cy))
            
            # Check if free
            if 0 <= cx < w and 0 <= cy < h and occupancy_map[cy, cx] == 0:
                return (cx, cy)
            
            # Add neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    queue.append((nx, ny, dist + 1))
        
        return None
    
    def _reconstruct_path(self,
                         came_from: Dict,
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
