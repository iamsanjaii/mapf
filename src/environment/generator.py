"""
src/environment/generator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reproducible random environment generator.

Generates grid + robot start/goal positions with guarantees:
- starts and goals are on FREE cells
- starts do not overlap each other or goals
- goals do not overlap each other
- a path MAY or MAY NOT exist (experiment runner detects unreachable goals)
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from src.environment.grid import Grid, CellType
from src.environment.obstacles import (
    Obstacle, Pit, Sandbag, ObstacleRegistry,
)


class EnvironmentGenerator:
    """
    Generates reproducible random environments.

    Parameters
    ----------
    width, height : int
        Grid dimensions.
    obstacle_density : float
        Fraction of cells that become obstacles (0.0–1.0).
    pit_density : float
        Fraction of cells that become pits (used for MAPF-RO).
    sandbag_count : int
        Number of sandbags to place (used for MAPF-RO).
    movement : int
        4 or 8.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        obstacle_density: float = 0.20,
        pit_density: float = 0.0,
        sandbag_count: int = 0,
        movement: int = 4,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.pit_density = pit_density
        self.sandbag_count = sandbag_count
        self.movement = movement
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, num_robots: int = 5
    ) -> Tuple[Grid, List[Tuple[int, int]], List[Tuple[int, int]], ObstacleRegistry]:
        """
        Generate a complete environment.

        Returns
        -------
        grid : Grid
        starts : list of (row, col) — one per robot
        goals  : list of (row, col) — one per robot
        registry : ObstacleRegistry
        """
        grid = Grid(self.width, self.height, self.movement)
        registry = ObstacleRegistry()

        total_cells = self.width * self.height
        n_obstacles = int(total_cells * self.obstacle_density)
        n_pits = int(total_cells * self.pit_density)

        all_positions = [
            (r, c) for r in range(self.height) for c in range(self.width)
        ]
        self._rng.shuffle(all_positions)

        used: set = set()

        # Place obstacles
        for pos in all_positions[:n_obstacles]:
            grid.set(pos[0], pos[1], CellType.OBSTACLE)
            registry.add_obstacle(Obstacle(position=pos, is_removable=False))
            used.add(pos)

        # Place pits (from remaining cells)
        remaining = [p for p in all_positions if p not in used]
        for pos in remaining[:n_pits]:
            grid.set(pos[0], pos[1], CellType.PIT)
            registry.add_pit(Pit(position=pos))
            used.add(pos)

        # Refresh free cells list
        free_cells = [
            p for p in all_positions
            if p not in used
        ]

        # Place sandbags
        sb_positions = self._rng.sample(
            free_cells, min(self.sandbag_count, len(free_cells))
        )
        for i, pos in enumerate(sb_positions):
            grid.set(pos[0], pos[1], CellType.SANDBAG)
            registry.add_sandbag(Sandbag(id=i, position=pos))
            used.add(pos)

        # Refresh after sandbags
        free_cells = [p for p in all_positions if p not in used]

        # Place robot starts and goals
        starts, goals = self._place_robots(free_cells, num_robots)

        return grid, starts, goals, registry

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _place_robots(
        self,
        free_cells: List[Tuple[int, int]],
        num_robots: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Choose non-overlapping start and goal positions from free_cells.
        Raises ValueError if the grid is too crowded.
        """
        needed = num_robots * 2
        if len(free_cells) < needed:
            raise ValueError(
                f"Not enough free cells ({len(free_cells)}) for "
                f"{num_robots} robots (need {needed})."
            )

        chosen = self._rng.sample(free_cells, needed)
        starts = chosen[:num_robots]
        goals = chosen[num_robots:]
        return starts, goals

    def from_config(self, cfg: Dict[str, Any], num_robots: int) -> Tuple[
        Grid, List[Tuple[int, int]], List[Tuple[int, int]], ObstacleRegistry
    ]:
        """Convenience wrapper — accepts a parsed YAML config dict."""
        return self.generate(num_robots=num_robots)
