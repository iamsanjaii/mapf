"""
src/planning/planner.py
~~~~~~~~~~~~~~~~~~~~~~~~
High-level Planner facade that wires together the grid, A* and heuristics.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

from src.environment.grid import Grid
from src.planning.astar import AStarPlanner, PlanResult
from src.planning.heuristics import get_heuristic, Heuristic


Pos = Tuple[int, int]


class Planner:
    """
    Convenience wrapper around AStarPlanner.

    Parameters
    ----------
    heuristic : str or callable
        Heuristic name ('manhattan', 'euclidean', 'chebyshev', 'octile')
        or a callable directly.
    move_cost : float
        Uniform cost per step.
    """

    def __init__(
        self,
        heuristic: str | Heuristic = "manhattan",
        move_cost: float = 1.0,
    ) -> None:
        if callable(heuristic):
            self._heuristic: Heuristic = heuristic
        else:
            self._heuristic = get_heuristic(heuristic)
        self._astar = AStarPlanner(move_cost=move_cost)

    # ------------------------------------------------------------------

    def plan(
        self,
        grid: Grid,
        start: Pos,
        goal: Pos,
        forbidden: Optional[set] = None,
    ) -> PlanResult:
        """Plan a path from start to goal using the configured heuristic."""
        return self._astar.plan(grid, start, goal, self._heuristic, forbidden)

    def plan_with_heuristic(
        self,
        grid: Grid,
        start: Pos,
        goal: Pos,
        heuristic: str | Heuristic,
        forbidden: Optional[set] = None,
    ) -> PlanResult:
        """Plan using an override heuristic (useful for experiments)."""
        h = heuristic if callable(heuristic) else get_heuristic(heuristic)
        return self._astar.plan(grid, start, goal, h, forbidden)
