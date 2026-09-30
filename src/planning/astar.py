"""
src/planning/astar.py
~~~~~~~~~~~~~~~~~~~~~~
A* path planner for a 2-D grid.

Returns a PlanResult containing the path, cost, number of nodes expanded,
and wall-clock runtime.  Returns None path when the goal is unreachable.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.environment.grid import Grid
from src.planning.heuristics import Heuristic, manhattan


Pos = Tuple[int, int]


@dataclass
class PlanResult:
    """Result returned by the A* planner."""
    path: Optional[List[Pos]]    # None if goal is unreachable
    cost: float                  # total path cost (number of steps)
    nodes_expanded: int          # number of nodes popped from the open list
    runtime_ms: float            # wall-clock planning time in milliseconds
    success: bool = field(init=False)

    def __post_init__(self) -> None:
        self.success = self.path is not None and len(self.path) > 0


@dataclass(order=True)
class _Node:
    """Priority-queue node for A*."""
    f: float
    g: float = field(compare=False)
    pos: Pos = field(compare=False)
    parent: Optional["_Node"] = field(compare=False, default=None)


class AStarPlanner:
    """
    Standard A* algorithm on a passable-cell grid.

    Parameters
    ----------
    move_cost : float
        Uniform step cost per cell (default 1.0).
    """

    def __init__(self, move_cost: float = 1.0) -> None:
        self.move_cost = move_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        grid: Grid,
        start: Pos,
        goal: Pos,
        heuristic: Heuristic = manhattan,
        forbidden: Optional[set] = None,
        verbose: bool = False,
        robot_id: Optional[int] = None,
    ) -> PlanResult:
        """
        Find the shortest path from *start* to *goal* on *grid*.

        Parameters
        ----------
        grid : Grid
            The environment grid.
        start : (row, col)
            Starting position.
        goal : (row, col)
            Goal position.
        heuristic : callable
            h(a, b) → float.
        forbidden : set of (row, col), optional
            Additional cells to treat as obstacles (used by prioritised MAPF).

        Returns
        -------
        PlanResult
        """
        t0 = time.perf_counter()

        forbidden = forbidden or set()

        # Validate start / goal
        if not grid.in_bounds(*start):
            return PlanResult(path=None, cost=0.0, nodes_expanded=0,
                              runtime_ms=0.0)
        if not grid.in_bounds(*goal):
            return PlanResult(path=None, cost=0.0, nodes_expanded=0,
                              runtime_ms=0.0)

        if start == goal:
            return PlanResult(path=[start], cost=0.0, nodes_expanded=0,
                              runtime_ms=0.0)

        tag = f"[A* Robot-{robot_id}]" if robot_id is not None else "[A*]"

        if verbose:
            print(f"\n  {tag} start={start}  goal={goal}")
            print(f"  {tag} heuristic h(start,goal) = {heuristic(start, goal):.2f}")
            print(f"  {'─'*50}")

        open_list: List[_Node] = []
        g_score: Dict[Pos, float] = {start: 0.0}
        closed_set: set = set()
        nodes_expanded = 0

        start_node = _Node(f=heuristic(start, goal), g=0.0, pos=start)
        heapq.heappush(open_list, start_node)

        while open_list:
            current = heapq.heappop(open_list)

            if current.pos in closed_set:
                continue

            closed_set.add(current.pos)
            nodes_expanded += 1

            if verbose:
                h_val = heuristic(current.pos, goal)
                print(f"  {tag} expand {current.pos}  "
                      f"g={current.g:.1f}  h={h_val:.2f}  f={current.f:.2f}  "
                      f"(open={len(open_list)}  closed={len(closed_set)})")

            if current.pos == goal:
                path = self._reconstruct(current)
                runtime = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"  {tag} ✓ GOAL REACHED  cost={current.g:.1f}  "
                          f"nodes_expanded={nodes_expanded}  "
                          f"runtime={runtime:.2f}ms")
                    print(f"  {tag} path: {' → '.join(str(p) for p in path)}")
                return PlanResult(
                    path=path,
                    cost=current.g,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=runtime,
                )

            for neighbour in grid.neighbours(*current.pos, passable_only=True):
                if neighbour in closed_set:
                    continue
                if neighbour in forbidden:
                    continue

                tentative_g = current.g + self.move_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    g_score[neighbour] = tentative_g
                    h = heuristic(neighbour, goal)
                    node = _Node(
                        f=tentative_g + h,
                        g=tentative_g,
                        pos=neighbour,
                        parent=current,
                    )
                    heapq.heappush(open_list, node)
                    if verbose:
                        print(f"    {tag}   → neighbour {neighbour}  "
                              f"g={tentative_g:.1f}  h={h:.2f}  f={tentative_g+h:.2f}")

        # Goal unreachable
        runtime = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"  {tag} ✗ UNREACHABLE  nodes_expanded={nodes_expanded}  "
                  f"runtime={runtime:.2f}ms")
        return PlanResult(path=None, cost=float("inf"),
                          nodes_expanded=nodes_expanded, runtime_ms=runtime)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct(node: _Node) -> List[Pos]:
        path: List[Pos] = []
        current: Optional[_Node] = node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        path.reverse()
        return path
