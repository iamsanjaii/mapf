"""
src/mapf/coordinator.py
~~~~~~~~~~~~~~~~~~~~~~~~
Rule-based multi-robot coordinator (Baseline 3 / bridge to agent architecture).

After paths are planned (by independent A* or prioritised planning),
the coordinator inspects conflicts and applies resolution rules:

1. WAIT   — one robot waits while the other passes
2. REROUTE — one robot takes a detour
3. REPLAN  — trigger full replanning for the affected robot

This deterministic coordinator is the precursor to the agentic architecture.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.environment.grid import Grid
from src.mapf.conflict import Conflict, ConflictDetector, ConflictType
from src.mapf.reservation import ReservationTable
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic
from src.robots.robot import Robot, RobotStatus


Pos = Tuple[int, int]
Path = List[Pos]


class ResolutionAction(Enum):
    WAIT = auto()
    REROUTE = auto()
    REPLAN = auto()
    UNRESOLVED = auto()


class CoordinationResult:
    def __init__(self) -> None:
        self.paths: Dict[int, Optional[Path]] = {}
        self.conflicts_before: List[Conflict] = []
        self.conflicts_after: List[Conflict] = []
        self.actions: Dict[int, ResolutionAction] = {}
        self.replan_count: int = 0
        self.runtime_ms: float = 0.0

    @property
    def conflict_reduction(self) -> int:
        return len(self.conflicts_before) - len(self.conflicts_after)


class RuleBasedCoordinator:
    """
    Applies rule-based conflict resolution to a set of planned paths.

    Strategy
    --------
    For each conflict (sorted by severity, highest first):
    1. Try to make the lower-priority robot wait one step.
    2. If waiting still causes conflict, reroute the lower-priority robot
       around the higher-priority robot's cells.
    3. If rerouting fails (e.g. no alternative path), mark as UNRESOLVED.

    Parameters
    ----------
    max_iterations : int
        Maximum conflict-resolution rounds before giving up.
    """

    def __init__(
        self,
        heuristic: str = "manhattan",
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
        max_iterations: int = 10,
    ) -> None:
        self._heuristic = get_heuristic(heuristic)
        self.move_cost = move_cost
        self.wait_cost = wait_cost
        self.max_iterations = max_iterations
        self._astar = AStarPlanner(move_cost=move_cost)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def coordinate(
        self,
        robots: List[Robot],
        paths: Dict[int, Optional[Path]],
        grid: Grid,
    ) -> CoordinationResult:
        t0 = time.perf_counter()
        result = CoordinationResult()
        result.paths = {rid: list(p) if p else None for rid, p in paths.items()}

        valid_paths = {rid: p for rid, p in result.paths.items() if p}
        result.conflicts_before = ConflictDetector.detect_all(valid_paths)

        for iteration in range(self.max_iterations):
            valid_paths = {rid: p for rid, p in result.paths.items() if p}
            conflicts = ConflictDetector.detect_all(valid_paths)

            if not conflicts:
                break

            # Resolve the most severe conflict first
            conflicts.sort(key=ConflictDetector.severity_score, reverse=True)
            conflict = conflicts[0]

            rid_a, rid_b = conflict.robots
            # Lower-priority robot yields
            robot_a = next(r for r in robots if r.id == rid_a)
            robot_b = next(r for r in robots if r.id == rid_b)

            yielding_robot = robot_a if robot_a.priority > robot_b.priority else robot_b
            priority_robot = robot_b if robot_a.priority > robot_b.priority else robot_a

            action = self._resolve(
                conflict,
                yielding_robot,
                priority_robot,
                result.paths,
                grid,
            )
            result.actions[yielding_robot.id] = action
            if action in (ResolutionAction.REROUTE, ResolutionAction.REPLAN):
                result.replan_count += 1
        else:
            pass  # Max iterations reached

        valid_paths = {rid: p for rid, p in result.paths.items() if p}
        result.conflicts_after = ConflictDetector.detect_all(valid_paths)
        result.runtime_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve(
        self,
        conflict: Conflict,
        yielding: Robot,
        priority: Robot,
        paths: Dict[int, Optional[Path]],
        grid: Grid,
    ) -> ResolutionAction:
        # Strategy 1: Insert a wait step at the beginning of yielding robot's path
        path = paths.get(yielding.id)
        if path and len(path) >= 1:
            wait_path = [path[0]] + path   # stay one extra step
            paths[yielding.id] = wait_path

            valid = {rid: p for rid, p in paths.items() if p}
            if not ConflictDetector.detect_all(valid):
                return ResolutionAction.WAIT

        # Strategy 2: Reroute — plan around priority robot's cells
        priority_path = paths.get(priority.id) or []
        forbidden = set(priority_path)

        plan = self._astar.plan(
            grid,
            yielding.current_pos,
            yielding.goal,
            self._heuristic,
            forbidden=forbidden,
        )
        if plan.path:
            paths[yielding.id] = plan.path
            return ResolutionAction.REROUTE

        # Strategy 3: Cannot resolve
        return ResolutionAction.UNRESOLVED
