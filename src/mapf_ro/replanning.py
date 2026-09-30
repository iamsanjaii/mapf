"""
src/mapf_ro/replanning.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Replanning loop for MAPF-RO.

After an environment modification (e.g. pit filled):
1. Invalidate affected robot paths
2. Re-run the planning agent
3. Detect new conflicts
4. Coordinate
5. Continue

This implements the SENSE → PLAN → ACT → ENVIRONMENT CHANGES → REPLAN loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.environment.grid import Grid, CellType
from src.mapf_ro.pit import PitManager
from src.mapf_ro.removal import RemovalCostModel, TrafficIndex
from src.mapf_ro.sandbag import SandbagManager
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic
from src.robots.robot import Robot, RobotStatus


Pos = Tuple[int, int]
Path = List[Pos]


@dataclass
class ROPlanningResult:
    """Result from MAPF-RO replanning."""
    paths: Dict[int, Optional[Path]] = field(default_factory=dict)
    total_robot_cost: float = 0.0
    total_removal_cost: float = 0.0
    total_sandbag_cost: float = 0.0
    pits_filled: int = 0
    replan_iterations: int = 0
    success_count: int = 0
    fail_count: int = 0
    runtime_ms: float = 0.0

    @property
    def total_energy(self) -> float:
        return self.total_robot_cost + self.total_removal_cost + self.total_sandbag_cost

    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total else 0.0


class MAPFROPlanner:
    """
    Extends the base planner with obstacle-removal capability.

    Strategy
    --------
    1. Plan paths for all robots ignoring pits (they are impassable).
    2. Compute traffic index for each pit.
    3. For each pit with traffic > 0, estimate removal cost vs detour cost.
    4. If removal is cheaper, fill the pit (update grid) and replan.
    5. Repeat until no more beneficial removals or max iterations reached.

    Parameters
    ----------
    enable_removal : bool
        If False, behaves like a standard planner (no obstacle removal).
    """

    def __init__(
        self,
        heuristic: str = "manhattan",
        move_cost: float = 1.0,
        fill_cost: float = 1.0,
        sandbag_move_cost: float = 4.0,
        max_iterations: int = 10,
        enable_removal: bool = True,
    ) -> None:
        self._h = get_heuristic(heuristic)
        self.move_cost = move_cost
        self.fill_cost = fill_cost
        self.sandbag_move_cost = sandbag_move_cost
        self.max_iterations = max_iterations
        self.enable_removal = enable_removal
        self._astar = AStarPlanner(move_cost=move_cost)

    # ------------------------------------------------------------------

    def plan(
        self,
        robots: List[Robot],
        grid: Grid,
        pit_manager: PitManager,
        sandbag_manager: SandbagManager,
    ) -> ROPlanningResult:
        t0 = time.perf_counter()
        result = ROPlanningResult()

        working_grid = grid.copy()

        for iteration in range(self.max_iterations):
            result.replan_iterations = iteration + 1

            # Plan all robots
            paths: Dict[int, Optional[Path]] = {}
            for robot in robots:
                pr = self._astar.plan(working_grid, robot.current_pos, robot.goal, self._h)
                paths[robot.id] = pr.path

            result.paths = paths

            if not self.enable_removal:
                break

            # Compute traffic index
            pit_positions = pit_manager.unfilled_pits()
            if not pit_positions:
                break

            traffic_idx = TrafficIndex(pit_positions)
            traffic_map = traffic_idx.compute(paths)

            cost_model = RemovalCostModel(
                fill_cost=self.fill_cost,
                robot_move_cost=self.move_cost,
                sandbag_manager=sandbag_manager,
            )

            # Find most beneficial pit to fill
            best_candidate = None
            best_benefit = 0.0

            for pit_pos in sorted(pit_positions, key=lambda p: traffic_map[p], reverse=True):
                t_count = traffic_map[pit_pos]
                if t_count == 0:
                    continue

                # Estimate detour cost: extra steps each affected robot would take
                detour_total = 0.0
                for robot in robots:
                    path_with_pit = paths.get(robot.id)
                    if path_with_pit is None:
                        # Robot stuck — check if pit is on its direct path
                        direct = self._astar.plan(working_grid, robot.current_pos, robot.goal, self._h)
                        if not direct.path:
                            detour_total += 20 * self.move_cost
                        continue
                    # Plan without this pit (temporarily make it free)
                    test_grid = working_grid.copy()
                    test_grid.set(pit_pos[0], pit_pos[1], CellType.FREE)
                    free_result = self._astar.plan(test_grid, robot.current_pos, robot.goal, self._h)
                    if free_result.path:
                        savings = (len(path_with_pit) - len(free_result.path)) * self.move_cost
                        detour_total += max(0.0, savings)

                estimate = cost_model.estimate(pit_pos, t_count, detour_total)
                if estimate.recommended and estimate.net_benefit > best_benefit:
                    best_benefit = estimate.net_benefit
                    best_candidate = (pit_pos, estimate)

            if best_candidate is None:
                break

            # Fill the pit
            pit_pos, estimate = best_candidate
            sb = sandbag_manager.nearest_free_sandbag(pit_pos)
            if sb is None:
                break

            deploy_cost = sandbag_manager.deploy(sb.id, pit_pos, working_grid)
            pit_manager.fill_pit(pit_pos, sb.id, self.fill_cost)

            result.pits_filled += 1
            result.total_removal_cost += self.fill_cost
            result.total_sandbag_cost += deploy_cost

        # Final metrics
        for robot in robots:
            path = result.paths.get(robot.id)
            if path:
                result.success_count += 1
                robot.path = list(path)
                robot.status = RobotStatus.PLANNING
                result.total_robot_cost += (len(path) - 1) * self.move_cost
            else:
                result.fail_count += 1
                robot.status = RobotStatus.STUCK

        result.runtime_ms = (time.perf_counter() - t0) * 1000
        return result
