"""
src/mapf/prioritized.py
~~~~~~~~~~~~~~~~~~~~~~~~
Prioritised Planning baseline for multi-robot MAPF.

Robots plan in strict priority order.  Each robot reserves its cells in
the ReservationTable before the next robot plans.  Lower-priority robots
must route around already-reserved space-time cells.

This is Baseline 2 (stronger than independent A*).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from src.environment.grid import Grid
from src.mapf.reservation import ReservationTable
from src.planning.astar import AStarPlanner, PlanResult
from src.planning.heuristics import get_heuristic, Heuristic
from src.robots.robot import Robot, RobotStatus


Pos = Tuple[int, int]
Path = List[Pos]


class PrioritizedMAPFResult:
    """Aggregated result from a full prioritised planning run."""

    def __init__(self) -> None:
        self.paths: Dict[int, Optional[Path]] = {}
        self.costs: Dict[int, float] = {}
        self.nodes_expanded: Dict[int, int] = {}
        self.runtime_ms: float = 0.0
        self.success_count: int = 0
        self.fail_count: int = 0

    @property
    def total_cost(self) -> float:
        return sum(v for v in self.costs.values() if v != float("inf"))

    @property
    def makespan(self) -> int:
        lens = [len(p) for p in self.paths.values() if p]
        return max(lens) - 1 if lens else 0

    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total else 0.0


class PrioritizedPlanner:
    """
    Prioritised MAPF planner (Baseline 2).

    Plans robots one by one in priority order.
    Each robot's path is reserved in a space-time table so later
    robots avoid those cells.

    A time-extended A* approach is used: forbidden cells at each
    timestep are derived from the reservation table.

    Parameters
    ----------
    heuristic : str
        Heuristic name.
    move_cost : float
    wait_cost : float
    max_timesteps : int
        Upper bound on path length (prevents infinite loops).
    """

    def __init__(
        self,
        heuristic: str = "manhattan",
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
        max_timesteps: int = 200,
    ) -> None:
        self._heuristic: Heuristic = get_heuristic(heuristic)
        self.move_cost = move_cost
        self.wait_cost = wait_cost
        self.max_timesteps = max_timesteps
        self._astar = AStarPlanner(move_cost=move_cost)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        robots: List[Robot],
        grid: Grid,
        verbose: bool = False,
    ) -> PrioritizedMAPFResult:
        """
        Run prioritised planning for all robots.

        Robots are sorted by their .priority attribute (ascending).
        """
        t0 = time.perf_counter()
        result = PrioritizedMAPFResult()
        reservation = ReservationTable()

        sorted_robots = sorted(robots, key=lambda r: r.priority)

        for robot in sorted_robots:
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Planning for Robot-{robot.id} (priority {robot.priority})  "
                      f"start={robot.start}  goal={robot.goal}")

            path, plan_result = self._plan_with_reservation(
                robot, grid, reservation, verbose=verbose
            )

            result.paths[robot.id] = path
            result.nodes_expanded[robot.id] = plan_result.nodes_expanded

            if path:
                result.costs[robot.id] = plan_result.cost
                result.success_count += 1
                robot.path = list(path)
                robot.status = RobotStatus.PLANNING
                reservation.reserve_path(robot.id, path)
            else:
                result.costs[robot.id] = float("inf")
                result.fail_count += 1
                robot.status = RobotStatus.STUCK

        result.runtime_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plan_with_reservation(
        self,
        robot: Robot,
        grid: Grid,
        reservation: ReservationTable,
        verbose: bool = False,
    ) -> Tuple[Optional[Path], PlanResult]:
        """
        Space-time A* that avoids cells reserved by higher-priority robots
        at the same timestep.

        Expands nodes as (pos, t) tuples.  At each (pos, t) we check the
        reservation table for t and t+1 (to catch edge conflicts).
        Waiting in place is also allowed (adds wait_cost).
        """
        import heapq
        import time as _time

        t0 = _time.perf_counter()

        start = robot.current_pos
        goal = robot.goal

        if not grid.in_bounds(*start) or not grid.in_bounds(*goal):
            return None, PlanResult(path=None, cost=0.0, nodes_expanded=0, runtime_ms=0.0)

        if start == goal:
            return [start], PlanResult(path=[start], cost=0.0, nodes_expanded=0, runtime_ms=0.0)

        # Priority queue entries: (f, g, pos, t, parent_key)
        # parent map: (pos, t) → (parent_pos, parent_t)
        open_heap = []
        g_score: dict = {}
        parent_map: dict = {}

        h0 = self._heuristic(start, goal)
        heapq.heappush(open_heap, (h0, 0.0, start, 0, None))
        g_score[(start, 0)] = 0.0
        nodes_expanded = 0
        closed_set: set = set()
        
        tag = f"[SpaceTime-A* Robot-{robot.id}]"
        if verbose:
            print(f"\n  {tag} start={start}  goal={goal}")
            print(f"  {tag} heuristic h(start,goal) = {h0:.2f}")
            print(f"  {'─'*50}")

        while open_heap:
            f, g, pos, t, par_key = heapq.heappop(open_heap)

            state = (pos, t)
            if state in closed_set:
                continue
            closed_set.add(state)
            nodes_expanded += 1
            parent_map[state] = par_key
            
            if verbose:
                h_val = self._heuristic(pos, goal)
                print(f"  {tag} expand {pos}@t={t}  "
                      f"g={g:.1f}  h={h_val:.2f}  f={f:.2f}  "
                      f"(open={len(open_heap)}  closed={len(closed_set)})")

            if pos == goal:
                # Reconstruct path
                path = []
                cur = state
                while cur is not None:
                    path.append(cur[0])
                    cur = parent_map[cur]
                path.reverse()
                runtime = (_time.perf_counter() - t0) * 1000
                result = PlanResult(path=path, cost=g, nodes_expanded=nodes_expanded, runtime_ms=runtime)
                if verbose:
                    print(f"  {tag} ✓ GOAL REACHED  cost={g:.1f}  nodes_expanded={nodes_expanded}  runtime={runtime:.2f}ms")
                return path, result

            if t >= self.max_timesteps:
                continue

            # Try neighbours (move)
            for nb in grid.neighbours(*pos, passable_only=True):
                next_t = t + 1
                # Check vertex reservation at next_t
                if reservation.is_reserved(nb, next_t, exclude_robot=robot.id):
                    continue
                # Check edge conflict: don't swap with another robot
                if reservation.is_reserved(pos, next_t, exclude_robot=robot.id):
                    continue
                new_g = g + self.move_cost
                ns = (nb, next_t)
                if new_g < g_score.get(ns, float("inf")):
                    g_score[ns] = new_g
                    h = self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (new_g + h, new_g, nb, next_t, state))
                    if verbose:
                        print(f"    {tag}   → neighbour {nb}@t={next_t}  g={new_g:.1f}  h={h:.2f}  f={new_g+h:.2f}")

            # Try waiting in place
            next_t = t + 1
            if not reservation.is_reserved(pos, next_t, exclude_robot=robot.id):
                new_g = g + self.wait_cost
                ns = (pos, next_t)
                if new_g < g_score.get(ns, float("inf")):
                    g_score[ns] = new_g
                    h = self._heuristic(pos, goal)
                    heapq.heappush(open_heap, (new_g + h, new_g, pos, next_t, state))
                    if verbose:
                        print(f"    {tag}   → wait {pos}@t={next_t}  g={new_g:.1f}  h={h:.2f}  f={new_g+h:.2f}")

        runtime = (_time.perf_counter() - t0) * 1000
        empty = PlanResult(path=None, cost=float("inf"), nodes_expanded=nodes_expanded, runtime_ms=runtime)
        return None, empty

