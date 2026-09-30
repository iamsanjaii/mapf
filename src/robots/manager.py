"""
src/robots/manager.py
~~~~~~~~~~~~~~~~~~~~~~
RobotManager — central controller for spawning and tracking robots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.robots.robot import Robot, RobotStatus


Pos = Tuple[int, int]


class RobotManager:
    """
    Manages all robots in the simulation.

    Responsibilities
    ----------------
    - Spawn robots from start/goal lists
    - Assign priorities (can be overridden)
    - Track positions and statuses
    - Step all robots forward one timestep
    """

    def __init__(
        self,
        starts: List[Pos],
        goals: List[Pos],
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
    ) -> None:
        if len(starts) != len(goals):
            raise ValueError("starts and goals must have equal length.")

        self.move_cost = move_cost
        self.wait_cost = wait_cost
        self.robots: List[Robot] = []
        self.timestep: int = 0

        for i, (start, goal) in enumerate(zip(starts, goals)):
            r = Robot(id=i, start=start, goal=goal, priority=i)
            self.robots.append(r)

        self._pos_index: Dict[Pos, Robot] = {r.start: r for r in self.robots}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, robot_id: int) -> Robot:
        return self.robots[robot_id]

    def active_robots(self) -> List[Robot]:
        """Robots that have not yet reached their goals."""
        return [r for r in self.robots if not r.is_done() and not r.is_stuck()]

    def done_robots(self) -> List[Robot]:
        return [r for r in self.robots if r.is_done()]

    def stuck_robots(self) -> List[Robot]:
        return [r for r in self.robots if r.is_stuck()]

    def all_done(self) -> bool:
        return all(r.is_done() or r.is_stuck() for r in self.robots)

    def success_count(self) -> int:
        return len(self.done_robots())

    # ------------------------------------------------------------------
    # Priority assignment
    # ------------------------------------------------------------------

    def assign_priorities_by_distance(self) -> None:
        """
        Robots closer to their goal get lower priority values (plan first).
        This is a simple, common heuristic.
        """
        from src.planning.heuristics import manhattan
        sorted_robots = sorted(
            self.robots,
            key=lambda r: manhattan(r.start, r.goal),
        )
        for rank, robot in enumerate(sorted_robots):
            robot.priority = rank

    def assign_priorities_fixed(self, order: List[int]) -> None:
        """Manually assign priorities from a list of robot ids."""
        if sorted(order) != list(range(len(self.robots))):
            raise ValueError("order must contain exactly one entry per robot id.")
        for rank, robot_id in enumerate(order):
            self.robots[robot_id].priority = rank

    def robots_by_priority(self) -> List[Robot]:
        """Return robots sorted by ascending priority value."""
        return sorted(self.robots, key=lambda r: r.priority)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step_all(self) -> Dict[int, Optional[Pos]]:
        """
        Advance all robots one timestep along their paths.

        Returns a dict mapping robot_id → new_position (or None if stayed).
        """
        self.timestep += 1
        result: Dict[int, Optional[Pos]] = {}

        for robot in self.robots:
            if robot.is_done() or robot.is_stuck():
                result[robot.id] = robot.current_pos
                continue

            if robot.status == RobotStatus.WAITING:
                robot.wait(self.wait_cost)
                result[robot.id] = robot.current_pos
            elif robot.path:
                new_pos = robot.step_along_path(self.move_cost)
                result[robot.id] = new_pos
            else:
                result[robot.id] = robot.current_pos

        self._rebuild_pos_index()
        return result

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def current_positions(self) -> Dict[int, Pos]:
        return {r.id: r.current_pos for r in self.robots}

    def position_to_robot(self) -> Dict[Pos, int]:
        return {r.current_pos: r.id for r in self.robots}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rebuild_pos_index(self) -> None:
        self._pos_index = {r.current_pos: r for r in self.robots}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_robots": len(self.robots),
            "done": self.success_count(),
            "stuck": len(self.stuck_robots()),
            "active": len(self.active_robots()),
            "timestep": self.timestep,
        }
