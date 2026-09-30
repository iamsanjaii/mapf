"""
src/robots/robot.py
~~~~~~~~~~~~~~~~~~~~
Robot data class and status enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


Pos = Tuple[int, int]


class RobotStatus(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    WAITING = auto()
    DONE = auto()
    STUCK = auto()       # goal is unreachable


@dataclass
class Robot:
    """
    Represents a single robot in the simulation.

    Attributes
    ----------
    id : int
        Unique robot identifier.
    start : Pos
        Starting cell (row, col).
    goal : Pos
        Goal cell (row, col).
    current_pos : Pos
        Current cell (updated each timestep).
    path : list of Pos
        Planned path from current_pos to goal (including current pos at index 0).
    status : RobotStatus
        Current lifecycle status.
    total_cost : float
        Accumulated movement cost.
    wait_steps : int
        Number of timesteps spent waiting.
    replan_count : int
        Number of times the robot has been replanned.
    priority : int
        Lower value → higher priority in prioritised MAPF.
    """

    id: int
    start: Pos
    goal: Pos
    current_pos: Pos = field(init=False)
    path: List[Pos] = field(default_factory=list)
    status: RobotStatus = RobotStatus.IDLE
    total_cost: float = 0.0
    wait_steps: int = 0
    replan_count: int = 0
    priority: int = 0

    def __post_init__(self) -> None:
        self.current_pos = self.start

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self.status == RobotStatus.DONE

    def is_stuck(self) -> bool:
        return self.status == RobotStatus.STUCK

    def at_goal(self) -> bool:
        return self.current_pos == self.goal

    def step_along_path(self, move_cost: float = 1.0) -> Optional[Pos]:
        """
        Advance one step along the planned path.
        Returns the new position, or None if no path remains.
        """
        if len(self.path) <= 1:
            if self.at_goal():
                self.status = RobotStatus.DONE
            return None

        # Move to next cell in path
        self.path.pop(0)
        self.current_pos = self.path[0]
        self.total_cost += move_cost

        if self.at_goal():
            self.status = RobotStatus.DONE

        return self.current_pos

    def wait(self, wait_cost: float = 1.0) -> None:
        """Increment wait counter and cost without moving."""
        self.wait_steps += 1
        self.total_cost += wait_cost
        self.status = RobotStatus.WAITING

    def __repr__(self) -> str:
        return (
            f"Robot(id={self.id}, pos={self.current_pos}, "
            f"goal={self.goal}, status={self.status.name})"
        )
