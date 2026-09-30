"""
src/mapf_ro/sandbag.py
~~~~~~~~~~~~~~~~~~~~~~~
Sandbag handling for MAPF-RO.

Sandbags are movable resources.  A robot can carry a sandbag toward a pit
and drop it in, paying a per-step movement cost > robot_move_cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.environment.grid import Grid, CellType


Pos = Tuple[int, int]


@dataclass
class SandbagState:
    """Live state of a sandbag during simulation."""
    id: int
    position: Pos
    move_cost_per_step: float = 4.0
    in_pit: bool = False
    target_pit: Optional[Pos] = None
    assigned_robot: Optional[int] = None
    total_distance_moved: int = 0

    def move_to(self, new_pos: Pos) -> float:
        """Record movement and return the step cost incurred."""
        self.position = new_pos
        self.total_distance_moved += 1
        return self.move_cost_per_step

    def deploy_in_pit(self, pit_pos: Pos) -> None:
        self.position = pit_pos
        self.in_pit = True
        self.target_pit = pit_pos


class SandbagManager:
    """
    Manages sandbag placement and movement.

    Exposes methods for:
    - Finding the nearest free sandbag to a given position
    - Routing a sandbag toward a pit (returns movement cost)
    - Deploying a sandbag into a pit
    """

    def __init__(
        self,
        sandbag_positions: List[Pos],
        move_cost_per_step: float = 4.0,
    ) -> None:
        self._sandbags: Dict[int, SandbagState] = {
            i: SandbagState(id=i, position=pos, move_cost_per_step=move_cost_per_step)
            for i, pos in enumerate(sandbag_positions)
        }

    # ------------------------------------------------------------------

    def nearest_free_sandbag(self, reference: Pos) -> Optional[SandbagState]:
        """Return the unassigned, in-field sandbag closest to *reference*."""
        available = [
            sb for sb in self._sandbags.values()
            if not sb.in_pit and sb.assigned_robot is None
        ]
        if not available:
            return None
        return min(
            available,
            key=lambda sb: abs(sb.position[0] - reference[0])
                           + abs(sb.position[1] - reference[1]),
        )

    def move_cost_estimate(self, sandbag_id: int, target: Pos) -> float:
        """Estimate total cost to move sandbag to target (Manhattan × move_cost)."""
        sb = self._sandbags[sandbag_id]
        dist = abs(sb.position[0] - target[0]) + abs(sb.position[1] - target[1])
        return dist * sb.move_cost_per_step

    def deploy(self, sandbag_id: int, pit_pos: Pos, grid: Grid) -> float:
        """
        Move sandbag to pit and deploy it.
        Updates the grid cell from PIT → FREE.
        Returns the total movement cost paid.
        """
        sb = self._sandbags[sandbag_id]
        dist = abs(sb.position[0] - pit_pos[0]) + abs(sb.position[1] - pit_pos[1])
        cost = dist * sb.move_cost_per_step
        sb.deploy_in_pit(pit_pos)
        grid.set(pit_pos[0], pit_pos[1], CellType.FREE)
        return cost

    def total_move_cost(self) -> float:
        return sum(
            sb.total_distance_moved * sb.move_cost_per_step
            for sb in self._sandbags.values()
        )

    def all_sandbags(self) -> List[SandbagState]:
        return list(self._sandbags.values())
