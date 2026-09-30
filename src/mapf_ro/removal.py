"""
src/mapf_ro/removal.py
~~~~~~~~~~~~~~~~~~~~~~~
Obstacle removal cost model and traffic index for MAPF-RO.

The key research question:
    Is modifying the environment (filling a pit) cheaper than
    forcing robots to navigate around it?

Traffic index
-------------
Counts how many planned robot paths pass through or adjacent to
each obstacle/pit cell.  Higher traffic → higher value of removal.

Cost model
----------
removal_cost(pit) =
    nearest_sandbag_travel_cost
  + pit_fill_cost
  + robot_detour_savings   (negative term — the benefit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.mapf_ro.sandbag import SandbagManager


Pos = Tuple[int, int]
Path = List[Pos]


@dataclass
class RemovalCostEstimate:
    """Cost estimate for removing (filling) one pit."""
    pit_position: Pos
    sandbag_travel_cost: float
    fill_cost: float
    traffic_index: int
    total_removal_cost: float
    detour_savings: float          # how much detour cost is avoided
    net_benefit: float             # detour_savings - total_removal_cost
    recommended: bool              # True if removal is cheaper than detour


class TrafficIndex:
    """
    Counts the number of robot paths that pass through or immediately
    adjacent to each pit/obstacle cell.
    """

    def __init__(self, pit_positions: List[Pos]) -> None:
        self._pits = pit_positions
        self._counts: Dict[Pos, int] = {p: 0 for p in pit_positions}

    def compute(self, paths: Dict[int, Optional[Path]]) -> Dict[Pos, int]:
        """
        Analyse all paths and count how many paths pass within
        1-cell Manhattan distance of each pit.

        Returns dict mapping pit_pos → traffic count.
        """
        self._counts = {p: 0 for p in self._pits}

        for path in paths.values():
            if not path:
                continue
            for pit in self._pits:
                # Check adjacency (Manhattan ≤ 1)
                for cell in path:
                    if abs(cell[0] - pit[0]) + abs(cell[1] - pit[1]) <= 1:
                        self._counts[pit] += 1
                        break   # count each path at most once per pit

        return self._counts

    def get(self, pos: Pos) -> int:
        return self._counts.get(pos, 0)

    def sorted_by_traffic(self) -> List[Tuple[Pos, int]]:
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)


class RemovalCostModel:
    """
    Computes removal cost estimates and recommends REMOVE vs DETOUR.

    Parameters
    ----------
    fill_cost : float
        Base cost to fill one pit (in addition to sandbag travel).
    robot_move_cost : float
        Cost per robot step.
    """

    def __init__(
        self,
        fill_cost: float = 1.0,
        robot_move_cost: float = 1.0,
        sandbag_manager: Optional[SandbagManager] = None,
    ) -> None:
        self.fill_cost = fill_cost
        self.robot_move_cost = robot_move_cost
        self.sandbag_manager = sandbag_manager

    def estimate(
        self,
        pit_pos: Pos,
        traffic_count: int,
        detour_cost: float,
    ) -> RemovalCostEstimate:
        """
        Estimate cost of filling the pit vs accepting the detour.

        Parameters
        ----------
        pit_pos : (row, col) of the pit
        traffic_count : number of robot paths near this pit
        detour_cost : total extra cost of all affected detours
        """
        if self.sandbag_manager:
            sb = self.sandbag_manager.nearest_free_sandbag(pit_pos)
            sandbag_travel = (
                self.sandbag_manager.move_cost_estimate(sb.id, pit_pos)
                if sb else float("inf")
            )
        else:
            sandbag_travel = float("inf")

        total_removal = sandbag_travel + self.fill_cost
        net_benefit = detour_cost - total_removal

        return RemovalCostEstimate(
            pit_position=pit_pos,
            sandbag_travel_cost=sandbag_travel,
            fill_cost=self.fill_cost,
            traffic_index=traffic_count,
            total_removal_cost=total_removal,
            detour_savings=detour_cost,
            net_benefit=net_benefit,
            recommended=net_benefit > 0,
        )
