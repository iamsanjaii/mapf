"""
src/mapf_ro/pit.py
~~~~~~~~~~~~~~~~~~~
Pit semantics for MAPF-RO.

A pit is an impassable cell that can be rendered passable by dropping a
sandbag into it.  The cost of doing so is accounted for separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Pos = Tuple[int, int]


@dataclass
class PitState:
    """Live state of a pit during simulation."""
    position: Pos
    filled: bool = False
    filled_by: Optional[int] = None      # sandbag id used
    fill_cost_paid: float = 0.0

    def fill(self, sandbag_id: int, cost: float) -> None:
        self.filled = True
        self.filled_by = sandbag_id
        self.fill_cost_paid = cost


class PitManager:
    """
    Tracks all pits in the environment and exposes helpers
    for checking passability and recording fills.
    """

    def __init__(self, pit_positions: List[Pos]) -> None:
        self._pits: Dict[Pos, PitState] = {
            pos: PitState(position=pos) for pos in pit_positions
        }

    def is_pit(self, pos: Pos) -> bool:
        return pos in self._pits

    def is_filled(self, pos: Pos) -> bool:
        state = self._pits.get(pos)
        return state is not None and state.filled

    def is_passable(self, pos: Pos) -> bool:
        """A pit is passable only if it has been filled."""
        return not self.is_pit(pos) or self.is_filled(pos)

    def fill_pit(self, pos: Pos, sandbag_id: int, cost: float) -> bool:
        """
        Fill a pit with a sandbag.
        Returns True on success, False if already filled or not a pit.
        """
        state = self._pits.get(pos)
        if state is None or state.filled:
            return False
        state.fill(sandbag_id, cost)
        return True

    def unfilled_pits(self) -> List[Pos]:
        return [pos for pos, s in self._pits.items() if not s.filled]

    def all_pits(self) -> List[Pos]:
        return list(self._pits.keys())

    def total_fill_cost(self) -> float:
        return sum(s.fill_cost_paid for s in self._pits.values())
