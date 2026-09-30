"""
src/environment/obstacles.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data classes for static obstacles, pits, and sandbags used in MAPF-RO.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Obstacle:
    """A static, impassable obstacle on the grid."""
    position: Tuple[int, int]
    removal_cost: float = float("inf")   # infinite → cannot be removed
    is_removable: bool = False

    def __hash__(self) -> int:
        return hash(self.position)


@dataclass
class Pit:
    """
    A pit cell.  A robot cannot traverse a pit unless a sandbag has been
    placed in it first, converting it to a FREE cell.
    """
    position: Tuple[int, int]
    filled: bool = False                 # True once a sandbag fills it
    fill_cost: float = 1.0              # additional cost to fill the pit

    def __hash__(self) -> int:
        return hash(self.position)


@dataclass
class Sandbag:
    """
    A movable resource.  Robots can spend extra cost to relocate a sandbag
    and use it to fill a pit.
    """
    id: int
    position: Tuple[int, int]
    move_cost_per_step: float = 4.0     # cost per cell moved
    in_pit: bool = False                # True once it has been used to fill a pit
    target_pit: Optional[Tuple[int, int]] = None

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class ObstacleRegistry:
    """Tracks all obstacle types in the current environment."""
    obstacles: dict = field(default_factory=dict)   # pos → Obstacle
    pits: dict = field(default_factory=dict)        # pos → Pit
    sandbags: dict = field(default_factory=dict)    # id  → Sandbag

    def add_obstacle(self, obs: Obstacle) -> None:
        self.obstacles[obs.position] = obs

    def add_pit(self, pit: Pit) -> None:
        self.pits[pit.position] = pit

    def add_sandbag(self, sb: Sandbag) -> None:
        self.sandbags[sb.id] = sb

    def get_pit(self, pos: Tuple[int, int]) -> Optional[Pit]:
        return self.pits.get(pos)

    def get_sandbag_at(self, pos: Tuple[int, int]) -> Optional[Sandbag]:
        for sb in self.sandbags.values():
            if sb.position == pos and not sb.in_pit:
                return sb
        return None
