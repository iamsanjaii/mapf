"""
src/mapf/conflict.py
~~~~~~~~~~~~~~~~~~~~~
Conflict detection for multi-robot paths.

Detects:
- VERTEX conflict   : two robots at the same cell at the same timestep
- EDGE   conflict   : two robots swap positions between t and t+1
- GOAL   conflict   : two robots share the same goal cell
- BLOCKING conflict : detected as a vertex conflict at the last planned cell
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


Pos = Tuple[int, int]
# Time-indexed path: list[Pos], index == timestep
Path = List[Pos]


class ConflictType(Enum):
    VERTEX = auto()
    EDGE = auto()
    GOAL = auto()
    BLOCKING = auto()


@dataclass
class Conflict:
    """A detected conflict between two robots."""
    conflict_type: ConflictType
    robots: Tuple[int, int]       # (robot_id_a, robot_id_b)
    timestep: int
    location: Pos                  # primary location of the conflict
    location_b: Optional[Pos] = None   # secondary (used for edge conflict)

    def __repr__(self) -> str:
        loc = f"{self.location}"
        if self.location_b:
            loc += f"↔{self.location_b}"
        return (
            f"Conflict({self.conflict_type.name}, "
            f"robots={self.robots}, t={self.timestep}, loc={loc})"
        )


class ConflictDetector:
    """
    Detects all types of conflicts from a dict of robot paths.
    """

    @staticmethod
    def detect_all(
        paths: Dict[int, Path],
    ) -> List[Conflict]:
        """
        Detect all conflicts across all pairs of robot paths.

        Parameters
        ----------
        paths : dict mapping robot_id → time-indexed path

        Returns
        -------
        list of Conflict objects (may contain duplicates from different pairs)
        """
        conflicts: List[Conflict] = []

        robot_ids = list(paths.keys())
        n = len(robot_ids)

        for i in range(n):
            for j in range(i + 1, n):
                rid_a = robot_ids[i]
                rid_b = robot_ids[j]
                path_a = paths[rid_a]
                path_b = paths[rid_b]

                conflicts.extend(
                    ConflictDetector._check_pair(rid_a, path_a, rid_b, path_b)
                )

        # Goal conflicts (same goal cell)
        goal_map: Dict[Pos, int] = {}
        for rid, path in paths.items():
            if path:
                goal = path[-1]
                if goal in goal_map:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.GOAL,
                            robots=(goal_map[goal], rid),
                            timestep=-1,
                            location=goal,
                        )
                    )
                else:
                    goal_map[goal] = rid

        return conflicts

    @staticmethod
    def _check_pair(
        rid_a: int, path_a: Path, rid_b: int, path_b: Path
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        max_t = max(len(path_a), len(path_b))

        def pos_at(path: Path, t: int) -> Pos:
            """Clamp to last position once path ends (robot stays at goal)."""
            if not path:
                return (-1, -1)
            return path[min(t, len(path) - 1)]

        for t in range(max_t):
            a = pos_at(path_a, t)
            b = pos_at(path_b, t)

            # Vertex conflict
            if a == b and a != (-1, -1):
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.VERTEX,
                        robots=(rid_a, rid_b),
                        timestep=t,
                        location=a,
                    )
                )

            # Edge conflict: robots swap positions between t and t+1
            if t + 1 < max_t:
                a_next = pos_at(path_a, t + 1)
                b_next = pos_at(path_b, t + 1)
                if a == b_next and b == a_next and a != b:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.EDGE,
                            robots=(rid_a, rid_b),
                            timestep=t,
                            location=a,
                            location_b=b,
                        )
                    )

        return conflicts

    @staticmethod
    def count_by_type(conflicts: List[Conflict]) -> Dict[str, int]:
        counts: Dict[str, int] = {ct.name: 0 for ct in ConflictType}
        for c in conflicts:
            counts[c.conflict_type.name] += 1
        return counts

    @staticmethod
    def severity_score(conflict: Conflict) -> int:
        """Simple severity score — higher is worse."""
        scores = {
            ConflictType.GOAL:     4,
            ConflictType.VERTEX:   3,
            ConflictType.EDGE:     2,
            ConflictType.BLOCKING: 1,
        }
        return scores.get(conflict.conflict_type, 0)
