"""
src/mapf/reservation.py
~~~~~~~~~~~~~~~~~~~~~~~~
Space-time reservation table for prioritised MAPF.

Tracks which robot has reserved which (row, col) at each timestep.
Used by PrioritizedPlanner to make lower-priority robots avoid
already-planned paths.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple


SpaceTimeKey = Tuple[int, int, int]   # (row, col, timestep)
Pos = Tuple[int, int]


class ReservationTable:
    """
    A sparse map from (row, col, t) → robot_id.

    Robots reserve their cells at each timestep.
    When a robot's path is complete, we hold its goal position
    indefinitely to prevent other robots from walking through it.
    """

    def __init__(self) -> None:
        self._table: Dict[SpaceTimeKey, int] = {}
        # After a robot reaches its goal, that cell is blocked at all t > arrival
        self._goal_locks: Dict[Pos, Tuple[int, int]] = {}   # pos → (robot_id, arrival_t)

    # ------------------------------------------------------------------
    # Reservation
    # ------------------------------------------------------------------

    def reserve_path(
        self,
        robot_id: int,
        path: list,
        start_t: int = 0,
    ) -> None:
        """
        Reserve all cells in *path* starting at *start_t*.
        Also reserve the goal indefinitely (goal lock).
        """
        for offset, pos in enumerate(path):
            t = start_t + offset
            self._table[(pos[0], pos[1], t)] = robot_id

        if path:
            goal = path[-1]
            arrival_t = start_t + len(path) - 1
            self._goal_locks[goal] = (robot_id, arrival_t)

    def clear_robot(self, robot_id: int) -> None:
        """Remove all reservations for a given robot."""
        keys_to_del = [k for k, v in self._table.items() if v == robot_id]
        for k in keys_to_del:
            del self._table[k]
        self._goal_locks = {
            pos: val
            for pos, val in self._goal_locks.items()
            if val[0] != robot_id
        }

    def clear_all(self) -> None:
        self._table.clear()
        self._goal_locks.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_reserved(self, pos: Pos, t: int, exclude_robot: int = -1) -> bool:
        """
        Return True if *pos* is reserved at time *t* by any robot other
        than *exclude_robot*.
        """
        key: SpaceTimeKey = (pos[0], pos[1], t)
        occupant = self._table.get(key)
        if occupant is not None and occupant != exclude_robot:
            return True

        # Check goal locks: if the cell is locked after some arrival time
        lock = self._goal_locks.get(pos)
        if lock is not None:
            locking_robot, arrival = lock
            if locking_robot != exclude_robot and t >= arrival:
                return True

        return False

    def forbidden_cells_at(self, t: int, exclude_robot: int = -1) -> Set[Pos]:
        """
        Return all positions reserved at timestep *t* by robots other
        than *exclude_robot*.  Used to construct the *forbidden* set for A*.
        """
        forbidden: Set[Pos] = set()

        for (r, c, ts), rid in self._table.items():
            if ts == t and rid != exclude_robot:
                forbidden.add((r, c))

        for pos, (locking_robot, arrival) in self._goal_locks.items():
            if locking_robot != exclude_robot and t >= arrival:
                forbidden.add(pos)

        return forbidden

    def get_occupant(self, pos: Pos, t: int) -> Optional[int]:
        return self._table.get((pos[0], pos[1], t))

    def __len__(self) -> int:
        return len(self._table)
