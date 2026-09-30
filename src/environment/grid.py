"""
src/environment/grid.py
~~~~~~~~~~~~~~~~~~~~~~~
Core grid representation for the AMR-MAPF simulator.

CellType enum defines all valid cell states.
Grid wraps a numpy array and provides spatial queries,
neighbour enumeration and path validation helpers.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Tuple, Optional
import numpy as np


class CellType(IntEnum):
    """Integer codes stored in the numpy grid array."""
    FREE = 0
    OBSTACLE = 1
    PIT = 2
    SANDBAG = 3
    ROBOT = 4        # transient — set by renderer only
    GOAL = 5         # transient — set by renderer only


# Human-readable labels used by the visualiser
CELL_LABELS = {
    CellType.FREE:     ".",
    CellType.OBSTACLE: "X",
    CellType.PIT:      "P",
    CellType.SANDBAG:  "S",
    CellType.ROBOT:    "R",
    CellType.GOAL:     "G",
}

# 4-directional movement (N, S, E, W)
DIRECTIONS_4 = [(-1, 0), (1, 0), (0, 1), (0, -1)]

# 8-directional movement (adds diagonals)
DIRECTIONS_8 = DIRECTIONS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class Grid:
    """
    2-D grid environment backed by a numpy int8 array.

    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.
    movement : int
        4 or 8 — controls which neighbours are returned.
    """

    def __init__(self, width: int, height: int, movement: int = 4) -> None:
        self.width = width
        self.height = height
        self.movement = movement
        self._directions = DIRECTIONS_4 if movement == 4 else DIRECTIONS_8
        # dtype int8 keeps memory small for large grids
        self._cells: np.ndarray = np.zeros((height, width), dtype=np.int8)

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> CellType:
        return CellType(self._cells[row, col])

    def set(self, row: int, col: int, cell_type: CellType) -> None:
        self._cells[row, col] = int(cell_type)

    def __getitem__(self, pos: Tuple[int, int]) -> CellType:
        return self.get(pos[0], pos[1])

    def __setitem__(self, pos: Tuple[int, int], cell_type: CellType) -> None:
        self.set(pos[0], pos[1], cell_type)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        """Returns True if the cell can be occupied by a robot."""
        return (
            self.in_bounds(row, col)
            and self._cells[row, col] == CellType.FREE
        )

    def is_passable(self, row: int, col: int) -> bool:
        """
        Returns True if a robot can *enter* the cell.
        PIT requires a sandbag to be passable (handled by MAPF-RO logic).
        For the base planner, FREE and SANDBAG are passable.
        OBSTACLE and PIT block movement.
        """
        if not self.in_bounds(row, col):
            return False
        ct = CellType(self._cells[row, col])
        return ct in (CellType.FREE, CellType.SANDBAG)

    def neighbours(
        self, row: int, col: int, passable_only: bool = True
    ) -> List[Tuple[int, int]]:
        """Return valid neighbour cells."""
        result = []
        for dr, dc in self._directions:
            nr, nc = row + dr, col + dc
            if passable_only:
                if self.is_passable(nr, nc):
                    result.append((nr, nc))
            else:
                if self.in_bounds(nr, nc):
                    result.append((nr, nc))
        return result

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def obstacle_positions(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self._cells == CellType.OBSTACLE)
        return list(zip(rows.tolist(), cols.tolist()))

    def free_positions(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self._cells == CellType.FREE)
        return list(zip(rows.tolist(), cols.tolist()))

    def pit_positions(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self._cells == CellType.PIT)
        return list(zip(rows.tolist(), cols.tolist()))

    def sandbag_positions(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self._cells == CellType.SANDBAG)
        return list(zip(rows.tolist(), cols.tolist()))

    # ------------------------------------------------------------------
    # Grid copy / serialisation
    # ------------------------------------------------------------------

    def copy(self) -> "Grid":
        """Return a deep copy of this grid."""
        new_grid = Grid(self.width, self.height, self.movement)
        new_grid._cells = self._cells.copy()
        return new_grid

    @property
    def array(self) -> np.ndarray:
        """Read-only view of the underlying numpy array."""
        return self._cells

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rows = []
        for r in range(self.height):
            row_str = " ".join(
                CELL_LABELS[CellType(self._cells[r, c])]
                for c in range(self.width)
            )
            rows.append(row_str)
        return "\n".join(rows)
