"""
src/planning/heuristics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Heuristic functions for A* on a 2-D grid.

All heuristics share the same signature:
    heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float

Admissibility note
------------------
- For 4-directional movement, Manhattan is perfectly admissible (never overestimates).
- Euclidean is admissible for 4-directional movement (straight-line ≤ Manhattan path).
- Chebyshev is admissible for 8-directional movement but can overestimate for 4-dir.

The experiments explicitly record the movement model alongside the heuristic so that
results are interpreted correctly.
"""

from __future__ import annotations

import math
from typing import Tuple, Callable


Pos = Tuple[int, int]
Heuristic = Callable[[Pos, Pos], float]


def manhattan(a: Pos, b: Pos) -> float:
    """
    Manhattan (L1) distance.
    Optimal for 4-directional grid movement.

    |r1 - r2| + |c1 - c2|
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Pos, b: Pos) -> float:
    """
    Euclidean (L2) distance.
    Admissible; tends to expand fewer nodes than Manhattan when the path
    is relatively straight, but may not be optimal for purely grid movement.

    sqrt((r1-r2)^2 + (c1-c2)^2)
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: Pos, b: Pos) -> float:
    """
    Chebyshev (L-inf) distance.
    Admissible for 8-directional movement; may overestimate for 4-directional.

    max(|r1-r2|, |c1-c2|)
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def octile(a: Pos, b: Pos) -> float:
    """
    Octile distance — optimal for 8-directional movement.
    Included as an optional fourth heuristic for extended experiments.

    D * (dx + dy) + (sqrt(2) - 2*D) * min(dx, dy)  where D = 1
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


# Registry — maps string names to callables for config-driven selection
HEURISTICS: dict[str, Heuristic] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile":    octile,
}


def get_heuristic(name: str) -> Heuristic:
    """Return a heuristic callable by name (case-insensitive)."""
    name = name.lower()
    if name not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. "
            f"Choose from: {list(HEURISTICS.keys())}"
        )
    return HEURISTICS[name]
