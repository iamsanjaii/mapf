"""Path primitives: passability, BFS, canonical shortest path, and the dream-path rent."""
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.environment.grid import CellType, Grid

Pos = Tuple[int, int]
PassFn = Callable[[Pos], bool]

DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


_TABLES: Dict[bytes, Tuple[List[List[bool]], List[List[bool]]]] = {}


def _tables(grid: Grid) -> Tuple[List[List[bool]], List[List[bool]]]:
    key = (grid.height, grid.width, grid.array.tobytes())
    hit = _TABLES.get(key)
    if hit is None:
        free = [[bool(grid.is_passable(r, c)) for c in range(grid.width)] for r in range(grid.height)]
        pit = [[grid.get(r, c) == CellType.PIT for c in range(grid.width)] for r in range(grid.height)]
        if len(_TABLES) > 64:
            _TABLES.clear()
        hit = _TABLES[key] = (free, pit)
    return hit


def passable_fn(grid: Grid, open_pits: FrozenSet[Pos], closed: FrozenSet[Pos] = frozenset()) -> PassFn:
    free, pit = _tables(grid)
    h, w = grid.height, grid.width

    def passable(p: Pos) -> bool:
        r, c = p
        if not (0 <= r < h and 0 <= c < w) or p in closed:
            return False
        return free[r][c] or (pit[r][c] and p in open_pits)
    return passable


def bfs_dist_map(passable: PassFn, source: Pos, height: int, width: int) -> Dict[Pos, int]:
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for dr, dc in DIRS:
            n = (cur[0] + dr, cur[1] + dc)
            if n in dist or not (0 <= n[0] < height and 0 <= n[1] < width) or not passable(n):
                continue
            dist[n] = dist[cur] + 1
            queue.append(n)
    return dist


def shortest_path(passable: PassFn, start: Pos, goal: Pos, height: int, width: int,
                  pit_cells: FrozenSet[Pos] = frozenset()) -> Optional[List[Pos]]:
    """A* over lexicographic (steps, pits_used); the insertion counter is the canonical tie-break."""
    if not passable(start) or not passable(goal):
        return None
    if start == goal:
        return [start]

    def h(p: Pos) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    best: Dict[Pos, Tuple[int, int]] = {start: (0, 0)}
    parent: Dict[Pos, Pos] = {}
    counter = 0
    heap = [(h(start), 0, counter, start, 0)]
    while heap:
        _f, pits_used, _c, cur, g = heapq.heappop(heap)
        if best[cur] != (g, pits_used):
            continue
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        for dr, dc in DIRS:
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < height and 0 <= n[1] < width) or not passable(n):
                continue
            cand = (g + 1, pits_used + (1 if n in pit_cells else 0))
            if n in best and best[n] <= cand:
                continue
            best[n] = cand
            parent[n] = cur
            counter += 1
            heapq.heappush(heap, (cand[0] + h(n), cand[1], counter, n, cand[0]))
    return None


@dataclass(frozen=True)
class DreamResult:
    d_block: int
    d_open: int
    bundle: FrozenSet[Pos]
    rent: int
    path_open: Optional[Tuple[Pos, ...]]


def dream_path(grid: Grid, pits: Sequence[Pos], filled: FrozenSet[Pos], start: Pos, goal: Pos,
               unreachable: float, hard_blocked: FrozenSet[Pos] = frozenset()) -> DreamResult:
    h, w = grid.height, grid.width
    unfilled = frozenset(pits) - filled
    blocked = shortest_path(passable_fn(grid, filled, closed=unfilled | hard_blocked), start, goal, h, w)
    d_block = len(blocked) - 1 if blocked is not None else unreachable
    opened = shortest_path(passable_fn(grid, frozenset(pits), closed=hard_blocked), start, goal, h, w,
                           pit_cells=unfilled)
    if opened is None:
        return DreamResult(d_block, unreachable, frozenset(), 0, None)
    bundle = frozenset(c for c in opened if c in unfilled)
    d_open = len(opened) - 1
    rent = max(0, d_block - d_open) if bundle else 0
    return DreamResult(d_block, d_open, bundle, rent, tuple(opened))


def single_pit_rents(grid: Grid, pits: Sequence[Pos], filled: FrozenSet[Pos], start: Pos, goal: Pos,
                     unreachable: float, hard_blocked: FrozenSet[Pos] = frozenset()) -> Dict[Pos, int]:
    h, w = grid.height, grid.width
    unfilled = frozenset(pits) - filled

    def length(open_pits: FrozenSet[Pos], closed: FrozenSet[Pos]) -> float:
        path = shortest_path(passable_fn(grid, open_pits, closed=closed | hard_blocked), start, goal, h, w)
        return len(path) - 1 if path is not None else unreachable

    d_block = length(filled, unfilled)
    rents: Dict[Pos, int] = {}
    for p in sorted(unfilled):
        rent = max(0, d_block - length(filled | {p}, unfilled - {p}))
        if rent > 0:
            rents[p] = rent
    return rents
