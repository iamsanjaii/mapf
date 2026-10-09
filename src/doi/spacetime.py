"""Local space-time A* with vertex and swap reservations, used by agents for collision-aware motion."""
import heapq
from typing import Dict, List, Optional, Set, Tuple

from src.doi.paths import DIRS, PassFn

Pos = Tuple[int, int]


def plan_spacetime(passable: PassFn, start: Pos, goal: Pos, t0: int,
                   reserved: Set[Tuple[Pos, int]], h: Dict[Pos, int],
                   max_len: int, window: Optional[int] = None) -> Optional[List[Pos]]:
    """path[k] is the cell at tick t0 + k; waiting is a repeated cell. None when the goal is out of reach.

    With `window = P` the search with reservations stops at depth P and the path continues down the BFS
    gradient `h` to the goal without reservations (the caller replans before the tail is reached).
    """
    if start == goal:
        return [start]
    if start not in h:
        return None
    counter = 0
    heap = [(h[start], 0, counter, start)]
    parent: Dict[Tuple[Pos, int], Tuple[Pos, int]] = {}
    seen = {(start, 0)}
    while heap:
        _f, k, _c, cur = heapq.heappop(heap)
        if cur == goal or (window is not None and k >= window):
            path = [cur]
            node = (cur, k)
            while node in parent:
                node = parent[node]
                path.append(node[0])
            path = path[::-1]
            while path[-1] != goal:
                last = path[-1]
                step = min((n for n in ((last[0] + dr, last[1] + dc) for dr, dc in DIRS)
                            if n in h and h[n] == h[last] - 1), default=None)
                if step is None:
                    return None
                path.append(step)
            return path
        if k >= max_len:
            continue
        now, nxt = t0 + k, t0 + k + 1
        cur_taken = (cur, nxt) in reserved
        options = [(cur[0] + dr, cur[1] + dc) for dr, dc in DIRS]
        options = [n for n in options if passable(n) and n in h and (n, nxt) not in reserved
                   and not (cur_taken and (n, now) in reserved)]
        if not cur_taken:
            options.append(cur)
        for n in options:
            if (n, k + 1) in seen:
                continue
            seen.add((n, k + 1))
            parent[(n, k + 1)] = (cur, k)
            counter += 1
            heapq.heappush(heap, (k + 1 + h[n], k + 1, counter, n))
    return None
