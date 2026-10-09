"""Abstract push model: requests served one at a time at their exact shortest-path distance.

There are no moving robots, no congestion and no walking cost. A configuration is the set of obstacles (cell, kind)
sorted by cell. Request j costs the BFS distance from its origin to its destination with the obstacles closed, or
U = 4 * (H + W) if an endpoint is blocked or the destination is unreachable (endpoints are not treated as open).
An action pushes one obstacle `steps` cells in a straight line at cost fee + steps * kappa * weight(kind). Whether
a robot could reach the approach cell is not checked: this is a relaxation, stated in the theory.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from src.doi.abstract.core import round_robin
from src.doi.kinds import weight
from src.doi.paths import bfs_dist_map, dream_path, passable_fn
from src.doi.pushplan import DIRS, _dead
from src.doi.rng import stream
from src.doi.scenarios import scenario_from_ascii
from src.environment.grid import Grid

Pos = Tuple[int, int]
Config = Tuple[Tuple[Pos, str], ...]


@dataclass(frozen=True)
class Action:
    obstacle: Pos
    kind: str
    direction: Pos
    steps: int
    landing: Pos
    cost: float

    def key(self) -> Tuple:
        return (self.obstacle, self.direction, self.steps)


@dataclass
class Instance:
    grid: Grid
    obstacles: Dict[Pos, str]
    requests: List[Tuple[Pos, Pos]]
    agents: List[int]                 # agents[j] serves request j
    kappa: float = 1.0
    fee: float = 1.0
    push_max: int = 6

    def unreachable(self) -> float:
        return float(4 * (self.grid.height + self.grid.width))

    def initial(self) -> Config:
        return tuple(sorted(self.obstacles.items()))


def instance_from_ascii(rows: List[str], requests: List[Tuple[Pos, Pos]], agents: Optional[List[int]] = None,
                        kappa: float = 1.0, fee: float = 1.0, push_max: int = 6) -> Instance:
    s = scenario_from_ascii(rows, [], [])
    return Instance(s.grid, dict(s.obstacles), list(requests),
                    list(agents) if agents is not None else [0] * len(requests), kappa, fee, push_max)


class Model:
    """Serving costs and legal actions of one instance, with caches."""

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        self.grid = inst.grid
        self.U = inst.unreachable()
        self._dist: Dict[Tuple[FrozenSet[Pos], Pos], Dict[Pos, int]] = {}
        self._actions: Dict[Config, List[Action]] = {}
        self._free_static = passable_fn(inst.grid)

    def blocked(self, x: Config) -> FrozenSet[Pos]:
        return frozenset(cell for cell, _ in x)

    def serve(self, x: Config, j: int) -> float:
        origin, dest = self.inst.requests[j]
        blocked = self.blocked(x)
        if origin in blocked or dest in blocked:
            return self.U
        key = (blocked, origin)
        dist = self._dist.get(key)
        if dist is None:
            dist = bfs_dist_map(passable_fn(self.grid, closed=blocked), origin, self.grid.height, self.grid.width)
            self._dist[key] = dist
        return float(dist[dest]) if dest in dist else self.U

    def legal_actions(self, x: Config) -> List[Action]:
        hit = self._actions.get(x)
        if hit is not None:
            return hit
        free = passable_fn(self.grid, closed=self.blocked(x))
        out: List[Action] = []
        for obstacle, kind in x:
            for d in DIRS:
                if not free((obstacle[0] - d[0], obstacle[1] - d[1])):
                    continue
                for k in range(1, self.inst.push_max + 1):
                    landing = (obstacle[0] + k * d[0], obstacle[1] + k * d[1])
                    if not free(landing):
                        break
                    if _dead(self._free_static, landing):
                        continue
                    out.append(Action(obstacle, kind, d, k, landing,
                                      self.inst.fee + k * self.inst.kappa * weight(kind)))
        out.sort(key=Action.key)
        self._actions[x] = out
        return out

    def apply(self, x: Config, a: Action) -> Config:
        return tuple(sorted([(c, k) for c, k in x if c != a.obstacle] + [(a.landing, a.kind)]))

    def bundle_cells(self, x: Config, j: int) -> FrozenSet[Pos]:
        origin, dest = self.inst.requests[j]
        return dream_path(self.grid, sorted(self.blocked(x)), origin, dest, self.U).bundle


G1_ROWS = ["..#....",
           "..L....",
           "..#....",
           "..#....",
           "......."]
G1_REQUEST = ((1, 0), (1, 6))


def g1(m: int, n_agents: int = 1, fee: float = 1.0) -> Instance:
    return instance_from_ascii(G1_ROWS, [G1_REQUEST] * m, round_robin(m, n_agents), kappa=1.0, fee=fee, push_max=6)


def random_small_instance(seed: int, n_obstacles: int, T: int, n_agents: int) -> Instance:
    if not 1 <= n_obstacles <= 3:
        raise ValueError("n_obstacles must be 1, 2 or 3")
    H, W = 7, 9
    gaps = (1, 3, 5)[:n_obstacles]
    rows = []
    for r in range(H):
        row = ["."] * W
        if r <= 5:
            row[4] = "L" if r in gaps else "#"
        rows.append("".join(row))
    west = sorted((r, c) for r in range(H) for c in range(0, 4))
    east = sorted((r, c) for r in range(H) for c in range(5, W))
    rng = stream(seed, "abstract-req")
    requests: List[Tuple[Pos, Pos]] = []
    for j in range(T):
        src, dst = (west, east) if j % 2 == 0 else (east, west)
        requests.append((rng.choice(src), rng.choice(dst)))
    return instance_from_ascii(rows, requests, round_robin(T, n_agents), kappa=1.0, fee=1.0, push_max=4)
