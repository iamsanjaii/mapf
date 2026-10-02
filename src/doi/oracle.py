"""Hindsight-optimal fill set and the OPT lower bound (spec 2.6); static-pit scenarios only."""
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

from src.doi.config import SimConfig
from src.doi.paths import bfs_dist_map, dream_path, passable_fn
from src.doi.scenarios import Scenario

Pos = Tuple[int, int]
EXHAUSTIVE_LIMIT = 10


@dataclass(frozen=True)
class Hindsight:
    s_star: FrozenSet[Pos]
    lb_star: float
    lb_empty: float
    buy_lb: Dict[Pos, float]
    relevant: Tuple[Pos, ...]
    exhaustive: bool


def task_pairs(scenario: Scenario) -> List[Tuple[Pos, Pos]]:
    pairs: List[Tuple[Pos, Pos]] = []
    for start, goals in zip(scenario.starts, scenario.tasks):
        prev = start
        for goal in goals:
            pairs.append((prev, goal))
            prev = goal
    return pairs


def _require_static(scenario: Scenario) -> None:
    if scenario.family == "D":
        raise ValueError("no hindsight benchmark for incident scenarios (spec 2.6)")


def buy_lb(scenario: Scenario, cfg: SimConfig) -> Dict[Pos, float]:
    _require_static(scenario)
    grid, pits = scenario.grid, frozenset(scenario.pits)
    passable = passable_fn(grid, pits)
    depot_maps = [bfs_dist_map(passable, d, grid.height, grid.width)
                  for d, stock in sorted(scenario.depots.items()) if stock > 0]
    out: Dict[Pos, float] = {}
    for p in sorted(pits):
        best = float("inf")
        for dist in depot_maps:
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                n = (p[0] + dr, p[1] + dc)
                if n in dist:
                    best = min(best, cfg.fee + cfg.kappa * dist[n])
        out[p] = best
    return out


def hindsight(scenario: Scenario, cfg: SimConfig) -> Hindsight:
    _require_static(scenario)
    grid = scenario.grid
    h, w = grid.height, grid.width
    unreachable = cfg.unreachable_cost_for(h, w)
    buys = buy_lb(scenario, cfg)
    pairs = task_pairs(scenario)
    capacity = sum(scenario.depots.values())

    bundles = set()
    for origin, dest in pairs:
        bundles |= dream_path(grid, scenario.pits, frozenset(), origin, dest, unreachable).bundle
    relevant = tuple(sorted(bundles))

    cache: Dict[FrozenSet[Pos], float] = {}

    def lb(subset: FrozenSet[Pos]) -> float:
        if subset in cache:
            return cache[subset]
        if len(subset) > capacity:
            cache[subset] = float("inf")
            return cache[subset]
        passable = passable_fn(grid, subset)
        maps: Dict[Pos, Dict[Pos, int]] = {}
        total = 0.0
        for origin, dest in pairs:
            if origin not in maps:
                maps[origin] = bfs_dist_map(passable, origin, h, w)
            total += maps[origin].get(dest, unreachable)
        total += sum(buys[p] for p in subset)
        cache[subset] = total
        return total

    lb_empty = lb(frozenset())
    if len(relevant) <= EXHAUSTIVE_LIMIT:
        best, best_lb = frozenset(), lb_empty
        for size in range(1, len(relevant) + 1):
            for combo in combinations(relevant, size):
                value = lb(frozenset(combo))
                if value < best_lb:
                    best, best_lb = frozenset(combo), value
        return Hindsight(best, best_lb, lb_empty, buys, relevant, True)

    current, current_lb = frozenset(), lb_empty
    while True:
        gains = [(lb(current | {p}), p) for p in relevant if p not in current]
        gains = [(v, p) for v, p in gains if v < current_lb]
        if not gains:
            break
        current_lb, pick = min(gains)
        current = current | {pick}
    for p in sorted(current):
        for q in relevant:
            if q in current or p not in current:
                continue
            swapped = (current - {p}) | {q}
            if lb(swapped) < current_lb:
                current, current_lb = swapped, lb(swapped)
    return Hindsight(current, current_lb, lb_empty, buys, relevant, False)
