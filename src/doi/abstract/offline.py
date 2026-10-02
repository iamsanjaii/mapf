"""Offline benchmarks for the abstract model: the exact optimum (layered Dijkstra) and the vanish lower bound."""
import heapq
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.doi.abstract.instance import Action, Config, Instance, Model
from src.doi.kinds import weight

MAX_STATES = 20000
MAX_LB_OBSTACLES = 12


class StateSpaceTooLarge(Exception):
    """More configurations are reachable than the caller allowed."""


@dataclass(frozen=True)
class OptResult:
    cost: float
    schedule: Tuple[Tuple[int, Action], ...]   # (request index the action is applied before, action), in order
    n_states: int


def reachable_configs(model: Model, start: Config, max_states: int = MAX_STATES) -> List[Config]:
    """BFS over legal actions from start, in discovery order with start first."""
    seen = {start}
    order = [start]
    head = 0
    while head < len(order):
        x = order[head]
        head += 1
        for a in model.legal_actions(x):
            y = model.apply(x, a)
            if y in seen:
                continue
            seen.add(y)
            order.append(y)
            if len(order) > max_states:
                raise StateSpaceTooLarge(f"more than {max_states} configurations")
    return order


def exact_opt(inst: Instance, max_states: int = MAX_STATES) -> OptResult:
    """Optimum over all push schedules: Dijkstra over (requests served, configuration)."""
    model = Model(inst)
    start = inst.initial()
    configs = reachable_configs(model, start, max_states)
    index = {x: k for k, x in enumerate(configs)}
    succ = [[(index[model.apply(x, a)], a) for a in model.legal_actions(x)] for x in configs]
    T = len(inst.requests)
    source = (0, index[start])
    dist: Dict[Tuple[int, int], float] = {source: 0.0}
    pred: Dict[Tuple[int, int], Tuple[Tuple[int, int], Optional[Action]]] = {}
    done = set()
    heap = [(0.0, 0, index[start])]
    while heap:
        d, i, k = heapq.heappop(heap)
        node = (i, k)
        if node in done:
            continue
        done.add(node)
        if i == T:
            schedule: List[Tuple[int, Action]] = []
            cur = node
            while cur != source:
                prev, a = pred[cur]
                if a is not None:
                    schedule.append((cur[0], a))
                cur = prev
            return OptResult(d, tuple(reversed(schedule)), len(configs))
        edges: List[Tuple[Tuple[int, int], float, Optional[Action]]] = [((i, k2), a.cost, a) for k2, a in succ[k]]
        if i < T:
            edges.append(((i + 1, k), model.serve(configs[k], i), None))
        for nxt, w, a in edges:
            nd = d + w
            if nxt not in done and (nxt not in dist or nd < dist[nxt]):
                dist[nxt] = nd
                pred[nxt] = (node, a)
                heapq.heappush(heap, (nd, nxt[0], nxt[1]))
    raise RuntimeError("no schedule serves every request")


def vanish_lower_bound(inst: Instance) -> float:
    """A valid lower bound on exact_opt: moved obstacles are charged one push and vanish.

    Every obstacle that a schedule ever moves costs at least one push (fee + kappa*w). The obstacles it never moves
    stay where they started, so at every request the blocked set contains initial \\ M, and serving costs can only
    rise as the blocked set grows."""
    start = inst.initial()
    if len(start) > MAX_LB_OBSTACLES:
        raise ValueError(f"more than {MAX_LB_OBSTACLES} obstacles")
    model = Model(inst)
    best = float("inf")
    for size in range(len(start) + 1):
        for moved in combinations(start, size):
            gone = {cell for cell, _ in moved}
            rest = tuple(o for o in start if o[0] not in gone)
            cost = sum(inst.fee + inst.kappa * weight(kind) for _, kind in moved)
            cost += sum(model.serve(rest, j) for j in range(len(inst.requests)))
            best = min(best, cost)
    return best
