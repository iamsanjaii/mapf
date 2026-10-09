"""Carry plans: pick an obstacle up, haul it to a slot or a pit, and walk on to the goal.

Two modes. `carry` lifts an obstacle on the robot's own route and puts it on a rack or dump slot that accepts its
kind, so the obstacle is gone from the map and parked off every route. `fill` lifts any debris and drops it into a
pit on the route, which removes both the debris and the pit. A plan is judged like a push plan: by what it costs
this robot for this task (`total`) and by what it does to the believed map (`before` to `after`).

Per obstacle (and per pit, for fills) only the cheapest plan is kept: the evidence for a plan depends only on
`before` and `after`, which are the same whichever slot is used.
"""
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from src.doi.kinds import KINDS, can_carry, can_fill, weight
from src.doi.paths import passable_fn
from src.environment.grid import Grid

Pos = Tuple[int, int]
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class CarryPlan:
    obstacle: Pos             # what is lifted
    kind: str
    mode: str                 # "carry" or "fill"
    approach: Pos             # where the robot stands to lift it
    target: Pos               # slot cell, or pit cell for a fill
    access: Pos               # where the robot stands to drop it
    walk_in: float            # robot to the approach cell
    haul: float               # steps walked loaded
    walk_on: float            # access cell to the goal, with the obstacle (and pit) gone
    pick_cost: float
    haul_cost: float          # haul * kappa_c * weight
    drop_cost: float
    before: FrozenSet[Pos]    # believed blocked cells now
    after: FrozenSet[Pos]     # believed blocked cells once it is done

    @property
    def total(self) -> float:
        return self.walk_in + self.pick_cost + self.haul_cost + self.drop_cost + self.walk_on

    @property
    def landing(self) -> Pos:          # where the obstacle ends up (read by triggers and captions)
        return self.target

    @property
    def steps(self) -> int:
        return int(self.haul)

    @property
    def direction(self) -> Pos:
        return (0, 0)

    @property
    def key(self) -> tuple:
        return (self.mode, self.obstacle, self.target)


def carry_plans(grid: Grid, distance: Callable[[Pos, Pos, FrozenSet[Pos]], float], blocked: FrozenSet[Pos],
                carry_sources: Iterable[Pos], fill_sources: Iterable[Pos], pits: FrozenSet[Pos],
                kind_of: Callable[[Pos], Optional[str]], pos: Pos, goal: Pos, kappa_c: float, pick_fee: float,
                drop_fee: float, unreachable: float, slots: Dict[Pos, str], full: FrozenSet[Pos],
                skip: FrozenSet[Pos] = frozenset()) -> List[CarryPlan]:
    """Cheapest plan per carried obstacle and per (debris, pit) pair, cheapest first.

    `carry_sources` are obstacles on the robot's own route; `fill_sources` are any obstacles it may lift to fill a
    pit in `pits`. `slots` is every rack and dump cell, `full` those believed occupied."""
    free = passable_fn(grid, closed=blocked)
    best: Dict[tuple, Tuple[tuple, CarryPlan]] = {}

    def offer(plan: CarryPlan) -> None:
        rank = (plan.total, plan.obstacle, plan.target, plan.approach, plan.access)
        if plan.key not in best or rank < best[plan.key][0]:
            best[plan.key] = (rank, plan)

    def neighbours(cell: Pos) -> List[Pos]:
        return [(cell[0] + dr, cell[1] + dc) for dr, dc in DIRS]

    def plans_for(obstacle: Pos, targets: List[Tuple[Pos, str]]) -> None:
        kind = kind_of(obstacle) or "pallet"
        if not can_carry(kind):
            return
        lifted = blocked - {obstacle}
        free_lifted = passable_fn(grid, closed=lifted)
        for approach in neighbours(obstacle):
            if not free(approach):
                continue
            walk_in = distance(pos, approach, blocked)
            if walk_in >= unreachable:
                continue
            for target, mode in targets:
                after = lifted - {target} if mode == "fill" else lifted
                for access in neighbours(target):
                    if not free_lifted(access):
                        continue
                    haul = distance(approach, access, lifted)
                    if haul >= unreachable:
                        continue
                    walk_on = distance(access, goal, after)
                    offer(CarryPlan(obstacle, kind, mode, approach, target, access, walk_in, haul, walk_on,
                                    pick_fee, haul * kappa_c * weight(kind), drop_fee, blocked, frozenset(after)))

    for obstacle in sorted(set(carry_sources) - skip):
        kind = kind_of(obstacle) or "pallet"
        if not can_carry(kind):
            continue
        targets = [(cell, "carry") for cell, slot_type in sorted(slots.items())
                   if slot_type in KINDS[kind].slots and cell not in full]
        plans_for(obstacle, targets)
    for obstacle in sorted(set(fill_sources) - skip):
        kind = kind_of(obstacle) or "pallet"
        if can_fill(kind):
            plans_for(obstacle, [(pit, "fill") for pit in sorted(pits) if pit != obstacle])

    carried: Dict[Pos, Tuple[tuple, CarryPlan]] = {}
    for (mode, obstacle, _target), (rank, plan) in best.items():          # cheapest slot per carried obstacle
        if mode == "carry" and (obstacle not in carried or rank < carried[obstacle][0]):
            carried[obstacle] = (rank, plan)
    keep = [v for v in carried.values()] + [v for (m, _o, _t), v in best.items() if m == "fill"]
    return [p for _, p in sorted(keep, key=lambda x: x[0])]
