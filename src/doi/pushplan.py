"""Push plans: walk to an approach cell, push one obstacle k steps in a straight line, then walk on to the goal.

A plan is judged by what it costs this robot for this task (`total`), and by what it does to the believed map:
the obstacle leaves its cell and sits on `landing`, so `after` differs from `before` by those two cells.
Only obstacles on the robot's own open route are considered, and only straight runs: no detour is made for
anyone else's benefit.
"""
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

from src.doi.kinds import weight
from src.doi.paths import passable_fn
from src.environment.grid import Grid

Pos = Tuple[int, int]
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def _dead(free_static: Callable[[Pos], bool], cell: Pos) -> bool:
    """True if an obstacle parked on `cell` could never be pushed again: no axis has free floor on both sides."""
    r, c = cell
    return not ((free_static((r - 1, c)) and free_static((r + 1, c))) or
                (free_static((r, c - 1)) and free_static((r, c + 1))))


@dataclass(frozen=True)
class PushPlan:
    obstacle: Pos
    kind: str
    direction: Pos
    approach: Pos
    steps: int
    landing: Pos
    end: Pos                  # where the robot stands after the last push
    walk_in: float            # distance from the robot to the approach cell
    push_cost: float          # steps * kappa * weight + fee
    walk_on: float            # distance from `end` to the goal with the obstacle on `landing`
    before: FrozenSet[Pos]    # believed blocked cells now
    after: FrozenSet[Pos]     # believed blocked cells once the push is done

    def landing_after(self, steps_done: int) -> Pos:
        """Where the obstacle sits once `steps_done` pushes of this plan have been made."""
        return (self.obstacle[0] + steps_done * self.direction[0], self.obstacle[1] + steps_done * self.direction[1])

    @property
    def total(self) -> float:
        return self.walk_in + self.push_cost + self.walk_on


def candidate_plans(grid: Grid, distance: Callable[[Pos, Pos, FrozenSet[Pos]], float], blocked: FrozenSet[Pos],
                    candidates: Iterable[Pos], kind_of: Callable[[Pos], Optional[str]], pos: Pos, goal: Pos,
                    kappa: float, fee: float, max_steps: int, unreachable: float,
                    skip: FrozenSet[Pos] = frozenset(), dead_end_guard: bool = True) -> List[PushPlan]:
    """For each obstacle and each of the four sides, the cheapest straight push (for pos -> goal), cheapest first.

    The policy then picks among these by what each does to the whole fleet, not only by what it costs this robot."""
    free = passable_fn(grid, closed=blocked)
    free_static = passable_fn(grid)
    plans: List[Tuple[tuple, PushPlan]] = []
    for obstacle in sorted(set(candidates) - skip):
        kind = kind_of(obstacle) or "pallet"
        for d in DIRS:
            approach = (obstacle[0] - d[0], obstacle[1] - d[1])
            if not free(approach):
                continue
            walk_in = distance(pos, approach, blocked)
            if walk_in >= unreachable:
                continue
            best: Optional[Tuple[tuple, PushPlan]] = None
            for k in range(1, max_steps + 1):
                landing = (obstacle[0] + k * d[0], obstacle[1] + k * d[1])
                if not free(landing):                       # wall, another obstacle or off the map: no room
                    break
                if landing == goal or (dead_end_guard and _dead(free_static, landing)):
                    continue
                end = (obstacle[0] + (k - 1) * d[0], obstacle[1] + (k - 1) * d[1])
                after = (blocked - {obstacle}) | {landing}
                walk_on = distance(end, goal, after)
                total = walk_in + k * kappa * weight(kind) + fee + walk_on
                key = (total, obstacle, d, k)
                if best is None or key < best[0]:
                    best = (key, PushPlan(obstacle, kind, d, approach, k, landing, end, walk_in,
                                          k * kappa * weight(kind) + fee, walk_on, blocked, frozenset(after)))
            if best is not None:
                plans.append(best)
    return [p for _, p in sorted(plans, key=lambda x: x[0])]


def best_push_plan(*args, **kwargs) -> Optional[PushPlan]:
    """The single cheapest plan for this robot (see candidate_plans)."""
    plans = candidate_plans(*args, **kwargs)
    return plans[0] if plans else None


def all_step_plans(grid: Grid, distance: Callable[[Pos, Pos, FrozenSet[Pos]], float], blocked: FrozenSet[Pos],
                   candidates: Iterable[Pos], kind_of: Callable[[Pos], Optional[str]], pos: Pos, goal: Pos,
                   kappa: float, fee: float, max_steps: int, unreachable: float,
                   skip: FrozenSet[Pos] = frozenset()) -> List[PushPlan]:
    """Like candidate_plans but keeps every push length, not only the cheapest per side.

    First legs of a two-step plan need this: the cheapest single push can leave the obstacle just outside a doorway,
    which only the second push (or a longer first push) clears."""
    free = passable_fn(grid, closed=blocked)
    free_static = passable_fn(grid)
    out: List[PushPlan] = []
    for obstacle in sorted(set(candidates) - skip):
        kind = kind_of(obstacle) or "pallet"
        for d in DIRS:
            approach = (obstacle[0] - d[0], obstacle[1] - d[1])
            if not free(approach):
                continue
            walk_in = distance(pos, approach, blocked)
            if walk_in >= unreachable:
                continue
            for k in range(1, max_steps + 1):
                landing = (obstacle[0] + k * d[0], obstacle[1] + k * d[1])
                if not free(landing):
                    break
                if landing == goal or _dead(free_static, landing):
                    continue
                end = (obstacle[0] + (k - 1) * d[0], obstacle[1] + (k - 1) * d[1])
                after = (blocked - {obstacle}) | {landing}
                out.append(PushPlan(obstacle, kind, d, approach, k, landing, end, walk_in,
                                    k * kappa * weight(kind) + fee, distance(end, goal, after), blocked,
                                    frozenset(after)))
    return sorted(out, key=lambda p: (p.total, p.obstacle, p.direction, p.steps))


@dataclass(frozen=True)
class BundlePlan:
    """Two pushes in sequence, judged as one plan; the robot carries out the first leg (see agent.decide)."""
    first: PushPlan
    second: PushPlan

    @property
    def obstacle(self) -> Pos:
        return self.first.obstacle

    @property
    def kind(self) -> str:
        return self.first.kind

    @property
    def landing(self) -> Pos:
        return self.first.landing

    @property
    def steps(self) -> int:
        return self.first.steps

    @property
    def before(self) -> FrozenSet[Pos]:
        return self.first.before

    @property
    def after(self) -> FrozenSet[Pos]:
        return self.second.after

    @property
    def total(self) -> float:
        return (self.first.walk_in + self.first.push_cost + self.second.walk_in + self.second.push_cost
                + self.second.walk_on)


def bundle_plans(grid: Grid, distance: Callable[[Pos, Pos, FrozenSet[Pos]], float], blocked: FrozenSet[Pos],
                 firsts: List[PushPlan], eligible_after: Callable[[PushPlan], List[Pos]],
                 kind_of: Callable[[Pos], Optional[str]], goal: Pos, kappa: float, fee: float, max_steps: int,
                 unreachable: float) -> List[BundlePlan]:
    """Every first plan followed by every plan for the next obstacle on the route that its landing leaves behind."""
    out: List[BundlePlan] = []
    for p1 in firsts:
        def kind_of2(c: Pos, p1: PushPlan = p1) -> Optional[str]:
            return p1.kind if c == p1.landing else kind_of(c)
        for p2 in candidate_plans(grid, distance, p1.after, eligible_after(p1), kind_of2, p1.end, goal, kappa, fee,
                                  max_steps, unreachable):
            out.append(BundlePlan(p1, p2))
    return sorted(out, key=lambda b: (b.total, b.first.obstacle, b.first.direction, b.first.steps,
                                      b.second.obstacle, b.second.direction, b.second.steps))
