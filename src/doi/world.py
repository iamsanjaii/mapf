"""Ground-truth world: physics, move arbitration, pushing, incidents, sensing and the override log."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

from src.doi.config import SimConfig
from src.doi.kinds import weight
from src.doi.scenarios import Scenario
from src.environment.grid import CellType

Pos = Tuple[int, int]
COUNTER_KEYS = ("moves", "waits", "push_steps", "push_rejected", "wrong_class_attempts")


@dataclass(frozen=True)
class Move:
    to: Pos


@dataclass(frozen=True)
class Wait:
    pass


@dataclass(frozen=True)
class Push:
    """Step into the neighbouring cell `to`, sliding the obstacle there one cell onward."""
    to: Pos


Action = Union[Move, Wait, Push]


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    reason: str
    pos: Pos
    blocked_ticks: int


@dataclass(frozen=True)
class Observation:
    robot_cells: FrozenSet[Pos]
    blocked_cells: FrozenSet[Pos]          # removable obstacles within sensing range
    kinds: Tuple[Tuple[Pos, str], ...]     # their kinds
    clearable: FrozenSet[Pos]              # those that are plainly ordinary obstacles a robot may push
    scanned: FrozenSet[Pos]                # cells seen to be free


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class World:
    def __init__(self, scenario: Scenario, cfg: SimConfig) -> None:
        self.scenario = scenario
        self.cfg = cfg
        self.grid = scenario.grid.copy()
        n = len(scenario.starts)
        self.pos: Dict[int, Pos] = {i: scenario.starts[i] for i in range(n)}
        self.blocked_ticks: Dict[int, int] = {i: 0 for i in range(n)}
        self.obstacles: Dict[Pos, str] = dict(scenario.obstacles)
        self.plain: Set[Pos] = set(self.obstacles)          # scenario obstacles; incident obstructions are not plain
        for c in self.obstacles:
            self.grid.set(c[0], c[1], CellType.OBSTACLE)
        self.appeared_at: Dict[int, int] = {}
        self.overrides = 0
        self.removals = 0                       # push runs
        self.push_cost = 0.0                    # sum of kappa * weight over push steps
        self.push_log: List[dict] = []          # one entry per run
        self._open_run: Dict[int, dict] = {}
        self.trajectory: Dict[int, List[Pos]] = {i: [self.pos[i]] for i in range(n)}
        self.obstacle_trace: List[Dict[Pos, str]] = [dict(self.obstacles)]
        self.tick_cost: List[float] = []
        self.counters: Dict[int, Dict[str, int]] = {i: {k: 0 for k in COUNTER_KEYS} for i in range(n)}
        self.incident_cls: Dict[Pos, str] = {c: inc.cls for inc in scenario.incidents for c in inc.cells}
        self._suppressed: Set[int] = set()

    def prefill(self, cells: Iterable[Pos]) -> None:
        """Remove obstacles at tick 0 (benchmark arms): the obstacle vanishes, it is not relocated."""
        gone = set(cells)
        for inc in self.scenario.incidents:
            if gone & set(inc.cells):
                self._suppressed.add(inc.oid)
        for c in sorted(gone & set(self.obstacles)):
            del self.obstacles[c]
            self.plain.discard(c)
            self.grid.set(c[0], c[1], CellType.FREE)
        self.obstacle_trace[0] = dict(self.obstacles)

    def begin_tick(self, t: int) -> None:
        occupied = set(self.pos.values())
        for inc in sorted(self.scenario.incidents, key=lambda i: i.oid):
            if inc.oid in self.appeared_at or inc.oid in self._suppressed or inc.appear_tick > t:
                continue
            if any(c in occupied or c in self.obstacles for c in inc.cells):
                continue
            for c in inc.cells:
                self.obstacles[c] = inc.kind
                self.grid.set(c[0], c[1], CellType.OBSTACLE)
            self.appeared_at[inc.oid] = t

    def active_ids(self) -> List[int]:
        return sorted(self.pos)

    def despawn(self, robot_id: int) -> None:
        self.pos.pop(robot_id, None)

    def observe(self, robot_id: int, r_sense: int) -> Observation:
        me = self.pos[robot_id]
        scanned = set()
        for r in range(max(0, me[0] - r_sense), min(self.grid.height, me[0] + r_sense + 1)):
            span = r_sense - abs(r - me[0])
            for c in range(max(0, me[1] - span), min(self.grid.width, me[1] + span + 1)):
                if self.grid.get(r, c) != CellType.OBSTACLE:
                    scanned.add((r, c))
        others = frozenset(p for j, p in self.pos.items() if j != robot_id and _manhattan(me, p) <= r_sense)
        near = sorted((c, k) for c, k in self.obstacles.items() if _manhattan(me, c) <= r_sense)
        return Observation(robot_cells=others, blocked_cells=frozenset(c for c, _ in near),
                           kinds=tuple(near), clearable=frozenset(c for c, _ in near if c in self.plain),
                           scanned=frozenset(scanned))

    def _key(self, i: int) -> Tuple[int, int]:
        return (-self.blocked_ticks[i], i)

    def _arbitrate(self, movers: Dict[int, Pos]) -> Dict[int, str]:
        """Returns {robot: reason} for every mover removed by vertex/occupied/swap/cycle rules."""
        cand = dict(movers)
        removed: Dict[int, str] = {}
        occupant = {p: j for j, p in self.pos.items()}

        def drop(i: int, reason: str) -> None:
            removed[i] = reason
            del cand[i]

        changed = True
        while changed:
            changed = False
            by_target: Dict[Pos, List[int]] = {}
            for i, to in cand.items():
                by_target.setdefault(to, []).append(i)
            for to in sorted(by_target):
                group = by_target[to]
                if len(group) > 1:
                    keep = min(group, key=self._key)
                    for i in group:
                        if i != keep:
                            drop(i, "vertex")
                            changed = True
            for i in sorted(cand):
                j = occupant.get(cand[i])
                if j is not None and j not in cand:
                    drop(i, "occupied")
                    changed = True
            for i in sorted(cand):
                if i not in cand:
                    continue
                j = occupant.get(cand[i])
                if j is not None and j in cand and cand[j] == self.pos[i]:
                    drop(i, "swap")
                    drop(j, "swap")
                    changed = True
            state: Dict[int, int] = {}
            for start in sorted(cand):
                if start in state:
                    continue
                path, cur = [], start
                while cur is not None and cur in cand and cur not in state:
                    state[cur] = 1
                    path.append(cur)
                    cur = occupant.get(cand[cur])
                if cur is not None and cur in cand and state.get(cur) == 1 and cur in path:
                    cycle = path[path.index(cur):]
                    if len(cycle) >= 3:
                        for i in cycle:
                            drop(i, "cycle")
                        changed = True
                for i in path:
                    state[i] = 2
        return removed

    def apply_actions(self, t: int, actions: Dict[int, Action]) -> Dict[int, ActionResult]:
        old_pos = dict(self.pos)
        reasons: Dict[int, str] = {}
        moved_ok: Set[int] = set()
        blocked_now: Set[int] = set()
        cost = 0.0

        for i in sorted((i for i, a in actions.items() if i in self.pos and isinstance(a, Push)), key=self._key):
            reason = self._push(i, actions[i].to, t)
            if reason:
                reasons[i] = reason
                blocked_now.add(i)
                self.counters[i]["push_rejected"] += 1
            else:
                moved_ok.add(i)
                kind_w = self._open_run[i]["weight"]
                step = self.cfg.kappa * kind_w
                cost += step + (self.cfg.fee if self._open_run[i]["steps"] == 1 else 0.0)
                self.push_cost += step

        legal: Dict[int, Pos] = {}
        for i, a in sorted(actions.items()):
            if i not in self.pos or not isinstance(a, Move):
                continue
            to = a.to
            if (_manhattan(self.pos[i], to) != 1 or not self.grid.in_bounds(to[0], to[1])
                    or not self.grid.is_passable(to[0], to[1])):
                reasons[i] = "wall"
                blocked_now.add(i)
            else:
                legal[i] = to
        removed = self._arbitrate(legal)
        for i, reason in removed.items():
            reasons[i] = reason
            blocked_now.add(i)
            self.overrides += 1
        for i, to in legal.items():
            if i in removed:
                continue
            self.pos[i] = to
            moved_ok.add(i)
            self.counters[i]["moves"] += 1
            cost += 1.0

        for i in self.pos:
            self.blocked_ticks[i] = self.blocked_ticks[i] + 1 if i in blocked_now else 0
        for i in actions:
            if i in self.pos and i not in moved_ok:
                self.counters[i]["waits"] += 1
                cost += 1.0
        for i, run in list(self._open_run.items()):       # a run ends when its robot does anything but push on
            if run["end_tick"] != t:
                del self._open_run[i]

        if self.cfg.debug_checks:
            self._check(old_pos)
        results = {}
        for i in actions:
            if i not in self.pos:
                continue
            reason = reasons.get(i, "")
            results[i] = ActionResult(reason == "", reason, self.pos[i], self.blocked_ticks[i])
        for i in self.pos:
            self.trajectory[i].append(self.pos[i])
        self.obstacle_trace.append(dict(self.obstacles))
        self.tick_cost.append(cost)
        return results

    def _push(self, i: int, to: Pos, t: int) -> str:
        """Slide the obstacle at `to` one cell further from robot i and move the robot in. '' on success."""
        p = self.pos[i]
        if _manhattan(p, to) != 1:
            return "not_adjacent"
        kind = self.obstacles.get(to)
        if kind is None:
            return "no_obstacle"
        if self.incident_cls.get(to) == "needs_human":
            self.counters[i]["wrong_class_attempts"] += 1
            return "needs_human"
        d = (to[0] - p[0], to[1] - p[1])
        beyond = (to[0] + d[0], to[1] + d[1])
        if not self.grid.in_bounds(beyond[0], beyond[1]) or not self.grid.is_passable(beyond[0], beyond[1]):
            return "no_room"
        if beyond in self.pos.values():
            return "no_room"
        del self.obstacles[to]
        self.obstacles[beyond] = kind
        self.grid.set(to[0], to[1], CellType.FREE)
        self.grid.set(beyond[0], beyond[1], CellType.OBSTACLE)
        clearable = to in self.plain or self.incident_cls.get(to) == "robot_clearable"
        if to in self.incident_cls:
            self.incident_cls[beyond] = self.incident_cls.pop(to)
        if clearable:                                    # an obstacle that has just been pushed is plainly clearable
            self.plain.discard(to)
            self.plain.add(beyond)
        self.pos[i] = to
        self.counters[i]["moves"] += 1
        self.counters[i]["push_steps"] += 1
        run = self._open_run.get(i)
        if run is not None and run["dir"] == d and run["landing"] == to and run["end_tick"] == t - 1:
            run["steps"] += 1
            run["landing"], run["end_tick"] = beyond, t
            self.push_log[run["index"]].update(landing=beyond, end_tick=t, steps=run["steps"])
        else:
            self.removals += 1
            run = {"dir": d, "landing": beyond, "end_tick": t, "steps": 1, "weight": weight(kind),
                   "index": len(self.push_log)}
            self._open_run[i] = run
            self.push_log.append({"robot": i, "kind": kind, "origin": to, "landing": beyond, "start_tick": t,
                                  "end_tick": t, "steps": 1})
        run["weight"] = weight(kind)
        return ""

    def _check(self, old_pos: Dict[int, Pos]) -> None:
        cells = list(self.pos.values())
        assert len(set(cells)) == len(cells), "two robots share a cell"
        ids = sorted(self.pos)
        for a in ids:
            for b in ids:
                if a < b and a in old_pos and b in old_pos and old_pos[a] != old_pos[b]:
                    assert not (self.pos[a] == old_pos[b] and self.pos[b] == old_pos[a]), "robots swapped"
