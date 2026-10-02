"""Ground-truth world: physics, move arbitration, incidents, edits, sensing and the override log."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple, Union

from src.doi.config import SimConfig
from src.doi.scenarios import Scenario
from src.environment.grid import CellType

Pos = Tuple[int, int]
COUNTER_KEYS = ("moves", "carried_steps", "waits", "pickups", "drops", "rejected_drops",
                "wrong_class_attempts")


@dataclass(frozen=True)
class Move:
    to: Pos


@dataclass(frozen=True)
class Wait:
    pass


@dataclass(frozen=True)
class Pickup:
    pass


@dataclass(frozen=True)
class Drop:
    pit: Pos


@dataclass(frozen=True)
class Return:
    pass


Action = Union[Move, Wait, Pickup, Drop, Return]


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    reason: str
    pos: Pos
    blocked_ticks: int
    carrying: bool


@dataclass(frozen=True)
class Observation:
    filled_pits: FrozenSet[Pos]
    robot_cells: FrozenSet[Pos]
    blocked_cells: FrozenSet[Pos]
    scanned: FrozenSet[Pos]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class World:
    def __init__(self, scenario: Scenario, cfg: SimConfig) -> None:
        self.scenario = scenario
        self.cfg = cfg
        self.grid = scenario.grid.copy()
        n = len(scenario.starts)
        self.pos: Dict[int, Pos] = {i: scenario.starts[i] for i in range(n)}
        self.carrying: Dict[int, bool] = {i: False for i in range(n)}
        self.blocked_ticks: Dict[int, int] = {i: 0 for i in range(n)}
        self.stock: Dict[Pos, int] = dict(scenario.depots)
        self.filled: Set[Pos] = set()
        self.filled_at: Dict[Pos, int] = {}
        self.blocked: Set[Pos] = set(scenario.pits)
        self.appeared_at: Dict[int, int] = {}
        self.overrides = 0
        self.fills = 0
        self.trajectory: Dict[int, List[Pos]] = {i: [self.pos[i]] for i in range(n)}
        self.carry_trace: Dict[int, List[bool]] = {i: [False] for i in range(n)}
        self.counters: Dict[int, Dict[str, int]] = {i: {k: 0 for k in COUNTER_KEYS} for i in range(n)}
        self.incident_cls: Dict[Pos, str] = {c: inc.cls for inc in scenario.incidents for c in inc.cells}
        self._suppressed: Set[int] = set()

    def prefill(self, pits: Iterable[Pos]) -> None:
        cells = set(pits)
        for inc in self.scenario.incidents:
            if cells & set(inc.cells):
                self._suppressed.add(inc.oid)
                for c in inc.cells:
                    self.filled.add(c)
                    self.filled_at[c] = -1
        for p in sorted(cells & set(self.scenario.pits)):
            self.grid.set(p[0], p[1], CellType.FREE)
            self.blocked.discard(p)
            self.filled.add(p)
            self.filled_at[p] = -1

    def begin_tick(self, t: int) -> None:
        occupied = set(self.pos.values())
        for inc in sorted(self.scenario.incidents, key=lambda i: i.oid):
            if inc.oid in self.appeared_at or inc.oid in self._suppressed or inc.appear_tick > t:
                continue
            if any(c in occupied for c in inc.cells):
                continue
            for c in inc.cells:
                self.grid.set(c[0], c[1], CellType.PIT)
                self.blocked.add(c)
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
        return Observation(filled_pits=frozenset(c for c in self.filled if _manhattan(me, c) <= r_sense),
                           robot_cells=others,
                           blocked_cells=frozenset(c for c in self.blocked if _manhattan(me, c) <= r_sense),
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
        survivors = {i: to for i, to in legal.items() if i not in removed}
        for i, to in survivors.items():
            self.pos[i] = to
            moved_ok.add(i)
            self.counters[i]["moves"] += 1
            if self.carrying[i]:
                self.counters[i]["carried_steps"] += 1

        for i in self.pos:
            self.blocked_ticks[i] = self.blocked_ticks[i] + 1 if i in blocked_now else 0
        for i in actions:
            if i in self.pos and i not in moved_ok:
                self.counters[i]["waits"] += 1

        order = sorted((i for i in actions if i in self.pos), key=self._key)
        for i in order:
            if isinstance(actions[i], Pickup):
                reasons[i] = self._pickup(i)
        self._drops({i: actions[i].pit for i in order if isinstance(actions[i], Drop)}, t, reasons)
        for i in order:
            if isinstance(actions[i], Return):
                reasons[i] = self._return(i)

        if self.cfg.debug_checks:
            self._check(old_pos)
        results = {}
        for i in actions:
            if i not in self.pos:
                continue
            reason = reasons.get(i, "")
            results[i] = ActionResult(reason == "", reason, self.pos[i], self.blocked_ticks[i], self.carrying[i])
        for i in self.pos:
            self.trajectory[i].append(self.pos[i])
            self.carry_trace[i].append(self.carrying[i])
        return results

    def _pickup(self, i: int) -> str:
        if self.carrying[i]:
            return "has_bag"
        p = self.pos[i]
        if p not in self.stock:
            return "not_depot"
        if self.stock[p] <= 0:
            return "empty"
        self.stock[p] -= 1
        self.carrying[i] = True
        self.counters[i]["pickups"] += 1
        return ""

    def _return(self, i: int) -> str:
        if not self.carrying[i]:
            return "no_bag"
        p = self.pos[i]
        if p not in self.stock:
            return "not_depot"
        self.stock[p] += 1
        self.carrying[i] = False
        return ""

    def _drops(self, drops: Dict[int, Pos], t: int, reasons: Dict[int, str]) -> None:
        candidates: Dict[Pos, List[int]] = {}
        for i, pit in drops.items():
            reason = ""
            if not self.carrying[i]:
                reason = "no_bag"
            elif _manhattan(self.pos[i], pit) != 1:
                reason = "not_adjacent"
            elif pit in self.filled:
                reason = "already_filled"
            elif pit not in self.blocked:
                reason = "not_blocked"
            elif self.incident_cls.get(pit) == "needs_human":
                reason = "needs_human"
                self.counters[i]["wrong_class_attempts"] += 1
            if reason:
                reasons[i] = reason
                self.counters[i]["rejected_drops"] += 1
            else:
                candidates.setdefault(pit, []).append(i)
        for pit, group in sorted(candidates.items()):
            winner = min(group, key=self._key)
            for i in group:
                if i != winner:
                    reasons[i] = "lost_priority"
                    self.counters[i]["rejected_drops"] += 1
            self.grid.set(pit[0], pit[1], CellType.FREE)
            self.blocked.discard(pit)
            self.filled.add(pit)
            self.filled_at[pit] = t
            self.carrying[winner] = False
            self.counters[winner]["drops"] += 1
            self.fills += 1

    def _check(self, old_pos: Dict[int, Pos]) -> None:
        cells = list(self.pos.values())
        assert len(set(cells)) == len(cells), "two robots share a cell"
        ids = sorted(self.pos)
        for a in ids:
            for b in ids:
                if a < b and a in old_pos and b in old_pos and old_pos[a] != old_pos[b]:
                    assert not (self.pos[a] == old_pos[b] and self.pos[b] == old_pos[a]), "robots swapped"
