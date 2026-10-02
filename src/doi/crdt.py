"""State-based CRDTs for the per-robot belief: sets, max registers, stock counters, claims, records.

Every component can carry a shared `Clock`; local writes and merge-ins stamp the touched entry with a fresh
version so `delta(v)` can return only the entries that changed after version `v` (delta-state gossip).
"""
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

Pos = Tuple[int, int]
Ticket = Tuple[int, int]
BundleKey = Tuple[Pos, ...]


class Clock:
    """Local version counter shared by the components of one belief."""

    def __init__(self, n: int = 0) -> None:
        self.n = n


class _Versioned:
    _clock: Optional[Clock] = None

    def _init_versions(self) -> None:
        self._clock = None
        self._ver: Dict[Any, int] = {}

    def _bump(self, key: Any) -> None:
        if self._clock is not None:
            self._clock.n += 1
            self._ver[key] = self._clock.n

    def _newer(self, key: Any, v: int) -> bool:
        return self._ver.get(key, 0) > v

    def _copy_versions_to(self, other: "_Versioned") -> None:
        other._ver = dict(self._ver)


class GSet(_Versioned):
    def __init__(self) -> None:
        self._init_versions()
        self._items: set = set()

    def add(self, x: Any) -> None:
        if x not in self._items:
            self._items.add(x)
            self._bump(x)

    def __contains__(self, x: Any) -> bool:
        return x in self._items

    def items(self) -> FrozenSet:
        return frozenset(self._items)

    def merge(self, other: "GSet") -> bool:
        changed = False
        for x in other._items:
            if x not in self._items:
                self._items.add(x)
                self._bump(x)
                changed = True
        return changed

    def units(self) -> int:
        return len(self._items)

    def copy(self) -> "GSet":
        out = GSet()
        out._items = set(self._items)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "GSet":
        out = GSet()
        out._items = {x for x in self._items if self._newer(x, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._items)


class MaxRegisterMap(_Versioned):
    def __init__(self, default: int = -1) -> None:
        self._init_versions()
        self.default = default
        self._d: Dict[Any, int] = {}

    def raise_to(self, key: Any, value: int) -> None:
        if value > self._d.get(key, self.default):
            self._d[key] = value
            self._bump(key)

    def get(self, key: Any) -> int:
        return self._d.get(key, self.default)

    def keys(self) -> List:
        return sorted(self._d)

    def merge(self, other: "MaxRegisterMap") -> bool:
        changed = False
        for k, v in other._d.items():
            if v > self._d.get(k, self.default):
                self._d[k] = v
                self._bump(k)
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "MaxRegisterMap":
        out = MaxRegisterMap(self.default)
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "MaxRegisterMap":
        out = MaxRegisterMap(self.default)
        out._d = {k: x for k, x in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return tuple(sorted(self._d.items()))


class PNStock(_Versioned):
    def __init__(self, initial: Dict[Pos, int]) -> None:
        self._init_versions()
        self.initial = dict(initial)
        self._takes: Dict[Pos, Dict[int, int]] = {d: {} for d in initial}
        self._returns: Dict[Pos, Dict[int, int]] = {d: {} for d in initial}

    def take(self, depot: Pos, robot_id: int, n: int = 1) -> None:
        t = self._takes[depot]
        t[robot_id] = t.get(robot_id, 0) + n
        self._bump(("t", depot, robot_id))

    def give_back(self, depot: Pos, robot_id: int, n: int = 1) -> None:
        r = self._returns[depot]
        r[robot_id] = r.get(robot_id, 0) + n
        self._bump(("r", depot, robot_id))

    def mark_empty(self, depot: Pos, robot_id: int) -> None:
        self.take(depot, robot_id, self.remaining(depot))

    def remaining(self, depot: Pos) -> int:
        value = (self.initial[depot] - sum(self._takes[depot].values())
                 + sum(self._returns[depot].values()))
        return max(0, value)

    def _merge_maps(self, kind: str, depot: Pos, mine: Dict[int, int], theirs: Dict[int, int]) -> bool:
        changed = False
        for k, v in theirs.items():
            if v > mine.get(k, 0):
                mine[k] = v
                self._bump((kind, depot, k))
                changed = True
        return changed

    def merge(self, other: "PNStock") -> bool:
        changed = False
        for d in self._takes:
            changed |= self._merge_maps("t", d, self._takes[d], other._takes.get(d, {}))
            changed |= self._merge_maps("r", d, self._returns[d], other._returns.get(d, {}))
        return changed

    def units(self) -> int:
        return sum(len(m) for m in self._takes.values()) + sum(len(m) for m in self._returns.values())

    def copy(self) -> "PNStock":
        out = PNStock(self.initial)
        out._takes = {d: dict(m) for d, m in self._takes.items()}
        out._returns = {d: dict(m) for d, m in self._returns.items()}
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "PNStock":
        out = PNStock(self.initial)
        for d in self._takes:
            out._takes[d] = {k: x for k, x in self._takes[d].items() if self._newer(("t", d, k), v)}
            out._returns[d] = {k: x for k, x in self._returns[d].items() if self._newer(("r", d, k), v)}
        return out

    def canonical(self) -> Any:
        return tuple((d, tuple(sorted(self._takes[d].items())), tuple(sorted(self._returns[d].items())))
                     for d in sorted(self._takes))


@dataclass(frozen=True)
class Claim:
    pits: Tuple[Pos, ...]
    hauler: int


class ClaimSet(_Versioned):
    """Ticket -> (claim, expiry) with max-merge expiry; released tickets are grow-only tombstones."""

    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[Ticket, Tuple[Claim, int]] = {}
        self._released: set = set()

    def issue(self, ticket: Ticket, claim: Claim, expiry: int) -> None:
        before = self._d.get(ticket)
        if before is not None:
            old, old_exp = before
            claim = min(old, claim, key=lambda c: (c.pits, c.hauler))
            expiry = max(expiry, old_exp)
        self._d[ticket] = (claim, expiry)
        if before != self._d[ticket]:
            self._bump(ticket)

    def renew(self, ticket: Ticket, expiry: int) -> None:
        if ticket in self._d:
            claim, old = self._d[ticket]
            if expiry > old:
                self._d[ticket] = (claim, expiry)
                self._bump(ticket)

    def release(self, ticket: Ticket) -> None:
        if ticket not in self._released:
            self._released.add(ticket)
            self._bump(("rel", ticket))

    def effective(self, pit: Pos, now: int) -> Optional[Tuple[Ticket, Claim]]:
        live = [(t, c) for t, (c, exp) in self._d.items()
                if pit in c.pits and exp > now and t not in self._released]
        return min(live, key=lambda x: x[0]) if live else None

    def tickets_of(self, robot_id: int) -> List[Ticket]:
        return sorted(t for t, (c, _) in self._d.items() if c.hauler == robot_id)

    def merge(self, other: "ClaimSet") -> bool:
        changed = False
        for t, (c, exp) in other._d.items():
            before = self._d.get(t)
            self.issue(t, c, exp)
            changed |= before != self._d[t]
        for t in other._released:
            if t not in self._released:
                self._released.add(t)
                self._bump(("rel", t))
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d) + len(self._released)

    def copy(self) -> "ClaimSet":
        out = ClaimSet()
        out._d = dict(self._d)
        out._released = set(self._released)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "ClaimSet":
        out = ClaimSet()
        out._d = {t: x for t, x in self._d.items() if self._newer(t, v)}
        out._released = {t for t in self._released if self._newer(("rel", t), v)}
        return out

    def canonical(self) -> Any:
        return (tuple(sorted((t, c.pits, c.hauler, exp) for t, (c, exp) in self._d.items())),
                frozenset(self._released))


@dataclass(frozen=True)
class RentRecord:
    robot: int
    task_idx: int
    origin: Pos
    dest: Pos
    tick: int
    rent: float


class RecordSet(_Versioned):
    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[Tuple[int, int], RentRecord] = {}

    def add(self, rec: RentRecord) -> None:
        key = (rec.robot, rec.task_idx)
        if key not in self._d:
            self._d[key] = rec
            self._bump(key)

    def records(self) -> List[RentRecord]:
        return [self._d[k] for k in sorted(self._d)]

    def merge(self, other: "RecordSet") -> bool:
        changed = False
        for k, rec in other._d.items():
            if k not in self._d:
                self._d[k] = rec
                self._bump(k)
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "RecordSet":
        out = RecordSet()
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "RecordSet":
        out = RecordSet()
        out._d = {k: r for k, r in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())


AggKey = Tuple[int, Pos, Pos, int]


class AggRecordSet(_Versioned):
    """Per (robot, origin, dest, epoch): (rent_sum, count). Only the owner writes; fields only grow."""

    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[AggKey, Tuple[float, int]] = {}

    def add(self, robot: int, origin: Pos, dest: Pos, epoch: int, rent: float) -> None:
        key = (robot, origin, dest, epoch)
        total, count = self._d.get(key, (0.0, 0))
        self._d[key] = (total + rent, count + 1)
        self._bump(key)

    def entries(self) -> List[Tuple[AggKey, Tuple[float, int]]]:
        return [(k, self._d[k]) for k in sorted(self._d)]

    def merge(self, other: "AggRecordSet") -> bool:
        changed = False
        for k, (total, count) in other._d.items():
            mine = self._d.get(k, (0.0, 0))
            new = (max(mine[0], total), max(mine[1], count))
            if new != mine:
                self._d[k] = new
                self._bump(k)
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "AggRecordSet":
        out = AggRecordSet()
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "AggRecordSet":
        out = AggRecordSet()
        out._d = {k: x for k, x in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())


@dataclass(frozen=True)
class ObstructionRecord:
    report_id: str
    node: int
    location: str
    cells: Tuple[Pos, ...]
    kind: str
    cls: str
    est_kits: int
    confidence: float
    rationale: str


def _obstruction_order(rec: ObstructionRecord) -> Tuple:
    return (rec.node, rec.location, rec.cells, rec.kind, rec.cls, rec.est_kits, rec.confidence, rec.rationale)


class ObstructionSet(_Versioned):
    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[str, ObstructionRecord] = {}

    def add(self, rec: ObstructionRecord) -> None:
        old = self._d.get(rec.report_id)
        if old is None or _obstruction_order(rec) < _obstruction_order(old):
            self._d[rec.report_id] = rec
            self._bump(rec.report_id)

    def get(self, report_id: str) -> Optional[ObstructionRecord]:
        return self._d.get(report_id)

    def records(self) -> List[ObstructionRecord]:
        return [self._d[k] for k in sorted(self._d)]

    def merge(self, other: "ObstructionSet") -> bool:
        before = dict(self._d)
        for rec in other._d.values():
            self.add(rec)
        return self._d != before

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "ObstructionSet":
        out = ObstructionSet()
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "ObstructionSet":
        out = ObstructionSet()
        out._d = {k: r for k, r in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())


ApprovalKey = Tuple[Tuple[Pos, ...], Ticket]


class ApprovalSet(_Versioned):
    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[ApprovalKey, str] = {}

    def set(self, key: ApprovalKey, decision: str) -> None:
        if self._d.get(key) != "veto" and self._d.get(key) != decision:
            self._d[key] = decision
            self._bump(key)

    def get(self, key: ApprovalKey) -> Optional[str]:
        return self._d.get(key)

    def merge(self, other: "ApprovalSet") -> bool:
        before = dict(self._d)
        for k, v in other._d.items():
            self.set(k, v)
        return self._d != before

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "ApprovalSet":
        out = ApprovalSet()
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "ApprovalSet":
        out = ApprovalSet()
        out._d = {k: d for k, d in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())
