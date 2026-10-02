"""State-based CRDTs for the per-robot belief: sets, max registers, stock counters, claims, records."""
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

Pos = Tuple[int, int]
Ticket = Tuple[int, int]
BundleKey = Tuple[Pos, ...]


class GSet:
    def __init__(self) -> None:
        self._items: set = set()

    def add(self, x: Any) -> None:
        self._items.add(x)

    def __contains__(self, x: Any) -> bool:
        return x in self._items

    def items(self) -> FrozenSet:
        return frozenset(self._items)

    def merge(self, other: "GSet") -> bool:
        before = len(self._items)
        self._items |= other._items
        return len(self._items) != before

    def units(self) -> int:
        return len(self._items)

    def copy(self) -> "GSet":
        out = GSet()
        out._items = set(self._items)
        return out

    def canonical(self) -> Any:
        return frozenset(self._items)


class MaxRegisterMap:
    def __init__(self, default: int = -1) -> None:
        self.default = default
        self._d: Dict[Any, int] = {}

    def raise_to(self, key: Any, value: int) -> None:
        if value > self._d.get(key, self.default):
            self._d[key] = value

    def get(self, key: Any) -> int:
        return self._d.get(key, self.default)

    def keys(self) -> List:
        return sorted(self._d)

    def merge(self, other: "MaxRegisterMap") -> bool:
        changed = False
        for k, v in other._d.items():
            if v > self._d.get(k, self.default):
                self._d[k] = v
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "MaxRegisterMap":
        out = MaxRegisterMap(self.default)
        out._d = dict(self._d)
        return out

    def canonical(self) -> Any:
        return tuple(sorted(self._d.items()))


class PNStock:
    def __init__(self, initial: Dict[Pos, int]) -> None:
        self.initial = dict(initial)
        self._takes: Dict[Pos, Dict[int, int]] = {d: {} for d in initial}
        self._returns: Dict[Pos, Dict[int, int]] = {d: {} for d in initial}

    def take(self, depot: Pos, robot_id: int, n: int = 1) -> None:
        t = self._takes[depot]
        t[robot_id] = t.get(robot_id, 0) + n

    def give_back(self, depot: Pos, robot_id: int, n: int = 1) -> None:
        r = self._returns[depot]
        r[robot_id] = r.get(robot_id, 0) + n

    def mark_empty(self, depot: Pos, robot_id: int) -> None:
        self.take(depot, robot_id, self.remaining(depot))

    def remaining(self, depot: Pos) -> int:
        value = (self.initial[depot] - sum(self._takes[depot].values())
                 + sum(self._returns[depot].values()))
        return max(0, value)

    @staticmethod
    def _merge_maps(mine: Dict[int, int], theirs: Dict[int, int]) -> bool:
        changed = False
        for k, v in theirs.items():
            if v > mine.get(k, 0):
                mine[k] = v
                changed = True
        return changed

    def merge(self, other: "PNStock") -> bool:
        changed = False
        for d in self._takes:
            changed |= self._merge_maps(self._takes[d], other._takes.get(d, {}))
            changed |= self._merge_maps(self._returns[d], other._returns.get(d, {}))
        return changed

    def units(self) -> int:
        return sum(len(m) for m in self._takes.values()) + sum(len(m) for m in self._returns.values())

    def copy(self) -> "PNStock":
        out = PNStock(self.initial)
        out._takes = {d: dict(m) for d, m in self._takes.items()}
        out._returns = {d: dict(m) for d, m in self._returns.items()}
        return out

    def canonical(self) -> Any:
        return tuple((d, tuple(sorted(self._takes[d].items())), tuple(sorted(self._returns[d].items())))
                     for d in sorted(self._takes))


@dataclass(frozen=True)
class Claim:
    pits: Tuple[Pos, ...]
    hauler: int


class ClaimSet:
    """Ticket -> (claim, expiry) with max-merge expiry; released tickets are grow-only tombstones."""

    def __init__(self) -> None:
        self._d: Dict[Ticket, Tuple[Claim, int]] = {}
        self._released: set = set()

    def issue(self, ticket: Ticket, claim: Claim, expiry: int) -> None:
        if ticket in self._d:
            old, old_exp = self._d[ticket]
            claim = min(old, claim, key=lambda c: (c.pits, c.hauler))
            expiry = max(expiry, old_exp)
        self._d[ticket] = (claim, expiry)

    def renew(self, ticket: Ticket, expiry: int) -> None:
        if ticket in self._d:
            claim, old = self._d[ticket]
            self._d[ticket] = (claim, max(old, expiry))

    def release(self, ticket: Ticket) -> None:
        self._released.add(ticket)

    def effective(self, pit: Pos, now: int) -> Optional[Tuple[Ticket, Claim]]:
        live = [(t, c) for t, (c, exp) in self._d.items()
                if pit in c.pits and exp > now and t not in self._released]
        return min(live, key=lambda x: x[0]) if live else None

    def tickets_of(self, robot_id: int) -> List[Ticket]:
        return sorted(t for t, (c, _) in self._d.items() if c.hauler == robot_id)

    def merge(self, other: "ClaimSet") -> bool:
        before = (dict(self._d), set(self._released))
        for t, (c, exp) in other._d.items():
            self.issue(t, c, exp)
        self._released |= other._released
        return (self._d, self._released) != before

    def units(self) -> int:
        return len(self._d) + len(self._released)

    def copy(self) -> "ClaimSet":
        out = ClaimSet()
        out._d = dict(self._d)
        out._released = set(self._released)
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


class RecordSet:
    def __init__(self) -> None:
        self._d: Dict[Tuple[int, int], RentRecord] = {}

    def add(self, rec: RentRecord) -> None:
        self._d.setdefault((rec.robot, rec.task_idx), rec)

    def records(self) -> List[RentRecord]:
        return [self._d[k] for k in sorted(self._d)]

    def merge(self, other: "RecordSet") -> bool:
        changed = False
        for k, rec in other._d.items():
            if k not in self._d:
                self._d[k] = rec
                changed = True
        return changed

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "RecordSet":
        out = RecordSet()
        out._d = dict(self._d)
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


class ObstructionSet:
    def __init__(self) -> None:
        self._d: Dict[str, ObstructionRecord] = {}

    def add(self, rec: ObstructionRecord) -> None:
        old = self._d.get(rec.report_id)
        if old is None or _obstruction_order(rec) < _obstruction_order(old):
            self._d[rec.report_id] = rec

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
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())


ApprovalKey = Tuple[Tuple[Pos, ...], Ticket]


class ApprovalSet:
    def __init__(self) -> None:
        self._d: Dict[ApprovalKey, str] = {}

    def set(self, key: ApprovalKey, decision: str) -> None:
        if self._d.get(key) != "veto":
            self._d[key] = decision

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
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())
