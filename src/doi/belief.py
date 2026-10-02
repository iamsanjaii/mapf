"""BeliefState: one robot's CRDT view of records, edits, cell status, stock, claims and approvals."""
from typing import Any, Dict, FrozenSet, Sequence, Tuple

from src.doi.crdt import (AggRecordSet, ApprovalSet, ClaimSet, Clock, GSet, MaxRegisterMap, ObstructionRecord,
                          ObstructionSet, PNStock, RecordSet, RentRecord, Ticket)

Pos = Tuple[int, int]
CLASS_CODE = {"unknown": 0, "robot_clearable": 1, "needs_human": 2}


class BeliefState:
    def __init__(self, robot_id: int, depots: Dict[Pos, int], static_pits: Sequence[Pos]) -> None:
        self.robot_id = robot_id
        self.records = RecordSet()
        self.agg = AggRecordSet()
        self.filled = GSet()
        self.stock = PNStock(depots)
        self.claims = ClaimSet()
        self.census = GSet()
        self.report_tick = MaxRegisterMap(-1)
        self.blocked_tick = MaxRegisterMap(-1)
        self.free_tick = MaxRegisterMap(-1)
        self.cls = MaxRegisterMap(0)
        self.obstructions = ObstructionSet()
        self.approvals = ApprovalSet()
        self.lamport = 0
        for p in static_pits:
            self.blocked_tick.raise_to(p, 0)
            self.cls.raise_to(p, CLASS_CODE["robot_clearable"])
        self.clock = Clock()
        for name in self._COMPONENTS:
            getattr(self, name)._clock = self.clock

    @property
    def version(self) -> int:
        return self.clock.n

    _COMPONENTS = ("records", "agg", "filled", "stock", "claims", "census", "report_tick", "blocked_tick",
                   "free_tick", "cls", "obstructions", "approvals")

    def add_record(self, rec: RentRecord) -> None:
        self.records.add(rec)
        self.census.add(rec.robot)

    def add_rent(self, rec: RentRecord, epoch: int) -> None:
        """Aggregated variant of add_record: one (rent_sum, count) entry per (robot, origin, dest, epoch)."""
        self.agg.add(rec.robot, rec.origin, rec.dest, rec.tick // epoch, rec.rent)
        self.census.add(rec.robot)

    def add_obstruction(self, rec: ObstructionRecord, t: int) -> None:
        self.obstructions.add(rec)
        for cell in rec.cells:
            self.report_tick.raise_to(cell, t)
            self.cls.raise_to(cell, CLASS_CODE[rec.cls])

    def observe_cell(self, cell: Pos, blocked: bool, t: int) -> None:
        (self.blocked_tick if blocked else self.free_tick).raise_to(cell, t)

    def mark_needs_human(self, cell: Pos) -> None:
        self.cls.raise_to(cell, CLASS_CODE["needs_human"])

    def status(self, cell: Pos) -> str:
        if cell in self.filled:
            return "filled"
        blocked, free, reported = self.blocked_tick.get(cell), self.free_tick.get(cell), self.report_tick.get(cell)
        if blocked >= 0 and blocked >= free:
            return "confirmed"
        if reported >= 0 and reported > free:
            return "reported"
        if free >= 0:
            return "refuted"
        return "none"

    def believed_blocked(self) -> FrozenSet[Pos]:
        cells = set(self.report_tick.keys()) | set(self.blocked_tick.keys()) | set(self.free_tick.keys())
        return frozenset(c for c in cells if self.status(c) in ("confirmed", "reported"))

    def hard_blocked(self) -> FrozenSet[Pos]:
        return frozenset(c for c in self.believed_blocked() if self.cls.get(c) == CLASS_CODE["needs_human"])

    def editable(self) -> Tuple[Pos, ...]:
        return tuple(sorted((self.believed_blocked() - self.hard_blocked()) | self.filled.items()))

    def next_ticket(self) -> Ticket:
        self.lamport += 1
        return (self.lamport, self.robot_id)

    def merge(self, other: "BeliefState") -> bool:
        changed = False
        for name in self._COMPONENTS:
            changed |= getattr(self, name).merge(getattr(other, name))
        if other.lamport > self.lamport:
            self.lamport = other.lamport
            changed = True
        return changed

    def snapshot(self) -> "BeliefState":
        out = BeliefState.__new__(BeliefState)
        out.robot_id = self.robot_id
        out.clock = Clock(self.clock.n)
        for name in self._COMPONENTS:
            comp = getattr(self, name).copy()
            comp._clock = out.clock
            setattr(out, name, comp)
        out.lamport = self.lamport
        return out

    def delta_since(self, v: int) -> "BeliefState":
        """Only the entries written or merged in after local version `v`."""
        out = BeliefState.__new__(BeliefState)
        out.robot_id = self.robot_id
        out.clock = Clock()
        for name in self._COMPONENTS:
            comp = getattr(self, name).delta(v)
            comp._clock = out.clock
            setattr(out, name, comp)
        out.lamport = self.lamport
        return out

    def units(self) -> int:
        return sum(getattr(self, name).units() for name in self._COMPONENTS)

    def canonical(self) -> Any:
        return tuple(getattr(self, name).canonical() for name in self._COMPONENTS) + (self.lamport,)
