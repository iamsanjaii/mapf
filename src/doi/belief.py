"""BeliefState: one robot's CRDT view of the traffic records and of which cells are blocked, and by what.

A cell's state is never "removed for good": obstacles move, so blocked and free are last-seen ticks that merge by
max, and the kind of obstacle on a cell is another max register (tick-encoded)."""
from typing import Any, Dict, FrozenSet, Optional, Tuple

from src.doi.crdt import AggRecordSet, Clock, GSet, MaxRegisterMap, ObstructionRecord, ObstructionSet, RecordSet, RentRecord
from src.doi.kinds import code, name_of

Pos = Tuple[int, int]
CLASS_CODE = {"unknown": 0, "robot_clearable": 1, "needs_human": 2}


KIND_SLOTS = 8          # kind codes are below this, so tick * KIND_SLOTS + code orders by tick first


class BeliefState:
    def __init__(self, robot_id: int, initial_obstacles: Dict[Pos, str]) -> None:
        self.robot_id = robot_id
        self.records = RecordSet()
        self.agg = AggRecordSet()
        self.census = GSet()
        self.report_tick = MaxRegisterMap(-1)
        self.blocked_tick = MaxRegisterMap(-1)
        self.free_tick = MaxRegisterMap(-1)
        self.cls = MaxRegisterMap(0)
        self.kind = MaxRegisterMap(-1)
        self.obstructions = ObstructionSet()
        self.slots_full = GSet()            # slots known to hold an obstacle; a slot is never emptied
        for cell, kind in initial_obstacles.items():       # obstacles on the map at tick 0 are known to everyone
            self.blocked_tick.raise_to(cell, 0)
            self.cls.raise_to(cell, CLASS_CODE["robot_clearable"])
            self.kind.raise_to(cell, code(kind))
        self.clock = Clock()
        for name in self._COMPONENTS:
            getattr(self, name)._clock = self.clock

    @property
    def version(self) -> int:
        return self.clock.n

    _COMPONENTS = ("records", "agg", "census", "report_tick", "blocked_tick", "free_tick", "cls", "kind",
                   "obstructions", "slots_full")

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
            self.kind.raise_to(cell, t * KIND_SLOTS + code(rec.kind))

    def observe_cell(self, cell: Pos, blocked: bool, t: int, kind: Optional[str] = None) -> None:
        (self.blocked_tick if blocked else self.free_tick).raise_to(cell, t)
        if blocked and kind is not None:
            self.kind.raise_to(cell, t * KIND_SLOTS + code(kind))

    def kind_of(self, cell: Pos) -> Optional[str]:
        v = self.kind.get(cell)
        return name_of(v % KIND_SLOTS) if v >= 0 else None

    def mark_needs_human(self, cell: Pos) -> None:
        self.cls.raise_to(cell, CLASS_CODE["needs_human"])

    def status(self, cell: Pos) -> str:
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
        """Believed-blocked cells a robot may remove (everything but obstructions that need a human)."""
        return tuple(sorted(self.believed_blocked() - self.hard_blocked()))

    def merge(self, other: "BeliefState") -> bool:
        changed = False
        for name in self._COMPONENTS:
            changed |= getattr(self, name).merge(getattr(other, name))
        return changed

    def snapshot(self) -> "BeliefState":
        out = BeliefState.__new__(BeliefState)
        out.robot_id = self.robot_id
        out.clock = Clock(self.clock.n)
        for name in self._COMPONENTS:
            comp = getattr(self, name).copy()
            comp._clock = out.clock
            setattr(out, name, comp)
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
        return out

    def units(self) -> int:
        return sum(getattr(self, name).units() for name in self._COMPONENTS)

    def canonical(self) -> Any:
        return tuple(getattr(self, name).canonical() for name in self._COMPONENTS)
