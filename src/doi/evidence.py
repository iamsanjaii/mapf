"""EvidenceEngine: residual rent evidence recomputed from task records under the current edit set."""
from typing import Dict, FrozenSet, Optional, Tuple

from src.doi.belief import BeliefState
from src.doi.crdt import BundleKey
from src.doi.paths import dream_path, single_pit_rents
from src.environment.grid import Grid

Pos = Tuple[int, int]


class EvidenceEngine:
    def __init__(self, grid: Grid, unreachable: float, record_epoch: Optional[int] = None) -> None:
        self.grid = grid
        self.unreachable = unreachable
        self.record_epoch = record_epoch
        self._cache: Dict[tuple, object] = {}
        self._contributors: Dict[BundleKey, FrozenSet[int]] = {}

    def contributors(self) -> Dict[BundleKey, FrozenSet[int]]:
        return dict(self._contributors)

    def _lookup(self, origin: Pos, dest: Pos, pits: Tuple[Pos, ...], filled: FrozenSet[Pos],
                hard: FrozenSet[Pos], per_pit: bool):
        key = (origin, dest, pits, filled, hard, per_pit)
        if key not in self._cache:
            if per_pit:
                self._cache[key] = single_pit_rents(self.grid, pits, filled, origin, dest, self.unreachable, hard)
            else:
                self._cache[key] = dream_path(self.grid, pits, filled, origin, dest, self.unreachable, hard)
        return self._cache[key]

    def evidence(self, belief: BeliefState, now: int, window: Optional[int] = None,
                 extrapolate: bool = False, per_pit: bool = False) -> Dict[BundleKey, float]:
        pits = belief.editable()
        filled = frozenset(belief.filled.items())
        hard = belief.hard_blocked()
        ev: Dict[BundleKey, float] = {}
        who: Dict[BundleKey, set] = {}
        for r in belief.records.records():
            if window is not None and r.tick < now - window:
                continue
            res = self._lookup(r.origin, r.dest, pits, filled, hard, per_pit)
            if per_pit:
                for p, rent in sorted(res.items()):
                    c = min(r.rent, rent)
                    if c > 0:
                        ev[(p,)] = ev.get((p,), 0.0) + c
                        who.setdefault((p,), set()).add(r.robot)
            else:
                bundle = tuple(sorted(res.bundle))
                c = min(r.rent, res.rent)
                if bundle and c > 0:
                    ev[bundle] = ev.get(bundle, 0.0) + c
                    who.setdefault(bundle, set()).add(r.robot)
        for (robot, origin, dest, epoch), (rent_sum, count) in belief.agg.entries():
            if window is not None and (epoch + 1) * self.record_epoch - 1 < now - window:
                continue
            res = self._lookup(origin, dest, pits, filled, hard, per_pit)
            if per_pit:
                for p, rent in sorted(res.items()):
                    c = min(rent_sum, count * rent)
                    if c > 0:
                        ev[(p,)] = ev.get((p,), 0.0) + c
                        who.setdefault((p,), set()).add(robot)
            else:
                bundle = tuple(sorted(res.bundle))
                c = min(rent_sum, count * res.rent)
                if bundle and c > 0:
                    ev[bundle] = ev.get(bundle, 0.0) + c
                    who.setdefault(bundle, set()).add(robot)
        if extrapolate:
            census = len(belief.census.items())
            for key in ev:
                ev[key] *= census / max(1, len(who[key]))
        self._contributors = {k: frozenset(v) for k, v in who.items()}
        return {k: float(v) for k, v in ev.items()}
