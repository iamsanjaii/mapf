"""Simulated human supervisor: delayed approve/veto decisions (a declared ground-truth exception)."""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.doi.config import SimConfig
from src.doi.crdt import ApprovalKey
from src.doi.rng import stream, u01
from src.doi.scenarios import Scenario

Pos = Tuple[int, int]


@dataclass(frozen=True)
class Decision:
    robot: int
    key: ApprovalKey
    decision: str
    needs_human: Tuple[Pos, ...]
    due_tick: int


class Supervisor:
    def __init__(self, scenario: Scenario, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.truth_needs_human = frozenset(c for inc in scenario.incidents if inc.cls == "needs_human"
                                           for c in inc.cells)
        self._due: Dict[int, List[Decision]] = {}

    def request(self, robot: int, key: ApprovalKey, pits: Tuple[Pos, ...], t: int) -> None:
        cfg = self.cfg
        z = stream(cfg.seed, f"sup-{robot}-{t}").gauss(0, 1)
        latency = max(1, round(cfg.sup_latency_median * math.exp(cfg.sup_latency_sigma * z)))
        bad = tuple(sorted(p for p in pits if p in self.truth_needs_human))
        if bad and u01(cfg.seed, robot, t, 91) < cfg.p_catch:
            d = Decision(robot, key, "veto", bad, t + latency)
        else:
            d = Decision(robot, key, "approve", (), t + latency)
        self._due.setdefault(d.due_tick, []).append(d)

    def due(self, t: int) -> List[Decision]:
        return sorted(self._due.pop(t, []), key=lambda d: (d.robot, d.key))
