"""Fill policies (the experimental arms), the shared per-run state, and make_policy."""
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple

from src.doi.belief import BeliefState, CLASS_CODE
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.oracle import buy_lb, hindsight
from src.doi.paths import bfs_dist_map, passable_fn

if TYPE_CHECKING:
    from src.doi.agent import RobotAgent, TaskPlanInfo
    from src.doi.scenarios import Scenario

Pos = Tuple[int, int]
NEIGHBOURS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


@dataclass
class Shared:
    triggers: List[dict] = field(default_factory=list)
    engine: Any = None
    global_belief: Any = None
    supervisor: Any = None
    agents: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class HaulProposal:
    pits: Tuple[Pos, ...]
    evidence: float
    buy: float
    buy_per_pit: Optional[Tuple[float, ...]] = None


class FillPolicy(ABC):
    name: str = "policy"
    needs_single_rents: bool = False
    uses_gossip: bool = True

    def prepare(self, scenario: "Scenario", cfg: SimConfig, shared: Shared) -> None:
        return None

    def on_task_planned(self, agent: "RobotAgent", info: "TaskPlanInfo", t: int) -> None:
        return None

    def propose(self, agent: "RobotAgent", t: int) -> Optional[HaulProposal]:
        return None

    def prefill_set(self, scenario: "Scenario", cfg: SimConfig) -> FrozenSet[Pos]:
        return frozenset()


_BUY_CACHE: Dict[tuple, Tuple[float, Tuple[float, ...]]] = {}


def buy_cost_for(belief: BeliefState, grid, pits: Tuple[Pos, ...], cfg: SimConfig) -> Tuple[float, Tuple[float, ...]]:
    """Estimated cost of filling `pits` (fee plus kappa times the carry from the nearest stocked depot)."""
    opened = frozenset(pits)
    filled = frozenset(belief.filled.items())
    closed = belief.believed_blocked() - opened
    stocked = tuple(d for d in sorted(belief.stock.initial) if belief.stock.remaining(d) > 0)
    key = (grid.array.tobytes(), tuple(pits), filled, closed, stocked, cfg.fee, cfg.kappa)
    hit = _BUY_CACHE.get(key)
    if hit is not None:
        return hit
    pf = passable_fn(grid, filled | opened, closed=closed)
    depot_maps = [bfs_dist_map(pf, d, grid.height, grid.width) for d in stocked]
    per: List[float] = []
    for p in pits:
        best = float("inf")
        for dist in depot_maps:
            for dr, dc in NEIGHBOURS:
                n = (p[0] + dr, p[1] + dc)
                if n in dist:
                    best = min(best, cfg.fee + cfg.kappa * dist[n])
        per.append(best)
    out = (float(sum(per)), tuple(per))
    if len(_BUY_CACHE) > 4096:
        _BUY_CACHE.clear()
    _BUY_CACHE[key] = out
    return out


def buy_cost(agent: "RobotAgent", pits: Tuple[Pos, ...]) -> Tuple[float, Tuple[float, ...]]:
    return buy_cost_for(agent.belief, agent.grid, pits, agent.cfg)


def record_rent(belief: BeliefState, rec: RentRecord, cfg: SimConfig) -> None:
    if cfg.record_epoch:
        belief.add_rent(rec, cfg.record_epoch)
    else:
        belief.add_record(rec)


def _eligible(belief: BeliefState, pits: Tuple[Pos, ...]) -> bool:
    """Spec 4.4: only confirmed cells whose class is robot_clearable may appear in a firing set."""
    return all(belief.status(p) == "confirmed" and belief.cls.get(p) == CLASS_CODE["robot_clearable"]
               for p in pits)


def _deferred(agent: "RobotAgent", pits: Tuple[Pos, ...], t: int) -> bool:
    return any(agent.deferred_until.get(p, -1) > t for p in pits)


def _claimed_by_other(belief: BeliefState, pits: Tuple[Pos, ...], me: int, t: int) -> bool:
    for p in pits:
        eff = belief.claims.effective(p, t)
        if eff is not None and eff[1].hauler != me:
            return True
    return False


class NeverFillPolicy(FillPolicy):
    name = "never"
    uses_gossip = False


class RoFPolicy(FillPolicy):
    def __init__(self, name: str, theta: float, keys: str = "bundle", scope: str = "gossip",
                 extrapolate: bool = False, window: Optional[int] = None) -> None:
        self.name = name
        self.theta = theta
        self.keys = keys
        self.scope = scope
        self.extrapolate = extrapolate
        self.window = window
        self.uses_gossip = scope == "gossip"
        self.shared: Optional[Shared] = None

    def prepare(self, scenario, cfg, shared) -> None:
        self.shared = shared

    def on_task_planned(self, agent, info, t) -> None:
        if info.rent > 0:
            record_rent(agent.belief, RentRecord(agent.id, info.task_idx, info.start, info.goal, t, info.rent),
                        agent.cfg)

    def propose(self, agent, t) -> Optional[HaulProposal]:
        belief = agent.belief
        ev = self.shared.engine.evidence(belief, t, window=self.window, extrapolate=self.extrapolate,
                                         per_pit=(self.keys == "pit"))
        best = None
        for T, known in ev.items():
            if known <= 0 or not _eligible(belief, T) or _deferred(agent, T, t):
                continue
            if agent.cfg.claim and _claimed_by_other(belief, T, agent.id, t):
                continue
            buy, per = buy_cost(agent, T)
            if buy == float("inf") or known < self.theta * buy:
                continue
            rank = (-(known - buy), len(T), T)
            if best is None or rank < best[0]:
                best = (rank, T, known, buy, per)
        if best is None:
            return None
        _, T, known, buy, per = best
        self.shared.triggers.append({"tick": t, "robot": agent.id, "pits": T, "known": known, "buy": buy})
        return HaulProposal(T, known, buy, per)


class MyopicPolicy(FillPolicy):
    name = "myopic"
    uses_gossip = False

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta
        self.current: Dict[int, "TaskPlanInfo"] = {}
        self.shared: Optional[Shared] = None

    def prepare(self, scenario, cfg, shared) -> None:
        self.shared = shared

    def on_task_planned(self, agent, info, t) -> None:
        self.current[agent.id] = info

    def propose(self, agent, t) -> Optional[HaulProposal]:
        info = self.current.get(agent.id)
        if info is None or info.task_idx != agent.task_idx or info.rent <= 0:
            return None
        T = tuple(p for p in info.bundle if p not in agent.belief.filled)
        if not T or not _eligible(agent.belief, T) or _deferred(agent, T, t):
            return None
        buy, per = buy_cost(agent, T)
        if buy == float("inf") or info.rent < self.theta * buy:
            return None
        self.shared.triggers.append({"tick": t, "robot": agent.id, "pits": T, "known": float(info.rent),
                                     "buy": buy})
        return HaulProposal(T, float(info.rent), buy, per)


class CentralPolicy(FillPolicy):
    """Omniscient arm: one global ledger refreshed from ground truth, dispatching the cheapest idle hauler."""
    name = "central"
    uses_gossip = False

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta
        self.shared: Optional[Shared] = None
        self.cfg: Optional[SimConfig] = None
        self.grid = None
        self.assigned: set = set()

    def prepare(self, scenario, cfg, shared) -> None:
        self.shared, self.cfg, self.grid = shared, cfg, scenario.grid
        shared.global_belief = BeliefState(-1, dict(scenario.depots), list(scenario.pits))

    def on_task_planned(self, agent, info, t) -> None:
        if info.rent > 0:
            record_rent(self.shared.global_belief,
                        RentRecord(agent.id, info.task_idx, info.start, info.goal, t, info.rent), self.cfg)

    def sync(self, world, agents, t: int) -> None:
        g = self.shared.global_belief
        for c in sorted(world.filled):
            g.filled.add(c)
        for c in sorted(world.blocked):
            g.observe_cell(c, True, t)
        for c, cls in world.incident_cls.items():
            g.cls.raise_to(c, CLASS_CODE[cls])
        for a in agents:
            if a.finished:
                continue
            before = len(a.belief.filled.items())
            for c in world.filled:
                a.belief.filled.add(c)
            for c in world.blocked:
                if a.belief.status(c) not in ("confirmed", "filled"):
                    a.belief.observe_cell(c, True, t)
                    a.replan_needed = True
            if len(a.belief.filled.items()) != before:
                a.replan_needed = True
        self.assigned = {p for p in self.assigned if p not in g.filled}

    def dispatch(self, agents, t: int) -> None:
        g = self.shared.global_belief
        ev = self.shared.engine.evidence(g, t)
        best = None
        for T, known in ev.items():
            if known <= 0 or not _eligible(g, T) or any(p in self.assigned for p in T):
                continue
            buy, per = buy_cost_for(g, self.grid, T, self.cfg)
            if buy == float("inf") or known < self.theta * buy:
                continue
            rank = (-(known - buy), len(T), T)
            if best is None or rank < best[0]:
                best = (rank, T, known, buy, per)
        if best is None:
            return
        _, T, known, buy, per = best
        from src.doi.hauler import HaulState, select_haul
        chosen = None
        for a in agents:
            if a.finished or a.hauler.state != HaulState.NONE:
                continue
            sel = select_haul(a, T, open_all=True)
            if sel is None:
                continue
            h = sel[3] + self.cfg.kappa * sel[4]
            if chosen is None or (h, a.id) < chosen[0]:
                chosen = ((h, a.id), a)
        if chosen is None:
            return
        self.shared.triggers.append({"tick": t, "robot": chosen[1].id, "pits": T, "known": known, "buy": buy})
        self.assigned |= set(T)
        chosen[1].hauler.assign(HaulProposal(T, known, buy, per), t)


class HindsightPolicy(FillPolicy):
    name = "hindsight"
    uses_gossip = False

    def __init__(self) -> None:
        self.hindsight_buy = 0.0

    def prefill_set(self, scenario, cfg) -> FrozenSet[Pos]:
        h = hindsight(scenario, cfg)
        self.hindsight_buy = float(sum(h.buy_lb[p] for p in h.s_star))
        return h.s_star


class FreeOpenPolicy(FillPolicy):
    name = "free"
    uses_gossip = False

    def prefill_set(self, scenario, cfg) -> FrozenSet[Pos]:
        return frozenset(scenario.pits) | frozenset(c for inc in scenario.incidents for c in inc.cells)


def make_policy(cfg: SimConfig) -> FillPolicy:
    name = cfg.policy
    if name == "never":
        return NeverFillPolicy()
    if name == "myopic":
        return MyopicPolicy(theta=cfg.theta)
    if name == "eager":
        return RoFPolicy("eager", theta=0.0)
    if name == "rof":
        return RoFPolicy("rof", theta=cfg.theta)
    if name == "rof_pit":
        return RoFPolicy("rof_pit", theta=cfg.theta, keys="pit")
    if name == "rof_local":
        return RoFPolicy("rof_local", theta=cfg.theta, scope="local")
    if name == "rof_w":
        return RoFPolicy("rof_w", theta=cfg.theta, window=cfg.window)
    if name == "rof_x":
        return RoFPolicy("rof_x", theta=cfg.theta, extrapolate=True)
    if name == "central":
        return CentralPolicy(theta=cfg.theta)
    if name == "hindsight":
        return HindsightPolicy()
    return FreeOpenPolicy()
