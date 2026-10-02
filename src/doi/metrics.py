"""RunResult, per-run cost accounting, and post-hoc metrics that read ground truth."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from src.doi.config import SimConfig
from src.doi.paths import dream_path

Pos = Tuple[int, int]


@dataclass
class RunResult:
    cfg: Dict[str, Any]
    policy: str
    J: float
    J_censored: float
    delay: float
    throughput: float
    moves: int
    waits: int
    carried_steps: int
    fills: int
    fee_total: float
    unfinished_tasks: int
    stalled: bool
    ticks: int
    messages: Dict[str, int]
    traffic_messages: Dict[str, int]
    overrides: int
    fill_ticks: List[int] = field(default_factory=list)
    triggers: List[dict] = field(default_factory=list)
    edits: List[dict] = field(default_factory=list)
    plan_infos: List[Any] = field(default_factory=list)
    filled_at: Dict[Pos, int] = field(default_factory=dict)
    appeared_at: Dict[int, int] = field(default_factory=dict)
    wasted_haul_cost: float = 0.0
    wrong_class_attempts: int = 0
    unconfirmed_hauls: int = 0
    false_report_hauls: int = 0
    claims: Dict[str, int] = field(default_factory=lambda: {"issued": 0, "lost": 0, "aborts": 0})
    approvals: Dict[str, int] = field(default_factory=lambda: {
        "requested": 0, "approved": 0, "vetoed": 0, "timeout_approved": 0, "deferred": 0})
    intake: Dict[str, int] = field(default_factory=lambda: {"reports": 0, "records": 0, "rejected": 0})
    final_stock: Dict[Pos, int] = field(default_factory=dict)
    hindsight_buy: float = 0.0
    runtime_ms: float = 0.0
    trajectory: Dict[int, List[Pos]] = field(default_factory=dict)
    scenario: Any = None


def _unreachable(result: RunResult) -> float:
    g = result.scenario.grid
    return SimConfig(**result.cfg).unreachable_cost_for(g.height, g.width)


def _truth_filled(result: RunResult, t: int) -> FrozenSet[Pos]:
    return frozenset(p for p, ft in result.filled_at.items() if ft <= t)


def _incident_cells_by(result: RunResult, t: int) -> FrozenSet[Pos]:
    cells = set()
    for inc in result.scenario.incidents:
        at = result.appeared_at.get(inc.oid)
        if at is not None and at <= t:
            cells.update(inc.cells)
    return frozenset(cells)


def stale_detour_cost(result: RunResult, scenario, cfg: SimConfig) -> float:
    """Extra steps planned because the robot's belief about edits lagged the truth."""
    U = cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)
    total = 0.0
    for p in result.plan_infos:
        truth = _truth_filled(result, p.tick)
        if truth == p.belief_filled:
            continue
        pits = scenario.pits if scenario.family != "D" else sorted(p.belief_blocked | truth)
        d_truth = dream_path(scenario.grid, pits, truth, p.start, p.goal, U).d_block
        total += max(0, p.d_block - d_truth)
    return float(total)


def false_report_cost(result: RunResult, scenario, cfg: SimConfig) -> float:
    """Extra steps planned around cells believed blocked that were never blocked (spec 7.4)."""
    U = cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)
    static = frozenset(scenario.pits)
    total = 0.0
    for p in result.plan_infos:
        real = static | _incident_cells_by(result, p.tick)
        phantom = frozenset(c for c in p.belief_blocked if c not in real)
        if not phantom:
            continue
        d_without = dream_path(scenario.grid, sorted(p.belief_blocked - phantom), p.belief_filled,
                               p.start, p.goal, U, p.hard_blocked - phantom).d_block
        total += max(0, p.d_block - d_without)
    return float(total)


def false_report_hauls(result: RunResult, scenario) -> int:
    real = set(scenario.pits) | {c for inc in scenario.incidents for c in inc.cells}
    return sum(1 for trig in result.triggers if any(c not in real for c in trig["pits"]))


def true_rent_at_triggers(result: RunResult) -> List[Dict[str, float]]:
    """Rent a perfectly informed ledger would hold at each trigger, and the ledger's coverage of it."""
    scenario, U = result.scenario, _unreachable(result)
    out = []
    for trig in result.triggers:
        tick = trig["tick"]
        filled_now = _truth_filled(result, tick)
        target = tuple(sorted(set(trig["pits"]) - filled_now))
        true_rent = 0.0
        for info in result.plan_infos:
            if info.tick > tick:
                continue
            pits = scenario.pits if scenario.family != "D" else sorted(
                set(scenario.pits) | _incident_cells_by(result, info.tick))
            d = dream_path(scenario.grid, pits, _truth_filled(result, info.tick), info.start, info.goal, U)
            residual = tuple(sorted(set(d.bundle) - filled_now))
            if residual and residual == target:
                true_rent += d.rent
        row = dict(trig)
        row["true_rent"] = float(true_rent)
        row["coverage"] = trig["known"] / true_rent if true_rent > 0 else float("nan")
        out.append(row)
    return out


def hindsight_ratios(alg: RunResult, hindsight: RunResult, free: RunResult) -> Dict[str, float]:
    assert hindsight.policy == "hindsight" and free.policy == "free"
    j_hind = hindsight.J_censored + hindsight.hindsight_buy
    hr = alg.J_censored / j_hind
    gap = j_hind - free.J_censored
    hr_av = (alg.J_censored - free.J_censored) / gap if gap >= 1 else float("nan")
    return {"hr": hr, "hr_av": hr_av}


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def summary_row(result: RunResult, ratios: Optional[Dict[str, float]] = None,
                pod: Optional[float] = None) -> dict:
    cfg = SimConfig(**result.cfg)
    scenario = result.scenario
    ratios = ratios or {}
    nan = float("nan")
    return {
        "policy": result.policy, "seed": cfg.seed, "scenario": cfg.scenario,
        "family": scenario.family if scenario is not None else "", "n_robots": cfg.n_robots,
        "r_comm": cfg.r_comm, "r_traffic": cfg.r_traffic, "loss": cfg.loss, "latency": cfg.latency,
        "theta": cfg.theta, "claim": cfg.claim, "gate": cfg.gate, "intake": cfg.intake, "window": cfg.window,
        "J": result.J, "J_censored": result.J_censored, "delay": result.delay,
        "throughput": result.throughput, "fills": result.fills, "carried_steps": result.carried_steps,
        "wasted_haul_cost": result.wasted_haul_cost, "stalled": result.stalled,
        "unfinished_tasks": result.unfinished_tasks, "ticks": result.ticks,
        "broadcasts": result.messages["broadcasts"], "transmissions": result.messages["transmissions"],
        "dropped": result.messages["dropped"], "message_units": result.messages["units"],
        "traffic_units": result.traffic_messages["units"],
        "overrides_per_1000": 1000.0 * result.overrides / max(1, result.ticks),
        "stale_detour_cost": stale_detour_cost(result, scenario, cfg) if scenario is not None else nan,
        "false_report_cost": false_report_cost(result, scenario, cfg) if scenario is not None else nan,
        "mean_coverage": _mean([r["coverage"] for r in true_rent_at_triggers(result)])
        if scenario is not None else nan,
        "mean_B_est": _mean([e["B_est"] for e in result.edits]),
        "mean_B_real": _mean([e["B_real"] for e in result.edits]),
        "mean_approval_wait": _mean([e["approval_wait"] for e in result.edits]),
        "claims_issued": result.claims["issued"], "claims_lost": result.claims["lost"],
        "aborts": result.claims["aborts"], "unconfirmed_hauls": result.unconfirmed_hauls,
        "false_report_hauls": false_report_hauls(result, scenario) if scenario is not None else 0,
        "wrong_class_attempts": result.wrong_class_attempts,
        "approvals_requested": result.approvals["requested"], "approvals_vetoed": result.approvals["vetoed"],
        "intake_records": result.intake["records"], "intake_rejected": result.intake["rejected"],
        "runtime_ms": result.runtime_ms,
        "hr": ratios.get("hr", nan), "hr_av": ratios.get("hr_av", nan), "pod": pod if pod is not None else nan,
    }
