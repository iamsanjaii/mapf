"""RunResult, per-run cost accounting, and post-hoc metrics that read ground truth."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from src.doi.config import SimConfig
from src.doi.paths import bfs_dist_map, passable_fn

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
    push_steps: int
    push_cost: float
    removals: int
    fee_total: float
    unfinished_tasks: int
    stalled: bool
    ticks: int
    messages: Dict[str, int]
    traffic_messages: Dict[str, int]
    overrides: int
    triggers: List[dict] = field(default_factory=list)
    pushes: List[dict] = field(default_factory=list)          # one entry per push run
    push_rejected: int = 0
    plan_infos: List[Any] = field(default_factory=list)
    appeared_at: Dict[int, int] = field(default_factory=dict)
    wrong_class_attempts: int = 0
    intake: Dict[str, int] = field(default_factory=lambda: {"reports": 0, "records": 0, "rejected": 0})
    hindsight_buy: float = 0.0
    runtime_ms: float = 0.0
    trajectory: Dict[int, List[Pos]] = field(default_factory=dict)
    obstacle_trace: List[Dict[Pos, str]] = field(default_factory=list)   # obstacles after t ticks
    tick_cost: List[float] = field(default_factory=list)                 # fleet cost paid in each tick
    scenario: Any = None
    picks: int = 0
    drops: int = 0
    carries: int = 0                 # obstacles taken to a rack or dump slot
    fills: int = 0                   # pits filled
    carry_steps: int = 0
    carry_cost: float = 0.0          # loaded steps, at kappa_c * weight each
    slot_conflicts: int = 0          # drops refused because the slot was full or the pit already filled
    carry_log: List[dict] = field(default_factory=list)
    lift_ticks: List[Tuple[int, int]] = field(default_factory=list)      # (robot, tick) of every pick-up and drop
    pick_log: List[dict] = field(default_factory=list)                   # one entry per pick-up: robot, tick, kind
    notice_reach: Dict[str, int] = field(default_factory=dict)          # notice id -> robots that knew it at the end
    forecast_log: List[dict] = field(default_factory=list)              # rof_a: one row per forecast requested
    forecast_cases: List[Any] = field(default_factory=list)             # rof_a: the case behind each answered one


def _unreachable(result: RunResult) -> float:
    g = result.scenario.grid
    return SimConfig(**result.cfg).unreachable_cost_for(g.height, g.width)


def _obstacles_at(result: RunResult, tick: int) -> FrozenSet[Pos]:
    trace = result.obstacle_trace
    return frozenset(trace[min(tick, len(trace) - 1)]) if trace else frozenset()


def _dist(grid, closed: FrozenSet[Pos], start: Pos, goal: Pos, unreachable: float) -> float:
    return float(bfs_dist_map(passable_fn(grid, closed=closed), start, grid.height, grid.width)
                 .get(goal, unreachable))


def stale_detour_cost(result: RunResult, scenario, cfg: SimConfig) -> float:
    """Extra steps planned because the robot's belief about the obstacles lagged the truth."""
    U = _unreachable(result)
    total = 0.0
    for p in result.plan_infos:
        truth = _obstacles_at(result, p.tick)
        total += max(0.0, p.d_block - _dist(scenario.grid, truth, p.start, p.goal, U))
    return float(total)


def false_report_cost(result: RunResult, scenario, cfg: SimConfig) -> float:
    """Extra steps planned around cells believed blocked that held no obstacle at the time."""
    U = _unreachable(result)
    total = 0.0
    for p in result.plan_infos:
        phantom = p.belief_blocked - _obstacles_at(result, p.tick)
        if not phantom:
            continue
        without = _dist(scenario.grid, p.belief_blocked - phantom, p.start, p.goal, U)
        total += max(0.0, p.d_block - without)
    return float(total)


def collateral_cost(result: RunResult, scenario, cfg: SimConfig) -> float:
    """Detour paid because a pushed obstacle was parked where it blocks a route, measured on the true map.

    For each planned task: route length with the obstacles as they stood, minus the route length if the obstacles
    that earlier pushes had left on a landing cell were not there."""
    U = _unreachable(result)
    total = 0.0
    for p in result.plan_infos:
        now = _obstacles_at(result, p.tick)
        parked = frozenset(run["landing"] for run in result.pushes if run["end_tick"] < p.tick) & now
        if not parked:
            continue
        total += max(0.0, _dist(scenario.grid, now, p.start, p.goal, U)
                     - _dist(scenario.grid, now - parked, p.start, p.goal, U))
    return float(total)


def loaded_at(result: RunResult, robot: int, t: int) -> bool:
    """Is the robot carrying something at trajectory index t (the state after t ticks)? Lifts alternate pick, drop."""
    return sum(1 for r, tick in result.lift_ticks if r == robot and tick < t) % 2 == 1


def carried_kind(result: RunResult, robot: int, t: int) -> Optional[str]:
    """The kind the robot is carrying at trajectory index t, or None."""
    if not loaded_at(result, robot, t):
        return None
    picks = [e for e in result.pick_log if e["robot"] == robot and e["tick"] < t]
    return picks[-1]["kind"] if picks else None


def slots_full_at(result: RunResult, t: int) -> FrozenSet[Pos]:
    """Rack and dump cells holding an obstacle after t ticks."""
    return frozenset(e["target"] for e in result.carry_log if e["mode"] == "carry" and e["tick"] < t)


def near_slot_waits(result: RunResult, radius: int = 2) -> int:
    """Ticks a robot stood still within `radius` cells of a slot, not counting the ticks it spent lifting or dropping.

    A rough measure of queueing at the racks and the dump region."""
    slots = list(result.scenario.slots) if result.scenario is not None else []
    if not slots:
        return 0
    lifting = set(result.lift_ticks)
    total = 0
    for robot, path in result.trajectory.items():
        for t in range(len(path) - 1):
            if path[t] != path[t + 1] or (robot, t) in lifting:
                continue
            if min(abs(path[t][0] - r) + abs(path[t][1] - c) for r, c in slots) <= radius:
                total += 1
    return total


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


def forecast_columns(log: List[dict]) -> dict:
    """Counts over a run's forecast log. A wrong yes pays the price early; a wrong no pays detours for longer."""
    scored = [r for r in log if r["answer"] is not None and r["truth"] is not None]
    return {
        "forecasts": len(log),
        "forecast_failed": sum(1 for r in log if r["failed"]),
        "forecast_skipped": sum(1 for r in log if r["skipped"]),
        "forecast_acc": (sum(1 for r in scored if r["answer"] == r["truth"]) / len(scored)) if scored
        else float("nan"),
        "wrong_yes": sum(1 for r in scored if r["answer"] and not r["truth"]),
        "wrong_no": sum(1 for r in scored if not r["answer"] and r["truth"]),
    }


def summary_row(result: RunResult, ratios: Optional[Dict[str, float]] = None,
                pod: Optional[float] = None, cheap: bool = False) -> dict:
    """`cheap` skips the metrics that recompute paths per planned task (needed at large scale)."""
    cfg = SimConfig(**result.cfg)
    scenario = result.scenario
    ratios = ratios or {}
    nan = float("nan")
    full = scenario is not None and not cheap
    if full:
        U = _unreachable(result)
        static_travel = float(sum(_dist(scenario.grid, _obstacles_at(result, p.tick), p.start, p.goal, U)
                                  for p in result.plan_infos))
    else:
        static_travel = nan
    return {
        "policy": result.policy, "seed": cfg.seed, "scenario": cfg.scenario,
        "family": scenario.family if scenario is not None else "", "n_robots": cfg.n_robots,
        "r_comm": cfg.r_comm, "r_traffic": cfg.r_traffic, "loss": cfg.loss, "latency": cfg.latency,
        "theta": cfg.theta, "intake": cfg.intake,
        "J": result.J, "J_censored": result.J_censored, "delay": result.delay,
        "throughput": result.throughput, "removals": result.removals, "push_steps": result.push_steps,
        "push_rejected": result.push_rejected, "stalled": result.stalled,
        "carries": result.carries, "fills": result.fills, "picks": result.picks, "drops": result.drops,
        "carry_steps": result.carry_steps, "carry_cost": result.carry_cost, "slot_conflicts": result.slot_conflicts,
        "slot_utilisation": (sum(1 for e in result.carry_log if e["mode"] == "carry") / len(scenario.slots)
                             if scenario is not None and scenario.slots else 0.0),
        "near_slot_waits": near_slot_waits(result) if full else nan,
        "unfinished_tasks": result.unfinished_tasks, "ticks": result.ticks,
        "broadcasts": result.messages["broadcasts"], "transmissions": result.messages["transmissions"],
        "dropped": result.messages["dropped"], "message_units": result.messages["units"],
        "traffic_units": result.traffic_messages["units"],
        "overrides_per_1000": 1000.0 * result.overrides / max(1, result.ticks),
        "stale_detour_cost": stale_detour_cost(result, scenario, cfg) if full else nan,
        "false_report_cost": false_report_cost(result, scenario, cfg) if full else nan,
        "collateral_cost": collateral_cost(result, scenario, cfg) if full else nan,
        "wrong_class_attempts": result.wrong_class_attempts,
        "intake_records": result.intake["records"], "intake_rejected": result.intake["rejected"],
        "runtime_ms": result.runtime_ms,
        "static_travel": static_travel,
        "congestion_excess": result.J - result.push_cost - result.fee_total - static_travel if full else nan,
        "hr": ratios.get("hr", nan), "hr_av": ratios.get("hr_av", nan), "pod": pod if pod is not None else nan,
        "forecaster": cfg.forecaster, **forecast_columns(result.forecast_log),
    }
