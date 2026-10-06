"""ForecastCase: what one robot knows about one candidate action, frozen into plain data, and the truth label.

A forecaster sees a case and nothing else, so forecasters can be tested, and models scored, without a simulation.
The fields a model may see are listed in VISIBLE; `truth`, `numeric_forecast`, `robot` and `tick` are not shown.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from src.doi.carryplan import CarryPlan
from src.doi.pushplan import BundlePlan
from src.doi.rng import stream

Pos = Tuple[int, int]
ORIGINS_PER_ZONE, DESTS_PER_ZONE = 6, 4
VISIBLE = ("mode", "kind", "obstacle", "landing", "plan_key", "price", "known_saving", "notices", "ledger", "zones",
           "trip_saving")


@dataclass(frozen=True)
class ForecastCase:
    case_id: str
    robot: int
    tick: int
    mode: str                                        # "push" | "bundle" | "carry" | "fill"
    kind: str
    obstacle: Pos
    landing: Pos
    plan_key: Tuple
    price: float
    known_saving: float
    notices: Tuple[Tuple[str, int, str], ...]        # (notice id, tick issued, text), sorted by id
    ledger: Dict[str, float]
    zones: Dict[str, Dict[str, Tuple[int, int]]]     # name -> {"rows": (lo, hi), "cols": (lo, hi)}
    trip_saving: Dict[str, Dict[str, float]]         # "from|to" -> {"mean_saving", "share_saving", "pairs"}
    numeric_forecast: bool
    truth: Optional[bool]


def plan_key(plan) -> Tuple:
    legs = (plan.first, plan.second) if isinstance(plan, BundlePlan) else (plan,)
    return tuple(p.key for p in legs)


def plan_mode(plan) -> str:
    if isinstance(plan, BundlePlan):
        return "bundle"
    return plan.mode if isinstance(plan, CarryPlan) else "push"


def true_total_saving(scenario, engine, before: FrozenSet[Pos], after: FrozenSet[Pos]) -> float:
    """Fall in total route length over every task of every robot, past and future, if `before` became `after`.

    A static measure: it ignores congestion and later changes to the map. A trip's own endpoints count as open,
    as in EvidenceEngine.fleet_cost."""
    total = 0.0
    for start, goals in zip(scenario.starts, scenario.tasks):
        prev = start
        for goal in goals:
            if goal != prev:
                ends = {prev, goal}
                total += engine.distance(prev, goal, before - ends) - engine.distance(prev, goal, after - ends)
            prev = goal
    return total


def numeric_saving(ledger: Dict[str, float]) -> float:
    """The arithmetic forecast of LedgerPolicy.saving(forecast=True), from the ledger numbers alone."""
    seen, expected = ledger["trips_recorded"], ledger["trips_expected"]
    return (ledger["saving_on_my_current_trip"]
            + ledger["saving_on_recorded_trips"] * max(0.0, expected - seen) / max(1.0, seen))


def _sample(cells, n: int, seed: int, name: str) -> list:
    pool = sorted(cells)
    stream(seed, name).shuffle(pool)
    return pool[:n]


def trip_table(zones: Dict[str, Tuple[Pos, ...]], engine, before: FrozenSet[Pos], after: FrozenSet[Pos],
               seed: int) -> Dict[str, Dict[str, float]]:
    """For every ordered pair of zones: what one trip between them would save, on a seeded sample of cell pairs."""
    blocked = before | after
    table: Dict[str, Dict[str, float]] = {}
    for a in sorted(zones):
        origins = _sample([c for c in zones[a] if c not in blocked], ORIGINS_PER_ZONE, seed, f"trip-o-{a}")
        for b in sorted(zones):
            dests = _sample([c for c in zones[b] if c not in blocked], DESTS_PER_ZONE, seed, f"trip-d-{b}")
            savings = [engine.distance(o, d, before) - engine.distance(o, d, after)
                       for o in origins for d in dests if o != d]
            n = len(savings)
            table[f"{a}|{b}"] = {"mean_saving": round(sum(savings) / n, 3) if n else 0.0,
                                 "share_saving": round(sum(1 for v in savings if v > 0) / n, 3) if n else 0.0,
                                 "pairs": float(n)}
    return table


def _freeze(x: Any) -> Any:
    return tuple(_freeze(v) for v in x) if isinstance(x, (list, tuple)) else x


def _case_id(fields: Dict[str, Any]) -> str:
    visible = {k: fields[k] for k in VISIBLE}
    return hashlib.sha256(json.dumps(visible, sort_keys=True, default=list).encode()).hexdigest()[:16]


def build_case(agent, plan, tick: int, price: float, known: float, scenario, engine) -> ForecastCase:
    """Freeze what `agent` knows about `plan` at `tick`. `scenario` supplies the zones and the hidden truth."""
    b = agent.belief
    before, after = plan.before, plan.after
    robots = max(1, len(b.census.items()))
    seen = len(b.records.records()) + sum(c for _, (_, c) in b.agg.entries())
    ledger = {
        "robots_known": float(robots),
        "tasks_per_robot": float(len(agent.tasks)),
        "trips_recorded": float(seen),
        "trips_expected": float(robots * len(agent.tasks)),
        "saving_on_recorded_trips": round(engine.fleet_cost(b, before) - engine.fleet_cost(b, after), 3),
        "saving_on_my_current_trip": round(engine.distance(agent.pos, agent.goal, before)
                                           - engine.distance(agent.pos, agent.goal, after), 3),
        "my_tasks_done": float(agent.task_idx),
        "my_tasks_left": float(len(agent.tasks) - agent.task_idx),
    }
    zones = {name: {"rows": (min(r for r, _ in cells), max(r for r, _ in cells)),
                    "cols": (min(c for _, c in cells), max(c for _, c in cells))}
             for name, cells in sorted(scenario.zones.items()) if cells}
    fields: Dict[str, Any] = {
        "mode": plan_mode(plan), "kind": plan.kind, "obstacle": tuple(plan.obstacle), "landing": tuple(plan.landing),
        "plan_key": _freeze(plan_key(plan)), "price": round(float(price), 3), "known_saving": round(float(known), 3),
        "notices": tuple((r.notice_id, r.tick, r.text) for r in b.notices.records()),
        "ledger": ledger, "zones": zones,
        "trip_saving": trip_table(scenario.zones, engine, before, after, agent.cfg.seed),
    }
    return ForecastCase(case_id=_case_id(fields), robot=agent.id, tick=tick,
                        numeric_forecast=numeric_saving(ledger) >= price,
                        truth=true_total_saving(scenario, engine, before, after) >= price, **fields)


def case_to_dict(case: ForecastCase) -> dict:
    return asdict(case)


def case_from_dict(d: dict) -> ForecastCase:
    d = dict(d)
    for name in ("obstacle", "landing", "plan_key", "notices"):
        d[name] = _freeze(d[name])
    d["zones"] = {name: {k: tuple(v) for k, v in box.items()} for name, box in d["zones"].items()}
    return ForecastCase(**d)
