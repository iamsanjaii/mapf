"""Push policies (the experimental arms), the shared per-run state, and make_policy.

An arm answers one question for a robot whose own route runs through a removable obstacle: push it out of the
way, or go round? Every arm sees the same candidate plan (see pushplan.py); they differ in what they weigh.

  never      go round, always
  myopic     push if it is cheaper for me than the detour (ignores the ledger)
  eager      push if the ledger says it would have saved the fleet anything
  rof        push if the ledger's saving reaches theta times the price (the method, gossip ledger)
  rof_local  the same with a ledger of the robot's own tasks only
  rof_f      push if the saving forecast for the rest of the run reaches the price
  central    the same as rof with one omniscient ledger and an omniscient map
  free       benchmark: every obstacle is gone at tick 0, for free
  hindsight  benchmark: the best set of obstacles is gone at tick 0, charged its lowest possible price
"""
import math
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional, Tuple

from src.doi.belief import CLASS_CODE, BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.forecast.case import build_case, plan_key
from src.doi.forecast.forecasters import make_forecaster
from src.doi.oracle import hindsight
from src.doi.pushplan import BundlePlan, PushPlan
from src.doi.rng import u01

if TYPE_CHECKING:
    from src.doi.agent import RobotAgent, TaskPlanInfo
    from src.doi.scenarios import Scenario

Pos = Tuple[int, int]


@dataclass
class Shared:
    triggers: List[dict] = field(default_factory=list)
    engine: Any = None
    global_belief: Any = None
    agents: List[Any] = field(default_factory=list)
    forecast_log: List[dict] = field(default_factory=list)       # one row per forecast a robot requested
    forecast_cases: List[Any] = field(default_factory=list)      # the ForecastCase of each answered request


class PushPolicy:
    name: str = "policy"
    pushes: bool = False             # does this arm ever consider a push?
    uses_gossip: bool = True         # do robots exchange their beliefs?
    records_traffic: bool = False    # does the arm keep a ledger of planned tasks?

    def prepare(self, scenario: "Scenario", cfg: SimConfig, shared: Shared) -> None:
        self.shared, self.cfg = shared, cfg

    def on_task_planned(self, agent: "RobotAgent", info: "TaskPlanInfo", t: int) -> None:
        if self.records_traffic:
            rec = RentRecord(agent.id, info.task_idx, info.start, info.goal, t, info.rent)
            record_traffic(agent.belief, rec, agent.cfg)

    def on_tick(self, agents, t: int) -> None:
        """Called once per tick before the robots decide. Most arms have nothing to do."""

    def assess(self, agent: "RobotAgent", plan: PushPlan, info: "TaskPlanInfo") -> Optional[Tuple[float, float, float]]:
        """None to reject the plan, else (score, saving, price); the robot takes the plan with the best score."""
        return None

    def prefill_set(self, scenario: "Scenario", cfg: SimConfig) -> FrozenSet[Pos]:
        return frozenset()


def record_traffic(belief: BeliefState, rec: RentRecord, cfg: SimConfig) -> None:
    if cfg.record_epoch:
        belief.add_rent(rec, cfg.record_epoch)
    else:
        belief.add_record(rec)


def price_of(plan: PushPlan, info: "TaskPlanInfo") -> float:
    """What the push costs over the ideal route through the obstacle (the premium for removing it now)."""
    return max(0.0, plan.total - info.d_open)


class NeverPolicy(PushPolicy):
    name = "never"
    uses_gossip = False


class MyopicPolicy(PushPolicy):
    name = "myopic"
    pushes = True
    uses_gossip = False

    def assess(self, agent, plan, info):
        if plan.total > info.d_block:
            return None
        return info.d_block - plan.total, float(info.rent), price_of(plan, info)


class LedgerPolicy(PushPolicy):
    """Push when the ledger's saving justifies the price. Variants differ in theta, scope and forecasting."""
    pushes = True
    records_traffic = True

    def __init__(self, name: str, theta: float, scope: str = "gossip", forecast: bool = False) -> None:
        self.name, self.theta, self.scope, self.forecast = name, theta, scope, forecast
        self.uses_gossip = scope == "gossip"

    def _belief(self, agent) -> BeliefState:
        return agent.belief

    @staticmethod
    def _own(agent):
        start = next((i.start for i in reversed(agent.plan_infos) if i.task_idx == agent.task_idx), agent.pos)
        return (agent.id, agent.task_idx, agent.pos, agent.goal, start)

    def saving(self, agent, plan: PushPlan) -> float:
        belief = self._belief(agent)
        ledger, mine = self.shared.engine.parts(belief, plan.before, plan.after, own=self._own(agent))
        if not self.forecast:
            return ledger + mine
        # forecast: the saving on all traffic seen so far, scaled to the tasks still to come, plus the task in hand
        engine = self.shared.engine
        everything = engine.fleet_cost(belief, plan.before) - engine.fleet_cost(belief, plan.after)
        seen = len(belief.records.records()) + sum(c for _, (_, c) in belief.agg.entries())
        expected = max(1, len(belief.census.items())) * len(agent.tasks)
        return mine + everything * max(0, expected - seen) / max(1, seen)

    def assess(self, agent, plan, info):
        known = self.saving(agent, plan)
        price = price_of(plan, info)
        if known <= 0 or known < self.theta * price:
            return None
        return known - price, known, price


class RandomizedPolicy(LedgerPolicy):
    """Rent-or-Fill with a random threshold z in [0, 1] (density e^z / (e - 1)), shared by all robots per obstacle."""

    def __init__(self, theta: float = 1.0) -> None:
        super().__init__("rof_r", theta)

    def assess(self, agent, plan, info):
        known = self.saving(agent, plan)
        price = price_of(plan, info)
        z = math.log(1 + u01(self.cfg.seed, plan.obstacle[0], plan.obstacle[1], 31) * (math.e - 1))
        if known <= 0 or known < z * price:
            return None
        return known - price, known, price


class PredictedPolicy(LedgerPolicy):
    """Rent-or-Fill whose threshold is lam if the robot's forecast says the push will pay, else 1 / lam."""

    def __init__(self, lam: float) -> None:
        super().__init__("rof_p", theta=1.0, forecast=False)
        self.lam = lam

    def _forecast(self, agent, plan) -> float:
        self.forecast = True
        try:
            return self.saving(agent, plan)
        finally:
            self.forecast = False

    def assess(self, agent, plan, info):
        price = price_of(plan, info)
        predictions = getattr(agent, "predictions", None)
        if predictions is None:
            predictions = agent.predictions = {}
        legs = (plan.first, plan.second) if isinstance(plan, BundlePlan) else (plan,)
        key = tuple(p.key for p in legs)
        if key not in predictions:
            predictions[key] = self._forecast(agent, plan) >= price
        thr = self.lam if predictions[key] else 1.0 / self.lam
        known = self.saving(agent, plan)
        if known <= 0 or known < thr * price:
            return None
        return known - price, known, price


@dataclass
class ForecastState:
    asked: int                  # tick the forecast was requested
    due: int                    # tick from which the answer is visible to the guard
    answer: Optional[bool]      # None: failed or skipped, so the classical threshold stays


class GuardedPolicy(PredictedPolicy):
    """rof_a: a forecaster says whether the action will pay; the predicted-threshold rule guards the answer.

    The threshold is lam on a yes, 1 / lam on a no, and 1 while there is no visible answer (Proposition 5).
    A forecast is requested once per robot and plan, and only when the known saving reaches lam * price: below
    that neither threshold can fire, so the answer could not matter."""

    def __init__(self, lam: float, forecaster=None) -> None:
        super().__init__(lam)
        self.name = "rof_a"
        self._given = forecaster

    def prepare(self, scenario: "Scenario", cfg: SimConfig, shared: Shared) -> None:
        super().prepare(scenario, cfg, shared)
        self.scenario = scenario
        self.forecaster = self._given or make_forecaster(cfg.forecaster, cfg)
        self.requests = 0

    def on_tick(self, agents, t: int) -> None:
        for a in agents:                 # an answer that lands now is a reason to look at the plans again
            if any(st.due == t and st.asked < t and st.answer is not None for st in a.forecasts.values()):
                a.forecast_epoch += 1

    def _request(self, agent, plan, key, t: int, price: float, known: float) -> ForecastState:
        row = {"tick": t, "robot": agent.id, "plan_key": key, "forecaster": self.forecaster.name,
               "known": float(known), "price": float(price)}
        if self.requests >= self.cfg.agent_max_forecasts:
            state = ForecastState(t, t, None)
            row.update(case_id="", answer=None, truth=None, numeric_forecast=None, confidence=None, failed="",
                       skipped=True, due=t, latency_s=0.0, reason="")
        else:
            self.requests += 1
            case = build_case(agent, plan, t, price, known, self.scenario, self.shared.engine)
            res = self.forecaster.forecast(case)
            due = t if res.latency_s <= 0 else t + max(1, math.ceil(res.latency_s / self.cfg.tick_seconds))
            state = ForecastState(t, due, res.answer)
            self.shared.forecast_cases.append(case)
            row.update(case_id=case.case_id, answer=res.answer, truth=case.truth,
                       numeric_forecast=case.numeric_forecast, confidence=res.confidence, failed=res.failed,
                       skipped=False, due=due, latency_s=res.latency_s, reason=res.reason)
        self.shared.forecast_log.append(row)
        return state

    def assess(self, agent, plan, info):
        price = price_of(plan, info)
        known = self.saving(agent, plan)
        if known <= 0:
            return None
        key = plan_key(plan)
        state = agent.forecasts.get(key)
        if state is None and price > 0 and self.lam < 1 and known >= self.lam * price:
            state = agent.forecasts[key] = self._request(agent, plan, key, info.tick, price, known)
        thr = 1.0
        if state is not None and state.answer is not None and info.tick >= state.due:
            thr = self.lam if state.answer else 1.0 / self.lam
        if known < thr * price:
            return None
        return known - price, known, price


class CentralPolicy(LedgerPolicy):
    """Omniscient arm: one global ledger and the true map, refreshed every tick; robots still push on their own routes."""

    def __init__(self, theta: float = 1.0) -> None:
        super().__init__("central", theta)
        self.uses_gossip = False

    def prepare(self, scenario, cfg, shared) -> None:
        super().prepare(scenario, cfg, shared)
        shared.global_belief = BeliefState(-1, dict(scenario.obstacles))

    def _belief(self, agent) -> BeliefState:
        return self.shared.global_belief

    def on_task_planned(self, agent, info, t) -> None:
        record_traffic(self.shared.global_belief, RentRecord(agent.id, info.task_idx, info.start, info.goal, t,
                                                             info.rent), self.cfg)

    def sync(self, world, agents, t: int) -> None:
        """Copy the true obstacle map into the global belief and into every robot's belief."""
        for belief in [self.shared.global_belief] + [a.belief for a in agents if not a.finished]:
            changed = False
            for c, kind in world.obstacles.items():
                if belief.status(c) != "confirmed":
                    belief.observe_cell(c, True, t, kind=kind)
                    changed = True
                if c in world.plain:
                    belief.cls.raise_to(c, CLASS_CODE["robot_clearable"])
            for c in sorted(world.slot_items):
                belief.slots_full.add(c)
            for c in sorted(belief.believed_blocked() - set(world.obstacles)):
                belief.observe_cell(c, False, t + 1)
                changed = True
            if changed:
                for a in agents:
                    if a.belief is belief:
                        a.replan_needed = True


class FreePolicy(PushPolicy):
    name = "free"
    uses_gossip = False

    def prefill_set(self, scenario, cfg) -> FrozenSet[Pos]:
        return frozenset(scenario.obstacles) | frozenset(c for inc in scenario.incidents for c in inc.cells)


class HindsightPolicy(PushPolicy):
    name = "hindsight"
    uses_gossip = False

    def __init__(self) -> None:
        self.hindsight_buy = 0.0

    def prefill_set(self, scenario, cfg) -> FrozenSet[Pos]:
        h = hindsight(scenario, cfg)
        self.hindsight_buy = float(sum(h.buy_lb[p] for p in h.s_star))
        return h.s_star


def trigger(t: int, agent, plan: PushPlan, known: float, price: float) -> dict:
    return {"tick": t, "robot": agent.id, "cells": (plan.obstacle,), "kind": plan.kind, "landing": plan.landing,
            "steps": plan.steps, "known": float(known), "buy": float(price),
            "bundle": isinstance(plan, BundlePlan), "mode": getattr(plan, "mode", "push")}


def make_policy(cfg: SimConfig) -> PushPolicy:
    name = cfg.policy
    if name == "never":
        return NeverPolicy()
    if name == "myopic":
        return MyopicPolicy()
    if name == "eager":
        return LedgerPolicy("eager", theta=0.0)
    if name == "rof":
        return LedgerPolicy("rof", theta=cfg.theta)
    if name == "rof_local":
        return LedgerPolicy("rof_local", theta=cfg.theta, scope="local")
    if name == "rof_f":
        return LedgerPolicy("rof_f", theta=cfg.theta, forecast=True)
    if name == "rof_r":
        return RandomizedPolicy(theta=cfg.theta)
    if name == "rof_p":
        return PredictedPolicy(lam=cfg.lam)
    if name == "rof_a":
        return GuardedPolicy(lam=cfg.lam)
    if name == "central":
        return CentralPolicy(theta=cfg.theta)
    if name == "hindsight":
        return HindsightPolicy()
    return FreePolicy()
