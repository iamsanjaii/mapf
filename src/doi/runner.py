"""run_episode: the tick loop tying world, network, agents and policy together."""
import dataclasses
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.evidence import EvidenceEngine
from src.doi.metrics import RunResult
from src.doi.crdt import ObstructionRecord
from src.doi.incidents import location_names, locate, render_report, report_rng
from src.doi.network import Network
from src.doi.policies import CentralPolicy, HindsightPolicy, PushPolicy, Shared, make_policy
from src.doi.scenarios import Scenario, build_scenario
from src.doi.rng import u01
from src.doi.world import Drop, Move, Pick, Push, World


class Intake:
    """Delivers incident reports to the nearest robot as records, per the configured intake mode."""

    def __init__(self, scenario: Scenario, cfg: SimConfig) -> None:
        self.scenario, self.cfg = scenario, cfg
        self.counts = {"reports": 0, "records": 0, "rejected": 0}
        self.by_tick: Dict[int, List[Tuple[int, object]]] = {}
        for idx, r in enumerate(scenario.reports):
            self.by_tick.setdefault(r.emit_tick, []).append((idx, r))
        self.pending: Dict[int, List[Tuple[int, ObstructionRecord]]] = {}
        self.cache = None
        self.names = location_names(scenario)
        if cfg.intake.startswith("llm:"):
            from src.doi.llm.intake import IntakeCache
            self.model_key = cfg.intake.split(":", 1)[1]
            self.cache = IntakeCache(cfg.intake_cache, self.model_key)

    @staticmethod
    def nearest(agents, cell) -> Optional[int]:
        live = [a for a in agents if not a.finished]
        if not live:
            return None
        return min(live, key=lambda a: (abs(a.pos[0] - cell[0]) + abs(a.pos[1] - cell[1]), a.id)).id

    def emit(self, t: int, agents) -> None:
        cfg, scenario = self.cfg, self.scenario
        for idx, report in self.by_tick.get(t, []):
            cells = locate(scenario, report.location)
            node = self.nearest(agents, cells[0])
            if node is None:
                continue
            self.counts["reports"] += 1
            if cfg.intake == "none":
                continue
            if cfg.intake == "oracle":
                rec, due = ObstructionRecord(report.report_id, node, report.location, cells, report.kind,
                                             report.cls, 1, 1.0, "oracle"), t
            else:
                from src.doi.llm.intake import cache_key, to_record
                text = render_report(report, report_rng(scenario, report))[0]
                res = self.cache.get(cache_key(text, self.names))
                if res is None:
                    raise KeyError(f"no cached intake result for report {report.report_id}; run "
                                   f"experiments/doi_intake_run.py --model-key {self.model_key} "
                                   f"--scenario {scenario.name} --seeds {scenario.meta.get('seed')}..")
                if not res.ok:
                    self.counts["rejected"] += 1
                    continue
                rec = to_record(res, report.report_id, node, scenario)
                due = t + max(1, math.ceil(res.latency_s / cfg.tick_seconds))
            if report.cls == "needs_human" and u01(cfg.seed, idx, 77) < cfg.p_wrong_class:
                rec = dataclasses.replace(rec, cls="robot_clearable")
            self.pending.setdefault(due, []).append((node, rec))

    def deliver(self, t: int, agents) -> None:
        for node, rec in self.pending.pop(t, []):
            target = node if not agents[node].finished else self.nearest(agents, rec.cells[0])
            if target is None:
                continue
            agents[target].ingest_record(rec, t)
            self.counts["records"] += 1


def run_episode(cfg: SimConfig, scenario: Optional[Scenario] = None,
                policy: Optional[PushPolicy] = None) -> RunResult:
    started = time.perf_counter()
    scenario = scenario or build_scenario(cfg)
    policy = policy or make_policy(cfg)
    shared = Shared(engine=EvidenceEngine(scenario.grid,
                                          cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width),
                                          cfg.record_epoch))
    policy.prepare(scenario, cfg, shared)
    world = World(scenario, cfg)
    world.prefill(policy.prefill_set(scenario, cfg))
    network = Network(cfg)
    agents = [RobotAgent(i, scenario, cfg, policy, shared) for i in range(len(scenario.starts))]
    shared.agents = agents
    intake = Intake(scenario, cfg)
    for a in agents:                        # obstacles removed at tick 0 (benchmark arms) are known to everyone
        for c in sorted(set(scenario.obstacles) - set(world.obstacles)):
            a.belief.observe_cell(c, False, 1)      # seen free later than the tick-0 sighting

    ticks, stalled, idle = 0, False, 0
    livelock_ticks = 20 * (scenario.grid.height + scenario.grid.width)
    last_done, last_pushes, last_gain = 0, 0, 0
    gain_carry_cost = 0.0
    waits_at_progress = {i: 0 for i in world.counters}
    gain_tick, gain_counters, gain_push_cost = -1, {i: dict(c) for i, c in world.counters.items()}, 0.0
    for t in range(cfg.horizon or cfg.max_ticks):
        active = [a for a in agents if not a.finished]
        if not active:
            break
        world.begin_tick(t)
        if isinstance(policy, CentralPolicy):
            policy.sync(world, active, t)
        intake.emit(t, agents)
        intake.deliver(t, agents)
        for a in active:
            a.sense(world.observe(a.id, cfg.r_sense), t)
        inbox = network.deliver(t)
        for a in active:
            a.receive(inbox.get(a.id, []), t)
        actions = {a.id: a.decide(t) for a in active}
        positions = dict(world.pos)
        for a in active:
            for m in a.outgoing(t):
                network.send(m, t, positions)
        results = world.apply_actions(t, actions)
        progress = False
        for a in active:
            res = results[a.id]
            a.after_action(res, t)
            if res.ok and isinstance(actions[a.id], (Move, Push, Pick, Drop)):
                progress = True
            if a.finished:
                progress = True
        for a in active:
            if a.finished:
                world.despawn(a.id)
        ticks = t + 1
        idle = 0 if progress else idle + 1
        if progress:
            waits_at_progress = {i: c["waits"] for i, c in world.counters.items()}
        if idle >= cfg.stall_ticks:
            stalled = True
            # Waiting out the stall window is not a cost of the run: charge nothing after the last progress tick,
            # so J does not depend on stall_ticks.
            for i, c in world.counters.items():
                c["waits"] = waits_at_progress[i]
            world.tick_cost[-idle:] = [0.0] * idle
            break
        done_now = sum(a.task_idx for a in agents)
        lifts = world.removals + sum(c["picks"] + c["drops"] for c in world.counters.values())
        if done_now != last_done or lifts != last_pushes:        # robots wandering is not progress
            last_done, last_pushes, last_gain = done_now, lifts, t
            gain_carry_cost = world.carry_cost
            gain_tick, gain_counters, gain_push_cost = t, {i: dict(c) for i, c in world.counters.items()}, world.push_cost
        elif t - last_gain >= livelock_ticks:
            stalled = True
            # Robots shuffling after the last finished task or push are not a cost of the run either.
            for i, c in gain_counters.items():
                world.counters[i].update(c)
            world.push_cost = gain_push_cost
            world.carry_cost = gain_carry_cost
            world.tick_cost[gain_tick + 1:] = [0.0] * (len(world.tick_cost) - gain_tick - 1)
            break
    result = build_result(cfg, policy, world, network, agents, shared, ticks, stalled,
                          (time.perf_counter() - started) * 1000.0, scenario)
    result.intake = dict(intake.counts)
    return result


def build_result(cfg, policy, world, network, agents, shared, ticks, stalled, runtime_ms, scenario) -> RunResult:
    moves = sum(c["moves"] for c in world.counters.values())
    waits = sum(c["waits"] for c in world.counters.values())
    push_steps = sum(c["push_steps"] for c in world.counters.values())
    removals = world.removals
    carry_steps = sum(c["carry_steps"] for c in world.counters.values())
    picks = sum(c["picks"] for c in world.counters.values())
    drops = sum(c["drops"] for c in world.counters.values())
    J = float((moves - push_steps - carry_steps) + waits + world.push_cost + cfg.fee * removals
              + world.carry_cost + cfg.pick_fee * picks + cfg.drop_fee * drops)
    U = cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)
    unfinished = sum(len(a.tasks) - a.task_idx for a in agents)
    done = sum(a.task_idx for a in agents)
    delay = float(sum(a.completed_tick + 1 for a in agents if a.finished)
                  + (cfg.horizon or cfg.max_ticks) * sum(1 for a in agents if not a.finished))
    infos = sorted((i for a in agents for i in a.plan_infos), key=lambda i: (i.robot, i.task_idx))
    return RunResult(
        cfg=cfg.to_dict(), policy=policy.name, J=J, J_censored=J if cfg.horizon else J + U * unfinished, delay=delay,
        throughput=1000.0 * done / max(1, ticks), moves=moves, waits=waits, push_steps=push_steps,
        push_cost=float(world.push_cost), removals=removals, fee_total=cfg.fee * removals,
        unfinished_tasks=unfinished, stalled=stalled, ticks=ticks,
        messages=network.stats.as_dict(), traffic_messages=network.traffic_stats.as_dict(),
        overrides=world.overrides, triggers=list(shared.triggers), pushes=[dict(r) for r in world.push_log],
        push_rejected=sum(c["push_rejected"] for c in world.counters.values()), plan_infos=infos,
        appeared_at=dict(world.appeared_at),
        wrong_class_attempts=sum(c["wrong_class_attempts"] for c in world.counters.values()),
        runtime_ms=runtime_ms,
        hindsight_buy=policy.hindsight_buy if isinstance(policy, HindsightPolicy) else 0.0,
        trajectory={i: list(p) for i, p in world.trajectory.items()},
        obstacle_trace=[dict(o) for o in world.obstacle_trace], tick_cost=list(world.tick_cost),
        scenario=scenario, picks=picks, drops=drops, carries=sum(1 for e in world.carry_log if e["mode"] == "carry"),
        fills=world.fills, carry_steps=carry_steps, carry_cost=float(world.carry_cost),
        slot_conflicts=sum(c["slot_conflicts"] for c in world.counters.values()),
        carry_log=[dict(e) for e in world.carry_log], lift_ticks=list(world.lift_ticks))


def run_arms(cfg: SimConfig, arms: Sequence[str]) -> Dict[str, RunResult]:
    scenario = build_scenario(cfg)
    return {name: run_episode(cfg.replace(policy=name), scenario=scenario) for name in arms}
