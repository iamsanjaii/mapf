"""run_episode: the tick loop tying world, network, agents and policy together."""
import time
from typing import Dict, Optional, Sequence

from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.evidence import EvidenceEngine
from src.doi.metrics import RunResult
from src.doi.network import Network
from src.doi.policies import CentralPolicy, FillPolicy, HindsightPolicy, Shared, make_policy
from src.doi.scenarios import Scenario, build_scenario
from src.doi.world import Drop, Move, Pickup, Return, World


def run_episode(cfg: SimConfig, scenario: Optional[Scenario] = None,
                policy: Optional[FillPolicy] = None) -> RunResult:
    started = time.perf_counter()
    scenario = scenario or build_scenario(cfg)
    policy = policy or make_policy(cfg)
    shared = Shared(engine=EvidenceEngine(scenario.grid,
                                          cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)))
    policy.prepare(scenario, cfg, shared)
    world = World(scenario, cfg)
    world.prefill(policy.prefill_set(scenario, cfg))
    network = Network(cfg)
    agents = [RobotAgent(i, scenario, cfg, policy, shared) for i in range(len(scenario.starts))]
    shared.agents = agents
    for a in agents:
        for c in sorted(world.filled):
            a.belief.filled.add(c)

    ticks, stalled, idle = 0, False, 0
    for t in range(cfg.max_ticks):
        active = [a for a in agents if not a.finished]
        if not active:
            break
        world.begin_tick(t)
        if isinstance(policy, CentralPolicy):
            policy.sync(world, active, t)
            policy.dispatch(active, t)
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
            if res.ok and isinstance(actions[a.id], (Move, Pickup, Drop, Return)):
                progress = True
            if a.finished:
                progress = True
        for a in active:
            if a.finished:
                world.despawn(a.id)
        ticks = t + 1
        idle = 0 if progress else idle + 1
        if idle >= cfg.stall_ticks:
            stalled = True
            break
    return build_result(cfg, policy, world, network, agents, shared, ticks, stalled,
                        (time.perf_counter() - started) * 1000.0, scenario)


def build_result(cfg, policy, world, network, agents, shared, ticks, stalled, runtime_ms, scenario) -> RunResult:
    moves = sum(c["moves"] for c in world.counters.values())
    waits = sum(c["waits"] for c in world.counters.values())
    carried = sum(c["carried_steps"] for c in world.counters.values())
    fills = world.fills
    J = float((moves - carried) + waits + cfg.kappa * carried + cfg.fee * fills)
    U = cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)
    unfinished = sum(len(a.tasks) - a.task_idx for a in agents)
    done = sum(a.task_idx for a in agents)
    delay = float(sum(a.completed_tick + 1 for a in agents if a.finished)
                  + cfg.max_ticks * sum(1 for a in agents if not a.finished))
    infos = sorted((i for a in agents for i in a.plan_infos), key=lambda i: (i.robot, i.task_idx))
    return RunResult(
        cfg=cfg.to_dict(), policy=policy.name, J=J, J_censored=J + U * unfinished, delay=delay,
        throughput=1000.0 * done / max(1, ticks), moves=moves, waits=waits, carried_steps=carried,
        fills=fills, fee_total=cfg.fee * fills, unfinished_tasks=unfinished, stalled=stalled, ticks=ticks,
        messages=network.stats.as_dict(), traffic_messages=network.traffic_stats.as_dict(),
        overrides=world.overrides,
        fill_ticks=sorted(v for v in world.filled_at.values() if v >= 0),
        triggers=list(shared.triggers), plan_infos=infos, filled_at=dict(world.filled_at),
        appeared_at=dict(world.appeared_at),
        edits=sorted((e for a in agents for e in a.hauler.edits), key=lambda e: (e["fill_tick"], e["pit"])),
        wasted_haul_cost=float(cfg.kappa * sum(a.hauler.wasted_steps for a in agents)),
        unconfirmed_hauls=sum(a.hauler.stats["unconfirmed"] for a in agents),
        claims={"issued": sum(a.hauler.stats["issued"] for a in agents),
                "lost": sum(a.hauler.stats["lost"] for a in agents),
                "aborts": sum(a.hauler.stats["aborts"] for a in agents)},
        wrong_class_attempts=sum(c["wrong_class_attempts"] for c in world.counters.values()),
        final_stock=dict(world.stock), runtime_ms=runtime_ms,
        hindsight_buy=policy.hindsight_buy if isinstance(policy, HindsightPolicy) else 0.0,
        trajectory={i: list(p) for i, p in world.trajectory.items()}, scenario=scenario)


def run_arms(cfg: SimConfig, arms: Sequence[str]) -> Dict[str, RunResult]:
    scenario = build_scenario(cfg)
    return {name: run_episode(cfg.replace(policy=name), scenario=scenario) for name in arms}
