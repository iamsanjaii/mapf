import math
from types import SimpleNamespace

import pytest
from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.forecast.result import ForecastResult
from src.doi.metrics import forecast_columns, summary_row
from src.doi.policies import GuardedPolicy, Shared, make_policy
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario, scenario_from_ascii

ROWS = ["..#....",
        "..L....",
        "..#....",
        "..#....",
        "......."]


class Scripted:
    """A forecaster that returns one fixed result and counts how often it is asked."""
    name = "scripted"

    def __init__(self, answer, latency_s=0.0, failed=""):
        self.result = ForecastResult(answer, None, "", failed, latency_s)
        self.asked = 0

    def forecast(self, case):
        self.asked += 1
        return self.result


def hand(forecaster, lam=0.5, **cfg_kw):
    """One robot at (1, 0) heading for (1, 6); known saving 12 for lifting the pallet out of the gap."""
    s = scenario_from_ascii(ROWS, starts=[(1, 0), (4, 0)], tasks=[[(1, 6), (1, 0), (1, 6)], [(4, 6)]])
    cfg = SimConfig(n_robots=2, tasks_per_robot=3, policy="rof_a", lam=lam, **cfg_kw)
    shared = Shared(engine=EvidenceEngine(s.grid, cfg.unreachable_cost_for(5, 7)))
    policy = GuardedPolicy(lam, forecaster)
    policy.prepare(s, cfg, shared)
    agent = RobotAgent(0, s, cfg, policy, shared)
    agent.belief.add_record(RentRecord(0, 0, (1, 0), (1, 6), 0, 6))
    agent.belief.add_record(RentRecord(1, 0, (1, 0), (1, 6), 0, 6))
    return policy, agent, shared


def plan(total):
    return SimpleNamespace(obstacle=(1, 2), kind="pallet", landing=(1, 2), key=("lift", (1, 2)),
                           before=frozenset({(1, 2)}), after=frozenset(), total=total)


def info(tick):
    return SimpleNamespace(tick=tick, d_open=6)        # price = plan.total - 6


def test_config_knows_the_arm_and_checks_the_forecaster():
    assert SimConfig(policy="rof_a").forecaster == "numeric" and SimConfig().agent_max_forecasts == 200
    assert make_policy(SimConfig(policy="rof_a")).name == "rof_a"
    SimConfig(forecaster="projected")
    with pytest.raises(ValueError):
        SimConfig(forecaster="crystal_ball")
    with pytest.raises(ValueError):
        SimConfig(agent_max_forecasts=-1)


def test_threshold_follows_the_answer():
    for answer, total, fires in ((True, 26.0, True), (False, 26.0, False),      # price 20: 12 >= 10, 12 < 40
                                 (False, 14.0, False), (None, 14.0, True),      # price 8: 12 < 16; no answer: 12 >= 8
                                 (None, 26.0, False)):                          # no answer: 12 < 20
        policy, agent, _ = hand(Scripted(answer))
        verdict = policy.assess(agent, plan(total), info(5))
        assert (verdict is not None) is fires, (answer, total)
        if fires:
            assert verdict == (12.0 - (total - 6), 12.0, total - 6)


def test_a_late_answer_is_ignored_until_it_is_due():
    policy, agent, shared = hand(Scripted(True, latency_s=2.0))                 # 2 s at 0.5 s per tick: 4 ticks
    p = plan(26.0)                                                              # price 20
    assert policy.assess(agent, p, info(5)) is None
    state = agent.forecasts[(("lift", (1, 2)),)]
    assert (state.asked, state.due, state.answer) == (5, 9, True)
    assert policy.assess(agent, p, info(8)) is None
    policy.on_tick([agent], 8)
    assert agent.forecast_epoch == 0
    policy.on_tick([agent], 9)
    assert agent.forecast_epoch == 1
    assert policy.assess(agent, p, info(9)) == (-8.0, 12.0, 20.0)
    assert policy.forecaster.asked == 1 and len(shared.forecast_log) == 1
    row = shared.forecast_log[0]
    assert (row["tick"], row["robot"], row["due"], row["answer"], row["skipped"]) == (5, 0, 9, True, False)
    assert row["truth"] is False and row["numeric_forecast"] is True and row["forecaster"] == "scripted"


def test_no_forecast_below_the_lazy_line_at_zero_price_or_with_lam_one():
    policy, agent, _ = hand(Scripted(True))
    assert policy.assess(agent, plan(36.0), info(5)) is None                    # price 30: 12 < 15, nothing asked
    assert agent.forecasts == {} and policy.forecaster.asked == 0
    assert policy.assess(agent, plan(6.0), info(5)) == (12.0, 12.0, 0.0)        # price 0: fires, nothing asked
    assert policy.forecaster.asked == 0
    policy, agent, _ = hand(Scripted(False), lam=1.0)
    assert policy.assess(agent, plan(14.0), info(5)) == (4.0, 12.0, 8.0)        # lam 1: the classical rule
    assert policy.forecaster.asked == 0


def test_budget_of_zero_skips_every_request():
    policy, agent, shared = hand(Scripted(True), agent_max_forecasts=0)
    assert policy.assess(agent, plan(26.0), info(5)) is None                    # no answer: threshold 1
    assert policy.forecaster.asked == 0 and shared.forecast_log[0]["skipped"] is True
    assert shared.forecast_cases == []


def _run(scenario, policy_obj=None, **kw):
    cfg = SimConfig(scenario=scenario, n_robots=6, tasks_per_robot=6, seed=2, **kw)
    s = build_scenario(cfg)
    return run_episode(cfg, scenario=s, policy=policy_obj), s


def test_a_forecaster_that_always_fails_leaves_the_classical_rule():
    rof, _ = _run("single_block", policy="rof")
    failing, _ = _run("single_block", GuardedPolicy(0.5, Scripted(None, failed="client_error")), policy="rof_a")
    assert failing.J == rof.J == 519.0 and failing.pushes == rof.pushes
    assert failing.forecast_log and all(r["failed"] == "client_error" for r in failing.forecast_log)


def test_requests_are_lazy_and_made_once_per_robot_and_plan():
    res, _ = _run("multi_block_wall", GuardedPolicy(0.5, Scripted(True)), policy="rof_a")
    rows = res.forecast_log
    assert rows and all(r["price"] > 0 and r["known"] >= 0.5 * r["price"] for r in rows)
    assert len({(r["robot"], r["plan_key"]) for r in rows}) == len(rows) == len(res.forecast_cases)
    assert not res.stalled and res.unfinished_tasks == 0


@pytest.mark.parametrize("mode", ["true", "false", "missing", "quiet"])
@pytest.mark.parametrize("forecaster", ["numeric", "keyword", "oracle", "inverted"])
def test_rof_a_finishes_shift_notice(mode, forecaster):
    cfg = SimConfig(scenario="shift_notice", n_robots=6, tasks_per_robot=12, seed=3, policy="rof_a",
                    forecaster=forecaster, scenario_params={"notice_mode": mode})
    res = run_episode(cfg)
    assert not res.stalled and res.unfinished_tasks == 0 and res.forecast_log
    answers = [(r["answer"], r["truth"]) for r in res.forecast_log]
    if forecaster == "oracle":
        assert all(a == t for a, t in answers)
    if forecaster == "inverted":
        assert all(a != t for a, t in answers)
    row = summary_row(res)
    assert row["forecaster"] == forecaster and row["forecasts"] == len(res.forecast_log)
    if forecaster in ("oracle", "inverted"):
        assert row["forecast_acc"] == (1.0 if forecaster == "oracle" else 0.0)


def test_a_scenario_with_no_zones_and_no_notices_still_runs():
    res, _ = _run("single_block", policy="rof_a", forecaster="keyword")
    assert not res.stalled and res.forecast_cases
    assert all(c.zones == {} and c.trip_saving == {} and c.notices == () for c in res.forecast_cases)
    assert all(r["answer"] == r["numeric_forecast"] for r in res.forecast_log)   # keyword falls back to numeric


def test_forecast_columns():
    log = [dict(answer=True, truth=True, failed="", skipped=False),
           dict(answer=True, truth=False, failed="", skipped=False),
           dict(answer=False, truth=True, failed="", skipped=False),
           dict(answer=None, truth=True, failed="no_answer", skipped=False),
           dict(answer=None, truth=None, failed="", skipped=True)]
    cols = forecast_columns(log)
    assert cols == {"forecasts": 5, "forecast_failed": 1, "forecast_skipped": 1, "forecast_acc": pytest.approx(1 / 3),
                    "wrong_yes": 1, "wrong_no": 1}
    empty = forecast_columns([])
    assert empty["forecasts"] == 0 and math.isnan(empty["forecast_acc"])
