from src.doi.config import SimConfig
from src.doi.metrics import collateral_cost, hindsight_ratios, stale_detour_cost, summary_row
from src.doi.runner import run_arms, run_episode
from src.doi.scenarios import build_scenario, scenario_from_ascii

ROWS = ["...#...", "...S...", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def arms(tasks=FOUR, names=("never", "rof", "hindsight", "free")):
    s = scenario_from_ascii(ROWS, [(1, 2)], [tasks])
    cfg = SimConfig(n_robots=1, tasks_per_robot=len(tasks), debug_checks=True)
    return {n: run_episode(cfg.replace(policy=n), scenario=s) for n in names}, s, cfg


def test_hindsight_ratios_single_robot():
    out, s, cfg = arms()
    r = hindsight_ratios(out["rof"], out["hindsight"], out["free"])
    assert abs(r["hr"] - 31.0 / 17.0) < 1e-9
    assert abs(r["hr_av"] - (31.0 - 8.0) / (17.0 - 8.0)) < 1e-9
    n = hindsight_ratios(out["never"], out["hindsight"], out["free"])
    assert abs(n["hr_av"] - (40.0 - 8.0) / 9.0) < 1e-9


def test_stale_cost_zero_when_belief_correct():
    out, s, cfg = arms()
    assert stale_detour_cost(out["rof"], s, cfg) == 0.0


def test_collateral_cost_is_the_detour_a_parked_obstacle_causes():
    out, s, cfg = arms([(1, 4), (1, 6)], ("eager",))
    r = out["eager"]
    assert r.pushes[0]["landing"] == (1, 5)
    assert collateral_cost(r, s, cfg) == 2.0              # (1,4) -> (1,6) must go round the parked shelf
    never, _, _ = arms([(1, 4), (1, 6)], ("never",))
    assert collateral_cost(never["never"], s, cfg) == 0.0


def test_summary_row_has_all_keys():
    out, s, cfg = arms()
    row = summary_row(out["rof"], ratios={"hr": 1.7, "hr_av": 2.3})
    for k in ("policy", "J", "hr", "hr_av", "pod", "message_units", "traffic_units", "overrides_per_1000",
              "removals", "push_steps", "push_rejected", "collateral_cost", "stale_detour_cost", "stalled"):
        assert k in row
    assert row["removals"] == 1 and row["collateral_cost"] == 0.0


def test_run_arms_share_scenario():
    cfg = SimConfig(n_robots=3, tasks_per_robot=3, seed=4)
    out = run_arms(cfg, ["never", "rof", "hindsight", "free"])
    assert set(out) == {"never", "rof", "hindsight", "free"}
    assert out["never"].cfg["seed"] == out["rof"].cfg["seed"] == 4


def test_stale_cost_positive_without_comm():
    hits = 0
    for seed in range(5):
        cfg = SimConfig(scenario="single_block", n_robots=6, tasks_per_robot=15, seed=seed,
                        r_comm=0.0, r_sense=1, policy="rof")
        s = build_scenario(cfg)
        r = run_episode(cfg, scenario=s)
        if stale_detour_cost(r, s, cfg) > 0:
            hits += 1
    assert hits >= 1
