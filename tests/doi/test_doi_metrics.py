from src.doi.config import SimConfig
from src.doi.metrics import stale_detour_cost, true_rent_at_triggers, hindsight_ratios, summary_row
from src.doi.runner import run_arms, run_episode
from src.doi.scenarios import scenario_from_ascii, build_scenario

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def arms(claim=False):
    s = scenario_from_ascii(ROWS, [(1, 2)], [FOUR], depot_stock=2)
    cfg = SimConfig(n_robots=1, tasks_per_robot=4, claim=claim, debug_checks=True)
    return {n: run_episode(cfg.replace(policy=n), scenario=s)
            for n in ("never", "rof", "hindsight", "free")}, s, cfg


def test_hindsight_ratios_single_robot():
    out, s, cfg = arms()
    r = hindsight_ratios(out["rof"], out["hindsight"], out["free"])
    assert abs(r["hr"] - 29.0 / 17.0) < 1e-9
    assert abs(r["hr_av"] - (29.0 - 8.0) / (17.0 - 8.0)) < 1e-9
    n = hindsight_ratios(out["never"], out["hindsight"], out["free"])
    assert abs(n["hr_av"] - (40.0 - 8.0) / 9.0) < 1e-9


def test_true_rent_and_coverage_single_robot_is_complete():
    out, s, cfg = arms()
    t = true_rent_at_triggers(out["rof"])[0]
    assert t["true_rent"] == 16.0 and t["coverage"] == 1.0


def test_stale_cost_zero_when_belief_correct():
    out, s, cfg = arms()
    assert stale_detour_cost(out["rof"], s, cfg) == 0.0


def test_summary_row_has_all_keys():
    out, s, cfg = arms()
    row = summary_row(out["rof"], ratios={"hr": 1.7, "hr_av": 2.3})
    for k in ("policy", "J", "hr", "hr_av", "pod", "mean_coverage", "message_units", "traffic_units",
              "overrides_per_1000", "mean_B_real", "unconfirmed_hauls", "wasted_haul_cost", "stalled"):
        assert k in row


def test_run_arms_share_scenario():
    cfg = SimConfig(n_robots=3, tasks_per_robot=3, seed=4, claim=False)
    out = run_arms(cfg, ["never", "rof", "hindsight", "free"])
    assert set(out) == {"never", "rof", "hindsight", "free"}
    assert out["never"].cfg["seed"] == out["rof"].cfg["seed"] == 4


def test_stale_cost_positive_without_comm():
    hits = 0
    for seed in range(5):
        cfg = SimConfig(scenario="single_pit", n_robots=6, tasks_per_robot=15, seed=seed, claim=False,
                        r_comm=0.0, r_sense=1, policy="rof")
        s = build_scenario(cfg)
        r = run_episode(cfg, scenario=s)
        if stale_detour_cost(r, s, cfg) > 0:
            hits += 1
    assert hits >= 1
