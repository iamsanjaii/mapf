"""Hindsight price lower bounds with carry and fill: the cheapest conceivable way to make an obstacle vanish."""
import math

from src.doi.config import SimConfig
from src.doi.oracle import buy_lb, hindsight
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario, scenario_from_ascii

CFG = SimConfig(n_robots=1, tasks_per_robot=1)            # kappa 4, fee 1, pick 1, drop 1


def lb(rows):
    s = scenario_from_ascii(rows, [(0, 0)], [[(0, 1)]])
    return buy_lb(s, CFG)


def test_a_pit_costs_at_least_a_pick_and_a_drop():
    out = lb(["P.R"])
    assert out[(0, 0)] == 2.0                    # pick 1 + drop 1, if there is debris to lift at all


def test_a_pit_with_no_debris_cannot_be_removed():
    assert math.isinf(lb(["P.."])[(0, 0)])


def test_a_crate_that_has_a_rack_is_priced_by_the_cheaper_mode():
    out = lb([".C.", "T.."])
    assert out[(0, 1)] == 2.0                    # carry: pick 1 + drop 1, against a push of 1 + 4 * 0.5 = 3


def test_a_shelf_unit_with_only_a_rack_is_priced_by_the_push():
    out = lb([".S.", "T.."])                     # a rack does not take a shelf unit
    assert out[(0, 1)] == 1.0 + 4.0 * 2.0


def test_hindsight_removes_pits_for_the_benchmark_arm():
    cfg = SimConfig(scenario="site_pits", n_robots=4, tasks_per_robot=3, seed=1, policy="hindsight")
    s = build_scenario(cfg)
    h = hindsight(s, cfg)
    assert h.s_star and all(s.obstacles[c] in ("pit", "debris") for c in h.s_star)
    r = run_episode(cfg, scenario=s)
    assert r.unfinished_tasks == 0 and r.hindsight_buy > 0 and r.picks == 0
