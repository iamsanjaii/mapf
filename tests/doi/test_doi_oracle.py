import pytest
from src.doi.config import SimConfig
from src.doi.oracle import hindsight, buy_lb, task_pairs
from src.doi.scenarios import build_scenario, scenario_from_ascii

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
CFG = SimConfig(n_robots=1, tasks_per_robot=4)


def make(tasks):
    return scenario_from_ascii(ROWS, [(1, 2)], [tasks], depot_stock=2)


def test_buy_lb_and_pairs():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    assert buy_lb(s, CFG) == {(1, 3): 9.0}
    assert task_pairs(s)[0] == ((1, 2), (1, 4)) and len(task_pairs(s)) == 4


def test_fill_pays_off_with_four_crossings():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    h = hindsight(s, CFG)
    assert h.s_star == frozenset({(1, 3)})
    assert h.lb_empty == 40.0 and h.lb_star == 17.0 and h.exhaustive


def test_fill_does_not_pay_with_one_crossing():
    s = make([(1, 4)])
    h = hindsight(s, SimConfig(n_robots=1, tasks_per_robot=1))
    assert h.s_star == frozenset() and h.lb_star == h.lb_empty == 10.0


def test_series_hindsight_needs_both_pits():
    cfg = SimConfig(scenario="series_pits", n_robots=2, tasks_per_robot=10, seed=1)
    s = build_scenario(cfg)
    h = hindsight(s, cfg)
    assert h.s_star in (frozenset(), frozenset(s.pits))
    assert h.lb_star <= h.lb_empty


def test_family_d_rejected():
    cfg = SimConfig(scenario="incidents_room", n_robots=2, tasks_per_robot=2)
    s = build_scenario(cfg)
    with pytest.raises(ValueError):
        hindsight(s, cfg)
    with pytest.raises(ValueError):
        buy_lb(s, cfg)


def test_greedy_branch_for_many_relevant_pits():
    rows = ["..P.." for _ in range(12)]
    rows[0] = "..P.D"
    tasks = [[(r, 0), (r, 4)] * 3 for r in range(12)]
    s = scenario_from_ascii(rows, [(r, 4) for r in range(12)], tasks, depot_stock=12)
    cfg = SimConfig(n_robots=12, tasks_per_robot=6)
    h = hindsight(s, cfg)
    assert not h.exhaustive and len(h.relevant) == 12
    assert h.lb_star < h.lb_empty and h.s_star <= frozenset(s.pits) and len(h.s_star) >= 1
