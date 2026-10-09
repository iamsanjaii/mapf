import pytest
from src.doi.config import SimConfig
from src.doi.oracle import hindsight, buy_lb, task_pairs
from src.doi.scenarios import build_scenario, scenario_from_ascii

ROWS = ["...#...", "...L...", "...#...", "...#...", "...#...", "......."]
CFG = SimConfig(n_robots=1, tasks_per_robot=4)


def make(tasks, rows=ROWS):
    return scenario_from_ascii(rows, [(1, 2)], [tasks])


def test_buy_lb_and_pairs():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    assert buy_lb(s, CFG) == {(1, 3): 5.0}                      # fee 1 + kappa 4 * weight 1.0: one push step
    assert task_pairs(s)[0] == ((1, 2), (1, 4)) and len(task_pairs(s)) == 4
    heavy = scenario_from_ascii(["..S.."], [(0, 0)], [[(0, 4)]])
    assert buy_lb(heavy, CFG) == {(0, 2): 1.0 + 4.0 * 2.0}


def test_removal_pays_off_with_four_crossings():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    h = hindsight(s, CFG)
    assert h.s_star == frozenset({(1, 3)})
    assert h.lb_empty == 40.0 and h.lb_star == 13.0 and h.exhaustive


def test_removal_does_not_pay_when_the_fee_is_high():
    s = make([(1, 4)])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1, fee=20.0)
    h = hindsight(s, cfg)
    assert h.s_star == frozenset() and h.lb_star == h.lb_empty == 10.0


def test_an_obstacle_that_can_never_be_pushed_is_never_removed():
    s = scenario_from_ascii(["###", "#L#", "###"], [(1, 1)], [[(1, 1)]])
    assert buy_lb(s, CFG) == {(1, 1): float("inf")}
    h = hindsight(make([(1, 4)], ["...#...", "...#...", "...#...", "...#...", "...#...", "......."]), CFG)
    assert h.s_star == frozenset() and h.relevant == ()


def test_family_d_rejected():
    cfg = SimConfig(scenario="incidents_room", n_robots=2, tasks_per_robot=2)
    s = build_scenario(cfg)
    with pytest.raises(ValueError):
        hindsight(s, cfg)
    with pytest.raises(ValueError):
        buy_lb(s, cfg)


def test_greedy_branch_for_many_relevant_obstacles():
    rows = ["..L.." for _ in range(12)]
    tasks = [[(r, 0), (r, 4)] * 3 for r in range(12)]
    s = scenario_from_ascii(rows, [(r, 4) for r in range(12)], tasks)
    cfg = SimConfig(n_robots=12, tasks_per_robot=6)
    h = hindsight(s, cfg)
    assert not h.exhaustive and len(h.relevant) == 12
    assert h.lb_star < h.lb_empty and h.s_star <= frozenset(s.obstacles) and len(h.s_star) >= 1
