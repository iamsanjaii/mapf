import pytest
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario, scenario_from_ascii
from src.environment.grid import CellType


def test_single_block_geometry():
    s = build_scenario(SimConfig(scenario="single_block", n_robots=4, tasks_per_robot=5, seed=1))
    g = s.grid
    assert (g.height, g.width) == (15, 21)
    assert s.obstacles == {(3, 10): "pallet"}
    assert g.get(3, 10) == CellType.FREE                       # the obstacle is dynamic, the opening is floor
    assert g.get(0, 10) == CellType.OBSTACLE and g.get(11, 10) == CellType.OBSTACLE
    assert all(g.get(r, 10) == CellType.FREE for r in (12, 13, 14))
    assert len(s.starts) == 4 and len(set(s.starts)) == 4
    assert all(len(t) == 5 for t in s.tasks)
    assert all(c[1] < 10 for c in s.starts[:2]) and all(c[1] > 10 for c in s.starts[2:])
    assert not any(c in s.obstacles for t in s.tasks for c in t)


def test_series_blocks_geometry():
    s = build_scenario(SimConfig(scenario="series_blocks", n_robots=2, tasks_per_robot=2))
    g = s.grid
    assert sorted(s.obstacles) == [(3, 11), (3, 13)]
    assert [g.get(3, c) for c in range(10, 15)] == [CellType.FREE] * 5
    assert g.get(2, 12) == CellType.OBSTACLE and g.get(4, 12) == CellType.OBSTACLE


def test_kinds_cycle_over_the_block_cells():
    s = build_scenario(SimConfig(scenario="multi_block_wall", n_robots=2, tasks_per_robot=2,
                                 scenario_params={"kinds": ("pallet", "crate", "shelf_unit")}))
    assert [s.obstacles[c] for c in sorted(s.obstacles)] == ["pallet", "crate", "shelf_unit", "pallet"]
    with pytest.raises(ValueError, match="unknown obstacle kind"):
        build_scenario(SimConfig(scenario="single_block", scenario_params={"kinds": ("anvil",)}))
    with pytest.raises(ValueError, match="outside the barrier"):
        build_scenario(SimConfig(scenario="single_block", scenario_params={"block_cells": ((3, 4),)}))


def test_build_is_deterministic_and_seed_sensitive():
    a = build_scenario(SimConfig(seed=3, n_robots=3, tasks_per_robot=4))
    b = build_scenario(SimConfig(seed=3, n_robots=3, tasks_per_robot=4))
    c = build_scenario(SimConfig(seed=4, n_robots=3, tasks_per_robot=4))
    assert (a.starts, a.tasks) == (b.starts, b.tasks)
    assert (a.starts, a.tasks) != (c.starts, c.tasks)


def test_tasks_do_not_depend_on_other_robot_count():
    a = build_scenario(SimConfig(seed=3, n_robots=4, tasks_per_robot=4))
    b = build_scenario(SimConfig(seed=3, n_robots=6, tasks_per_robot=4))
    assert a.tasks[0] == b.tasks[0]


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        build_scenario(SimConfig(scenario_params={"bogus": 1}))


def test_ascii_scenario():
    rows = ["...#...", "...L..C", "...#...", "..S...."]
    s = scenario_from_ascii(rows, starts=[(1, 2)], tasks=[[(1, 4)]])
    assert s.obstacles == {(1, 3): "pallet", (1, 6): "crate", (3, 2): "shelf_unit"}
    assert s.grid.get(0, 3) == CellType.OBSTACLE and s.grid.get(1, 3) == CellType.FREE
    with pytest.raises(ValueError):
        scenario_from_ascii(["..", "..."], [(0, 0)], [[(0, 1)]])
    with pytest.raises(ValueError, match="bad ascii"):
        scenario_from_ascii(["..?"], [(0, 0)], [[(0, 1)]])


def test_random_blocks_strips_and_counts():
    cfg = SimConfig(scenario="random_blocks", n_robots=4, tasks_per_robot=5, seed=2,
                    scenario_params=dict(H=16, W=18, strips=4, strip_len_min=3, strip_len_max=6,
                                         pallets=2, crates=3, shelves=1))
    s = build_scenario(cfg)
    kinds = sorted(s.obstacles.values())
    assert kinds == ["crate"] * 3 + ["pallet"] * 2 + ["shelf_unit"]
    assert (s.grid.height, s.grid.width) == (16, 18)
    assert any(s.grid.get(r, c) == CellType.OBSTACLE for r in range(16) for c in range(18))
    assert all(s.grid.get(*c) == CellType.FREE for c in s.obstacles)
    assert not any(goal in s.obstacles for t in s.tasks for goal in t)
    assert not any(c in s.obstacles for c in s.starts)
    assert build_scenario(cfg).obstacles == s.obstacles
    with pytest.raises(ValueError, match="too many"):
        build_scenario(cfg.replace(scenario_params=dict(pallets=200)))


def test_shift_goals_follow_band():
    cfg = SimConfig(scenario="shift", n_robots=6, tasks_per_robot=20, seed=2)
    s = build_scenario(cfg)
    before = [t[k] for t in s.tasks for k in range(0, 10) if t[k][1] > 10]
    after = [t[k] for t in s.tasks for k in range(10, 20) if t[k][1] > 10]
    assert sum(1 for p in before if p[0] <= 6) / max(1, len(before)) > 0.6
    assert sum(1 for p in after if p[0] >= 8) / max(1, len(after)) > 0.6


def test_aisles_geometry_and_locations():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=4, tasks_per_robot=3, seed=1))
    g = s.grid
    assert (g.height, g.width) == (13, 21) and s.family == "D" and s.obstacles == {}
    assert g.get(1, 1) == CellType.OBSTACLE and g.get(0, 1) == CellType.FREE and g.get(6, 1) == CellType.FREE
    assert s.locations["aisle 2 bay 3"] == ((3, 2),)
    assert s.locations["aisle 2 bay 6"] == ((7, 2),)


def test_incidents_generated_and_goals_avoid_them():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=6, tasks_per_robot=5, seed=1))
    cells = [c for inc in s.incidents for c in inc.cells]
    assert len(s.incidents) == 4 and len(set(cells)) == 4
    assert all(c[1] not in (0, 20) for c in cells)
    assert all(s.grid.get(*c) == CellType.FREE for c in cells)
    assert all(inc.cls in ("robot_clearable", "needs_human") for inc in s.incidents)
    assert not any(goal in cells for t in s.tasks for goal in t)
    assert all(r.oid is not None for r in s.reports) and len(s.reports) <= 4


def test_false_reports_point_at_clear_cells():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=2, tasks_per_robot=2, seed=3,
                                 scenario_params={"p_false": 1.0}))
    false = [r for r in s.reports if r.oid is None]
    cells = {c for inc in s.incidents for c in inc.cells}
    assert len(false) == 4
    assert all(s.locations[r.location][0] not in cells for r in false)
    assert [r.report_id for r in s.reports] == [f"r{i}" for i in range(len(s.reports))]


def test_room_doors():
    s = build_scenario(SimConfig(scenario="incidents_room", n_robots=4, tasks_per_robot=3, seed=1))
    assert s.grid.get(2, 10) == CellType.FREE and s.grid.get(3, 10) == CellType.OBSTACLE
    assert s.locations["north door"] == ((2, 10),)
    with pytest.raises(ValueError):
        build_scenario(SimConfig(scenario="incidents_room", scenario_params={"n_incidents": 3}))


def test_the_default_fleet_barrier_has_all_three_kinds():
    s = build_scenario(SimConfig(scenario="multi_block_wall", n_robots=2, tasks_per_robot=2))
    assert sorted(set(s.obstacles.values())) == ["crate", "pallet", "shelf_unit"]
