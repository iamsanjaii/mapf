from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.maps import load_movingai
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario


def write_warehouse(path):
    from src.doi.maps import write_synthetic_warehouse
    return write_synthetic_warehouse(str(path))


def test_load_movingai(tmp_path):
    p = tmp_path / "m.map"
    p.write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\nG.T\n")
    g = load_movingai(str(p))
    assert (g.height, g.width) == (2, 3)
    assert g.is_passable(0, 0) and not g.is_passable(0, 1) and g.is_passable(1, 0) and not g.is_passable(1, 2)


def test_delta_merge_equals_full_merge():
    a, b = BeliefState(0, {(0, 0): 2}, []), BeliefState(1, {(0, 0): 2}, [])
    b.add_record(RentRecord(1, 0, (0, 0), (0, 3), 0, 5))
    a.merge(b)
    v = b.version
    b.add_record(RentRecord(1, 1, (0, 3), (0, 0), 4, 6))
    b.observe_cell((2, 2), True, 4)
    x, y = a.snapshot(), a.snapshot()
    x.merge(b.delta_since(v))
    y.merge(b)
    assert x.canonical() == y.canonical()


def test_aggregated_record_has_count_and_equal_evidence(tmp_path):
    from src.doi.evidence import EvidenceEngine
    from src.doi.scenarios import scenario_from_ascii
    rows = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
    s = scenario_from_ascii(rows, [(1, 2)], [[(1, 4)]], depot_stock=2)
    per_task, agg = BeliefState(0, s.depots, s.pits), BeliefState(0, s.depots, s.pits)
    for k, tick in enumerate((3, 40)):
        rec = RentRecord(0, k, (1, 2), (1, 4), tick, 8)
        per_task.add_record(rec)
        agg.add_rent(rec, 100)
    entries = agg.agg.entries()
    assert len(entries) == 1 and entries[0][1] == (16.0, 2)
    e1 = EvidenceEngine(s.grid, 144.0).evidence(per_task, 50)
    e2 = EvidenceEngine(s.grid, 144.0, record_epoch=100).evidence(agg, 50)
    assert e1 == e2 == {((1, 3),): 16.0}
    other_epoch = BeliefState(0, s.depots, s.pits)
    other_epoch.add_rent(RentRecord(0, 5, (1, 2), (1, 4), 150, 8), 100)
    assert len(other_epoch.agg.entries()) == 1
    assert EvidenceEngine(s.grid, 144.0, 100).evidence(agg, 400, window=100) == {}


def test_aggregated_merge_is_idempotent_and_commutative():
    a, b = BeliefState(0, {(0, 0): 1}, []), BeliefState(1, {(0, 0): 1}, [])
    a.add_rent(RentRecord(0, 0, (0, 0), (0, 3), 1, 5), 10)
    b.add_rent(RentRecord(1, 0, (0, 0), (0, 3), 2, 7), 10)
    x, y = a.snapshot(), b.snapshot()
    x.merge(b)
    y.merge(a)
    x.merge(b)
    assert x.canonical() == y.canonical()


def test_record_epoch_run_matches_per_task_run_on_single_robot():
    from src.doi.scenarios import scenario_from_ascii
    rows = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
    tasks = [[(1, 4), (1, 2), (1, 4), (1, 2)]]
    s = scenario_from_ascii(rows, [(1, 2)], tasks, depot_stock=2)
    base = dict(policy="rof", n_robots=1, tasks_per_robot=4, claim=False)
    a = run_episode(SimConfig(**base), scenario=s)
    b = run_episode(SimConfig(record_epoch=1000, **base), scenario=s)
    assert a.J == b.J == 29.0 and a.fills == b.fills == 1


def test_delta_gossip_sends_fewer_units_and_still_completes():
    base = dict(scenario="multi_pit_wall", n_robots=8, tasks_per_robot=8, seed=3, policy="rof", claim=True,
                r_comm=float("inf"))
    full = run_episode(SimConfig(**base))
    delta = run_episode(SimConfig(delta_gossip=True, full_sync_period=20, **base))
    assert not delta.stalled and delta.unfinished_tasks == 0
    assert delta.messages["units"] < full.messages["units"]
    assert delta.fills >= 1


def test_plan_window_path_reaches_goal_and_respects_reservations_inside_window():
    from src.doi.paths import bfs_dist_map, passable_fn
    from src.doi.scenarios import scenario_from_ascii
    from src.doi.spacetime import plan_spacetime
    s = scenario_from_ascii(["." * 12, "." * 12], [(0, 0)], [[(0, 11)]])
    pf = passable_fn(s.grid, frozenset())
    h = bfs_dist_map(pf, (0, 11), 2, 12)
    reserved = {((0, 2), 2)}
    path = plan_spacetime(pf, (0, 0), (0, 11), 0, reserved, h, 60, window=4)
    assert path[0] == (0, 0) and path[-1] == (0, 11)
    assert all((c, k) not in reserved for k, c in enumerate(path[:5]))
    assert len(path) - 1 >= 11


def test_plan_window_episode_completes():
    cfg = SimConfig(scenario="multi_pit_wall", n_robots=6, tasks_per_robot=5, seed=2, policy="never", plan_window=8)
    r = run_episode(cfg)
    assert not r.stalled and r.unfinished_tasks == 0


def test_warehouse_pits_scenario_and_horizon_run(tmp_path):
    path = write_warehouse(tmp_path / "w.map")
    cfg = SimConfig(scenario="warehouse_pits", n_robots=6, seed=1, map_path=path, horizon=120,
                    scenario_params={"n_pits": 4, "n_stations": 2}, policy="rof")
    s = build_scenario(cfg)
    assert s.family == "S" and len(s.pits) == 4 and len(s.depots) == 2
    assert all(len(t) == 120 // 5 + 50 for t in s.tasks)
    assert len({*s.starts}) == 6
    r = run_episode(cfg, scenario=s)
    assert r.ticks == 120 and r.J_censored == r.J and r.throughput > 0
    free = run_episode(cfg.replace(policy="free"), scenario=s)
    assert free.throughput >= r.throughput * 0.5


def test_warehouse_incidents_scenario(tmp_path):
    path = write_warehouse(tmp_path / "w.map")
    cfg = SimConfig(scenario="warehouse_incidents", n_robots=4, seed=2, map_path=path, horizon=100,
                    scenario_params={"n_incidents": 3, "appear_max": 40}, intake="oracle", policy="rof")
    s = build_scenario(cfg)
    assert s.family == "D" and len(s.incidents) == 3
    assert all(name.startswith("corridor ") for name in s.locations)
    r = run_episode(cfg, scenario=s)
    assert r.ticks == 100 and r.unconfirmed_hauls == 0


def test_warehouse_requires_map_path():
    import pytest
    with pytest.raises(ValueError):
        build_scenario(SimConfig(scenario="warehouse_pits"))
