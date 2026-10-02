from src.doi.paths import passable_fn, bfs_dist_map, shortest_path, dream_path, single_pit_rents
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario, scenario_from_ascii


def test_single_pit_dream_path_numbers():
    s = build_scenario(SimConfig(scenario="single_pit", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 11), unreachable=144.0)
    assert (r.d_block, r.d_open, r.rent) == (20, 2, 18)
    assert r.bundle == frozenset({(3, 10)})
    r2 = dream_path(s.grid, s.pits, frozenset({(3, 10)}), (3, 9), (3, 11), 144.0)
    assert (r2.d_block, r2.d_open, r2.rent, r2.bundle) == (2, 2, 0, frozenset())


def test_series_needs_bundle():
    s = build_scenario(SimConfig(scenario="series_pits", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 15), 144.0)
    assert (r.d_block, r.d_open, r.rent) == (24, 6, 18)
    assert r.bundle == frozenset({(3, 11), (3, 13)})
    assert single_pit_rents(s.grid, s.pits, frozenset(), (3, 9), (3, 15), 144.0) == {}
    r2 = dream_path(s.grid, s.pits, frozenset({(3, 13)}), (3, 9), (3, 15), 144.0)
    assert r2.bundle == frozenset({(3, 11)}) and r2.rent == 18


def test_parallel_pits_are_substitutes():
    s = build_scenario(SimConfig(scenario="two_pits_parallel", n_robots=1, tasks_per_robot=1))
    rents = single_pit_rents(s.grid, s.pits, frozenset(), (3, 9), (3, 11), 144.0)
    assert rents == {(3, 10): 18, (5, 10): 14}
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 11), 144.0)
    assert r.bundle == frozenset({(3, 10)})


def test_unreachable_uses_penalty():
    s = scenario_from_ascii(["..#..", "..#..", "..#.."], [(0, 0)], [[(0, 4)]])
    r = dream_path(s.grid, s.pits, frozenset(), (0, 0), (0, 4), unreachable=50.0)
    assert r.d_block == 50 and r.d_open == 50 and r.rent == 0 and r.bundle == frozenset()


def test_canonical_tiebreak_is_deterministic():
    s = scenario_from_ascii(["....", "....", "...."], [(0, 0)], [[(2, 3)]])
    pf = passable_fn(s.grid, frozenset())
    a = shortest_path(pf, (0, 0), (2, 3), 3, 4)
    b = shortest_path(pf, (0, 0), (2, 3), 3, 4)
    assert a == b and len(a) == 6


def test_bfs_dist_map():
    s = scenario_from_ascii(["...", ".#.", "..."], [(0, 0)], [[(2, 2)]])
    d = bfs_dist_map(passable_fn(s.grid, frozenset()), (0, 0), 3, 3)
    assert d[(0, 0)] == 0 and d[(2, 2)] == 4 and (1, 1) not in d


def test_dynamic_obstruction_and_hard_blocked():
    s = build_scenario(SimConfig(scenario="incidents_room", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, [(2, 10)], frozenset(), (2, 9), (2, 11), 144.0)
    assert (r.d_block, r.d_open, r.rent, r.bundle) == (12, 2, 10, frozenset({(2, 10)}))
    r2 = dream_path(s.grid, [(2, 10)], frozenset(), (2, 9), (2, 11), 144.0,
                    hard_blocked=frozenset({(7, 10)}))
    assert (r2.d_block, r2.d_open, r2.rent) == (22, 2, 20)
