import io
from contextlib import redirect_stdout

from run_doi import print_summary
from src.doi.config import SimConfig
from src.doi.paths import bfs_dist_map, passable_fn
from src.doi.runner import run_arms
from src.doi.scenarios import build_scenario


def _dist(grid, closed, a, b):
    return bfs_dist_map(passable_fn(grid, closed=frozenset(closed)), a, grid.height, grid.width).get(b, 10 ** 6)


def test_complements_one_doorway_saves_nothing():
    s = build_scenario(SimConfig(scenario="complements", n_robots=2, tasks_per_robot=4, seed=1))
    west, east = sorted(s.obstacles)
    both = set(s.obstacles)
    pairs = [(s.starts[i], s.tasks[i][0]) for i in range(2)]
    pairs += [(a, b) for i in range(2) for a, b in zip(s.tasks[i], s.tasks[i][1:])]
    for a, b in pairs:
        full = _dist(s.grid, both, a, b)
        assert _dist(s.grid, both - {west}, a, b) == full
        assert _dist(s.grid, both - {east}, a, b) == full
        assert _dist(s.grid, set(), a, b) < full


def test_single_step_arms_never_push_on_complements():
    runs = run_arms(SimConfig(scenario="complements", n_robots=6, tasks_per_robot=8, seed=1),
                    ["never", "rof", "central"])
    assert all(r.removals == 0 for r in runs.values())


def test_two_step_plans_push_on_complements():
    runs = run_arms(SimConfig(scenario="complements", n_robots=6, tasks_per_robot=8, seed=1, bundle_max=2),
                    ["never", "rof", "central"])
    assert runs["central"].removals >= 2 and runs["rof"].removals >= 2
    assert runs["rof"].J_censored < runs["never"].J_censored


def test_bundle_max_one_changes_nothing():
    for scen in ("single_block", "two_blocks_parallel", "multi_block_wall"):
        base = SimConfig(scenario=scen, n_robots=6, tasks_per_robot=6, seed=2)
        a = run_arms(base, ["rof"])["rof"]
        b = run_arms(base.replace(bundle_max=1), ["rof"])["rof"]
        assert a.J == b.J


def test_new_arms_run_and_finish():
    for scen in ("single_block", "complements"):
        runs = run_arms(SimConfig(scenario=scen, n_robots=4, tasks_per_robot=5, seed=1, bundle_max=2),
                        ["rof_r", "rof_p"])
        for r in runs.values():
            assert not r.stalled and r.unfinished_tasks == 0


def test_hindsight_row_includes_buy():
    runs = run_arms(SimConfig(scenario="single_block", n_robots=4, tasks_per_robot=5, seed=1),
                    ["free", "hindsight"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(["free", "hindsight"], runs)
    row = next(l for l in buf.getvalue().splitlines() if l.startswith("hindsight"))
    h = runs["hindsight"]
    assert float(row.split()[1]) == round(h.J_censored + h.hindsight_buy)
