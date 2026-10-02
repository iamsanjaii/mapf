import os

from src.doi.animate import animate_runs, cumulative_cost, goal_timelines
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def toy():
    s = scenario_from_ascii(ROWS, [(1, 2)], [FOUR], depot_stock=2)
    cfg = SimConfig(n_robots=1, tasks_per_robot=4, claim=False)
    return s, cfg, {p: run_episode(cfg.replace(policy=p), scenario=s) for p in ("never", "rof")}


def test_cost_counter_ends_at_the_run_total():
    s, cfg, runs = toy()
    for r in runs.values():
        assert cumulative_cost(r, cfg)[-1] == r.J


def test_goal_timeline_follows_the_task_list():
    s, cfg, runs = toy()
    line = goal_timelines(runs["rof"], s)[0]
    assert line[0] == (1, 4) and line[-1] is None


def test_gif_is_written(tmp_path):
    s, cfg, runs = toy()
    out = animate_runs(s, runs, cfg, path=str(tmp_path / "x.gif"), fps=4)
    assert os.path.getsize(out) > 5000
