import os

from src.doi.animate import animate_runs, cumulative_cost, goal_timelines, pushing_robots
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

ROWS = ["...#...", "...S...", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def toy():
    s = scenario_from_ascii(ROWS, [(1, 2)], [FOUR])
    cfg = SimConfig(n_robots=1, tasks_per_robot=4)
    return s, cfg, {p: run_episode(cfg.replace(policy=p), scenario=s) for p in ("never", "rof")}


def test_cost_counter_ends_at_the_run_total_and_never_goes_down():
    s, cfg, runs = toy()
    for r in runs.values():
        c = cumulative_cost(r)
        assert abs(c[-1] - r.J) < 1e-9 and all(b >= a for a, b in zip(c, c[1:]))


def test_goal_timeline_follows_the_task_list():
    s, cfg, runs = toy()
    line = goal_timelines(runs["rof"], s)[0]
    assert line[0] == (1, 4) and line[-1] is None


def test_pushing_robots_are_marked_only_while_a_push_run_is_on():
    s, cfg, runs = toy()
    r = runs["rof"]
    run = r.pushes[0]
    assert pushing_robots(r, run["start_tick"] + 1) == {0}
    assert pushing_robots(r, 3) == set() and pushing_robots(runs["never"], 3) == set()


def test_gif_and_html_are_written(tmp_path):
    s, cfg, runs = toy()
    out = animate_runs(s, runs, cfg, path=str(tmp_path / "x.gif"), html=str(tmp_path / "x.html"), fps=4)
    assert os.path.getsize(out) > 5000 and os.path.getsize(tmp_path / "x.html") > 5000
