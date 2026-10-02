from src.doi.config import SimConfig
from src.doi.scenarios import scenario_from_ascii, build_scenario
from src.doi.runner import run_episode

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]


def ascii_run(tasks, starts=None, **kw):
    starts = starts or [(1, 2)]
    s = scenario_from_ascii(ROWS, starts, tasks, depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=len(starts), tasks_per_robot=len(tasks[0]),
                    debug_checks=True, **kw)
    return run_episode(cfg, scenario=s)


def test_single_robot_pays_detour():
    r = ascii_run([[(1, 4), (1, 2), (1, 4), (1, 2)]])
    assert not r.stalled and r.unfinished_tasks == 0
    assert r.J == 40.0 and r.fills == 0 and r.waits == 0


def test_two_robots_finish_without_collision():
    tasks = [[(1, 4), (1, 2), (1, 4)], [(1, 2), (1, 4), (1, 2)]]
    r = ascii_run(tasks, starts=[(1, 2), (1, 4)])
    assert not r.stalled and r.unfinished_tasks == 0
    t0, t1 = r.trajectory[0], r.trajectory[1]
    for k in range(min(len(t0), len(t1))):
        assert t0[k] != t1[k]
        if k + 1 < min(len(t0), len(t1)):
            assert not (t0[k] == t1[k + 1] and t1[k] == t0[k + 1])


def test_plan_infos_record_rent_once_per_task():
    r = ascii_run([[(1, 4), (1, 2)]])
    assert [(i.task_idx, i.rent, i.bundle) for i in r.plan_infos] == [(0, 8, ((1, 3),)), (1, 8, ((1, 3),))]


def test_metrics_formula():
    r = ascii_run([[(1, 4)]])
    assert r.moves == 10 and r.J == 10.0 and r.J_censored == 10.0


def test_random_scenario_neverfill_completes():
    cfg = SimConfig(policy="never", scenario="multi_pit_wall", n_robots=6,
                    tasks_per_robot=4, seed=3, debug_checks=True)
    r = run_episode(cfg)
    assert r.unfinished_tasks == 0 and not r.stalled


def test_run_is_deterministic():
    cfg = SimConfig(policy="never", n_robots=6, tasks_per_robot=4, seed=5)
    a, b = run_episode(cfg), run_episode(cfg)
    assert (a.J, a.ticks, a.trajectory) == (b.J, b.ticks, b.trajectory)


def test_stale_belief_robot_never_planned_through_pit():
    r = ascii_run([[(1, 4)]])
    assert all(c != (1, 3) for c in r.trajectory[0])
