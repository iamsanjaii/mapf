import math
import pathlib
import pytest
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii, Incident, Report
from src.doi.metrics import false_report_cost

ROWS_D = ["...#...", ".......", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]
LOC = {"gap": ((1, 3),), "corner": ((5, 0),)}


def d_scenario(cls="robot_clearable", false_report=False):
    if false_report:
        inc, reps = [], [Report("r0", None, "gap", 0, "pallet", "robot_clearable")]
    else:
        inc = [Incident(0, ((1, 3),), 0, "pallet", cls)]
        reps = [Report("r0", 0, "gap", 0, "pallet", cls)]
    return scenario_from_ascii(ROWS_D, [(1, 2)], [FOUR],
                               incidents=inc, reports=reps, locations=LOC)


def d_run(s, **kw):
    base = dict(policy="rof", n_robots=1, tasks_per_robot=4, intake="oracle", debug_checks=True)
    base.update(kw)
    return run_episode(SimConfig(**base), scenario=s)


def test_oracle_intake_matches_a_static_obstacle():
    r = d_run(d_scenario())
    static = scenario_from_ascii(["...#...", "...L...", "...#...", "...#...", "...#...", "......."], [(1, 2)], [FOUR])
    ref = run_episode(SimConfig(policy="rof", n_robots=1, tasks_per_robot=4, debug_checks=True), scenario=static)
    assert r.removals == 1 and r.J == ref.J and r.intake["records"] == 1


def test_false_report_is_never_acted_on():
    s = d_scenario(false_report=True)
    r = d_run(s, r_sense=0)
    assert r.removals == 0 and len(r.triggers) == 0 and r.push_rejected == 0
    assert r.J == 40.0
    assert false_report_cost(r, s, SimConfig(n_robots=1, tasks_per_robot=4)) == 32.0


def test_a_wrong_class_report_is_caught_by_the_world():
    s = d_scenario(cls="needs_human")
    on = d_run(s, p_wrong_class=1.0)
    assert on.wrong_class_attempts == 1 and on.removals == 0 and on.unfinished_tasks == 0
    right = d_run(s, p_wrong_class=0.0)
    assert right.wrong_class_attempts == 0 and right.removals == 0


def test_llm_intake_reads_cache_only(tmp_path):
    from src.doi.incidents import render_report, report_rng, location_names
    from src.doi.llm.intake import IntakeCache, IntakeResult, cache_key
    s = d_scenario()
    text, _, _ = render_report(s.reports[0], report_rng(s, s.reports[0]))
    key = cache_key(text, location_names(s))
    with pytest.raises(KeyError):
        d_run(s, intake="llm:fake", intake_cache=str(tmp_path))
    IntakeCache(str(tmp_path), "fake").put(IntakeResult(key, True, "gap", "pallet", "robot_clearable", 1, 0.9,
                                                        "x", "", 1.2, 10, 10, "{}"))
    r = d_run(s, intake="llm:fake", intake_cache=str(tmp_path))
    assert r.intake["records"] == 1 and r.removals == 1


def test_llm_boundary_imports():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "doi"
    for name in ("policies", "pushplan", "pusher", "agent", "spacetime", "world", "evidence", "belief"):
        lines = [l for l in (root / f"{name}.py").read_text().splitlines() if "import" in l]
        assert not any("llm" in l for l in lines), name


def test_wrong_class_oracle_probability_is_deterministic():
    s = d_scenario(cls="needs_human")
    a = d_run(s, p_wrong_class=0.5)
    b = d_run(s, p_wrong_class=0.5)
    assert (a.J, a.wrong_class_attempts) == (b.J, b.wrong_class_attempts)


def test_unconfirmed_report_on_only_route_is_verified_not_deadlocked():
    rows = ["...#...", ".......", "...#..."]
    s = scenario_from_ascii(rows, [(1, 0)], [[(1, 6), (1, 0)]],
                            reports=[Report("r0", None, "gap", 0, "pallet", "robot_clearable")],
                            locations={"gap": ((1, 3),)})
    r = run_episode(SimConfig(policy="never", n_robots=1, tasks_per_robot=2, intake="oracle", debug_checks=True),
                    scenario=s)
    assert not r.stalled and r.unfinished_tasks == 0


def test_unknown_class_cells_are_never_edited_without_intake():
    r = d_run(d_scenario(), intake="none")
    assert r.removals == 0 and r.wrong_class_attempts == 0
