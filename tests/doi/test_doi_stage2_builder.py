"""Stage 2 builder: the `site` layout (barrier with pits, debris, racks and a dump region) and slot glyphs in map files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi
from src.doi.builder import BuildError, build, check
from src.doi.config import SimConfig
from src.doi.runner import run_episode

SITE = dict(rows=15, cols=21, wall_col=None, blocks=3, doors=3, kind="crate", crossing=0.8,
            pits=2, debris=3, racks=4, dump_rows=2, dump_cols=3)


def cfg(**kw):
    return SimConfig(n_robots=kw.pop("n_robots", 4), tasks_per_robot=kw.pop("tasks", 4), seed=kw.pop("seed", 0), **kw)


def test_site_layout_builds_the_requested_counts():
    c, s = build("site", cfg(), SITE)
    kinds = list(s.obstacles.values())
    assert kinds.count("crate") == 3 and kinds.count("pit") == 2 and kinds.count("debris") == 3
    assert sum(1 for k in s.slots.values() if k == "rack") == 4
    assert sum(1 for k in s.slots.values() if k == "dump") == 6                     # a 2 x 3 block
    assert all(p[1] == 10 for p, k in s.obstacles.items() if k in ("crate", "pit"))  # barrier openings
    assert s.name == "site" and len(s.starts) == 4


def test_site_layout_is_deterministic_and_runs():
    a, b = build("site", cfg(), SITE)[1], build("site", cfg(), SITE)[1]
    assert (a.obstacles, a.slots, a.starts, a.tasks) == (b.obstacles, b.slots, b.starts, b.tasks)
    c, s = build("site", cfg(), SITE)
    r = run_episode(c.replace(policy="central", r_comm=float("inf")), scenario=s)
    assert not r.stalled and r.unfinished_tasks == 0


def test_a_site_with_no_extras_is_just_a_barrier():
    c, s = build("site", cfg(), {**SITE, "pits": 0, "debris": 0, "racks": 0, "dump_rows": 0, "dump_cols": 0})
    assert s.slots == {} and set(s.obstacles.values()) == {"crate"}


@pytest.mark.parametrize("change, words", [
    (dict(pits=3, debris=1), "debris"),                          # every pit needs a piece of debris to fill it
    (dict(blocks=0, pits=0), "at least 1"),
    (dict(racks=-1), "cannot be negative"),
    (dict(dump_rows=1, dump_cols=0), "dump"),                    # a dump region needs both a height and a width
    (dict(dump_rows=12, dump_cols=9), "dump"),                   # too big for the room
    (dict(racks=60), "rack"),
])
def test_site_rejects_bad_input_with_a_reason(change, words):
    with pytest.raises(BuildError, match=words):
        build("site", cfg(), {**SITE, **change})


def test_check_warns_about_an_obstacle_that_can_never_go_anywhere():
    c, s = build("site", cfg(), {**SITE, "kind": "shelf_unit", "pits": 0, "debris": 0, "racks": 2,
                                 "dump_rows": 0, "dump_cols": 0})
    assert not any("can never be removed" in w for w in check(s, c))             # in the open, a push still works
    from src.doi.scenarios import scenario_from_ascii
    boxed = scenario_from_ascii(["#S#", "..."], [(1, 0)], [[(1, 2)]])           # walls both sides: no push, no dump
    assert any("can never be removed" in w for w in check(boxed, cfg()))
    with_dump = scenario_from_ascii(["#S#", "...", "D.."], [(1, 0)], [[(1, 2)]])  # a dump region gives it a way out
    assert not any("can never be removed" in w for w in check(with_dump, cfg()))


def test_check_warns_about_a_pit_with_no_debris():
    from src.doi.scenarios import scenario_from_ascii
    s = scenario_from_ascii(["..P..", "....."], [(0, 0)], [[(0, 4)]])
    assert any("pit" in w and "never be filled" in w for w in check(s, cfg()))


def test_a_map_file_may_use_pits_racks_and_dump_cells(tmp_path):
    path = tmp_path / "site.txt"
    path.write_text("T..DD\n..P..\n.R.C.\n")
    c, s = build("map", cfg(n_robots=2, tasks=2), dict(map=str(path)))
    assert s.obstacles == {(1, 2): "pit", (2, 1): "debris", (2, 3): "crate"}
    assert s.slots == {(0, 0): "rack", (0, 3): "dump", (0, 4): "dump"}
    assert not (set(s.starts) & {(0, 0)}) and len(s.starts) == 2


def test_the_runner_builds_a_site_from_flags(capsys):
    args = ["--layout", "site", "--yes", "--rows", "15", "--cols", "21", "--blocks", "3", "--doors", "3",
            "--pits", "2", "--debris", "3", "--racks", "4", "--dump-rows", "2", "--dump-cols", "3",
            "--robots", "4", "--tasks", "3", "--policy", "never,central", "--no-show", "--r-comm", "inf"]
    assert run_doi.main(args) == 0
    out = capsys.readouterr().out
    assert "carried" in out and "central" in out
