import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.doi.builder import BuildError, barrier_block_rows, build, check, has_push_room, potential_rent
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii
from src.environment.grid import CellType

BARRIER = dict(rows=15, cols=21, wall_col=None, blocks=4, doors=3, kind="pallet", crossing=0.8)
STRIPS = dict(rows=16, cols=18, strips=3, strip_min=3, strip_max=6, pallets=2, crates=2, shelves=1)


def cfg(**kw):
    return SimConfig(n_robots=kw.pop("n_robots", 4), tasks_per_robot=kw.pop("tasks", 6), seed=kw.pop("seed", 0), **kw)


def test_barrier_builds_the_requested_shape():
    c, s = build("barrier", cfg(), BARRIER)
    assert (s.grid.height, s.grid.width) == (15, 21) and len(s.obstacles) == 4 and len(s.starts) == 4
    assert all(p[1] == 10 for p in s.obstacles) and set(s.obstacles.values()) == {"pallet"}
    assert all(s.grid.get(r, 10) == CellType.FREE for r in (12, 13, 14))        # three door rows at the bottom
    assert s.name == "barrier"
    c, s = build("barrier", cfg(), {**BARRIER, "kind": "shelf_unit", "blocks": 2})
    assert set(s.obstacles.values()) == {"shelf_unit"}


def test_mixed_barrier_shows_every_kind():
    c, s = build("barrier", cfg(), {**BARRIER, "kind": "mixed"})
    assert set(s.obstacles.values()) == {"pallet", "crate", "shelf_unit"}


def test_obstacles_are_spread_and_stay_above_the_doors():
    assert barrier_block_rows(15, 3, 4) == [1, 4, 7, 10]
    assert barrier_block_rows(15, 3, 1) == [1]


@pytest.mark.parametrize("change, words", [
    (dict(blocks=12), "do not fit"), (dict(blocks=0), "at least 1"), (dict(rows=6), "at least 8 rows"),
    (dict(doors=20), "doors must be"), (dict(crossing=1.5), "between 0 and 1"),
    (dict(wall_col=19), "no room on one side"), (dict(kind="anvil"), "unknown obstacle kind"),
])
def test_barrier_rejects_bad_input_with_a_reason(change, words):
    with pytest.raises(BuildError, match=words):
        build("barrier", cfg(), {**BARRIER, **change})


def test_strips_layout_counts_each_kind_and_checks_ranges():
    c, s = build("strips", cfg(), STRIPS)
    assert sorted(s.obstacles.values()) == ["crate"] * 2 + ["pallet"] * 2 + ["shelf_unit"]
    assert (s.grid.height, s.grid.width) == (16, 18)
    for change, words in [(dict(strip_min=9), "strip length"), (dict(pallets=0, crates=0, shelves=0), "at least 1"),
                          (dict(crates=-1), "cannot be negative"), (dict(rows=3), "at least 5 rows"),
                          (dict(pallets=500), "too many")]:
        with pytest.raises(BuildError, match=words):
            build("strips", cfg(), {**STRIPS, **change})


def test_map_file(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text("...#...\n...L..C\n...#...\n...#...\n.......\n")
    c, s = build("map", cfg(n_robots=2), dict(map=str(f)))
    assert s.obstacles == {(1, 3): "pallet", (1, 6): "crate"} and len(s.starts) == 2
    assert not any(p in s.obstacles for p in s.starts)
    for text, words in [("...\n..\n", "same length"), ("..x\n...\n", "unknown characters"),
                        ("...\n...\n", "at least one removable"), ("", "empty")]:
        f.write_text(text)
        with pytest.raises(BuildError, match=words):
            build("map", cfg(), dict(map=str(f)))
    with pytest.raises(BuildError, match="cannot read"):
        build("map", cfg(), dict(map=str(tmp_path / "nope.txt")))


def test_check_flags_an_obstacle_that_can_never_be_pushed():
    s = scenario_from_ascii(["###", "#L#", "###"], [(1, 1)], [[(1, 1)]])
    assert not has_push_room(s, (1, 1))
    assert any("no room to be pushed" in w for w in check(s, cfg(n_robots=1)))
    c, s = build("barrier", cfg(), BARRIER)
    assert all(has_push_room(s, p) for p in s.obstacles)


def test_check_flags_a_map_with_nothing_to_remove():
    c, s = build("barrier", cfg(), BARRIER)
    assert potential_rent(s, c) > 0 and any(w.startswith("info:") for w in check(s, c))
    c, s = build("barrier", cfg(), {**BARRIER, "crossing": 0.0})
    assert potential_rent(s, c) == 0 and any("nothing worth removing" in w for w in check(s, c))


def test_built_scenario_runs():
    c, s = build("barrier", cfg(n_robots=3, tasks=5), BARRIER)
    assert run_episode(c.replace(policy="rof"), scenario=s).unfinished_tasks == 0
