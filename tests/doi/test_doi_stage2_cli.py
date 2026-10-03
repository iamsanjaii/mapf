"""Stage 2 in the runner: scenario list, cost flags, summary columns, event text and the ASCII replay."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi
from src.doi.config import SimConfig
from src.doi.narrate import SCENARIO_NOTES, events
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

DOOR_CRATE = ["...T#....", "....#....", "....C....", "....#....", "....#...."]
PIT_ROOM = ["....#....", "....#....", "....P....", "..R.#....", "....#...."]
STAGE2 = ("warehouse_racks", "dump_central", "site_pits", "mixed")


def run(rows):
    s = scenario_from_ascii(rows, [(2, 1)], [[(2, 7)]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1, policy="myopic", kappa=50.0)
    return run_episode(cfg, scenario=s), s


def test_every_stage2_scenario_is_described_in_the_list(capsys):
    assert run_doi.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in STAGE2:
        assert name in out and SCENARIO_NOTES[name][0] == "S" and len(SCENARIO_NOTES[name][1]) > 40


def test_cost_flags_reach_the_simulation(capsys):
    args = ["--scenario", "site_pits", "--robots", "2", "--tasks", "2", "--policy", "never,rof", "--no-show",
            "--kappa-c", "6", "--pick-fee", "3", "--drop-fee", "4"]
    assert run_doi.main(args) == 0
    out = capsys.readouterr().out
    assert "carried" in out and "filled" in out


def test_summary_table_has_carried_and_filled_columns(capsys):
    assert run_doi.main(["--scenario", "site_pits", "--robots", "4", "--tasks", "3", "--policy", "never,central",
                         "--no-show", "--r-comm", "inf"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("arm"))
    assert header.split()[:5] == ["arm", "cost", "pushes", "carried", "filled"]
    row = next(line.split() for line in lines if line.startswith("central"))
    assert int(row[4]) >= 1                                              # central fills at least one pit


def test_event_timeline_names_carries_and_fills():
    r, s = run(DOOR_CRATE)
    text = " ".join(t for _, t in events(r, s))
    assert "CARRIES the crate from [2, 4] to the rack at [0, 3]" in text
    r, s = run(PIT_ROOM)
    text = " ".join(t for _, t in events(r, s))
    assert "FILLS the pit at [2, 4] with debris from [3, 2]" in text


def test_ascii_replay_shows_a_loaded_robot_and_a_full_rack():
    r, s = run(DOOR_CRATE)
    frames = {t: run_doi.render(s, t, r) for t in range(7)}
    assert "T" in frames[0] and "t" not in frames[0]                    # an empty rack is T
    assert "@" not in frames[2] and "@" in frames[3] and "@" in frames[4] and "@" not in frames[5]
    assert "T" in frames[4] and "t" in frames[5]                        # a rack holding a crate is t, once dropped


def test_ascii_replay_shows_the_pit_until_it_is_filled():
    r, s = run(PIT_ROOM)
    assert "P" in run_doi.render(s, 0, r)
    assert "P" not in run_doi.render(s, len(r.trajectory[0]) - 1, r)
