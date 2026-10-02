import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import run_doi


def test_list_and_toy_demo_reproduce_hand_checked_numbers(capsys):
    assert run_doi.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "rof" in out and "single_block" in out and "pit" not in out.lower()
    assert run_doi.main(["--demo", "toy", "--no-show"]) == 0
    out = capsys.readouterr().out
    rows = {w[0]: w[1:] for w in (line.split() for line in out.splitlines()) if len(w) >= 3 and w[1].isdigit()}
    assert {k: v[:2] for k, v in rows.items()} == {
        "never": ["40", "0"], "myopic": ["40", "0"], "eager": ["23", "1"], "rof": ["31", "1"],
        "central": ["31", "1"]}
    assert "rof saves 9" in out
    assert "DECIDES" not in out and "ARM " not in out          # the full dump is behind --verbose


def test_verbose_keeps_the_full_timeline(capsys):
    assert run_doi.main(["--demo", "toy", "--no-show", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "ARM rof" in out and "J = 31" in out and "J = 40" in out
    assert "PUSHES the shelf_unit from [1, 3] to [1, 5]" in out and "DECIDES" in out
    assert "HR_av" in out and "hindsight" in out


def test_sim_only_prints_a_few_lines_and_writes_the_player(capsys, tmp_path):
    html, gif = tmp_path / "toy.html", tmp_path / "toy.gif"
    assert run_doi.main(["--sim", "toy", "--no-show", "--html", str(html), "--gif", str(gif)]) == 0
    out = capsys.readouterr().out
    assert "result: total cost 31, 1 push run" in out and len(out.strip().splitlines()) <= 6
    assert "anim" in html.read_text() and "<title>" in html.read_text()
    assert gif.stat().st_size > 5000


def test_sim_rejects_unknown_names():
    with pytest.raises(SystemExit):
        run_doi.main(["--sim", "nope", "--no-show"])


def test_family_d_run_with_oracle_intake(capsys):
    argv = ["--scenario", "incidents_room", "--intake", "oracle", "--robots", "3", "--tasks", "3",
            "--seed", "200", "--policy", "rof", "--no-benchmarks", "--verbose"]
    assert run_doi.main(argv) == 0
    assert "incident" in capsys.readouterr().out


def test_the_removed_pit_flags_are_gone():
    for flag in ("--claim", "--gate", "--pits", "--stock", "--depot-dist"):
        with pytest.raises(SystemExit):
            run_doi.main([flag, "1", "--no-show"])


def test_scatter_demo_has_obstacles_anywhere_and_one_trip_per_robot(capsys):
    assert run_doi.main(["--demo", "scatter", "--no-show"]) == 0
    out = capsys.readouterr().out
    assert "8 robots, 1 tasks each, seed 8" in out and "rof saves 17" in out
    scenario = run_doi.build(run_doi.argparse.Namespace(**_scatter_args()))[1]
    assert len(scenario.obstacles) == 42 and all(len(t) == 1 for t in scenario.tasks)
    walls = {(r, c) for r in range(scenario.grid.height) for c in range(scenario.grid.width)
             if scenario.grid.get(r, c).name == "OBSTACLE"}
    assert not walls                                      # no permanent wall at all: obstacles are everywhere


def _scatter_args():
    return dict(demo="scatter", scenario="single_block", seed=8, robots=12, tasks=20, kappa=1.0, fee=1.0, push_max=6,
                claim=None, r_comm=8.0, loss=0.0, latency=1, intake="none", max_ticks=20000, p_false=None,
                p_report=None, p_wrong_class=0.0, theta=1.0)


def test_play_asks_only_the_essentials_and_loops_until_told_to_stop():
    from argparse import Namespace
    answers = iter(["10", "12", "5", "8", "4", "3", "1", "3", "y",         # first go, then "try different numbers"
                    "", "", "", "12", "", "", "", "", "n"])                 # second go: only the pallets changed
    shown = []
    args = Namespace(no_show=True, robots_given=False, fps=6)
    assert run_doi.play(args, input_fn=lambda prompt: next(answers), out=shown.append) == 0
    text = "\n".join(str(x) for x in shown)
    assert text.count("Repeat this run") == 2 and "--pallets 8 --crates 4 --shelves 3" in text
    assert "--pallets 12 --crates 4 --shelves 3" in text           # the second run kept the other answers


def test_play_reports_an_impossible_request_and_asks_again():
    from argparse import Namespace
    answers = iter(["10", "12", "5", "400", "", "", "", "",          # too many obstacles for the floor
                    "", "", "", "20", "2", "1", "", "", "n"])
    shown = []
    run_doi.play(Namespace(no_show=True, robots_given=False, fps=6), input_fn=lambda p: next(answers),
                 out=shown.append)
    assert any("too many" in str(x) for x in shown) and any("Repeat this run" in str(x) for x in shown)


def test_play_cannot_be_combined_with_other_modes():
    with pytest.raises(SystemExit):
        run_doi.main(["--play", "--demo", "toy"])
