import os
import sys
from argparse import Namespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi
from src.doi.builder import BuildError
from src.doi.wizard import ask, configure, equivalent_command


def by_prompt(replies):
    """Answer each prompt by a fragment of its text (queued); anything else gets Enter."""
    queues = {k: list(v) for k, v in replies.items()}

    def fn(prompt):
        for key, q in queues.items():
            if key in prompt and q:
                return q.pop(0)
        return ""
    return fn


def blank_args(**kw):
    base = dict(layout=None, map=None, rows=None, cols=None, wall_col=None, blocks=None, doors=None, kind=None,
                crossing=None, strips=None, strip_min=None, strip_max=None, pallets=None, crates=None,
                shelves=None, robots=None, tasks=None, seed=None, kappa=None, fee=None, push_max=None,
                r_comm=None, loss=None, latency=None, policy=None, no_show=False)
    base.update(kw)
    return Namespace(**base)


def test_ask_reprompts_on_bad_input_and_accepts_defaults():
    lines = []
    seq = iter(["x", "0", "7"])
    assert ask("n", int, 5, lambda v: v >= 1, "at least 1", lambda p: next(seq), lines.append) == 7
    assert len(lines) == 2 and "at least 1" in lines[0]
    assert ask("n", int, 5, input_fn=lambda p: "") == 5

    def eof(prompt):
        raise EOFError
    assert ask("n", int, 5, input_fn=eof) == 5               # end of input keeps the default


def test_barrier_wizard_builds_what_was_typed():
    feed = by_prompt({"Layout": ["barrier"], "Grid rows": ["12"], "Grid columns": ["18"],
                      "Removable obstacles": ["3"], "Door rows": ["2"], "Kind of obstacle": ["crate"],
                      "Share of trips": ["0.9"], "Number of robots": ["3"], "Trips per robot": ["6"],
                      "Random seed": ["5"]})
    args = blank_args()
    cfg, s = configure(args, set(), lambda c, sc: None, input_fn=feed, out=lambda *_: None)
    assert (s.grid.height, s.grid.width) == (12, 18) and len(s.obstacles) == 3 and len(s.starts) == 3
    assert set(s.obstacles.values()) == {"crate"} and cfg.seed == 5 and cfg.tasks_per_robot == 6
    assert args.policy == "never,rof"


def test_a_bad_answer_reasks_only_the_map_questions():
    shown, asked = [], []
    feed = by_prompt({"Layout": ["barrier"], "Removable obstacles": ["99", "4"]})
    cfg, s = configure(blank_args(), set(), lambda c, sc: None, input_fn=lambda p: asked.append(p) or feed(p),
                       out=shown.append)
    assert any("do not fit" in str(x) for x in shown) and len(s.obstacles) == 4
    assert sum("Number of robots" in p for p in asked) == 1          # fleet questions were not repeated
    assert sum("Removable obstacles" in p for p in asked) == 2


def test_gives_up_instead_of_looping_when_the_error_cannot_change():
    def eof(prompt):
        raise EOFError
    with pytest.raises(BuildError, match="do not fit"):
        configure(blank_args(blocks=99, rows=12), {"blocks", "rows"}, lambda c, sc: None, input_fn=eof,
                  out=lambda *_: None)


def test_flags_are_not_asked_again_and_yes_accepts_defaults():
    args = blank_args(layout="strips", rows=10, cols=12, robots=2, tasks=3)
    asked = []
    cfg, s = configure(args, {"layout", "rows", "cols", "robots", "tasks"}, lambda c, sc: None, yes=True,
                       input_fn=lambda p: asked.append(p) or "", out=lambda *_: None)
    assert asked == [] and (s.grid.height, s.grid.width) == (10, 12) and len(s.starts) == 2


def test_equivalent_command_reproduces_the_run():
    args = blank_args(layout="barrier")
    cfg, s = configure(args, set(), lambda c, sc: None, yes=True, out=lambda *_: None)
    cmd = equivalent_command(args, "barrier")
    assert cmd.startswith("python run_doi.py --layout barrier --rows 15") and cmd.endswith("--yes")
    assert "--blocks 4" in cmd and "--kind mixed" in cmd


def test_cli_layout_yes_runs_end_to_end(capsys):
    argv = ["--layout", "barrier", "--yes", "--no-show", "--rows", "12", "--cols", "16", "--blocks", "2",
            "--robots", "3", "--tasks", "6", "--policy", "never,rof"]
    assert run_doi.main(argv) == 0
    out = capsys.readouterr().out
    assert "simulation: barrier (3 robots" in out and "Repeat this run exactly with" in out and "rof" in out


def test_cli_builds_strips_and_map_layouts(capsys, tmp_path):
    assert run_doi.main(["--layout", "strips", "--yes", "--no-show", "--robots", "3", "--tasks", "4",
                         "--policy", "never,eager"]) == 0
    assert "simulation: strips" in capsys.readouterr().out
    f = tmp_path / "m.txt"
    f.write_text(".....#......\n.....L..C...\n.....#......\n.....#......\n............\n")
    assert run_doi.main(["--layout", "map", "--map", str(f), "--yes", "--no-show", "--robots", "2",
                         "--tasks", "3", "--policy", "never"]) == 0
    assert "simulation: map" in capsys.readouterr().out


def test_cli_reports_a_bad_map_as_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        run_doi.main(["--layout", "barrier", "--yes", "--no-show", "--blocks", "50"])
    assert "do not fit" in capsys.readouterr().err


def test_cli_rejects_mixing_build_with_demo():
    with pytest.raises(SystemExit):
        run_doi.main(["--build", "--demo", "toy"])
