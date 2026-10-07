import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi

BASE = ["--scenario", "shift_notice", "--robots", "6", "--tasks", "12", "--seed", "3", "--no-show"]


def test_rof_a_runs_from_the_command_line_with_a_named_forecaster(capsys):
    assert run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "oracle"]) == 0
    out = capsys.readouterr().out
    assert "rof_a" in out and "STALLED" not in out


def test_agent_mode_is_accepted_and_a_bad_one_is_refused(capsys):
    assert run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "keyword", "--agent-mode", "single"]) == 0
    capsys.readouterr()
    try:
        run_doi.main(BASE + ["--policy", "rof_a", "--agent-mode", "chat"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("a bad --agent-mode should exit with a usage error")


def test_an_empty_store_stops_the_run_with_the_remedy_and_no_traceback(capsys, tmp_path):
    code = run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "llm:small", "--agent-cache", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2 and "--agent-live" in captured.err and "Traceback" not in captured.err


def test_the_new_flags_are_in_the_guide(capsys):
    assert run_doi.main(["--guide"]) == 0
    out = capsys.readouterr().out
    for flag in ("--forecaster", "--agent-mode", "--agent-live", "--agent-cache"):
        assert flag in out
