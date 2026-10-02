import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi


def test_list_and_toy_demo_reproduce_hand_checked_numbers(capsys):
    assert run_doi.main(["--list"]) == 0
    assert "rof" in capsys.readouterr().out
    assert run_doi.main(["--demo", "toy"]) == 0
    out = capsys.readouterr().out
    assert "ARM rof" in out and "J = 29" in out and "J = 40" in out and "FILLED pit (1, 3)" in out


def test_family_d_run_with_oracle_intake(capsys):
    argv = ["--scenario", "incidents_room", "--intake", "oracle", "--robots", "3", "--tasks", "3",
            "--seed", "200", "--policy", "rof", "--no-benchmarks"]
    assert run_doi.main(argv) == 0
    assert "incident" in capsys.readouterr().out
