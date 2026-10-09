import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arms_are_the_references_and_every_rule_based_forecaster():
    e9 = _load("doi_e9_agent")
    assert [a[0] for a in e9.ARMS] == ["never", "rof", "rof_p", "numeric", "keyword", "ledger", "oracle", "inverted"]


def test_usable_takes_the_columns_that_identify_a_point():
    headroom = _load("doi_e9_headroom")
    df = pd.DataFrame({"seed": [0, 0, 1, 1], "lam": [0.5, 0.5, 0.5, 0.5], "stalled": [False, True, False, False]})
    kept, dropped = headroom.usable(df, point=["seed", "lam"])
    assert dropped == 1 and list(kept.seed) == [1, 1]


def test_quick_run_writes_runs_and_a_summary(tmp_path, capsys):
    e9 = _load("doi_e9_agent")
    out = tmp_path / "e9"
    assert e9.main(["--quick", "--modes", "true", "--out", str(out)]) == 0
    runs = pd.read_csv(out / "runs.csv", dtype={"mode": str})
    assert sorted(runs.arm.unique()) == sorted(a[0] for a in e9.ARMS)
    assert sorted(runs.seed.unique()) == [200, 201, 202] and set(runs["mode"]) == {"true"}
    assert (runs.avoidable == runs.J - runs.J_free).all() and len(runs) == 3 * 8
    summary = pd.read_csv(out / "summary.csv")
    assert {"mode", "lam", "notice_tick", "compare", "median_diff", "lo", "hi", "n"} <= set(summary.columns)
    assert "rof - ledger" in set(summary["compare"])
    assert "rule-based forecasters only" in capsys.readouterr().out
