import importlib.util
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_headroom_quick_runs(tmp_path):
    out = str(tmp_path / "headroom")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "doi_e9_headroom.py"),
                           "--quick", "--out", out], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    runs = pd.read_csv(os.path.join(out, "runs.csv"))
    assert sorted(runs.arm.unique()) == ["inverted", "keyword", "numeric", "oracle", "rof"]
    assert sorted(runs["mode"].unique()) == ["false", "missing", "quiet", "true"]
    assert len(runs) == 3 * 4 * 5 and not runs.stalled.any()             # 3 seeds, 4 modes, 5 arms, 1 cost setting
    assert (runs.avoidable == runs.J - runs.J_free).all()
    summary = pd.read_csv(os.path.join(out, "summary.csv"))
    assert {"kappa", "fee", "mode", "compare", "mean_diff", "median_diff", "lo", "hi", "share_positive",
            "headroom_share"} <= set(summary.columns)
    assert "numeric - oracle" in set(summary["compare"])
    assert "no model was called" in proc.stdout


def _module():
    spec = importlib.util.spec_from_file_location("doi_e9_headroom", os.path.join(ROOT, "experiments", "doi_e9_headroom.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs(stalled_at=None):
    """Three seeds x five arms in one cell; the arm `inverted` on seed 1 optionally stalls with a huge cost."""
    rows = []
    for seed in range(3):
        for k, arm in enumerate(["rof", "numeric", "keyword", "oracle", "inverted"]):
            stall = stalled_at == (seed, arm)
            rows.append({"seed": seed, "mode": "true", "kappa": 4.0, "fee": 1.0, "arm": arm, "J": 0.0, "J_free": 0.0,
                         "avoidable": 9999.0 if stall else 100.0 + 10 * k + seed, "removals": 0, "stalled": stall,
                         "forecasts": 1, "forecast_acc": 1.0, "wrong_yes": 0, "wrong_no": 0})
    return pd.DataFrame(rows)


def test_a_stalled_arm_drops_its_whole_point_from_the_summary():
    mod = _module()
    kept, dropped = mod.usable(_runs(stalled_at=(1, "inverted")))
    assert dropped == 1 and sorted(kept.seed.unique()) == [0, 2] and not kept.stalled.any()
    summary = mod.summarise(_runs(stalled_at=(1, "inverted")))
    assert set(summary.n) == {2} and (summary.mean_avoidable_numeric == 111.0).all()   # numeric on seeds 0 and 2: 110, 112
    assert len(mod.summarise(_runs())) == len(summary) and set(mod.summarise(_runs()).n) == {3}


def test_resummarise_rereads_runs_and_reports_what_it_dropped(tmp_path):
    mod = _module()
    _runs(stalled_at=(1, "inverted")).to_csv(tmp_path / "runs.csv", index=False)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "doi_e9_headroom.py"),
                           "--resummarise", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "dropped 1 of 3 points" in proc.stdout and "no model was called" in proc.stdout
    assert (tmp_path / "summary.csv").exists()
