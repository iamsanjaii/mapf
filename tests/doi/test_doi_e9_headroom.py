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
