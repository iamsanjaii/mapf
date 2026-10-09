import csv
import subprocess
import sys


def test_e1_quick_runs(tmp_path):
    proc = subprocess.run([sys.executable, "experiments/doi_e1_ratio.py", "--quick", "--out", str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "core.csv").exists() and (tmp_path / "grid.csv").exists()
    rows = list(csv.DictReader(open(tmp_path / "core.csv")))
    assert rows and all(r["bound_holds"] == "True" for r in rows)
