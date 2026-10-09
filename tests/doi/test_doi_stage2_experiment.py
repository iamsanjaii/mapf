import csv
import subprocess
import sys


def test_s2_quick_runs_and_carries_need_racks(tmp_path):
    proc = subprocess.run([sys.executable, "experiments/doi_s2_modes.py", "--quick", "--out", str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "s2_modes_vs_racks.png").exists()
    cap = list(csv.DictReader(open(tmp_path / "capacity" / "runs.csv")))
    rof = [r for r in cap if r["policy"] == "rof"]
    assert all(int(r["carries"]) == 0 for r in rof if r["n_racks"] == "0")           # no rack, nothing carried
    assert any(int(r["carries"]) > 0 for r in rof if r["n_racks"] == "8")            # with racks, crates are carried
    dist = list(csv.DictReader(open(tmp_path / "distance" / "runs.csv")))
    assert {r["dump_far"] for r in dist} == {"True", "False"}

    def median_carries(far, kappa):
        vals = sorted(int(r["carries"]) for r in dist
                      if r["policy"] == "rof" and r["dump_far"] == far and float(r["axis_kappa"]) == kappa)
        return vals[len(vals) // 2]
    assert median_carries("False", 4.0) == 0 and median_carries("True", 4.0) == 0     # cheap pushes: never haul
    assert median_carries("False", 16.0) > median_carries("True", 16.0)                # dear pushes: near dump wins
    assert all(r["stalled"] == "False" for r in cap + dist)
