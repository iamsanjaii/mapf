import numpy as np
from src.doi.stats import bootstrap_ci, signflip_pvalue, spearman


def test_bootstrap_ci_contains_median_and_is_deterministic():
    x = list(range(1, 31))
    a = bootstrap_ci(x, seed=1)
    assert a == bootstrap_ci(x, seed=1)
    assert a[1] <= a[0] <= a[2]


def test_signflip_detects_shift_and_accepts_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    assert signflip_pvalue(a + 1.5, a) < 0.01
    assert signflip_pvalue(a, a + rng.normal(0, 0.01, 30)) > 0.05


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(spearman([1, 2, 2, 4], [1, 2, 2, 4]) - 1.0) < 1e-12


def test_spearman_constant_input_is_nan():
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_run_grid_jobs_equal(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    from doi_common import run_grid
    from src.doi.config import SimConfig
    base = SimConfig(n_robots=3, tasks_per_robot=3)
    a = run_grid(base, {"r_comm": [0.0, 8.0]}, ["never", "rof", "central"], [0, 1], str(tmp_path / "a"), jobs=1)
    b = run_grid(base, {"r_comm": [0.0, 8.0]}, ["never", "rof", "central"], [0, 1], str(tmp_path / "b"), jobs=2)
    assert a.drop(columns=["runtime_ms"]).equals(b.drop(columns=["runtime_ms"]))
    assert {"hr", "hr_av", "pod", "axis_r_comm", "git_commit"} <= set(a.columns)
    assert {"free", "hindsight"} <= set(a["policy"])


def test_run_grid_writes_files_scenario_params_axis_and_skips_hindsight_for_family_d(tmp_path):
    import json, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    from doi_common import run_grid
    from src.doi.config import SimConfig
    base = SimConfig(scenario="incidents_room", n_robots=2, tasks_per_robot=2, intake="oracle")
    out = str(tmp_path / "d")
    df = run_grid(base, {"scenario_params.p_report": [0.5, 0.9]}, ["never", "rof"], [200], out, jobs=1)
    assert "hindsight" not in set(df["policy"]) and "free" in set(df["policy"])
    assert "axis_scenario_params.p_report" in df.columns
    assert os.path.exists(os.path.join(out, "runs.csv")) and os.path.exists(os.path.join(out, "config.json"))
    assert json.load(open(os.path.join(out, "config.json")))["arms"] == ["never", "rof"]
    assert df["hr"].isna().all()
