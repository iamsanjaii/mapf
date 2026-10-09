"""Grid-of-configs runner: Cartesian sweeps over SimConfig fields, arms and seeds, with CSV output."""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.doi.config import SimConfig
from src.doi.metrics import hindsight_ratios, summary_row
from src.doi.oracle import buy_lb
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario

BASELINES = ("free", "hindsight")
PILOT_SEEDS = [1000, 1001, 1002]
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "experiments", "results", "doi")


def extra_columns(res, scenario, cfg, cheap: bool = False) -> dict:
    """Per-run quantities the experiment analyses need beyond summary_row."""
    rents = [i.rent for i in res.plan_infos]
    out = {"rent_total": float(sum(rents)), "r_max": float(max(rents)) if rents else 0.0,
           "n_triggers": len(res.triggers), "n_pushes": res.removals, "buy_lb_total": float("nan"),
           "hindsight_buy": float(res.hindsight_buy)}
    if scenario.family != "D" and not cheap:
        finite = [v for v in buy_lb(scenario, cfg).values() if math.isfinite(v)]
        out["buy_lb_total"] = float(sum(finite))
    return out


def git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def apply_axes(base: SimConfig, point: Dict[str, object], seed: int) -> SimConfig:
    fields: Dict[str, object] = {"seed": seed}
    params = dict(base.scenario_params)
    for key, value in point.items():
        if key.startswith("scenario_params."):
            params[key.split(".", 1)[1]] = value
        else:
            fields[key] = value
    fields["scenario_params"] = params
    return base.replace(**fields)


def run_point(task) -> List[dict]:
    base, point, arms, seed, commit = task
    cfg = apply_axes(base, point, seed)
    scenario = build_scenario(cfg)
    cheap = cfg.horizon is not None
    no_hindsight = scenario.family == "D" or cheap
    names = [a for a in arms if not (a == "hindsight" and no_hindsight)]
    for extra in BASELINES:
        if extra not in names and not (extra == "hindsight" and no_hindsight):
            names.append(extra)
    results = {name: run_episode(cfg.replace(policy=name), scenario=scenario) for name in names}
    have_hind = "hindsight" in results
    rows = []
    for name in names:
        res = results[name]
        ratios, pod = None, None
        if name not in BASELINES:
            if have_hind:
                ratios = hindsight_ratios(res, results["hindsight"], results["free"])
            if "central" in results and name != "central":
                pod = res.J_censored / results["central"].J_censored
        row = summary_row(res, ratios, pod, cheap=cheap)
        row.update(extra_columns(res, scenario, cfg, cheap=cheap))
        for key, value in point.items():
            row[f"axis_{key}"] = value
        row["git_commit"] = commit
        rows.append(row)
    return rows


def run_grid(base: SimConfig, axes: Dict[str, Sequence], arms: Sequence[str], seeds: Sequence[int],
             out_dir: str, jobs: int = 1) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"base": base.to_dict(), "axes": {k: list(v) for k, v in axes.items()},
                   "arms": list(arms), "seeds": list(seeds)}, f, indent=2, default=str)
    keys = list(axes)
    points = [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]
    commit = git_commit()
    tasks = [(base, point, list(arms), seed, commit) for point in points for seed in seeds]
    rows: List[dict] = []
    csv_path = os.path.join(out_dir, "runs.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    def consume(batch: List[dict]) -> None:
        rows.extend(batch)
        pd.DataFrame(batch).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for batch in pool.map(run_point, tasks):
                consume(batch)
    else:
        for task in tasks:
            consume(run_point(task))
    return pd.DataFrame(rows)


def make_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seeds", type=int, default=30, help="number of seeds")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="pilot seeds and the smallest axis values")
    return ap


def seed_list(args, first: int = 0) -> List[int]:
    return list(PILOT_SEEDS) if args.quick else list(range(first, first + args.seeds))


def out_dir(args, name: str) -> str:
    return args.out or os.path.join(RESULTS_ROOT, name + ("_quick" if args.quick else ""))


def timed_grid(base, axes, arms, seeds, out, jobs):
    started = time.time()
    df = run_grid(base, axes, arms, seeds, out, jobs)
    return df, time.time() - started


def project_hours(elapsed_s: float, quick_points: int, quick_seeds: int, full_points: int, full_seeds: int,
                  jobs: int) -> float:
    return elapsed_s / max(1, quick_points * quick_seeds) * full_points * full_seeds / max(1, jobs) / 3600.0


def n_points(axes: Dict[str, Sequence]) -> int:
    return int(np.prod([len(v) for v in axes.values()])) if axes else 1


def axis_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("axis_")]


def finite_x(v: float) -> float:
    return 1e3 if (isinstance(v, float) and math.isinf(v)) else float(v)


def median_ci(df: pd.DataFrame, by: Sequence[str], col: str, seed: int = 0) -> pd.DataFrame:
    from src.doi.stats import bootstrap_ci
    rows = []
    for key, g in df.groupby(list(by), dropna=False):
        vals = g[col].dropna().to_numpy(dtype=float)
        key = key if isinstance(key, tuple) else (key,)
        if len(vals) == 0:
            rows.append(dict(zip(by, key), **{f"{col}_median": float("nan"), f"{col}_lo": float("nan"),
                                              f"{col}_hi": float("nan"), "n": 0}))
            continue
        med, lo, hi = bootstrap_ci(vals, n=2000, seed=seed)
        rows.append(dict(zip(by, key), **{f"{col}_median": med, f"{col}_lo": lo, f"{col}_hi": hi, "n": len(vals)}))
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, out: str, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    parts = None
    for m in metrics:
        t = median_ci(df, by, m)
        parts = t if parts is None else parts.merge(t.drop(columns=["n"]), on=list(by), how="outer")
    parts.to_csv(os.path.join(out, "summary.csv"), index=False)
    return parts


def stalled_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("policy")["stalled"].mean().rename("stalled_rate").reset_index()


def save_fig(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def verdict(ok: Optional[bool]) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def paired_diff(df: pd.DataFrame, keys: Sequence[str], mask_a, mask_b, col: str):
    """Align two subsets of df on `keys` (plus seed) and return (a_values, b_values)."""
    k = list(keys) + ["seed"]
    a = df[mask_a][k + [col]].rename(columns={col: "a"})
    b = df[mask_b][k + [col]].rename(columns={col: "b"})
    m = a.merge(b, on=k).dropna()
    return m["a"].to_numpy(dtype=float), m["b"].to_numpy(dtype=float), m
