"""Grid-of-configs runner: Cartesian sweeps over SimConfig fields, arms and seeds, with CSV output."""
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.doi.config import SimConfig
from src.doi.metrics import hindsight_ratios, summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario

BASELINES = ("free", "hindsight")


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
    names = [a for a in arms if not (a == "hindsight" and scenario.family == "D")]
    for extra in BASELINES:
        if extra not in names and not (extra == "hindsight" and scenario.family == "D"):
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
        row = summary_row(res, ratios, pod)
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
