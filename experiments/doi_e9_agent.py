"""E9: does a forecaster behind the guard lower fleet cost on shift_notice, and how much could a better one win?

Avoidable cost is J_censored minus the `free` arm's on the same seed. A point (seed, mode, lam, notice_tick) in
which any arm stalled is left out of every summary, as in the headroom pilot. Every forecaster is rule-based, so
the run needs no credentials and nothing is stored or replayed.

Arms: never, rof, rof_p (numeric forecast inside the predicted rule), and rof_a with each forecaster: numeric,
keyword, ledger, oracle (best possible) and inverted (worst possible). The summary gives paired differences in
avoidable cost (A - B; positive means B is cheaper) with bootstrap 95% intervals.

Usage: venv/bin/python experiments/doi_e9_agent.py [--quick] [--seeds 100] [--jobs 4]
           [--modes true,false,missing,quiet] [--out DIR]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doi_e9_headroom import usable
from src.doi.config import SimConfig
from src.doi.metrics import summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario
from src.doi.stats import bootstrap_ci

ARMS = [("never", "never", "numeric"), ("rof", "rof", "numeric"), ("rof_p", "rof_p", "numeric"),
        ("numeric", "rof_a", "numeric"), ("keyword", "rof_a", "keyword"), ("ledger", "rof_a", "ledger"),
        ("oracle", "rof_a", "oracle"), ("inverted", "rof_a", "inverted")]     # (label, policy, forecaster)
MODES = ["true", "false", "missing", "quiet"]
PRIMARY = (0.5, 5)                                                       # (lam, notice_tick)
SECONDARY = [(0.25, 5), (1.0, 5), (0.5, 150)]                            # on the true and false modes only
POINT = ["seed", "mode", "lam", "notice_tick"]
FIRST_SEED = 200
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "e9_agent")


def make_cfg(seed: int, mode: str, lam: float, notice_tick: int, **kw) -> SimConfig:
    return SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, lam=lam, bundle_max=1,
                     scenario_params={"notice_mode": mode, "notice_tick": notice_tick}, **kw)


def run_point(task) -> List[dict]:
    seed, mode, lam, notice_tick = task
    cfg = make_cfg(seed, mode, lam, notice_tick)
    scenario = build_scenario(cfg)
    free = run_episode(cfg.replace(policy="free"), scenario=scenario).J_censored
    rows = []
    for label, policy, forecaster in ARMS:
        res = run_episode(cfg.replace(policy=policy, forecaster=forecaster), scenario=scenario)
        cols = summary_row(res, cheap=True)
        rows.append({"seed": seed, "mode": mode, "lam": lam, "notice_tick": notice_tick, "arm": label,
                     "J": res.J_censored, "J_free": free, "avoidable": res.J_censored - free,
                     "removals": res.removals, "stalled": res.stalled, "forecasts": cols["forecasts"],
                     "forecast_acc": cols["forecast_acc"], "wrong_yes": cols["wrong_yes"],
                     "wrong_no": cols["wrong_no"]})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Paired differences in avoidable cost for the hypotheses and for the headroom references."""
    kept, _ = usable(df, POINT)
    pairs = [("ref", m, a, b) for m in MODES for a, b in (("numeric", "oracle"), ("inverted", "oracle"),
                                                         ("rof", "numeric"), ("rof", "ledger"), ("never", "rof"),
                                                         ("numeric", "ledger"), ("ledger", "oracle"))]
    out = []
    for (lam, tick, mode), g in kept.groupby(["lam", "notice_tick", "mode"]):
        wide = g.pivot(index="seed", columns="arm", values="avoidable")
        for hyp, m, a, b in pairs:
            if m != mode or a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna().to_numpy(dtype=float)
            if len(d) == 0:
                continue
            med, lo, hi = bootstrap_ci(d, n=2000, seed=0)
            out.append({"lam": lam, "notice_tick": tick, "mode": mode, "hypothesis": hyp, "compare": f"{a} - {b}",
                        "mean_diff": float(d.mean()), "median_diff": med, "lo": lo, "hi": hi,
                        "share_positive": float((d > 0).mean()), "n": len(d)})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="seeds 200..202, no secondary sweeps")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or (DEFAULT_OUT + ("_quick" if args.quick else ""))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = list(range(FIRST_SEED, FIRST_SEED + (3 if args.quick else args.seeds)))

    def points(selected):
        base = [(m, *PRIMARY) for m in selected]
        extra = [] if args.quick else [(m, lam, tick) for m in selected if m in ("true", "false")
                                       for lam, tick in SECONDARY]
        return base + extra

    tasks = [(seed, m, lam, tick) for seed in seeds for m, lam, tick in points(modes)]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            batches = list(pool.map(run_point, tasks))
    else:
        batches = [run_point(t) for t in tasks]
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame([row for batch in batches for row in batch])
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    kept, dropped = usable(df, POINT)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    print(f"E9: shift_notice, 8 robots, 20 tasks, seeds {seeds[0]}..{seeds[-1]}, rule-based forecasters only")
    print(f"points with a stalled run are left out of every summary: dropped {dropped} of "
          f"{df.groupby(POINT).ngroups}")
    means = kept.groupby(["lam", "notice_tick", "mode", "arm"])["avoidable"].mean().unstack("arm").round(1)
    print("\nmean avoidable cost (J_censored minus the free arm's):")
    print(means.to_string())
    print("\npaired differences in avoidable cost, A - B (positive: B is cheaper); median, 95% interval, share above 0:")
    for _, r in summary.iterrows():
        print(f"  lam {r.lam:g} tick {int(r.notice_tick):3d} {r['mode']:8s} {r['compare']:26s} "
              f"median {r.median_diff:7.1f} [{r.lo:7.1f}, {r.hi:7.1f}] above 0: {r.share_positive:.2f}  n {int(r.n)}")
    print(f"\nstalled runs (all of them, kept in runs.csv): {int(df.stalled.sum())} of {len(df)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
