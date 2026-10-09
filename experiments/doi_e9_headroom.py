"""E9 headroom pilot: how much can any forecaster change fleet cost on shift_notice? Calls no model.

`oracle` is the best a forecaster can be and `inverted` the worst, so the gap between them, and between `numeric`
and `oracle`, bounds what a language model could add. Costs are compared as avoidable cost: J minus the cost of
the `free` arm on the same seed, because most of J is travel no rule can avoid.

A run that stalls is not a cost: its J is the censored cost of an unfinished run. Existing arms stall on this
scenario at high fees (a pallet is pushed onto another robot's pending goal and the rule never clears it), so a
stalled run is left out of every summary, together with the other arms on the same seed, mode and cost setting,
which a paired comparison cannot do without. `runs.csv` keeps every run; the printed output says how many points
were dropped.

Usage: venv/bin/python experiments/doi_e9_headroom.py [--quick] [--seeds 100] [--jobs 4] [--out DIR]
       venv/bin/python experiments/doi_e9_headroom.py --resummarise DIR    (recompute from DIR/runs.csv)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.metrics import summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario
from src.doi.stats import bootstrap_ci

ARMS = [("rof", "rof", "numeric"), ("numeric", "rof_a", "numeric"), ("keyword", "rof_a", "keyword"),
        ("oracle", "rof_a", "oracle"), ("inverted", "rof_a", "inverted")]      # (label, policy, forecaster)
COSTS = [(4.0, 1.0), (8.0, 1.0), (4.0, 20.0), (8.0, 40.0)]                     # (kappa, fee)
MODES = ["true", "false", "missing", "quiet"]
COMPARE = [("numeric", "oracle"), ("inverted", "oracle"), ("rof", "oracle"), ("numeric", "keyword"),
           ("rof", "numeric")]
FIRST_SEED = 200
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "e9_headroom")


def run_point(task) -> List[dict]:
    seed, mode, kappa, fee = task
    cfg = SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, kappa=kappa, fee=fee,
                    lam=0.5, scenario_params={"notice_mode": mode})
    scenario = build_scenario(cfg)
    free = run_episode(cfg.replace(policy="free"), scenario=scenario).J_censored
    rows = []
    for label, policy, forecaster in ARMS:
        res = run_episode(cfg.replace(policy=policy, forecaster=forecaster), scenario=scenario)
        cols = summary_row(res, cheap=True)
        rows.append({"seed": seed, "mode": mode, "kappa": kappa, "fee": fee, "arm": label, "J": res.J_censored,
                     "J_free": free, "avoidable": res.J_censored - free, "removals": res.removals,
                     "stalled": res.stalled, "forecasts": cols["forecasts"], "forecast_acc": cols["forecast_acc"],
                     "wrong_yes": cols["wrong_yes"], "wrong_no": cols["wrong_no"]})
    return rows


POINT = ["seed", "mode", "kappa", "fee"]


def usable(df: pd.DataFrame, point: List[str] = POINT) -> Tuple[pd.DataFrame, int]:
    """The runs of every point (seed, mode, cost setting) in which no arm stalled, and how many points were dropped.
    `point` names the columns that identify a point."""
    bad = df.groupby(point)["stalled"].transform("any").astype(bool)
    return df[~bad], int(df[bad].groupby(point).ngroups)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per cost setting and notice mode: paired differences in avoidable cost between arms, with a bootstrap
    interval on the median, and the share of the numeric arm's avoidable cost a perfect forecast would remove.
    Points in which any arm stalled are left out (see `usable`)."""
    out = []
    for (kappa, fee, mode), g in usable(df)[0].groupby(["kappa", "fee", "mode"]):
        wide = g.pivot(index="seed", columns="arm", values="avoidable")
        headroom = float((wide["numeric"] - wide["oracle"]).mean() / max(1e-9, wide["numeric"].mean()))
        for a, b in COMPARE:
            d = (wide[a] - wide[b]).to_numpy(dtype=float)
            med, lo, hi = bootstrap_ci(d, n=2000, seed=0)
            out.append({"kappa": kappa, "fee": fee, "mode": mode, "compare": f"{a} - {b}", "mean_diff": float(d.mean()),
                        "median_diff": med, "lo": lo, "hi": hi, "share_positive": float((d > 0).mean()),
                        "headroom_share": headroom, "mean_avoidable_numeric": float(wide["numeric"].mean()),
                        "n": len(d)})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="3 seeds and the default costs only")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resummarise", metavar="DIR", default=None, help="recompute summary.csv from DIR/runs.csv")
    args = ap.parse_args(argv)
    if args.resummarise:
        out = args.resummarise
        df = pd.read_csv(os.path.join(out, "runs.csv"), dtype={"mode": str})
    else:
        out = args.out or (DEFAULT_OUT + ("_quick" if args.quick else ""))
        os.makedirs(out, exist_ok=True)
        seeds = range(FIRST_SEED, FIRST_SEED + (3 if args.quick else args.seeds))
        costs = COSTS[:1] if args.quick else COSTS
        tasks = [(seed, mode, kappa, fee) for kappa, fee in costs for mode in MODES for seed in seeds]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                batches = list(pool.map(run_point, tasks))
        else:
            batches = [run_point(t) for t in tasks]
        df = pd.DataFrame([row for batch in batches for row in batch])
        df.to_csv(os.path.join(out, "runs.csv"), index=False)
    kept, dropped = usable(df)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    print(f"E9 headroom pilot: shift_notice, 8 robots, 20 tasks, lam 0.5, seeds {int(df.seed.min())}..{int(df.seed.max())}; "
          f"no model was called")
    print(f"stalled runs are left out of every summary below, with the other arms of the same seed, mode and cost "
          f"setting: dropped {dropped} of {df.groupby(POINT).ngroups} points")
    means = kept.groupby(["kappa", "fee", "mode", "arm"])["avoidable"].mean().unstack("arm").round(1)
    print("\nmean avoidable cost (J minus the free arm's J):")
    print(means[[label for label, _, _ in ARMS]].to_string())
    acc = kept[kept.arm.isin(["numeric", "keyword"])].groupby(["kappa", "fee", "mode", "arm"])["forecast_acc"].mean()
    print("\nmean forecast accuracy against the truth label:")
    print(acc.unstack("arm").round(2).to_string())
    print("\npaired differences in avoidable cost (median, 95% interval of the median, share of seeds above 0):")
    for _, r in summary.iterrows():
        print(f"  kappa {r.kappa:g} fee {r.fee:g} {r['mode']:8s} {r['compare']:20s} median {r.median_diff:7.1f} "
              f"[{r.lo:7.1f}, {r.hi:7.1f}]  above 0: {r.share_positive:.2f}  n {int(r.n)}")
    top = summary[summary["compare"] == "numeric - oracle"][["kappa", "fee", "mode", "headroom_share"]]
    print("\nshare of the numeric arm's avoidable cost a perfect forecast removes:")
    print(top.round(3).to_string(index=False))
    print(f"\nstalled runs (all of them, kept in runs.csv): {int(df.stalled.sum())} of {len(df)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
