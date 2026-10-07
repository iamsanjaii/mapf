"""E9: does a forecast agent behind the guard lower fleet cost on shift_notice, and how bad is it when the notices
are wrong? (Section 14.3 of the design.)

Avoidable cost is J_censored minus the `free` arm's on the same seed. A point (seed, mode, lam, notice_tick) in
which any arm stalled is left out of every summary, as in the headroom pilot. Model arms replay stored calls and
call nothing; a call that is not stored is an error unless --live, which prints an upper bound and asks first.

Hypotheses (fixed before a full run; the margin of H9c is fixed after the --quick pilot):
  H9a  true notices:  rof_a with a model costs less than rof_a with numeric
  H9b  false notices: rof_a with a model costs no more than rof_a with inverted; its cost against rof is reported
  H9c  no notices (missing, quiet): rof_a with a model is within a fixed margin of rof_a with numeric
  H9d  is scored by doi_agent_eval.py on the human cases, not here

Usage: venv/bin/python experiments/doi_e9_agent.py [--quick] [--seeds 100] [--model-seeds 30] [--jobs 4]
           [--modes true,false,missing,quiet] [--model-keys small,large] [--with-single] [--cache DIR] [--live] [--yes]
           [--out DIR]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doi_e9_headroom import usable
from src.doi.config import SimConfig
from src.doi.forecast.agent import MAX_TURNS
from src.doi.forecast.budget import confirm_calls
from src.doi.llm.chatcache import ReplayMiss
from src.doi.metrics import summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario
from src.doi.stats import bootstrap_ci

NON_MODEL_ARMS = [("never", "never", "numeric", "tools"), ("rof", "rof", "numeric", "tools"),
                  ("rof_p", "rof_p", "numeric", "tools"), ("numeric", "rof_a", "numeric", "tools"),
                  ("keyword", "rof_a", "keyword", "tools"), ("oracle", "rof_a", "oracle", "tools"),
                  ("inverted", "rof_a", "inverted", "tools")]          # (label, policy, forecaster, agent mode)
MODES = ["true", "false", "missing", "quiet"]
PRIMARY = (0.5, 5)                                                       # (lam, notice_tick)
SECONDARY = [(0.25, 5), (1.0, 5), (0.5, 150)]                            # on the true and false modes only
POINT = ["seed", "mode", "lam", "notice_tick"]
FIRST_SEED = 200
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "e9_agent")


def make_cfg(seed: int, mode: str, lam: float, notice_tick: int, **kw) -> SimConfig:
    return SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, lam=lam, bundle_max=1,
                     scenario_params={"notice_mode": mode, "notice_tick": notice_tick}, **kw)


def model_arms(keys: List[str], single: bool = True) -> list:
    """llm:<key> with tools for every key, and (if `single`) the first key again as a single call."""
    arms = [(f"llm:{k}", "rof_a", f"llm:{k}", "tools") for k in keys]
    if single:
        arms.append((f"llm:{keys[0]} single", "rof_a", f"llm:{keys[0]}", "single"))
    return arms


def store_has_calls(cache: str, key: str) -> bool:
    path = os.path.join(cache, key, "chat.jsonl")
    return os.path.exists(path) and os.path.getsize(path) > 0


def select_model_arms(keys: List[str], cache: str, live: bool, single: bool = False) -> Tuple[list, List[str]]:
    """Model arms that can run: all of them when live, else only those whose key has stored calls. The single-call
    arm needs its own stored calls, so it runs when live or when `single` says they are there."""
    notes, usable_keys = [], []
    for k in keys:
        if live or store_has_calls(cache, k):
            usable_keys.append(k)
        else:
            notes.append(f"no stored calls for llm:{k}: its arms are left out (fill the store with --live)")
    if not usable_keys:
        return [], notes
    return model_arms(usable_keys, single=live or single), notes


def run_point(task) -> List[dict]:
    seed, mode, lam, notice_tick, arms, cache, live = task
    cfg = make_cfg(seed, mode, lam, notice_tick, agent_cache=cache, agent_live=live)
    scenario = build_scenario(cfg)
    free = run_episode(cfg.replace(policy="free"), scenario=scenario).J_censored
    rows = []
    for label, policy, forecaster, agent_mode in arms:
        res = run_episode(cfg.replace(policy=policy, forecaster=forecaster, agent_mode=agent_mode), scenario=scenario)
        cols = summary_row(res, cheap=True)
        rows.append({"seed": seed, "mode": mode, "lam": lam, "notice_tick": notice_tick, "arm": label,
                     "J": res.J_censored, "J_free": free, "avoidable": res.J_censored - free,
                     "removals": res.removals, "stalled": res.stalled, "forecasts": cols["forecasts"],
                     "forecast_acc": cols["forecast_acc"], "wrong_yes": cols["wrong_yes"],
                     "wrong_no": cols["wrong_no"], "agent_calls": cols["agent_calls"],
                     "agent_latency_s": cols["agent_latency_s"], "agent_prompt_tokens": cols["agent_prompt_tokens"],
                     "agent_completion_tokens": cols["agent_completion_tokens"]})
    return rows


def hypotheses(labels) -> list:
    """(hypothesis, mode, arm A, arm B): the difference A - B in avoidable cost; positive means B is cheaper."""
    out = []
    for model in sorted(l for l in labels if l.startswith("llm:") and "single" not in l):
        out += [("H9a", "true", "numeric", model), ("H9b", "false", "inverted", model), ("H9b", "false", "rof", model),
                ("H9c", "missing", "numeric", model), ("H9c", "quiet", "numeric", model)]
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Paired differences in avoidable cost for the hypotheses and for the headroom references."""
    kept, _ = usable(df, POINT)
    labels = set(kept.arm)
    pairs = [(h, m, a, b) for h, m, a, b in hypotheses(labels)]
    pairs += [("ref", m, a, b) for m in MODES for a, b in (("numeric", "oracle"), ("inverted", "oracle"),
                                                         ("rof", "numeric"), ("never", "rof"))]
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


def count_requests(seeds, points, modes, cache) -> int:
    """Forecast requests of a run with the numeric forecaster: the number a model arm would make, roughly."""
    total = 0
    for seed in seeds:
        for mode, lam, tick in points(modes):
            cfg = make_cfg(seed, mode, lam, tick, policy="rof_a", forecaster="numeric", agent_cache=cache)
            res = run_episode(cfg, scenario=build_scenario(cfg))
            total += sum(1 for r in res.forecast_log if not r["skipped"])
    return total


def main(argv=None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="seeds 200..202, no secondary sweeps")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--model-seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--model-keys", default="small,large")
    ap.add_argument("--with-single", action="store_true",
                    help="also run the single-call arm from the store (it is always run with --live)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--live", action="store_true", help="let a call that is not stored reach the model")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cache = args.cache or SimConfig().agent_cache
    out = args.out or (DEFAULT_OUT + ("_quick" if args.quick else ""))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = list(range(FIRST_SEED, FIRST_SEED + (3 if args.quick else args.seeds)))
    model_seeds = set(seeds[:(3 if args.quick else args.model_seeds)])

    def points(selected):
        base = [(m, *PRIMARY) for m in selected]
        extra = [] if args.quick else [(m, lam, tick) for m in selected if m in ("true", "false")
                                       for lam, tick in SECONDARY]
        return base + extra

    keys = [k.strip() for k in args.model_keys.split(",") if k.strip()]
    model, notes = select_model_arms(keys, cache, args.live, args.with_single)
    for note in notes:
        print("NOTE:", note)
    jobs = args.jobs
    if args.live and model:
        per = MAX_TURNS * sum(1 for a in model if a[3] == "tools") + sum(1 for a in model if a[3] == "single")
        requests = count_requests(sorted(model_seeds), points, modes, cache)
        if not confirm_calls(requests, per, args.yes, confirm):
            print("aborted")
            return 1
        jobs = 1                                    # one process appends to the store at a time
    tasks = [(seed, m, lam, tick, NON_MODEL_ARMS + (model if seed in model_seeds else []), cache, args.live)
             for seed in seeds for m, lam, tick in points(modes)]
    try:
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                batches = list(pool.map(run_point, tasks))
        else:
            batches = [run_point(t) for t in tasks]
    except ReplayMiss as miss:
        print(f"error: {miss}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame([row for batch in batches for row in batch])
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    kept, dropped = usable(df, POINT)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    called = bool(args.live and model)
    print(f"E9: shift_notice, 8 robots, 20 tasks, seeds {seeds[0]}..{seeds[-1]}, model arms on "
          f"{len(model_seeds)} seeds; " + ("model calls were allowed" if called else "no model was called"))
    print(f"points with a stalled run are left out of every summary: dropped {dropped} of "
          f"{df.groupby(POINT).ngroups}")
    means = kept.groupby(["lam", "notice_tick", "mode", "arm"])["avoidable"].mean().unstack("arm").round(1)
    print("\nmean avoidable cost (J_censored minus the free arm's):")
    print(means.to_string())
    print("\npaired differences in avoidable cost, A - B (positive: B is cheaper); median, 95% interval, share above 0:")
    for _, r in summary.iterrows():
        print(f"  lam {r.lam:g} tick {int(r.notice_tick):3d} {r['mode']:8s} {r.hypothesis:3s} {r['compare']:26s} "
              f"median {r.median_diff:7.1f} [{r.lo:7.1f}, {r.hi:7.1f}] above 0: {r.share_positive:.2f}  n {int(r.n)}")
    print(f"\nstalled runs (all of them, kept in runs.csv): {int(df.stalled.sum())} of {len(df)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
