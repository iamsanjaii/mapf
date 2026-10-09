"""Collect forecast cases: every forecast `rof_a` requests on `shift_notice`, with the truth label. Calls no model.

The oracle forecaster plays the run, so the cases are the ones a good policy meets. One file per split:

  dev    seeds 0..19      notice bank dev     prompt development only
  test   seeds 100..149   notice bank test    held-out scoring
  human  seeds 300..319   notice bank human   held-out scoring, always reported apart

Seeds 200..229 are reserved for E9 and appear in no file. The `human` split needs data/forecasts/human_notices.jsonl
(see data/forecasts/README.md); without it the split is skipped and level 2 is not done.

Usage: venv/bin/python experiments/doi_agent_cases.py [--quick] [--splits dev,test,human] [--out DIR]
                                                      [--human-notices PATH] [--jobs 4]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.forecast.case import case_to_dict
from src.doi.runner import run_episode

SPLITS = {"dev": (range(0, 20), "dev"), "test": (range(100, 150), "test"), "human": (range(300, 320), "human")}
MODES = ("true", "false", "missing", "quiet")
DEFAULT_OUT = os.path.join(ROOT, "data", "forecasts")
DEFAULT_HUMAN = os.path.join(ROOT, "data", "forecasts", "human_notices.jsonl")


def collect(task) -> List[dict]:
    """One run of rof_a with the oracle forecaster; one row per forecast it requested."""
    split, seed, mode, bank, human = task
    cfg = SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, lam=0.5, bundle_max=1,
                    policy="rof_a", forecaster="oracle",
                    scenario_params={"notice_mode": mode, "notice_bank": bank, "human_notices": human})
    res = run_episode(cfg)
    meta = {"split": split, "seed": seed, "mode": mode, "bank": bank, "stalled": bool(res.stalled)}
    return [{"meta": meta, "case": case_to_dict(c)} for c in res.forecast_cases]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the first 2 seeds of each split")
    ap.add_argument("--splits", default="dev,test,human")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--human-notices", default=DEFAULT_HUMAN)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLITS:
            ap.error(f"unknown split {split!r}: choose from {', '.join(SPLITS)}")
        seeds, bank = SPLITS[split]
        if bank == "human" and not os.path.exists(args.human_notices):
            print(f"human split skipped: {args.human_notices} not found (see data/forecasts/README.md); "
                  f"level 2 is not done")
            continue
        seeds = list(seeds)[:2] if args.quick else list(seeds)
        tasks = [(split, seed, mode, bank, args.human_notices) for seed in seeds for mode in MODES]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                batches = list(pool.map(collect, tasks))
        else:
            batches = [collect(t) for t in tasks]
        seen, rows = set(), []
        for row in (r for batch in batches for r in batch):
            if row["case"]["case_id"] not in seen:
                seen.add(row["case"]["case_id"])
                rows.append(row)
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        total = sum(len(b) for b in batches)
        share = sum(1 for r in rows if r["case"]["truth"]) / max(1, len(rows))
        print(f"{split}: {len(rows)} distinct cases from {total} requests in {len(tasks)} runs; "
              f"truth is yes in {share:.2f} of them; wrote {path}")
    print("no model was called")
    return 0


if __name__ == "__main__":
    sys.exit(main())
