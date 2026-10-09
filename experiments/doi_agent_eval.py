"""Score forecasters on a file of forecast cases (levels 1 to 3 of section 14.1 of the design).

Accuracy, the two kinds of wrong answer, failures by code, calibration of the stated confidence, and the answer
latency. Every forecaster is scored on the same cases.

Usage: venv/bin/python experiments/doi_agent_eval.py --cases data/forecasts/test.jsonl
           [--forecasters numeric,keyword,ledger,projected,oracle,inverted] [--quick] [--sample N] [--out DIR]
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.forecast.case import ForecastCase, case_from_dict
from src.doi.forecast.forecasters import make_forecaster

DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "agent_eval")
DEFAULT_FORECASTERS = "numeric,keyword,ledger,oracle,inverted"
CODES = ("no_answer", "bad_answer")


def load_cases(path: str) -> List[Tuple[dict, ForecastCase]]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(r["meta"], case_from_dict(r["case"])) for r in rows]


def ece(pairs, bins: int = 10) -> float:
    """Expected calibration error of (confidence, answer was right) pairs, in equal-width bins."""
    if not pairs:
        return float("nan")
    total = 0.0
    for b in range(bins):
        group = [(c, ok) for c, ok in pairs if min(int(c * bins), bins - 1) == b]
        if group:
            total += len(group) / len(pairs) * abs(sum(ok for _, ok in group) / len(group)
                                                   - sum(c for c, _ in group) / len(group))
    return total


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def evaluate(name: str, forecaster, cases, tick_seconds: float) -> dict:
    results = [forecaster.forecast(case) for _, case in cases]
    n = len(results)
    scored = [(r, c) for r, (_, c) in zip(results, cases) if r.answer is not None and c.truth is not None]
    right = [r.answer == c.truth for r, c in scored]
    fails = Counter(r.failed for r in results if r.failed)
    latency = [r.latency_s for r in results]
    p50 = float(np.percentile(latency, 50)) if latency else float("nan")
    p95 = float(np.percentile(latency, 95)) if latency else float("nan")
    row = {
        "forecaster": name, "n": n, "answered": len(scored),
        "truth_share": _mean(c.truth for _, c in cases if c.truth is not None),
        "accuracy": _mean(right),
        "wrong_yes_rate": _mean(1.0 if r.answer and not c.truth else 0.0 for r, c in scored),
        "wrong_no_rate": _mean(1.0 if not r.answer and c.truth else 0.0 for r, c in scored),
        "failure_rate": sum(fails.values()) / n if n else float("nan"),
        "ece": ece([(r.confidence, ok) for (r, _), ok in zip(scored, right) if r.confidence is not None]),
        "latency_p50_s": p50, "latency_p95_s": p95,
        "latency_p50_ticks": p50 / tick_seconds, "latency_p95_ticks": p95 / tick_seconds,
    }
    for code in CODES:
        row[f"fail_{code}"] = fails[code]
    row["first_failure"] = next((f"{r.failed}: {r.reason}" for r in results if r.failed), "")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--forecasters", default=DEFAULT_FORECASTERS)
    ap.add_argument("--quick", action="store_true", help="the first 20 cases")
    ap.add_argument("--sample", type=int, default=0, help="N evenly spaced cases from the whole file (spread over "
                                                          "seeds and modes); overrides --quick's first 20")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    if args.sample > 0:
        step = max(1, len(cases) // args.sample)
        cases = cases[::step][:args.sample]
    elif args.quick:
        cases = cases[:20]
    names = [n.strip() for n in args.forecasters.split(",") if n.strip()]
    tick_seconds = SimConfig().tick_seconds
    rows = [evaluate(name, make_forecaster(name), cases, tick_seconds) for name in names]
    os.makedirs(args.out, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "eval.csv"), index=False)
    shown = ["forecaster", "n", "answered", "truth_share", "accuracy", "wrong_yes_rate", "wrong_no_rate",
             "failure_rate", "ece", "latency_p50_ticks"]
    print(f"scored {len(cases)} cases from {args.cases}")
    print(table[shown].round(3).to_string(index=False))
    for _, r in table.iterrows():
        if r["first_failure"]:
            print(f"{r['forecaster']}: first failure was {r['first_failure']}")
    print(f"wrote {os.path.join(args.out, 'eval.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
