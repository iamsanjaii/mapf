"""Score forecasters on a file of forecast cases (levels 1 to 3 of section 14.1 of the design).

Accuracy, the two kinds of wrong answer, failures by code, calibration of the stated confidence, and what a forecast
costs in calls, tokens and latency. The non-model forecasters are scored on the same cases as baselines. A model is
scored from the store; a call that is not stored is an error unless --live, which asks before it spends.

Usage: venv/bin/python experiments/doi_agent_eval.py --cases data/forecasts/test.jsonl
           [--forecasters numeric,keyword,oracle,inverted,llm:small] [--agent-mode tools|single]
           [--cache DIR] [--live] [--yes] [--quick] [--out DIR]
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
from src.doi.forecast.agent import MAX_TURNS
from src.doi.forecast.budget import confirm_calls
from src.doi.forecast.case import ForecastCase, case_from_dict
from src.doi.forecast.forecasters import make_forecaster
from src.doi.llm.chatcache import ReplayMiss

DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "agent_eval")
DEFAULT_FORECASTERS = "numeric,keyword,oracle,inverted"
CODES = ("no_tool_call", "no_answer", "bad_answer", "client_error")


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
        "calls_per_forecast": _mean(r.calls for r in results),
        "tool_calls_per_forecast": _mean(r.tool_calls for r in results),
        "latency_p50_s": p50, "latency_p95_s": p95,
        "latency_p50_ticks": p50 / tick_seconds, "latency_p95_ticks": p95 / tick_seconds,
        "prompt_tokens_per_forecast": _mean(r.prompt_tokens for r in results),
        "completion_tokens_per_forecast": _mean(r.completion_tokens for r in results),
    }
    for code in CODES:
        row[f"fail_{code}"] = fails[code]
    return row


def main(argv=None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--forecasters", default=DEFAULT_FORECASTERS)
    ap.add_argument("--agent-mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--cache", default=None, help="the model-call store (default: the SimConfig default)")
    ap.add_argument("--live", action="store_true", help="let a call that is not stored reach the model")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--quick", action="store_true", help="the first 20 cases; a model only if its store has calls")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    if args.quick:
        cases = cases[:20]
    names = [n.strip() for n in args.forecasters.split(",") if n.strip()]
    cache = args.cache or SimConfig().agent_cache
    if args.quick:
        names = [n for n in names if not n.startswith("llm:")
                 or os.path.exists(os.path.join(cache, n.split(":", 1)[1], "chat.jsonl"))]
    models = [n for n in names if n.startswith("llm:")]
    if args.live and models:
        per = MAX_TURNS if args.agent_mode == "tools" else 1
        if not confirm_calls(len(cases) * len(models), per, args.yes, confirm):
            print("aborted")
            return 1
    rows = []
    for name in names:
        cfg = SimConfig(forecaster=name, agent_mode=args.agent_mode, agent_cache=cache, agent_live=args.live)
        try:
            rows.append(evaluate(name, make_forecaster(name, cfg), cases, cfg.tick_seconds))
        except ReplayMiss as miss:
            print(f"error: {miss}", file=sys.stderr)
            return 2
    os.makedirs(args.out, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "eval.csv"), index=False)
    shown = ["forecaster", "n", "answered", "truth_share", "accuracy", "wrong_yes_rate", "wrong_no_rate",
             "failure_rate", "ece", "calls_per_forecast", "latency_p50_ticks"]
    print(f"scored {len(cases)} cases from {args.cases}" + ("" if models else "; no model was called"))
    print(table[shown].round(3).to_string(index=False))
    print(f"wrote {os.path.join(args.out, 'eval.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
