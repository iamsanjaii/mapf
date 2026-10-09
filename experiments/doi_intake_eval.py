"""Score cached intake results against the labelled items.

Usage: python experiments/doi_intake_eval.py --model-key hosted --split dev
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.incidents import read_jsonl
from src.doi.llm.intake import IntakeCache, cache_key

DEFAULT_CACHE = os.path.join("experiments", "results", "doi", "intake_cache")
OUT_DIR = os.path.join("experiments", "results", "doi", "intake")


def score(items, cache):
    rows, missing = [], 0
    for it in items:
        res = cache.get(cache_key(it.text, it.location_names))
        if res is None:
            missing += 1
        else:
            rows.append((it, res))

    def frac(num, den):
        return float(num) / den if den else float("nan")

    located = [(it, r) for it, r in rows if it.truth_location is not None]
    ambiguous = [(it, r) for it, r in rows if it.truth_location is None]
    ok = [(it, r) for it, r in rows if r.ok]
    schema_fail = [r for _, r in rows if not r.ok and not r.reject_reason.startswith("model_reject")]
    lat = [r.latency_s for _, r in rows]
    return {
        "items": len(items), "missing": missing, "scored": len(rows),
        "location_acc": frac(sum(1 for it, r in located if r.ok and r.location == it.truth_location), len(located)),
        "ambiguous_reject_rate": frac(sum(1 for _, r in ambiguous if not r.ok), len(ambiguous)),
        "class_acc": frac(sum(1 for it, r in ok if r.cls == it.truth_cls), len(ok)),
        "kind_acc": frac(sum(1 for it, r in ok if r.kind == it.truth_kind), len(ok)),
        "kits_mae": float(np.mean([abs(r.est_kits - it.truth_kits) for it, r in ok])) if ok else float("nan"),
        "schema_failure_rate": frac(len(schema_fail), len(rows)),
        "latency_p50": float(np.percentile(lat, 50)) if lat else float("nan"),
        "latency_p95": float(np.percentile(lat, 95)) if lat else float("nan"),
        "mean_prompt_tokens": float(np.mean([r.prompt_tokens for _, r in rows])) if rows else float("nan"),
        "mean_completion_tokens": float(np.mean([r.completion_tokens for _, r in rows])) if rows else float("nan"),
        "model": next((r.model for _, r in rows if r.model), ""),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test", "human"])
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)
    items = read_jsonl(os.path.join(ROOT, "data", "incidents", f"{args.split}.jsonl"))
    metrics = score(items, IntakeCache(args.cache_dir, args.model_key))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.model_key}-{args.split}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics))
        w.writeheader()
        w.writerow(metrics)
    for k, v in metrics.items():
        print(f"{k:24s} {v:.3f}" if isinstance(v, float) else f"{k:24s} {v}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
