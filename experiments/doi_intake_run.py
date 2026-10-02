"""Run an LLM over incident reports and write the intake cache. The only code that calls an LLM.

Usage:
  python experiments/doi_intake_run.py --model-key hosted --split dev
  python experiments/doi_intake_run.py --model-key hosted --scenario incidents_aisles --seeds 200..229 \
      --p-report 0.5,0.9 --p-false 0.0,0.1,0.2
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.incidents import location_names, read_jsonl, render_report, report_rng
from src.doi.llm.client import client_from_env
from src.doi.llm.intake import IntakeCache, cache_key, run_intake
from src.doi.scenarios import build_scenario

EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS = 400, 80
DEFAULT_CACHE = os.path.join("experiments", "results", "doi", "intake_cache")


def parse_seeds(spec: str):
    a, b = spec.split("..")
    return range(int(a), int(b) + 1)


def split_items(split: str):
    items = read_jsonl(os.path.join(ROOT, "data", "incidents", f"{split}.jsonl"))
    return [(it.text, it.location_names) for it in items]


def scenario_items(name: str, seeds, p_reports, p_falses):
    out = []
    for seed in seeds:
        for pr in p_reports:
            for pf in p_falses:
                cfg = SimConfig(scenario=name, n_robots=2, tasks_per_robot=2, seed=seed,
                                scenario_params={"p_report": pr, "p_false": pf})
                s = build_scenario(cfg)
                names = location_names(s)
                for r in s.reports:
                    out.append((render_report(r, report_rng(s, r))[0], names))
    return out


def pending(items, cache):
    seen, todo = set(), []
    for text, names in items:
        key = cache_key(text, names)
        if key not in seen and cache.get(key) is None:
            seen.add(key)
            todo.append((text, names))
    return todo


def main(argv=None, client=None, confirm=input) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--split", choices=["dev", "test", "human"])
    ap.add_argument("--scenario")
    ap.add_argument("--seeds")
    ap.add_argument("--p-report", default="0.9")
    ap.add_argument("--p-false", default="0.0")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = ap.parse_args(argv)
    if args.split:
        items = split_items(args.split)
    elif args.scenario and args.seeds:
        items = scenario_items(args.scenario, parse_seeds(args.seeds),
                               [float(x) for x in args.p_report.split(",")],
                               [float(x) for x in args.p_false.split(",")])
    else:
        ap.error("give --split or --scenario with --seeds")
    cache = IntakeCache(args.cache_dir, args.model_key)
    todo = pending(items, cache)
    print(f"{len(items)} reports, {len(todo)} not yet cached for model key '{args.model_key}'")
    print(f"estimated tokens for the uncached items: ~{len(todo) * EST_PROMPT_TOKENS} prompt, "
          f"~{len(todo) * EST_COMPLETION_TOKENS} completion (multiply by your model's price)")
    if not todo:
        return 0
    if not args.yes and confirm("Proceed with these calls? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 1
    client = client or client_from_env(args.model_key)
    failures = 0
    for i, (text, names) in enumerate(todo, 1):
        try:
            cache.put(run_intake(text, names, client))
            failures = 0
        except Exception as e:
            failures += 1
            print(f"item {i} failed: {type(e).__name__}: {e}", file=sys.stderr)
            if failures >= 5:
                print("five failures in a row, stopping", file=sys.stderr)
                return 2
    print(f"done, cache at {cache.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
