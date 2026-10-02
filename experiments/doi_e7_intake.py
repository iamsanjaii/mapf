"""E7 (H7): exception intake in family D. Cost of none/oracle/llm intake, false reports, safety invariants."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (RESULTS_ROOT, make_parser, n_points, out_dir, paired_diff, plt, project_hours, save_fig,
                        seed_list, stalled_rate, timed_grid, verdict, write_summary)
from src.doi.config import SimConfig

ARMS = ["rof", "central", "never"]
CACHE = os.path.join(RESULTS_ROOT, "intake_cache")


def llm_available(model_key: str = "hosted") -> bool:
    path = os.path.join(CACHE, model_key, "cache.jsonl")
    return os.path.exists(path) and os.path.getsize(path) > 0


def axes_for(quick: bool):
    if quick:
        return {"scenario": ["incidents_room", "incidents_aisles"], "intake": ["none", "oracle"],
                "scenario_params.p_report": [0.9], "scenario_params.p_false": [0.0, 0.2]}
    intakes = ["none", "oracle"] + (["llm:hosted"] if llm_available() else [])
    return {"scenario": ["incidents_room", "incidents_aisles"], "intake": intakes,
            "scenario_params.p_report": [0.5, 0.9], "scenario_params.p_false": [0.0, 0.1, 0.2]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = axes_for(args.quick)
    out = out_dir(args, "e7")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args, first=200)
    if not args.quick and not llm_available():
        print("NOTE: no cached llm:hosted intake results found; that arm is omitted (run doi_intake_run.py first).")
    base = SimConfig(n_robots=12, tasks_per_robot=10, intake_cache=CACHE, p_wrong_class=0.0)
    df, elapsed = timed_grid(base, axes, ARMS, seeds, out, args.jobs)
    if args.quick:
        full = {k: v for k, v in axes_for(False).items()}
        print(f"projected full E7 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(full), 30, args.jobs):.1f} h")
    write_summary(df, out, ["policy", "axis_scenario", "axis_intake", "axis_scenario_params.p_false"],
                  ["J_censored", "false_report_cost", "intake_rejected", "removals"])
    rof = df[df.policy == "rof"]
    fig, axp = plt.subplots(figsize=(7, 4))
    for scen, g in rof.groupby("axis_scenario"):
        m = g.groupby("axis_intake")["J_censored"].median()
        axp.bar([f"{scen[:9]}\n{i}" for i in m.index], m.values)
    axp.set_ylabel("median J_censored (RoF)")
    save_fig(fig, out, "e7_cost_vs_intake.png")
    fig, axp = plt.subplots(figsize=(6, 4))
    g = rof[rof.axis_intake != "none"].groupby("axis_scenario_params.p_false")["false_report_cost"].median()
    axp.plot(g.index, g.values, marker="o")
    axp.set_xlabel("p_false")
    axp.set_ylabel("median false_report_cost")
    save_fig(fig, out, "e7_false_report_cost.png")

    print(f"\nE7 / H7  (n seeds = {len(seeds)}; seeds {seeds[0]}..{seeds[-1]})")
    bad = df[df["wrong_class_attempts"] > 0]
    print(f"H7a no push attempted on an obstacle that needs a human, in every run ({len(df)} runs): "
          f"{verdict(len(bad) == 0)}  (needs p_wrong_class = 0, as set here)")
    keys = ["axis_scenario", "axis_scenario_params.p_report", "axis_scenario_params.p_false"]
    for intake in [i for i in axes["intake"] if i != "oracle"]:
        a, b, m = paired_diff(rof, keys, rof["axis_intake"] == intake, rof["axis_intake"] == "oracle", "J_censored")
        if len(a):
            print(f"J({intake}) / J(oracle), paired by seed: median {np.median(a / b):.3f} over {len(a)} pairs")
    print("H7b/H7c need the [pilot] thresholds of spec section 8 and the llm:hosted cache; not evaluated here.")
    fr = rof[rof.axis_intake == "oracle"].groupby("axis_scenario_params.p_false")["false_report_cost"].median()
    print("H7d false_report_cost by p_false (oracle intake): " + ", ".join(f"{k}: {v:.1f}" for k, v in fr.items()))
    print("intake_rejected total: " + str(int(df["intake_rejected"].sum())))
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
