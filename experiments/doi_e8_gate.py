"""E8 (H8): human-on-the-loop approval gate against a simulated supervisor."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (RESULTS_ROOT, make_parser, n_points, out_dir, paired_diff, plt, project_hours, save_fig,
                        seed_list, stalled_rate, timed_grid, verdict, write_summary)
from doi_e7_intake import llm_available
from src.doi.config import SimConfig

CACHE = os.path.join(RESULTS_ROOT, "intake_cache")


def axes_for(quick: bool):
    if quick:
        return {"scenario": ["incidents_room"], "intake": ["oracle"], "gate": [False, True],
                "sup_latency_median": [5, 30], "p_catch": [0.9]}
    intakes = ["oracle"] + (["llm:hosted"] if llm_available() else [])
    return {"scenario": ["incidents_room", "incidents_aisles"], "intake": intakes, "gate": [False, True],
            "sup_latency_median": [5, 30, 120], "p_catch": [0.5, 0.9]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = axes_for(args.quick)
    out = out_dir(args, "e8")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args, first=200)
    base = SimConfig(claim=True, n_robots=12, tasks_per_robot=10, intake_cache=CACHE, p_wrong_class=0.1,
                     sup_latency_sigma=0.5)
    df, elapsed = timed_grid(base, axes, ["rof"], seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E8 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(axes_for(False)), 30, args.jobs):.1f} h")
    rof = df[df.policy == "rof"]
    write_summary(rof, out, ["axis_scenario", "axis_intake", "axis_gate", "axis_sup_latency_median", "axis_p_catch"],
                  ["J_censored", "wrong_class_attempts", "mean_approval_wait", "pred_gate_cost"])
    keys = ["axis_scenario", "axis_intake", "axis_sup_latency_median", "axis_p_catch"]
    on, off, m = paired_diff(rof, keys, rof["axis_gate"] == True, rof["axis_gate"] == False, "J_censored")  # noqa: E712
    m["delta"] = m["a"] - m["b"]
    pred = rof[rof["axis_gate"] == True][keys + ["seed", "pred_gate_cost"]]  # noqa: E712
    m = m.merge(pred, on=keys + ["seed"])
    fig, axp = plt.subplots(figsize=(6, 4))
    g = m.groupby("axis_sup_latency_median")[["delta", "pred_gate_cost"]].median()
    axp.plot(g.index, g["delta"], marker="o", label="measured dJ (gate on - off)")
    axp.plot(g.index, g["pred_gate_cost"], marker="s", label="predicted sum(lambda_T * approval_wait)")
    axp.set_xlabel("supervisor latency median (ticks)")
    axp.legend(fontsize=7)
    save_fig(fig, out, "e8_cost_vs_latency.png")
    fig, axp = plt.subplots(figsize=(6, 4))
    w = rof.groupby(["axis_gate", "axis_p_catch"])["wrong_class_attempts"].mean().unstack()
    w.plot.bar(ax=axp)
    axp.set_ylabel("mean wrong_class_attempts")
    save_fig(fig, out, "e8_wrong_class.png")

    print(f"\nE8 / H8  (n seeds = {len(seeds)}). The supervisor is simulated; nothing here speaks to human usefulness.")
    for lat, g in m.groupby("axis_sup_latency_median"):
        meas, p = g["delta"].median(), g["pred_gate_cost"].median()
        ok = abs(meas - p) <= 0.25 * abs(p) if abs(p) > 1e-9 else None
        print(f"H8a latency {lat}: median measured dJ = {meas:.1f}, predicted = {p:.1f}, within 25%: {verdict(ok)}")
    for pc, g in rof.groupby("axis_p_catch"):
        w_on = g[g.axis_gate == True]["wrong_class_attempts"].sum()  # noqa: E712
        w_off = g[g.axis_gate == False]["wrong_class_attempts"].sum()  # noqa: E712
        ok = (w_on <= (1 - pc) * w_off) if w_off > 0 else None
        print(f"H8b p_catch {pc}: wrong_class_attempts on = {int(w_on)}, off = {int(w_off)}, bound (1-p_catch)*off = {(1 - pc) * w_off:.1f}: {verdict(ok)}")
    print("\nstalled rate:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
