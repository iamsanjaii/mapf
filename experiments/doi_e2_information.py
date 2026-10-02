"""E2 (H2): price of information. HR_av(RoF) against ledger range, loss, latency and fleet size."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (axis_cols, finite_x, make_parser, n_points, out_dir, plt, project_hours, save_fig,
                        seed_list, stalled_rate, timed_grid, verdict, write_summary)
from src.doi.config import SimConfig
from src.doi.stats import spearman

ARMS = ["rof", "rof_local", "central", "never"]
INF = float("inf")
AXES = {"scenario": ["single_pit", "multi_pit_wall"], "r_comm": [0.0, 2.0, 4.0, 8.0, 16.0, INF],
        "loss": [0.0, 0.2, 0.5], "latency": [1, 3], "n_robots": [6, 12, 24]}
AXES_QUICK = {"scenario": ["single_pit"], "r_comm": [0.0, 8.0, INF], "loss": [0.0], "latency": [1],
              "n_robots": [6]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = AXES_QUICK if args.quick else AXES
    out = out_dir(args, "e2")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args)
    df, elapsed = timed_grid(SimConfig(claim=True), axes, ARMS, seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E2 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 30, args.jobs):.1f} h")
    ax = axis_cols(df)
    write_summary(df, out, ["policy", "axis_scenario", "axis_r_comm", "axis_loss"],
                  ["hr_av", "pod", "mean_coverage", "J_censored", "message_units"])
    rof = df[df.policy == "rof"].copy()
    rof["x"] = rof["axis_r_comm"].map(finite_x)

    for name, col, ylabel in [("e2_hrav_vs_range.png", "hr_av", "median HR_av"),
                              ("e2_pod_vs_range.png", "pod", "median PoD = J_RoF / J_central"),
                              ("e2_coverage_vs_range.png", "mean_coverage", "median coverage at trigger")]:
        fig, axp = plt.subplots(figsize=(6.5, 4))
        for loss, g in rof.groupby("axis_loss"):
            m = g.groupby("x")[col].median()
            axp.plot(range(len(m)), m.values, marker="o", label=f"loss {loss}")
            axp.set_xticks(range(len(m)))
            axp.set_xticklabels(["inf" if v >= 1e3 else f"{v:g}" for v in m.index])
        axp.set_xlabel("ledger r_comm")
        axp.set_ylabel(ylabel)
        axp.legend(fontsize=7)
        save_fig(fig, out, name)

    print(f"\nE2 / H2  (n seeds = {len(seeds)}). Direction only; effect size is a Stage 1 result.")
    for scen, g in rof.groupby("axis_scenario"):
        rho = spearman(g["x"].to_numpy(), g["hr_av"].to_numpy())
        pods = g[(g["axis_r_comm"] == INF) & (g["axis_loss"] == 0.0)]["pod"]
        zero = g[g["axis_r_comm"] == 0.0]
        print(f"{scen}: Spearman(r_comm, HR_av(RoF)) = {rho:.2f} -> non-increasing: {verdict(rho <= 0 if not np.isnan(rho) else None)}; "
              f"median PoD at r_comm=inf, loss 0 = {pods.median():.3f} (threshold 1.05: {verdict(pods.median() <= 1.05 if len(pods) else None)}); "
              f"median HR_av at r_comm=0 = {zero['hr_av'].median():.2f} (n_robots + 1 = "
              f"{sorted(set(zero['n_robots'].tolist()))} + 1; equal-rent case not reproduced by random streams)")
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
