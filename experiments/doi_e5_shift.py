"""E5 (H5, stretch): non-stationary demand. Regret of RoF, RoF-W and RoF-X under a hotspot shift."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (make_parser, n_points, out_dir, plt, project_hours, save_fig, seed_list, stalled_rate,
                        timed_grid, write_summary)
from src.doi.config import SimConfig

ARMS = ["rof", "rof_w", "rof_x", "central", "never"]
AXES = {"scenario": ["shift", "multi_pit_wall"], "theta": [0.5, 1.0, 2.0], "window": [50, 200],
        "r_comm": [2.0, 4.0, 8.0]}
AXES_QUICK = {"scenario": ["shift", "multi_pit_wall"], "theta": [1.0], "window": [50], "r_comm": [8.0]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = AXES_QUICK if args.quick else AXES
    out = out_dir(args, "e5")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args)
    df, elapsed = timed_grid(SimConfig(claim=True), axes, ARMS, seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E5 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 30, args.jobs):.1f} h")
    # regret fraction (J_alg - J_hind) / (J_never - J_hind) using the hindsight J that includes its buy cost
    hind = df[df.policy == "hindsight"].assign(J_hind=lambda d: d["J_censored"])
    never = df[df.policy == "never"][["axis_scenario", "axis_theta", "axis_window", "axis_r_comm", "seed", "J_censored"]]
    keys = ["axis_scenario", "axis_theta", "axis_window", "axis_r_comm", "seed"]
    base = hind[keys + ["J_hind"]].merge(never.rename(columns={"J_censored": "J_never"}), on=keys)
    d = df[df.policy.isin(["rof", "rof_w", "rof_x"])].merge(base, on=keys)
    d["regret"] = (d["J_censored"] - d["J_hind"]) / (d["J_never"] - d["J_hind"]).where(lambda x: x.abs() >= 1)
    d.to_csv(os.path.join(out, "regret.csv"), index=False)
    write_summary(d, out, ["policy", "axis_scenario"], ["regret", "J_censored"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for a, scen in zip(axs, axes["scenario"]):
        g = d[d.axis_scenario == scen]
        a.boxplot([g[g.policy == p]["regret"].dropna() for p in ["rof", "rof_w", "rof_x"]],
                  tick_labels=["rof", "rof_w", "rof_x"])
        a.set_title(scen)
        a.set_ylabel("regret fraction")
    save_fig(fig, out, "e5_shift_regret.png")
    print(f"\nE5 / H5 (stretch, n seeds = {len(seeds)})")
    for scen in axes["scenario"]:
        g = d[d.axis_scenario == scen].groupby("policy")["regret"].median()
        print(f"{scen}: median regret " + ", ".join(f"{k} {v:.3f}" for k, v in g.items()))
    print("H5: RoF-W lower regret than RoF under shift; RoF-X lower than RoF when stationary (report negative results).")
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
