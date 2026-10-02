"""E1 (H1): is RoF within 2 + r_max/B_real (+0.15) of the avoidable hindsight cost on a single pit?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (axis_cols, make_parser, n_points, out_dir, plt, project_hours, save_fig, seed_list,
                        stalled_rate, timed_grid, verdict, write_summary)
from src.doi.config import SimConfig

ARMS = ["never", "eager", "myopic", "rof", "central"]
AXES = {"scenario_params.depot_dist": [1, 2, 4, 8], "fee": [1, 10, 50, 200], "kappa": [1, 4, 16],
        "tasks_per_robot": [2, 5, 10, 20, 40], "n_robots": [4, 12]}
AXES_QUICK = {"scenario_params.depot_dist": [2], "fee": [1, 50], "kappa": [4], "tasks_per_robot": [5, 10],
              "n_robots": [4]}


def main(argv=None) -> None:
    ap = make_parser(__doc__)
    ap.add_argument("--claim", choices=["off", "on"], default="off",
                    help="claim protocol (the plan fixes E1 at off; on is for the pilot comparison)")
    args = ap.parse_args(argv)
    axes = AXES_QUICK if args.quick else AXES
    out = out_dir(args, "e1" + ("_claim" if args.claim == "on" else ""))
    os.makedirs(out, exist_ok=True)
    base = SimConfig(scenario="single_pit", claim=(args.claim == "on"), r_comm=float("inf"), loss=0.0, gate=False,
                     r_traffic=8.0, loss_traffic=0.0)
    seeds = seed_list(args)
    df, elapsed = timed_grid(base, axes, ARMS, seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E1 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 30, args.jobs):.1f} h "
              f"on {args.jobs} job(s)")
    ax = axis_cols(df)
    never = df[df.policy == "never"].copy()
    never["RB"] = never["rent_total"] / never["buy_lb_total"]
    cells = never.groupby(ax)[["RB", "r_max"]].median().reset_index()
    rof = df[df.policy == "rof"]
    b_real = rof.groupby(ax)[["mean_B_real", "mean_B_est"]].median().reset_index()
    cells = cells.merge(b_real, on=ax, how="left")
    hr = df.groupby(ax + ["policy"])["hr_av"].median().reset_index().pivot(index=ax, columns="policy", values="hr_av")
    cells = cells.merge(hr.reset_index(), on=ax, how="left", suffixes=("", "_hr"))
    cells["bound"] = 2 + cells["r_max"] / cells["mean_B_real"] + 0.15
    # Diagnostic (not part of H1): charge hindsight the realised buy cost instead of the lower bound buy_lb.
    free = df[df.policy == "free"][ax + ["seed", "J_censored"]].rename(columns={"J_censored": "J_free"})
    hind = df[df.policy == "hindsight"][ax + ["seed", "J_censored", "n_prefilled"]].rename(
        columns={"J_censored": "J_hind_travel", "n_prefilled": "s_star"})
    d = rof[ax + ["seed", "J_censored", "mean_B_real"]].merge(free, on=ax + ["seed"]).merge(hind, on=ax + ["seed"])
    d["den"] = (d["J_hind_travel"] - d["J_free"]) + d["s_star"] * d["mean_B_real"].fillna(0)
    d["hr_av_breal"] = np.where(d["den"] >= 1, (d["J_censored"] - d["J_free"]) / d["den"], np.nan)
    cells = cells.merge(d.groupby(ax)["hr_av_breal"].median().reset_index(), on=ax, how="left")
    write_summary(df, out, ["policy"] + ax, ["J_censored", "hr_av", "fills", "stalled"])
    cells.to_csv(os.path.join(out, "cells.csv"), index=False)

    fig, axp = plt.subplots(figsize=(7, 4.5))
    for pol in ARMS:
        if pol in cells and pol not in ("central",):
            axp.scatter(cells["RB"], cells[pol], label=pol, s=18)
    axp.scatter(cells["RB"], cells["bound"], marker="_", color="k", label="2 + r_max/B_real + 0.15")
    axp.set_xscale("log")
    axp.set_xlabel("R / B (hindsight rent over buy_lb)")
    axp.set_ylabel("median HR_av")
    axp.legend(fontsize=7)
    save_fig(fig, out, "e1_hrav_vs_rent_over_buy.png")

    filled = cells.dropna(subset=["bound", "rof"])
    h1a = bool((filled["rof"] <= filled["bound"]).all()) if len(filled) else None
    low, high = cells[cells["RB"] <= 0.3], cells[cells["RB"] >= 5]
    diag = cells.dropna(subset=["bound", "hr_av_breal"])
    h1d = bool((diag["hr_av_breal"] <= diag["bound"]).all()) if len(diag) else None
    h1b = bool((low["eager"] > 3).any()) if len(low) else None
    h1c = bool((high["never"] > 3).any()) if len(high) else None
    print(f"\nE1 / H1  (n seeds = {len(seeds)}, cells = {len(cells)})")
    print(f"sweep reaches R/B <= 0.3: {len(low) > 0}   R/B >= 5: {len(high) > 0}   (min {cells['RB'].min():.2f}, max {cells['RB'].max():.2f})")
    print(f"HR_av defined in {100 * df[df.policy.isin(ARMS)]['hr_av'].notna().mean():.0f}% of non-baseline rows")
    print(f"H1a HR_av(RoF) <= 2 + r_max/B_real + 0.15 in every cell with an edit: {verdict(h1a)} "
          f"(worst margin {np.nanmax(filled['rof'] - filled['bound']) if len(filled) else float('nan'):.2f})")
    print(f"   diagnostic (not H1): same check with hindsight charged B_real instead of buy_lb: {verdict(h1d)} "
          f"(worst margin {np.nanmax(diag['hr_av_breal'] - diag['bound']) if len(diag) else float('nan'):.2f}; "
          f"median B_real/B_est {np.nanmedian(cells['mean_B_real'] / cells['mean_B_est']):.2f})")
    print(f"H1b HR_av(Eager) > 3 somewhere with R/B <= 0.3: {verdict(h1b)}")
    print(f"H1c HR_av(NeverFill) > 3 where R/B >= 5: {verdict(h1c)}")
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
