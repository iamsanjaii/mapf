"""E3 (H3): complements (series pits) and substitutes (parallel pits)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (make_parser, n_points, out_dir, plt, project_hours, save_fig, seed_list, stalled_rate,
                        timed_grid, verdict, write_summary)
from src.doi.config import SimConfig

ARMS = ["rof", "rof_pit", "myopic", "eager", "never"]
AXES = {"scenario": ["series_pits", "two_pits_parallel"], "n_robots": [4, 12], "tasks_per_robot": [10, 20]}
AXES_QUICK = {"scenario": ["series_pits", "two_pits_parallel"], "n_robots": [4], "tasks_per_robot": [10]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = AXES_QUICK if args.quick else AXES
    out = out_dir(args, "e3")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args)
    df, elapsed = timed_grid(SimConfig(claim=True), axes, ARMS, seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E3 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 30, args.jobs):.1f} h")
    write_summary(df, out, ["policy", "axis_scenario"], ["hr_av", "fills", "J_censored", "wasted_haul_cost"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for a, scen in zip(axs, axes["scenario"]):
        g = df[(df.axis_scenario == scen) & df.policy.isin(ARMS)]
        data = [g[g.policy == p]["hr_av"].dropna().to_numpy() for p in ARMS]
        a.boxplot(data, tick_labels=ARMS)
        a.set_title(scen)
        a.set_ylabel("HR_av")
    save_fig(fig, out, "e3_complements.png")

    print(f"\nE3 / H3  (n seeds = {len(seeds)})")
    s = df[df.axis_scenario == "series_pits"]
    rof, pit = s[s.policy == "rof"], s[s.policy == "rof_pit"]
    print(f"series: RoF-Pit makes no edit in every run: {verdict(bool((pit['fills'] == 0).all()))} "
          f"({int((pit['fills'] == 0).sum())}/{len(pit)}); RoF makes two edits: {int((rof['fills'] == 2).sum())}/{len(rof)}; "
          f"median HR_av(RoF) = {rof['hr_av'].median():.2f} (threshold 2.5: {verdict(rof['hr_av'].median() <= 2.5)})")
    p = df[(df.axis_scenario == "two_pits_parallel") & (df.policy == "rof")]
    second = p[p["fills"] >= 2]
    served_all = second[second["min_true_rent_after_first_fill"] == 0]
    print(f"parallel: RoF filled the second pit in {len(second)}/{len(p)} runs; in {len(served_all)} of those the first fill "
          f"already served every recorded task at the second trigger (H3b violation if > 0): {verdict(len(served_all) == 0)}")
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
