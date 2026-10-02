"""E4 (H4): claims, leases and wasted hauls under ledger loss."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (axis_cols, make_parser, n_points, out_dir, paired_diff, plt, project_hours, save_fig,
                        seed_list, stalled_rate, timed_grid, verdict, write_summary)
from src.doi.config import SimConfig

AXES = {"loss": [0.0, 0.1, 0.3, 0.5], "claim": [False, True], "lease_ticks": [4, 8, 16], "r_comm": [4.0, 8.0]}
AXES_QUICK = {"loss": [0.0, 0.5], "claim": [False, True], "lease_ticks": [8], "r_comm": [8.0]}


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    axes = AXES_QUICK if args.quick else AXES
    out = out_dir(args, "e4")
    os.makedirs(out, exist_ok=True)
    seeds = seed_list(args)
    base = SimConfig(scenario="single_pit")
    df, elapsed = timed_grid(base, axes, ["rof"], seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E4 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 30, args.jobs):.1f} h")
    rof = df[df.policy == "rof"]
    write_summary(rof, out, ["axis_loss", "axis_claim", "axis_lease_ticks"],
                  ["wasted_haul_cost", "fills", "claims_lost", "aborts", "J_censored"])
    fig, axp = plt.subplots(figsize=(6, 4))
    for claim, g in rof.groupby("axis_claim"):
        m = g.groupby("axis_loss")["wasted_haul_cost"].median()
        axp.plot(m.index, m.values, marker="o", label=f"claim {'on' if claim else 'off'}")
    axp.set_xlabel("ledger loss")
    axp.set_ylabel("median wasted_haul_cost")
    axp.legend()
    save_fig(fig, out, "e4_waste_vs_loss.png")

    print(f"\nE4 / H4  (n seeds = {len(seeds)})")
    print(f"never more fills than pits (1): {verdict(bool((rof['fills'] <= 1).all()))} (max fills {int(rof['fills'].max())})")
    keys = ["axis_loss", "axis_lease_ticks", "axis_r_comm"]
    for loss in sorted(rof["axis_loss"].unique()):
        sub = rof[rof["axis_loss"] == loss]
        off, on, _ = paired_diff(sub, keys, sub["axis_claim"] == False, sub["axis_claim"] == True, "wasted_haul_cost")  # noqa: E712
        ratio = off.mean() / on.mean() if len(on) and on.mean() > 0 else float("nan")
        note = ""
        if loss >= 0.3:
            ok = (off.mean() >= 2 * on.mean()) if len(on) else None
            note = f"  H4 (loss >= 0.3: off >= 2x on): {verdict(ok)}"
        print(f"loss {loss}: mean wasted cost claim off {off.mean():.1f} vs on {on.mean():.1f} (ratio {ratio:.2f}); "
              f"claims_lost(on) {rof[(rof['axis_loss'] == loss) & (rof['axis_claim'] == True)]['claims_lost'].sum()}{note}")  # noqa: E712
    print("\nstalled rate:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
