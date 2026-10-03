"""S2 (stage 2): the choice between push, carry and fill. How it moves with slot capacity and with haul distance.

Sweep 1 (capacity): warehouse_racks with 0 to 8 racks. With no rack nothing can be carried; with more racks more
crates are carried away and fewer are pushed aside.
Sweep 2 (distance): dump_central with the dump region near the barrier or in the far corner, and the push cost
kappa. At the default costs pushing is cheaper than any haul, so nothing is carried; when pushing gets dear a near
dump is worth the haul and a far one is not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doi_common import (make_parser, out_dir, plt, project_hours, save_fig, seed_list, stalled_rate, timed_grid,
                        write_summary)
from src.doi.config import SimConfig
from src.doi.scenarios import DEFAULTS
from src.doi.stats import spearman

ARMS = ["rof", "central", "never"]
ALL_RACKS = DEFAULTS["warehouse_racks"]["racks"]
DUMP_NEAR, DUMP_FAR = (9, 6, 10, 8), (13, 0, 14, 2)
CAPACITY = {"scenario_params.racks": [ALL_RACKS[:n] for n in (0, 1, 2, 4, 8)]}
CAPACITY_QUICK = {"scenario_params.racks": [ALL_RACKS[:n] for n in (0, 2, 8)]}
DISTANCE = {"scenario_params.dump": [DUMP_NEAR, DUMP_FAR], "kappa": [4.0, 8.0, 16.0, 32.0]}
DISTANCE_QUICK = {"scenario_params.dump": [DUMP_NEAR, DUMP_FAR], "kappa": [4.0, 16.0]}


def base(scenario: str, quick: bool) -> SimConfig:
    return SimConfig(scenario=scenario, n_robots=6 if quick else 12, tasks_per_robot=6 if quick else 12,
                     r_comm=float("inf"))


def main(argv=None) -> None:
    args = make_parser(__doc__).parse_args(argv)
    out = out_dir(args, "s2_modes")
    seeds = seed_list(args)
    os.makedirs(out, exist_ok=True)

    cap_dir, dist_dir = os.path.join(out, "capacity"), os.path.join(out, "distance")
    cap, elapsed = timed_grid(base("warehouse_racks", args.quick), CAPACITY_QUICK if args.quick else CAPACITY, ARMS,
                              seeds, cap_dir, args.jobs)
    dist, _ = timed_grid(base("dump_central", args.quick), DISTANCE_QUICK if args.quick else DISTANCE, ARMS, seeds,
                         dist_dir, args.jobs)
    cap["n_racks"] = cap["axis_scenario_params.racks"].map(len)
    dist["dump_far"] = dist["axis_scenario_params.dump"].map(lambda d: tuple(d) == DUMP_FAR)
    cap.to_csv(os.path.join(cap_dir, "runs.csv"), index=False)
    dist.to_csv(os.path.join(dist_dir, "runs.csv"), index=False)
    write_summary(cap, cap_dir, ["policy", "n_racks"], ["carries", "removals", "slot_conflicts", "J_censored"])
    write_summary(dist, dist_dir, ["policy", "dump_far", "axis_kappa"], ["carries", "removals", "J_censored"])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm in ("rof", "central"):
        g = cap[cap.policy == arm].groupby("n_racks")
        ax.plot(g["carries"].median().index, g["carries"].median().values, marker="o", label=f"{arm}: carried")
        ax.plot(g["removals"].median().index, g["removals"].median().values, marker="s", ls="--",
                label=f"{arm}: pushed")
    ax.set_xlabel("racks on the map")
    ax.set_ylabel("median obstacles removed per run")
    ax.legend(fontsize=7)
    save_fig(fig, out, "s2_modes_vs_racks.png")

    print(f"\nS2 (n seeds = {len(seeds)}). Direction only.")
    rof = cap[cap.policy == "rof"]
    rho = spearman(rof["n_racks"].to_numpy(dtype=float), rof["carries"].to_numpy(dtype=float))
    print(f"capacity: carries with no rack = {rof[rof.n_racks == 0]['carries'].sum():.0f} (must be 0); "
          f"Spearman(racks, carries) = {rho:.2f} (expect >= 0)")
    for (far, kappa), g in dist[dist.policy == "rof"].groupby(["dump_far", "axis_kappa"]):
        print(f"distance: dump {'far' if far else 'near'}, kappa {kappa:g}: median carries {g['carries'].median():.1f}, "
              f"median pushes {g['removals'].median():.1f}, median J {g['J_censored'].median():.0f}")
    if args.quick:
        print(f"projected full S2 runtime (capacity sweep, rough): "
              f"{project_hours(elapsed, len(CAPACITY_QUICK['scenario_params.racks']), len(seeds), 5, 30, args.jobs):.1f} h")
    print("\nstalled rate per arm:\n" + stalled_rate(cap).to_string(index=False))


if __name__ == "__main__":
    main()
