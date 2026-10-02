"""E6 (Stage 1): scaling on warehouse maps; H2 effect size and H6 (smallest r_comm with PoD <= 1.15).

Needs a MovingAI-format warehouse map (--map PATH). Without --map, --quick uses a small synthetic warehouse.
Maps are not committed; download them into data/maps/ first (see README).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from doi_common import (finite_x, make_parser, n_points, out_dir, plt, project_hours, save_fig, seed_list,
                        stalled_rate, timed_grid, verdict, write_summary)
from src.doi.config import SimConfig
from src.doi.maps import load_movingai, write_synthetic_warehouse
from src.doi.stats import spearman

ARMS = ["rof", "central", "never"]
INF = float("inf")
AXES = {"n_robots": [100, 200, 500], "r_comm": [0.0, 4.0, 8.0, 16.0, 32.0, INF], "loss": [0.0, 0.2]}
AXES_QUICK = {"n_robots": [12], "r_comm": [0.0, 8.0, INF], "loss": [0.0]}


def main(argv=None) -> None:
    ap = make_parser(__doc__)
    ap.add_argument("--map", default=None, help="path to a MovingAI .map file")
    ap.add_argument("--horizon", type=int, default=None)
    ap.set_defaults(seeds=10)
    args = ap.parse_args(argv)
    out = out_dir(args, "e6")
    os.makedirs(out, exist_ok=True)
    map_path = args.map
    if map_path is None:
        if not args.quick:
            ap.error("--map is required for a full run (download a warehouse map into data/maps/)")
        map_path = write_synthetic_warehouse(os.path.join(out, "synthetic.map"))
    grid = load_movingai(map_path)
    axes = AXES_QUICK if args.quick else AXES
    horizon = args.horizon or (300 if args.quick else 2000)
    base = SimConfig(scenario="warehouse_pits", map_path=map_path, horizon=horizon, claim=True,
                     record_epoch=100, delta_gossip=True, plan_window=32,
                     scenario_params={"n_pits": 6 if args.quick else 10, "n_stations": 2})
    seeds = seed_list(args)
    arms = ARMS + ["free"]
    df, elapsed = timed_grid(base, axes, arms, seeds, out, args.jobs)
    if args.quick:
        print(f"projected full E6 runtime: {project_hours(elapsed, n_points(axes), len(seeds), n_points(AXES), 10, args.jobs):.1f} h "
              f"(crude: the quick grid is far smaller than N = 500)")
    write_summary(df, out, ["policy", "axis_n_robots", "axis_r_comm", "axis_loss"],
                  ["throughput", "J", "pod", "runtime_ms", "overrides_per_1000"])
    df["msg_units_per_robot_tick"] = df["message_units"] / (df["n_robots"] * df["ticks"].clip(lower=1))
    rof = df[df.policy == "rof"].copy()
    rof["x"] = rof["axis_r_comm"].map(finite_x)

    fig, axp = plt.subplots(figsize=(6.5, 4))
    for n, g in rof.groupby("axis_n_robots"):
        m = g.groupby("x")["pod"].median()
        axp.plot(range(len(m)), m.values, marker="o", label=f"N = {n}")
        axp.set_xticks(range(len(m)))
        axp.set_xticklabels(["inf" if v >= 1e3 else f"{v:g}" for v in m.index])
    axp.axhline(1.15, color="k", ls="--", lw=0.8)
    axp.set_xlabel("ledger r_comm")
    axp.set_ylabel("median PoD = J_RoF / J_central")
    axp.legend(fontsize=7)
    save_fig(fig, out, "e6_pod_vs_range.png")

    print(f"\nE6 / Stage 1  (map {os.path.basename(map_path)} {grid.height}x{grid.width}, horizon {horizon}, "
          f"{len(seeds)} seeds)")
    longest = max(grid.height, grid.width)
    for n, g in rof.groupby("axis_n_robots"):
        rho = spearman(g["x"].to_numpy(), g["pod"].to_numpy())
        finite = g[g["x"] < 1e3]
        slope = np.polyfit(finite["x"], finite["pod"].fillna(finite["pod"].median()), 1)[0] if finite["x"].nunique() > 1 else float("nan")
        med = g.groupby("x")["pod"].median()
        ok = med[med <= 1.15]
        r_star = float(ok.index.min()) if len(ok) else float("nan")
        frac = r_star / longest if r_star == r_star and r_star < 1e3 else float("nan")
        print(f"N = {n}: Spearman(r_comm, PoD) = {rho:.2f}, slope of PoD against finite r_comm = {slope:.4f}, "
              f"smallest r_comm with PoD <= 1.15: {r_star:g} ({frac:.2f} of max(H, W); H6 needs <= 0.4: {verdict(frac <= 0.4 if frac == frac else None)})")
    print(f"median throughput per 1000 ticks by arm:\n{df.groupby('policy')['throughput'].median().round(1).to_string()}")
    print(f"median runtime_ms by N (RoF): {rof.groupby('axis_n_robots')['runtime_ms'].median().round(0).to_dict()}")
    print(f"message units per robot per tick (RoF): {rof['msg_units_per_robot_tick'].median():.2f}; "
          f"overrides per 1000 ticks (RoF): {rof['overrides_per_1000'].median():.1f}")
    print("\nstalled rate per arm:\n" + stalled_rate(df).to_string(index=False))


if __name__ == "__main__":
    main()
