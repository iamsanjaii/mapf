"""
experiments/mapf_ro.py
~~~~~~~~~~~~~~~~~~~~~~~
Experiment E — MAPF-RO: Obstacle Removal vs Detour Comparison.

Compares:
- No obstacle removal (standard planning with pits as impassable)
- With obstacle removal (pits can be filled by sandbags)

Metrics:
- Success rate
- Total energy (robot + sandbag + removal cost)
- Path length
- Pits filled
- Runtime

Usage:
    python experiments/mapf_ro.py
    python experiments/mapf_ro.py --seed 42 --runs 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.experiments import ExperimentManager
from src.visualization.renderer import GridRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E: MAPF-RO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Experiment E — MAPF-RO: Removal vs Detour")
    print("=" * 60)

    mgr = ExperimentManager(output_dir=args.output)
    df = mgr.experiment_e_mapfro(
        grid_sizes=[20, 30],
        obstacle_densities=[0.10, 0.20],
        pit_densities=[0.05, 0.10],
        robot_counts=[5, 10],
        runs_per_config=args.runs,
        seed=args.seed,
    )

    print("\n--- Removal vs No-Removal Comparison ---")
    cols = ["success_rate", "total_energy", "total_robot_cost",
            "total_sandbag_cost", "pits_filled"]
    available = [c for c in cols if c in df.columns]
    summary = df.groupby("enable_removal")[available].mean().round(3)
    print(summary.to_string())

    df["label"] = df["enable_removal"].map({True: "With Removal", False: "No Removal"})

    # Plot: Total energy comparison
    GridRenderer.plot_metric(
        df=df.groupby(["num_robots", "label"])["total_energy"]
            .mean().reset_index(),
        x_col="num_robots",
        y_col="total_energy",
        hue_col="label",
        title="Total Energy vs Robots (Removal vs No Removal)",
        xlabel="Number of Robots",
        ylabel="Mean Total Energy",
        save_path=os.path.join(args.output, "exp_e_energy.png"),
    )
    plt.close("all")

    # Plot: Success rate
    GridRenderer.plot_metric(
        df=df.groupby(["pit_density", "label"])["success_rate"]
            .mean().reset_index(),
        x_col="pit_density",
        y_col="success_rate",
        hue_col="label",
        title="Success Rate vs Pit Density (Removal vs No Removal)",
        xlabel="Pit Density",
        ylabel="Mean Success Rate",
        save_path=os.path.join(args.output, "exp_e_success.png"),
    )
    plt.close("all")

    print(f"\n[Done] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
