"""
experiments/q2_mapf.py
~~~~~~~~~~~~~~~~~~~~~~~
Experiments B, C, D, F — Multi-robot MAPF comparison.

Compares:
- Independent A* (Baseline 1)
- Prioritized MAPF (Baseline 2)
- Agentic MAPF (Proposed)

Across:
- Robot counts: 2, 5, 10, 15
- Obstacle densities: 10%, 20%, 30%
- Grid sizes: 20×20, 30×30, 40×40

Produces:
- experiments/results/exp_bcd_multiagent.csv
- experiments/results/exp_bcd_*.png

Usage:
    python experiments/q2_mapf.py
    python experiments/q2_mapf.py --seed 42 --runs 5
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
    parser = argparse.ArgumentParser(description="Experiments B/C/D/F: Multi-Robot MAPF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Experiments B/C/D/F — Multi-Robot MAPF Comparison")
    print("=" * 60)

    mgr = ExperimentManager(output_dir=args.output)
    df = mgr.experiment_bcd_multiagent(
        grid_sizes=[20, 30, 40],
        obstacle_densities=[0.10, 0.20, 0.30],
        robot_counts=[2, 5, 10, 15],
        algorithms=["independent", "prioritized", "agentic"],
        runs_per_config=args.runs,
        seed=args.seed,
    )

    # Filter to valid rows
    valid = df[df.get("error", df.get("conflict_count", None)).isna()
               if "error" in df.columns else df["conflict_count"].notna()]

    print("\n--- Algorithm Comparison (mean across all configs) ---")
    cols = ["success_rate", "total_cost", "makespan", "conflict_count",
            "replan_count", "runtime_ms"]
    available = [c for c in cols if c in df.columns]
    summary = df.groupby("algorithm")[available].mean().round(3)
    print(summary.to_string())

    # Plot: Robots vs Conflict Count
    GridRenderer.plot_metric(
        df=df.groupby(["num_robots", "algorithm"])["conflict_count"]
            .mean().reset_index(),
        x_col="num_robots",
        y_col="conflict_count",
        hue_col="algorithm",
        title="Number of Robots vs Conflict Count",
        xlabel="Number of Robots",
        ylabel="Mean Conflict Count",
        save_path=os.path.join(args.output, "exp_bcd_conflicts.png"),
    )
    plt.close("all")

    # Plot: Robots vs Runtime
    GridRenderer.plot_metric(
        df=df.groupby(["num_robots", "algorithm"])["runtime_ms"]
            .mean().reset_index(),
        x_col="num_robots",
        y_col="runtime_ms",
        hue_col="algorithm",
        title="Number of Robots vs Planning Runtime",
        xlabel="Number of Robots",
        ylabel="Mean Runtime (ms)",
        save_path=os.path.join(args.output, "exp_bcd_runtime.png"),
    )
    plt.close("all")

    # Plot: Obstacle density vs Success Rate
    GridRenderer.plot_metric(
        df=df.groupby(["obstacle_density", "algorithm"])["success_rate"]
            .mean().reset_index(),
        x_col="obstacle_density",
        y_col="success_rate",
        hue_col="algorithm",
        title="Obstacle Density vs Success Rate",
        xlabel="Obstacle Density",
        ylabel="Mean Success Rate",
        save_path=os.path.join(args.output, "exp_bcd_success.png"),
    )
    plt.close("all")

    # Plot: Grid size vs Runtime (scaling)
    GridRenderer.plot_metric(
        df=df.groupby(["grid_size", "algorithm"])["runtime_ms"]
            .mean().reset_index(),
        x_col="grid_size",
        y_col="runtime_ms",
        hue_col="algorithm",
        title="Grid Size vs Planning Runtime (Scaling)",
        xlabel="Grid Size (N×N)",
        ylabel="Mean Runtime (ms)",
        save_path=os.path.join(args.output, "exp_d_scaling.png"),
    )
    plt.close("all")

    print(f"\n[Done] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
