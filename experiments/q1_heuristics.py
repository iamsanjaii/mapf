"""
experiments/q1_heuristics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Experiment A — Single-robot heuristic comparison.

Compares Manhattan, Euclidean, and Chebyshev heuristics across:
- Grid sizes: 20×20, 30×30, 40×40
- Obstacle densities: 10%, 20%, 30%

Produces:
- experiments/results/exp_a_heuristics.csv
- experiments/results/exp_a_*.png  (comparison plots)

Usage:
    python experiments/q1_heuristics.py
    python experiments/q1_heuristics.py --seed 42 --runs 10
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.experiments import ExperimentManager
from src.visualization.renderer import GridRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A: Heuristic Comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per configuration")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Experiment A — Single-Robot Heuristic Comparison")
    print("=" * 60)

    mgr = ExperimentManager(output_dir=args.output)
    df = mgr.experiment_a_heuristics(
        grid_sizes=[20, 30, 40],
        obstacle_densities=[0.10, 0.20, 0.30],
        heuristics=["manhattan", "euclidean", "chebyshev"],
        runs_per_config=args.runs,
        seed=args.seed,
    )

    print("\n--- Summary Statistics ---")
    summary = (
        df.groupby("heuristic")[["path_cost", "nodes_expanded", "runtime_ms"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())

    # Plot 1: Nodes expanded by heuristic across grid sizes
    GridRenderer.plot_metric(
        df=df.groupby(["grid_size", "heuristic"])["nodes_expanded"]
            .mean().reset_index(),
        x_col="grid_size",
        y_col="nodes_expanded",
        hue_col="heuristic",
        title="Nodes Expanded vs Grid Size (by Heuristic)",
        xlabel="Grid Size (N×N)",
        ylabel="Mean Nodes Expanded",
        save_path=os.path.join(args.output, "exp_a_nodes_expanded.png"),
    )
    plt.close("all")

    # Plot 2: Runtime by heuristic
    GridRenderer.plot_metric(
        df=df.groupby(["grid_size", "heuristic"])["runtime_ms"]
            .mean().reset_index(),
        x_col="grid_size",
        y_col="runtime_ms",
        hue_col="heuristic",
        title="Planning Runtime vs Grid Size (by Heuristic)",
        xlabel="Grid Size (N×N)",
        ylabel="Mean Runtime (ms)",
        save_path=os.path.join(args.output, "exp_a_runtime.png"),
    )
    plt.close("all")

    # Plot 3: Path cost by obstacle density
    GridRenderer.plot_metric(
        df=df.groupby(["obstacle_density", "heuristic"])["path_cost"]
            .mean().reset_index(),
        x_col="obstacle_density",
        y_col="path_cost",
        hue_col="heuristic",
        title="Path Cost vs Obstacle Density (by Heuristic)",
        xlabel="Obstacle Density",
        ylabel="Mean Path Cost",
        save_path=os.path.join(args.output, "exp_a_path_cost.png"),
    )
    plt.close("all")

    print(f"\n[Done] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
