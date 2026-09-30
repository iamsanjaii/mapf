"""
src/visualization/renderer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Matplotlib-based static grid renderer.

Renders:
- Grid cells (free, obstacle, pit, sandbag)
- Robot starting positions
- Robot goal positions
- Planned paths (as colored lines)
- Conflict markers (red stars)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.environment.grid import Grid, CellType
from src.mapf.conflict import Conflict, ConflictType


Pos = Tuple[int, int]
Path = List[Pos]

# Colour palette
COLORS = {
    CellType.FREE:     "#F5F5F5",
    CellType.OBSTACLE: "#2C3E50",
    CellType.PIT:      "#8E44AD",
    CellType.SANDBAG:  "#E67E22",
}

ROBOT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#1ABC9C", "#9B59B6", "#E91E63", "#00BCD4",
    "#FF5722", "#607D8B", "#795548", "#FFEB3B",
]


class GridRenderer:
    """
    Renders the grid environment and robot paths using Matplotlib.

    Parameters
    ----------
    cell_size : float
        Figure size scaling factor per cell.
    """

    def __init__(self, cell_size: float = 0.5) -> None:
        self.cell_size = cell_size

    # ------------------------------------------------------------------

    def render(
        self,
        grid: Grid,
        starts: Optional[List[Pos]] = None,
        goals: Optional[List[Pos]] = None,
        paths: Optional[Dict[int, Optional[Path]]] = None,
        conflicts: Optional[List[Conflict]] = None,
        title: str = "MAPF Environment",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Render the grid with optional paths and conflict markers.

        Parameters
        ----------
        grid : Grid
        starts : list of (row, col) — robot start positions
        goals  : list of (row, col) — robot goal positions
        paths  : dict robot_id → path
        conflicts : list of Conflict
        title : str
        save_path : path to save PNG (None = don't save)
        show : whether to call plt.show()
        """
        h, w = grid.height, grid.width
        fig_w = max(6, w * self.cell_size)
        fig_h = max(5, h * self.cell_size)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#16213E")

        # Draw cells
        for r in range(h):
            for c in range(w):
                ct = CellType(grid.array[r, c])
                color = COLORS.get(ct, "#F5F5F5")
                rect = mpatches.FancyBboxPatch(
                    (c, h - 1 - r), 1, 1,
                    boxstyle="round,pad=0.05",
                    linewidth=0.3,
                    edgecolor="#0F3460",
                    facecolor=color,
                )
                ax.add_patch(rect)

        # Draw paths
        if paths:
            for robot_id, path in paths.items():
                if not path:
                    continue
                color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
                xs = [p[1] + 0.5 for p in path]
                ys = [h - p[0] - 0.5 for p in path]
                ax.plot(xs, ys, "-", color=color, linewidth=2.0,
                        alpha=0.8, zorder=3)

        # Draw starts
        if starts:
            for i, pos in enumerate(starts):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                ax.text(
                    pos[1] + 0.5, h - pos[0] - 0.5,
                    f"S{i}", ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"),
                    zorder=5,
                )

        # Draw goals
        if goals:
            for i, pos in enumerate(goals):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                ax.text(
                    pos[1] + 0.5, h - pos[0] - 0.5,
                    f"G{i}", ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color,
                              ec="white", linewidth=1.5),
                    zorder=5,
                )

        # Draw conflict markers
        if conflicts:
            for conflict in conflicts:
                pos = conflict.location
                ax.scatter(
                    pos[1] + 0.5, h - pos[0] - 0.5,
                    s=120, marker="*", color="#FF0000", zorder=6,
                    label="_conflict" if conflict != conflicts[0] else "Conflict",
                )

        # Legend
        legend_elements = [
            mpatches.Patch(fc=COLORS[CellType.FREE],     label="Free"),
            mpatches.Patch(fc=COLORS[CellType.OBSTACLE], label="Obstacle"),
            mpatches.Patch(fc=COLORS[CellType.PIT],      label="Pit"),
            mpatches.Patch(fc=COLORS[CellType.SANDBAG],  label="Sandbag"),
        ]
        if conflicts:
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                       markersize=10, label="Conflict")
            )
        ax.legend(
            handles=legend_elements, loc="upper right",
            fontsize=7, framealpha=0.8,
            facecolor="#0F3460", labelcolor="white",
        )

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=11, pad=8)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Convenience: experiment result plots
    # ------------------------------------------------------------------

    @staticmethod
    def plot_metric(
        df,
        x_col: str,
        y_col: str,
        hue_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generic grouped line/bar chart for experiment results."""
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#16213E")

        groups = df[hue_col].unique()
        palette = ROBOT_COLORS[: len(groups)]

        for grp, color in zip(groups, palette):
            subset = df[df[hue_col] == grp].sort_values(x_col)
            means = subset.groupby(x_col)[y_col].mean()
            ax.plot(means.index, means.values, "o-", label=str(grp),
                    color=color, linewidth=2.0, markersize=6)

        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel(xlabel, color="#AAAAAA")
        ax.set_ylabel(ylabel, color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.legend(title=hue_col, facecolor="#0F3460", labelcolor="white",
                  title_fontsize=8)
        ax.grid(alpha=0.2, color="#334455")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig
