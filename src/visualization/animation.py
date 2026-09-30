"""
src/visualization/animation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Matplotlib FuncAnimation-based animator.

Produces a frame-by-frame animation of robots moving along their planned
paths and saves it as a GIF (or MP4 if ffmpeg is available).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from src.environment.grid import Grid, CellType
from src.visualization.renderer import COLORS, ROBOT_COLORS

Pos = Tuple[int, int]
Path = List[Pos]


class GridAnimator:
    """
    Animates robot movement along planned paths.

    Parameters
    ----------
    grid : Grid
    paths : dict robot_id → time-indexed path
    starts : list of start positions
    goals : list of goal positions
    fps : int
        Frames per second.
    cell_size : float
    """

    def __init__(
        self,
        grid: Grid,
        paths: Dict[int, Optional[Path]],
        starts: Optional[List[Pos]] = None,
        goals: Optional[List[Pos]] = None,
        fps: int = 4,
        cell_size: float = 0.5,
    ) -> None:
        self.grid = grid
        self.paths = {rid: p for rid, p in paths.items() if p}
        self.starts = starts or []
        self.goals = goals or []
        self.fps = fps
        self.cell_size = cell_size
        self._max_t = max((len(p) for p in self.paths.values()), default=1)
        # Optional per-cell colour overrides: {(row,col): hex_colour}
        self.color_overrides: Dict[Pos, str] = {}

    # ------------------------------------------------------------------

    def animate(
        self,
        save_path: Optional[str] = "experiments/results/animation.gif",
        title: str = "AMR-MAPF Simulation",
        live: bool = False,
    ) -> None:
        """Render and save the animation, optionally showing it live."""
        h, w = self.grid.height, self.grid.width
        fig_w = max(6, w * self.cell_size)
        fig_h = max(5, h * self.cell_size)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#16213E")

        def _draw_base() -> None:
            ax.clear()
            ax.set_facecolor("#16213E")
            for r in range(h):
                for c in range(w):
                    ct = CellType(self.grid.array[r, c])
                    # Check per-cell override first (e.g. green for filled pits)
                    color = self.color_overrides.get(
                        (r, c), COLORS.get(ct, "#F5F5F5")
                    )
                    rect = mpatches.FancyBboxPatch(
                        (c, h - 1 - r), 1, 1,
                        boxstyle="round,pad=0.05",
                        linewidth=0.3,
                        edgecolor="#0F3460",
                        facecolor=color,
                    )
                    ax.add_patch(rect)

            # Draw path traces (faded)
            for rid, path in self.paths.items():
                color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
                xs = [p[1] + 0.5 for p in path]
                ys = [h - p[0] - 0.5 for p in path]
                ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=0.3, zorder=2)

            # Draw goals
            for i, pos in enumerate(self.goals):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                ax.text(
                    pos[1] + 0.5, h - pos[0] - 0.5, f"G{i}",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="white", lw=1.5),
                    zorder=4,
                )

            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        def _update(frame: int) -> None:
            _draw_base()
            ax.set_title(
                f"{title}  [t={frame}]",
                color="white", fontsize=10, pad=6,
            )

            for rid, path in self.paths.items():
                color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
                t = min(frame, len(path) - 1)
                pos = path[t]
                circle = plt.Circle(
                    (pos[1] + 0.5, h - pos[0] - 0.5),
                    0.35, color=color, zorder=5,
                )
                ax.add_patch(circle)
                ax.text(
                    pos[1] + 0.5, h - pos[0] - 0.5, str(rid),
                    ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=6,
                )

        anim = FuncAnimation(
            fig,
            _update,
            frames=self._max_t,
            interval=1000 // self.fps,
            repeat=live,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            writer = PillowWriter(fps=self.fps)
            anim.save(save_path, writer=writer)
            print(f"[Animation] saved → {save_path}")

        if live:
            plt.show()
        else:
            plt.close(fig)
