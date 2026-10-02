"""Matplotlib animation of one or more arms on the same scenario, with a running-cost counter per arm."""
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.doi.narrate import short_events
from src.environment.grid import CellType

Pos = Tuple[int, int]
COLORS = {
    "free": (0.96, 0.96, 0.93), "wall": (0.18, 0.19, 0.22), "pit": (0.82, 0.22, 0.20),
    "filled": (0.30, 0.72, 0.42), "depot": (0.96, 0.68, 0.18), "incident": (0.56, 0.30, 0.72),
}
TRAIL = 8


def cumulative_cost(result, cfg) -> np.ndarray:
    """Cost paid up to each tick, rebuilt from trajectories (equals result.J at the end for run arms)."""
    n_ticks = max(len(t) for t in result.trajectory.values())
    cost = np.zeros(n_ticks)
    for i, traj in result.trajectory.items():
        carry = result.carry_trace.get(i, [False] * len(traj))
        for t in range(len(traj) - 1):
            moved = traj[t + 1] != traj[t]
            cost[t + 1] += cfg.kappa if (moved and carry[t]) else 1.0
    for ft in result.filled_at.values():
        if ft >= 0 and ft + 1 < n_ticks:
            cost[ft + 1] += cfg.fee
    return np.cumsum(cost)


def goal_timelines(result, scenario) -> Dict[int, List[Optional[Pos]]]:
    """The goal each robot is heading to at each tick, derived from where it actually stops."""
    out = {}
    for i, traj in result.trajectory.items():
        goals = scenario.tasks[i]
        k, line = 0, []
        for pos in traj:
            if k < len(goals) and pos == goals[k]:
                k += 1
            line.append(goals[k] if k < len(goals) else None)
        out[i] = line
    return out


def _background(scenario, result, t: int) -> np.ndarray:
    g = scenario.grid
    img = np.zeros((g.height, g.width, 3))
    for r in range(g.height):
        for c in range(g.width):
            ct = g.get(r, c)
            key = {CellType.OBSTACLE: "wall", CellType.PIT: "pit", CellType.SANDBAG: "depot"}.get(ct, "free")
            img[r, c] = COLORS[key]
    for cell, ft in result.filled_at.items():
        if ft <= t and 0 <= cell[0] < g.height:
            img[cell] = COLORS["filled"]
    for inc in scenario.incidents:
        at = result.appeared_at.get(inc.oid)
        for cell in inc.cells:
            ft = result.filled_at.get(cell)
            if at is not None and at <= t and not (ft is not None and ft <= t):
                img[cell] = COLORS["incident"]
    return img


def animate_runs(scenario, results: Dict[str, object], cfg, path: Optional[str] = None, show: bool = False,
                 fps: int = 6, max_frames: int = 240, title: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Rectangle

    if not show:
        plt.switch_backend("Agg")
    names = list(results)
    g = scenario.grid
    horizon = max(max(len(t) for t in r.trajectory.values()) for r in results.values()) - 1
    step = max(1, math.ceil(horizon / max_frames))
    ticks = list(range(0, horizon + 1, step)) + [horizon] * 8
    costs = {n: cumulative_cost(r, cfg) for n, r in results.items()}
    goals = {n: goal_timelines(r, scenario) for n, r in results.items()}
    caps = {n: short_events(r, scenario) for n, r in results.items()}
    cmap = plt.get_cmap("tab10")

    panel_w = 5.6
    panel_h = panel_w * g.height / g.width
    fig = plt.figure(figsize=(panel_w * len(names), panel_h + 1.9))
    gs = fig.add_gridspec(2, len(names), height_ratios=[panel_h, 1.5], hspace=0.08)
    grid_axes = [fig.add_subplot(gs[0, k]) for k in range(len(names))]
    text_axes = [fig.add_subplot(gs[1, k]) for k in range(len(names))]
    fig.suptitle(title or f"{scenario.name}: {len(scenario.starts)} robot(s), same tasks for every arm", fontsize=11)
    fig.text(0.5, 0.005, "red P = pit   green = filled   orange = depot (kits)   purple = obstruction   "
                         "circle = robot (black ring + square = carrying a kit)   star = its goal",
             ha="center", fontsize=8)

    def draw(frame_idx: int) -> None:
        t = ticks[frame_idx]
        for k, name in enumerate(names):
            res, ax, tx = results[name], grid_axes[k], text_axes[k]
            ax.clear()
            tx.clear()
            ax.imshow(_background(scenario, res, t), interpolation="nearest")
            for r in range(g.height):
                for c in range(g.width):
                    if g.get(r, c) == CellType.PIT and not (res.filled_at.get((r, c), 1 << 30) <= t):
                        ax.text(c, r, "P", ha="center", va="center", fontsize=8, color="white", weight="bold")
                    elif g.get(r, c) == CellType.SANDBAG:
                        ax.text(c, r, "kit", ha="center", va="center", fontsize=6)
            for i, traj in sorted(res.trajectory.items()):
                color = cmap(i % 10)
                goal = goals[name][i][min(t, len(goals[name][i]) - 1)] if t < len(traj) else None
                if goal is not None:
                    ax.plot(goal[1], goal[0], marker="*", color=color, markersize=9, markeredgecolor="k",
                            markeredgewidth=0.4)
                if t >= len(traj):
                    continue
                trail = traj[max(0, t - TRAIL):t + 1]
                if len(trail) > 1:
                    ax.plot([p[1] for p in trail], [p[0] for p in trail], color=color, alpha=0.45, lw=2)
                r, c = traj[t]
                carrying = res.carry_trace.get(i, [False] * len(traj))[t]
                ax.add_patch(Circle((c, r), 0.38, facecolor=color, edgecolor="k" if carrying else "white",
                                    linewidth=2.2 if carrying else 0.8, zorder=5))
                ax.text(c, r, str(i), ha="center", va="center", fontsize=7, color="white", zorder=6, weight="bold")
                if carrying:
                    ax.add_patch(Rectangle((c + 0.12, r - 0.52), 0.34, 0.34, facecolor=COLORS["depot"],
                                           edgecolor="k", linewidth=0.8, zorder=7))
            ax.set_xlim(-0.5, g.width - 0.5)
            ax.set_ylim(g.height - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            so_far = costs[name][min(t, len(costs[name]) - 1)]
            ax.set_title(f"{name}    tick {t}    cost so far {so_far:.0f}", fontsize=10)
            tx.axis("off")
            shown = [text for tick, text in caps[name] if tick <= t][-4:]
            tx.text(0.0, 1.0, "\n".join(f"- {s}" for s in shown) if shown else "- (no edits yet)", va="top",
                    ha="left", fontsize=8, family="monospace", transform=tx.transAxes, wrap=True)
            if t >= horizon:
                tx.text(0.0, 0.0, f"final cost J = {costs[name][-1]:.0f}", va="bottom", ha="left", fontsize=9,
                        weight="bold", transform=tx.transAxes)

    anim = FuncAnimation(fig, draw, frames=len(ticks), interval=1000 / fps, repeat=False)
    if show:
        plt.show()
        return None
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path
