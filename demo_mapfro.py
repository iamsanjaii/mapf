"""
demo_mapfro.py
~~~~~~~~~~~~~~
Live demonstration of MAPF-RO: Pits filled by Sandbags.

SCENARIO  (10 wide × 10 tall)
──────────────────────────────
• A horizontal middle corridor (row 5) connects left to right.
• Two pits at (5,4) and (5,5) block the corridor.
• Two sandbags sit next to the pits at (5,3) and (5,6).
• The grid is open on the top (row 0) and bottom (row 9), giving
  robots a long detour if they skip filling.

WITHOUT filling  → robot detours via row 0: cost ≈ 19 steps
WITH    filling  → robot crosses straight: cost = 9 steps
Filling cost (sandbag 1 step away, fill_cost=0.5): ≈ 1.5 per pit

Result: MAPF-RO fills both pits  (huge net benefit).

Run:
    source venv/bin/activate
    python demo_mapfro.py          # static before/after window
    python demo_mapfro.py --live   # + live animation
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("MacOSX")          # macOS interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from src.environment.grid import Grid, CellType
from src.mapf_ro.pit import PitManager
from src.mapf_ro.sandbag import SandbagManager
from src.mapf_ro.replanning import MAPFROPlanner
from src.robots.robot import Robot
from src.visualization.renderer import COLORS, ROBOT_COLORS

# ── Colours ─────────────────────────────────────────────────────────────────

FILLED_PIT_COLOR = "#4CAF50"      # green = pit filled successfully

# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_panel(ax, grid: Grid,
                starts=None, goals=None, paths=None,
                filled_pits=None, pit_positions=None,
                sandbag_positions=None, title=""):
    """Render one grid panel onto *ax*."""
    filled_pits       = set(filled_pits       or [])
    pit_positions     = set(pit_positions     or [])
    sandbag_positions = set(sandbag_positions or [])

    H, W = grid.height, grid.width
    ax.clear()
    ax.set_facecolor("#16213E")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=9, pad=6, loc="center")

    for r in range(H):
        for c in range(W):
            pos = (r, c)
            ct  = CellType(grid.array[r, c])

            if pos in filled_pits:
                color = FILLED_PIT_COLOR
            else:
                color = COLORS.get(ct, "#F5F5F5")

            rect = mpatches.FancyBboxPatch(
                (c, H - 1 - r), 1, 1,
                boxstyle="round,pad=0.05",
                linewidth=0.4,
                edgecolor="#0F3460",
                facecolor=color,
                zorder=1,
            )
            ax.add_patch(rect)

    # Paths
    if paths:
        for rid, path in paths.items():
            if not path:
                continue
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            xs = [p[1] + 0.5 for p in path]
            ys = [H - p[0] - 0.5 for p in path]
            ax.plot(xs, ys, "-", color=color, linewidth=2.5,
                    alpha=0.85, zorder=3)

    # Starts
    for i, pos in enumerate(starts or []):
        _label(ax, pos, H, f"S{i}", ROBOT_COLORS[i % len(ROBOT_COLORS)], zorder=5)

    # Goals
    for i, pos in enumerate(goals or []):
        _label(ax, pos, H, f"G{i}", ROBOT_COLORS[i % len(ROBOT_COLORS)],
               edgecolor="white", zorder=5)

    # Annotations on filled pits
    for pos in filled_pits:
        r, c = pos
        ax.text(c + 0.5, H - r - 0.5, "✓",
                ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=6)


def _label(ax, pos, H, text, fc, edgecolor="none", zorder=4):
    r, c = pos
    ax.text(c + 0.5, H - r - 0.5, text,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec=edgecolor, lw=1.5),
            zorder=zorder)


def _add_legend(ax):
    elements = [
        mpatches.Patch(fc=COLORS[CellType.FREE],      label="Free"),
        mpatches.Patch(fc=COLORS[CellType.OBSTACLE],  label="Obstacle"),
        mpatches.Patch(fc=COLORS[CellType.PIT],       label="Pit  (impassable)"),
        mpatches.Patch(fc=COLORS[CellType.SANDBAG],   label="Sandbag"),
        mpatches.Patch(fc=FILLED_PIT_COLOR,           label="Pit  (filled ✓)"),
    ]
    ax.legend(handles=elements, loc="upper right",
              fontsize=7, framealpha=0.9,
              facecolor="#0D1B2A", labelcolor="white",
              edgecolor="#3A86FF")


# ── Scenario ─────────────────────────────────────────────────────────────────

def build_scenario():
    """
    10×10 grid.

    Top (row 0) and bottom (row 9) are entirely free — long detour routes.
    Middle corridor is row 5, columns 0-9.
    Interior rows 1-4 and 6-8 are all OBSTACLES (force detour).

    Pits:     (5,4) and (5,5)      — block the corridor
    Open 10×5 grid (no walls) with a single pit that blocks Robot 0's direct path.

    Grid layout (row, col):

        Row 0: . . . . . . . . . .
        Row 1: . . . . . . . . . .      ← Robot 1 travels here (adjacent to pit)
        Row 2: S . . SB [P] . . . . G  ← Robot 0, sandbag at (2,3), pit at (2,4)
        Row 3: . . . . . . . . . .
        Row 4: . . . . . . . . . .

    WITHOUT fill:
        Robot 0 detours via row 1: 9 steps  (direct would be 9 but pit blocks)
        Robot 1 travels row 1: 9 steps (path passes adjacent to pit → traffic +1)

    WITH fill:
        Sandbag (2,3) is 1 step from pit (2,4) → travel cost = 1.0
        fill_cost = 0.5  → total removal cost = 1.5
        Robot 0 savings = detour len - direct len (≥ 2 steps)
        Net benefit > 0 → fills the pit!
    """
    H, W = 5, 10
    grid = Grid(W, H)

    # Single pit blocking the direct path of Robot 0
    pit_positions = [(2, 5)]
    for pos in pit_positions:
        grid.set(pos[0], pos[1], CellType.PIT)

    # Sandbag 1 step to the left of the pit (passable by robots)
    sb_positions = [(2, 4)]
    for pos in sb_positions:
        grid.set(pos[0], pos[1], CellType.SANDBAG)

    starts = [(2, 0), (1, 0)]
    goals  = [(2, 9), (1, 9)]

    robots = [Robot(id=i, start=s, goal=g) for i, (s, g) in enumerate(zip(starts, goals))]
    for r in robots:
        r.current_pos = r.start

    # Managers
    pit_mgr = PitManager(pit_positions)
    sb_mgr  = SandbagManager(
        sandbag_positions=sb_positions,
        move_cost_per_step=1.0,
    )

    return grid, robots, starts, goals, pit_positions, sb_positions, pit_mgr, sb_mgr



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAPF-RO Pit-Fill Demo")
    parser.add_argument("--live", action="store_true",
                        help="Show live robot animation after pit-fill")
    parser.add_argument("--fps",  type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("  MAPF-RO Demo — Pits Filled by Sandbags")
    print("=" * 60)

    # ── Step 1: Plan WITHOUT removal ────────────────────────────────────────
    print("\n[Step 1]  WITHOUT pit removal  (detour baseline)...")
    (grid, robots, starts, goals,
     pit_pos, sb_pos, pit_mgr, sb_mgr) = build_scenario()

    planner_base = MAPFROPlanner(enable_removal=False,
                                 move_cost=1.0)
    result_base  = planner_base.plan(robots, grid, pit_mgr, sb_mgr)

    print(f"  Success: {result_base.success_count}/{len(robots)}")
    for rid, path in result_base.paths.items():
        cost = len(path) - 1 if path else "∞"
        print(f"  Robot {rid}: cost={cost}  "
              f"path_len={len(path) if path else 0}")
    base_energy = sum((len(p) - 1) for p in result_base.paths.values() if p)
    print(f"  Total energy (detour): {base_energy}")

    # ── Step 2: Plan WITH removal ────────────────────────────────────────────
    print("\n[Step 2]  WITH pit removal  (MAPF-RO)...")
    (grid2, robots2, starts2, goals2,
     pit_pos2, sb_pos2, pit_mgr2, sb_mgr2) = build_scenario()

    planner_ro = MAPFROPlanner(
        enable_removal=True,
        move_cost=1.0,
        fill_cost=0.5,           # cheap fill cost
        sandbag_move_cost=1.0,   # cheap sandbag movement
    )
    result_ro = planner_ro.plan(robots2, grid2, pit_mgr2, sb_mgr2)

    filled_pits = set(pit_mgr2.all_pits()) - set(pit_mgr2.unfilled_pits())

    # ── Update grid2 to reflect the post-fill state (for animation) ──────────
    # Mark filled pits as FREE (robots can now cross)
    for pit in filled_pits:
        grid2.set(pit[0], pit[1], CellType.FREE)
    # Remove deployed sandbags from original positions (they've moved to the pits)
    for sb in sb_mgr2.all_sandbags():
        if sb.in_pit:
            # Clear the original sandbag position from the grid
            for orig_pos in sb_pos2:
                if orig_pos != sb.position:   # position is now the pit
                    # find which original position this sandbag came from
                    pass
    # Simplest: clear ALL original sandbag positions (all were deployed)
    for pos in sb_pos2:
        if CellType(grid2.array[pos[0], pos[1]]) == CellType.SANDBAG:
            grid2.set(pos[0], pos[1], CellType.FREE)

    print(f"  Success: {result_ro.success_count}/{len(robots2)}")
    print(f"  Pits filled: {result_ro.pits_filled}  → {sorted(filled_pits)}")
    for rid, path in result_ro.paths.items():
        cost = len(path) - 1 if path else "∞"
        print(f"  Robot {rid}: cost={cost}  "
              f"path_len={len(path) if path else 0}")
    print(f"  Robot path energy:  {result_ro.total_robot_cost:.1f}")
    print(f"  Sandbag move cost:  {result_ro.total_sandbag_cost:.1f}")
    print(f"  Pit fill cost:      {result_ro.total_removal_cost:.1f}")
    print(f"  ─────────────────────────────")
    print(f"  Total energy (MAPF-RO): {result_ro.total_energy:.1f}")
    print(f"  Total energy (detour):  {base_energy:.1f}")
    saving = base_energy - result_ro.total_energy
    print(f"  Net saving:             {saving:.1f}  "
          f"({'✓ better' if saving > 0 else '✗ worse'})")

    # ── Step 3: Visualise before / after ────────────────────────────────────
    plt.close("all")
    fig = plt.figure(figsize=(14, 6.5), facecolor="#1A1A2E")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06)
    ax_b = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])

    # BEFORE
    _draw_panel(
        ax_b, grid,
        starts=starts, goals=goals,
        paths=result_base.paths,
        pit_positions=pit_pos,
        sandbag_positions=sb_pos,
        title=(f"BEFORE  —  pits block corridor\n"
               f"Robot 0 detours via row 0  (cost {base_energy})"),
    )

    # AFTER
    _draw_panel(
        ax_a, grid2,
        starts=starts2, goals=goals2,
        paths=result_ro.paths,
        filled_pits=filled_pits,
        sandbag_positions=sb_pos2,
        title=(f"AFTER  —  sandbags fill pits  ✓\n"
               f"Robot 0 crosses straight  (total energy {result_ro.total_energy:.1f})"),
    )
    _add_legend(ax_a)

    # "filled!" arrows
    for pit in filled_pits:
        r, c = pit
        H = grid2.height
        ax_a.annotate(
            "filled!", xy=(c + 0.5, H - r - 0.5),
            xytext=(c + 0.5, H - r + 1.4),
            fontsize=7, color=FILLED_PIT_COLOR, fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=FILLED_PIT_COLOR, lw=1.5),
        )

    fig.suptitle("MAPF-RO: Pit Filling by Sandbags",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/demo_mapfro_pitfill.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[Render] Saved → {out_path}")

    # ── Step 4: Live animation ───────────────────────────────────────────────
    if args.live:
        valid_paths = {rid: p for rid, p in result_ro.paths.items() if p}
        if valid_paths:
            plt.close("all")
            from src.visualization.animation import GridAnimator
            animator = GridAnimator(grid2, valid_paths, starts2, goals2, fps=args.fps)
            # Mark filled pits green so they stand out clearly
            animator.color_overrides = {pos: FILLED_PIT_COLOR for pos in filled_pits}
            print("\n[Animate] Live window  — green cells = filled pits (now passable)")
            animator.animate(
                save_path=None,
                title="MAPF-RO — Robots crossing filled pits  [green = filled]",
                live=True,
            )
        else:
            print("\n  No valid paths to animate.")
            plt.show()
    else:
        plt.show()

    print("\n[Done]")


if __name__ == "__main__":
    main()
