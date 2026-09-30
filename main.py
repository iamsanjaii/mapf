"""
main.py
~~~~~~~~
AMR-MAPF Agent Simulator — main entry point.

Provides a quick demonstration of the full pipeline:
  1. Generate a random environment
  2. Plan paths using all three algorithms
  3. Detect conflicts
  4. Render the grid + paths
  5. Save a GIF animation

Usage:
    python main.py
    python main.py --algo prioritized --robots 8 --seed 42
    python main.py --algo agentic --grid 30 --density 0.20
"""

from __future__ import annotations

import argparse
import os
import sys


def _ask(prompt: str, cast, default, validate=None):
    """Helper: prompt the user for a value with a default and optional validator."""
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
            value = cast(raw) if raw else default
            if validate and not validate(value):
                raise ValueError
            return value
        except (ValueError, TypeError):
            print(f"  ✗  Invalid input. Please try again.")


def _prompt_inputs(args):
    """Interactively ask the user for simulation parameters."""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║    AMR-MAPF Simulator — Configuration    ║")
    print("╚══════════════════════════════════════════╝")
    print("  Press Enter to keep the default value.\n")

    args.rows     = _ask("  Grid rows    (height)   ", int,   args.rows,
                         lambda v: v >= 5)
    args.cols     = _ask("  Grid columns (width)    ", int,   args.cols,
                         lambda v: v >= 5)
    args.robots   = _ask("  Number of agents        ", int,   args.robots,
                         lambda v: 1 <= v <= (args.rows * args.cols) // 4)
    args.density  = _ask("  Obstacle density 0–0.6  ", float, args.density,
                         lambda v: 0.0 <= v <= 0.6)
    args.pits     = _ask("  Pit density  0–0.3      ", float, args.pits,
                         lambda v: 0.0 <= v <= 0.3)
    args.sandbags = _ask("  Number of sandbags      ", int,   args.sandbags,
                         lambda v: v >= 0)

    print("\n  Algorithm choices: independent | prioritized | agentic")
    args.algo = _ask("  Algorithm               ", str,   args.algo,
                     lambda v: v in ("independent", "prioritized", "agentic"))

    live_raw = input("  Show live window? (y/n) [y]: ").strip().lower()
    args.live = live_raw != "n"
    
    verbose_raw = input("  Show A* calculations? (y/n) [n]: ").strip().lower()
    args.verbose = verbose_raw == "y"

    print()
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMR-MAPF Simulator Demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--algo",    default=None,
                        choices=["independent", "prioritized", "agentic"],
                        help="Planning algorithm")
    parser.add_argument("--rows",    type=int,   default=None, help="Grid rows (height)")
    parser.add_argument("--cols",    type=int,   default=None, help="Grid columns (width)")
    parser.add_argument("--robots",  type=int,   default=None, help="Number of robots (agents)")
    parser.add_argument("--density",  type=float, default=None, help="Obstacle density 0.0–0.6")
    parser.add_argument("--pits",     type=float, default=None, help="Pit density 0.0–0.3")
    parser.add_argument("--sandbags", type=int,   default=None, help="Number of sandbags")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--animate",  action="store_true", help="Save animation GIF")
    parser.add_argument("--live",     action="store_true", help="Show live simulation window")
    parser.add_argument("--verbose",  action="store_true", help="Print A* step-by-step logs")
    parser.add_argument("--output",   default="experiments/results")
    args = parser.parse_args()

    # If any of the key inputs are missing → interactive prompt
    needs_prompt = any(v is None for v in [args.rows, args.cols, args.robots, args.density, args.algo])
    if needs_prompt:
        args.rows     = args.rows     or 20
        args.cols     = args.cols     or 20
        args.robots   = args.robots   or 5
        args.density  = args.density  if args.density  is not None else 0.20
        args.pits     = args.pits     if args.pits     is not None else 0.0
        args.sandbags = args.sandbags if args.sandbags is not None else 0
        args.algo     = args.algo     or "agentic"
        args = _prompt_inputs(args)
    else:
        # Defaults when fully specified via CLI
        if args.pits     is None: args.pits     = 0.0
        if args.sandbags is None: args.sandbags = 0

    import matplotlib
    if not args.live:
        matplotlib.use("Agg")

    from src.environment.generator import EnvironmentGenerator
    from src.mapf.conflict import ConflictDetector
    from src.mapf.coordinator import RuleBasedCoordinator
    from src.mapf.prioritized import PrioritizedPlanner
    from src.planning.astar import AStarPlanner
    from src.planning.heuristics import get_heuristic
    from src.robots.manager import RobotManager
    from src.robots.robot import RobotStatus
    from src.visualization.renderer import GridRenderer

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"  AMR-MAPF Simulator  |  {args.algo.upper()}  |  {args.rows}×{args.cols}")
    print(f"  Robots: {args.robots}  |  Density: {args.density}  |  Seed: {args.seed}")
    print("=" * 60)

    # Generate environment
    gen = EnvironmentGenerator(
        width=args.cols, height=args.rows,
        obstacle_density=args.density,
        pit_density=args.pits,
        sandbag_count=args.sandbags,
        seed=args.seed,
    )
    print(f"  Pits: {int(args.rows * args.cols * args.pits)}  "
          f"Sandbags: {args.sandbags}")
    grid, starts, goals, registry = gen.generate(num_robots=args.robots)
    rm = RobotManager(starts, goals)

    print(f"\n[Env] Grid {args.rows}×{args.cols}  "
          f"Obstacles: {len(grid.obstacle_positions())}  "
          f"Free cells: {len(grid.free_positions())}")

    paths = {}

    # ── Independent A* ──────────────────────────────────────────────
    if args.algo == "independent":
        planner = AStarPlanner()
        h = get_heuristic("manhattan")
        for robot in rm.robots:
            print(f"\n{'─'*60}")
            print(f"  Planning for Robot-{robot.id}  "
                  f"start={robot.start}  goal={robot.goal}")
            result = planner.plan(
                grid, robot.start, robot.goal, h,
                verbose=getattr(args, 'verbose', False),
                robot_id=robot.id,
            )
            paths[robot.id] = result.path
            robot.status = RobotStatus.PLANNING if result.success else RobotStatus.STUCK
            status = "✓ PATH FOUND" if result.success else "✗ STUCK (no path)"
            print(f"  Robot-{robot.id}: {status}  "
                  f"cost={result.cost:.0f}  "
                  f"nodes={result.nodes_expanded}  "
                  f"time={result.runtime_ms:.2f}ms")

        valid = {rid: p for rid, p in paths.items() if p}
        conflicts = ConflictDetector.detect_all(valid)
        print(f"\n{'═'*60}")
        print(f"[Independent A*]  Paths found: {sum(1 for p in paths.values() if p)}/{args.robots}")
        print(f"                  Conflicts detected: {len(conflicts)}")

    # ── Prioritised MAPF ─────────────────────────────────────────────
    elif args.algo == "prioritized":
        rm.assign_priorities_by_distance()
        pp = PrioritizedPlanner()
        result = pp.plan(
            rm.robots, grid,
            verbose=getattr(args, 'verbose', False)
        )
        paths = result.paths

        valid = {rid: p for rid, p in paths.items() if p}
        conflicts = ConflictDetector.detect_all(valid)
        print(f"\n[Prioritized MAPF]  Success: {result.success_count}/{args.robots}")
        print(f"                    Makespan: {result.makespan}")
        print(f"                    Total cost: {result.total_cost:.1f}")
        print(f"                    Conflicts: {len(conflicts)}")
        print(f"                    Runtime: {result.runtime_ms:.1f} ms")

    # ── Agentic MAPF ─────────────────────────────────────────────────
    elif args.algo == "agentic":
        from src.agents.coordinator_agent import CoordinatorAgent
        coord = CoordinatorAgent()
        result = coord.coordinate(rm.robots, grid)
        paths = result.paths

        print(f"\n[Agentic MAPF]  Success: {result.success_count}/{args.robots}")
        print(f"               Makespan: {result.makespan}")
        print(f"               Total cost: {result.total_cost:.1f}")
        print(f"               Conflicts before: {result.conflicts_before}")
        print(f"               Conflicts after:  {result.conflicts_after}")
        print(f"               Replan count: {result.replan_count}")
        print(f"               Runtime: {result.runtime_ms:.1f} ms")

    # Render
    renderer = GridRenderer(cell_size=0.6)
    valid_paths = {rid: p for rid, p in paths.items() if p}
    valid_conflicts = ConflictDetector.detect_all(valid_paths)
    stuck_count = len(paths) - len(valid_paths)

    title = (f"{args.algo.upper()} — {args.rows}×{args.cols}  "
             f"Robots={args.robots}  Density={args.density}")

    if stuck_count > 0:
        print(f"\n  ⚠  {stuck_count} robot(s) could not find a path.")
        print(f"     Try reducing --density / --pits or using a larger grid.")

    # Close any lingering matplotlib figures
    import matplotlib.pyplot as _plt
    _plt.close("all")

    if not args.live:
        # Normal (non-live) mode: save static PNG
        save_path = os.path.join(args.output, f"demo_{args.algo}.png")
        renderer.render(
            grid=grid, starts=starts, goals=goals,
            paths=valid_paths, conflicts=valid_conflicts,
            title=title, save_path=save_path, show=False,
        )
        print(f"\n[Render] Grid saved → {save_path}")

        if args.animate and valid_paths:
            from src.visualization.animation import GridAnimator
            anim_path = os.path.join(args.output, f"demo_{args.algo}.gif")
            animator = GridAnimator(grid, valid_paths, starts, goals, fps=3)
            animator.animate(save_path=anim_path,
                             title=f"AMR-MAPF {args.algo.upper()}", live=False)
            print(f"[Animate] GIF saved → {anim_path}")

    else:
        # Live mode: animate if paths exist, else show static grid window
        if valid_paths:
            from src.visualization.animation import GridAnimator
            anim_path = (os.path.join(args.output, f"demo_{args.algo}.gif")
                         if args.animate else None)
            animator = GridAnimator(grid, valid_paths, starts, goals, fps=3)
            animator.animate(save_path=anim_path,
                             title=f"AMR-MAPF {args.algo.upper()}", live=True)
            if args.animate:
                print(f"[Animate] GIF saved → {anim_path}")
        else:
            print("\n  No valid paths — showing environment layout...")
            renderer.render(
                grid=grid, starts=starts, goals=goals,
                paths={}, conflicts=[],
                title=title + "  ⚠ NO PATHS FOUND",
                save_path=None, show=True,
            )

    print("\n[Done]")


if __name__ == "__main__":
    main()
