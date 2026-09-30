"""
src/metrics/experiments.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
ExperimentManager — runs the full experiment matrix automatically
and saves results as CSV.

Supports all experiment configurations from the PRD:
    Experiment A — single-robot heuristic comparison
    Experiment B — multi-robot scaling
    Experiment C — obstacle density sweep
    Experiment D — grid scaling
    Experiment E — MAPF-RO removal vs detour
    Experiment F — agentic coordination comparison
"""

from __future__ import annotations

import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.environment.generator import EnvironmentGenerator
from src.mapf.conflict import ConflictDetector
from src.mapf.coordinator import RuleBasedCoordinator
from src.mapf.prioritized import PrioritizedPlanner
from src.mapf_ro.pit import PitManager
from src.mapf_ro.replanning import MAPFROPlanner
from src.mapf_ro.sandbag import SandbagManager
from src.metrics.metrics import MetricsCollector, RunMetrics
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic
from src.robots.manager import RobotManager
from src.robots.robot import RobotStatus


class ExperimentManager:
    """
    Runs multi-dimensional experiment matrices and produces DataFrames / CSVs.

    Parameters
    ----------
    output_dir : str
        Directory where CSV files and plots are written.
    """

    ALGORITHMS = ["independent", "prioritized", "agentic"]

    def __init__(self, output_dir: str = "experiments/results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.collector = MetricsCollector()

    # ------------------------------------------------------------------
    # Experiment A — Single-robot heuristics
    # ------------------------------------------------------------------

    def experiment_a_heuristics(
        self,
        grid_sizes: List[int] = (20, 30, 40),
        obstacle_densities: List[float] = (0.10, 0.20, 0.30),
        heuristics: List[str] = ("manhattan", "euclidean", "chebyshev"),
        runs_per_config: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Experiment A: single-robot, compare 3 heuristics."""
        records = []

        for gs, density, heuristic in product(grid_sizes, obstacle_densities, heuristics):
            for run_idx in range(runs_per_config):
                run_seed = seed + run_idx
                gen = EnvironmentGenerator(
                    width=gs, height=gs,
                    obstacle_density=density,
                    seed=run_seed,
                )
                try:
                    grid, starts, goals, _ = gen.generate(num_robots=1)
                except ValueError:
                    continue

                h = get_heuristic(heuristic)
                planner = AStarPlanner()
                result = planner.plan(grid, starts[0], goals[0], h)

                records.append({
                    "grid_size": gs,
                    "obstacle_density": density,
                    "heuristic": heuristic,
                    "seed": run_seed,
                    "success": result.success,
                    "path_cost": result.cost,
                    "nodes_expanded": result.nodes_expanded,
                    "runtime_ms": round(result.runtime_ms, 4),
                    "path_length": len(result.path) if result.path else 0,
                })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, "exp_a_heuristics.csv"), index=False)
        print(f"[Exp A] {len(df)} rows → exp_a_heuristics.csv")
        return df

    # ------------------------------------------------------------------
    # Experiment B/C/D — Multi-robot scaling
    # ------------------------------------------------------------------

    def experiment_bcd_multiagent(
        self,
        grid_sizes: List[int] = (20, 30, 40),
        obstacle_densities: List[float] = (0.10, 0.20, 0.30),
        robot_counts: List[int] = (2, 5, 10, 15),
        algorithms: List[str] = ("independent", "prioritized", "agentic"),
        runs_per_config: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Experiments B/C/D: multi-robot, algorithm comparison, scaling."""
        records = []

        for gs, density, n_robots, algo in product(
            grid_sizes, obstacle_densities, robot_counts, algorithms
        ):
            for run_idx in range(runs_per_config):
                run_seed = seed + run_idx
                try:
                    rec = self._run_multiagent(
                        grid_size=gs,
                        density=density,
                        n_robots=n_robots,
                        algorithm=algo,
                        seed=run_seed,
                    )
                    records.append(rec)
                except Exception as e:
                    records.append({
                        "grid_size": gs,
                        "obstacle_density": density,
                        "num_robots": n_robots,
                        "algorithm": algo,
                        "seed": run_seed,
                        "error": str(e),
                    })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, "exp_bcd_multiagent.csv"), index=False)
        print(f"[Exp B/C/D] {len(df)} rows → exp_bcd_multiagent.csv")
        return df

    # ------------------------------------------------------------------
    # Experiment E — MAPF-RO
    # ------------------------------------------------------------------

    def experiment_e_mapfro(
        self,
        grid_sizes: List[int] = (20, 30),
        obstacle_densities: List[float] = (0.10, 0.20),
        pit_densities: List[float] = (0.05, 0.10),
        robot_counts: List[int] = (5, 10),
        runs_per_config: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Experiment E: MAPF-RO — removal vs no removal."""
        records = []

        for gs, density, pit_d, n_robots in product(
            grid_sizes, obstacle_densities, pit_densities, robot_counts
        ):
            for run_idx in range(runs_per_config):
                run_seed = seed + run_idx
                for enable_removal in (False, True):
                    try:
                        rec = self._run_mapfro(
                            grid_size=gs,
                            density=density,
                            pit_density=pit_d,
                            n_robots=n_robots,
                            enable_removal=enable_removal,
                            seed=run_seed,
                        )
                        records.append(rec)
                    except Exception as e:
                        records.append({
                            "grid_size": gs, "obstacle_density": density,
                            "pit_density": pit_d, "num_robots": n_robots,
                            "enable_removal": enable_removal, "seed": run_seed,
                            "error": str(e),
                        })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.output_dir, "exp_e_mapfro.csv"), index=False)
        print(f"[Exp E] {len(df)} rows → exp_e_mapfro.csv")
        return df

    # ------------------------------------------------------------------
    # Internal runners
    # ------------------------------------------------------------------

    def _run_multiagent(
        self,
        grid_size: int,
        density: float,
        n_robots: int,
        algorithm: str,
        seed: int,
    ) -> dict:
        gen = EnvironmentGenerator(
            width=grid_size, height=grid_size,
            obstacle_density=density,
            seed=seed,
        )
        grid, starts, goals, _ = gen.generate(num_robots=n_robots)
        rm = RobotManager(starts, goals)

        t0 = time.perf_counter()
        paths: Dict[int, Any] = {}
        conflict_count = 0
        replan_count = 0

        if algorithm == "independent":
            planner = AStarPlanner()
            h = get_heuristic("manhattan")
            for robot in rm.robots:
                pr = planner.plan(grid, robot.start, robot.goal, h)
                paths[robot.id] = pr.path
                if pr.success:
                    robot.status = RobotStatus.PLANNING
                else:
                    robot.status = RobotStatus.STUCK

            valid = {rid: p for rid, p in paths.items() if p}
            conflicts = ConflictDetector.detect_all(valid)
            conflict_count = len(conflicts)

        elif algorithm == "prioritized":
            pp = PrioritizedPlanner()
            rm.assign_priorities_by_distance()
            pr_result = pp.plan(rm.robots, grid)
            paths = pr_result.paths
            valid = {rid: p for rid, p in paths.items() if p}
            conflicts = ConflictDetector.detect_all(valid)
            conflict_count = len(conflicts)

        elif algorithm == "agentic":
            from src.agents.coordinator_agent import CoordinatorAgent
            coord = CoordinatorAgent()
            ag_result = coord.coordinate(rm.robots, grid)
            paths = ag_result.paths
            conflict_count = ag_result.conflicts_after
            replan_count = ag_result.replan_count

        runtime_ms = (time.perf_counter() - t0) * 1000

        success_count = sum(1 for p in paths.values() if p)
        path_lengths = {rid: len(p) for rid, p in paths.items() if p}
        total_cost = sum(max(0, l - 1) for l in path_lengths.values())
        makespan = max(path_lengths.values()) - 1 if path_lengths else 0

        return {
            "grid_size": grid_size,
            "obstacle_density": density,
            "num_robots": n_robots,
            "algorithm": algorithm,
            "seed": seed,
            "success_rate": round(success_count / n_robots, 4),
            "total_cost": total_cost,
            "makespan": makespan,
            "conflict_count": conflict_count,
            "replan_count": replan_count,
            "runtime_ms": round(runtime_ms, 2),
        }

    def _run_mapfro(
        self,
        grid_size: int,
        density: float,
        pit_density: float,
        n_robots: int,
        enable_removal: bool,
        seed: int,
    ) -> dict:
        gen = EnvironmentGenerator(
            width=grid_size, height=grid_size,
            obstacle_density=density,
            pit_density=pit_density,
            sandbag_count=max(3, int(grid_size * grid_size * pit_density * 0.5)),
            seed=seed,
        )
        grid, starts, goals, registry = gen.generate(num_robots=n_robots)
        rm = RobotManager(starts, goals)

        pit_positions = grid.pit_positions()
        sandbag_positions = grid.sandbag_positions()
        pit_mgr = PitManager(pit_positions)
        sb_mgr = SandbagManager(sandbag_positions)

        planner = MAPFROPlanner(enable_removal=enable_removal)
        ro_result = planner.plan(rm.robots, grid, pit_mgr, sb_mgr)

        return {
            "grid_size": grid_size,
            "obstacle_density": density,
            "pit_density": pit_density,
            "num_robots": n_robots,
            "enable_removal": enable_removal,
            "seed": seed,
            "success_rate": round(ro_result.success_rate(), 4),
            "total_robot_cost": ro_result.total_robot_cost,
            "total_sandbag_cost": ro_result.total_sandbag_cost,
            "total_removal_cost": ro_result.total_removal_cost,
            "total_energy": ro_result.total_energy,
            "pits_filled": ro_result.pits_filled,
            "replan_iterations": ro_result.replan_iterations,
            "runtime_ms": round(ro_result.runtime_ms, 2),
        }
