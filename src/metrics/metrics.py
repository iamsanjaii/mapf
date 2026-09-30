"""
src/metrics/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~
MetricsCollector — tracks M1–M8 from the PRD.

M1  Success rate
M2  Total path cost
M3  Makespan
M4  Conflict count
M5  Replanning count
M6  Computation time
M7  Environment modification cost
M8  Total energy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunMetrics:
    """
    Stores metrics for a single experiment run.

    All fields have sensible defaults so partial runs can still be recorded.
    """
    # Identification
    experiment_id:     int = 0
    algorithm:         str = ""
    grid_size:         int = 0
    obstacle_density:  float = 0.0
    num_robots:        int = 0
    seed:              int = 0

    # M1 — Success
    total_robots:      int = 0
    successful_robots: int = 0

    # M2 — Path cost
    total_path_cost:   float = 0.0

    # M3 — Makespan
    makespan:          int = 0

    # M4 — Conflicts
    conflict_count:    int = 0

    # M5 — Replanning
    replan_count:      int = 0

    # M6 — Computation time
    runtime_ms:        float = 0.0

    # M7 — Environment modification (MAPF-RO)
    pits_filled:       int = 0
    sandbag_cost:      float = 0.0
    removal_cost:      float = 0.0

    # M8 — Total energy
    total_energy:      float = 0.0

    # Per-robot details (optional)
    robot_costs:       Dict[int, float] = field(default_factory=dict)
    robot_lengths:     Dict[int, int]   = field(default_factory=dict)

    # ------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        if self.total_robots == 0:
            return 0.0
        return self.successful_robots / self.total_robots

    @property
    def avg_path_length(self) -> float:
        if not self.robot_lengths:
            return 0.0
        return sum(self.robot_lengths.values()) / len(self.robot_lengths)

    def to_dict(self) -> dict:
        return {
            "experiment_id":    self.experiment_id,
            "algorithm":        self.algorithm,
            "grid_size":        self.grid_size,
            "obstacle_density": self.obstacle_density,
            "num_robots":       self.num_robots,
            "seed":             self.seed,
            "success_rate":     round(self.success_rate, 4),
            "total_path_cost":  self.total_path_cost,
            "makespan":         self.makespan,
            "conflict_count":   self.conflict_count,
            "replan_count":     self.replan_count,
            "runtime_ms":       round(self.runtime_ms, 2),
            "pits_filled":      self.pits_filled,
            "sandbag_cost":     self.sandbag_cost,
            "removal_cost":     self.removal_cost,
            "total_energy":     self.total_energy,
            "avg_path_length":  round(self.avg_path_length, 2),
        }


class MetricsCollector:
    """Accumulates RunMetrics across multiple experiment runs."""

    def __init__(self) -> None:
        self._runs: List[RunMetrics] = []
        self._next_id: int = 0

    def new_run(self, **kwargs) -> RunMetrics:
        """Create and register a new RunMetrics."""
        run = RunMetrics(experiment_id=self._next_id, **kwargs)
        self._next_id += 1
        self._runs.append(run)
        return run

    def record(self, run: RunMetrics) -> None:
        if run not in self._runs:
            self._runs.append(run)

    def to_records(self) -> List[dict]:
        return [r.to_dict() for r in self._runs]

    def __len__(self) -> int:
        return len(self._runs)
