"""
src/agents/coordinator_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Coordinator Agent — orchestrates all sub-agents and makes final decisions.

Architecture:
                 CoordinatorAgent
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
 PathAgent        ConflictAgent       CostAgent
      │                 │                 │
      └─────────────────┼─────────────────┘
                        ▼
                  Decision / Negotiation
                        │
            ┌───────────┼───────────┐
            │           │           │
         WAIT        REROUTE     MODIFY
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.conflict_agent import ConflictAgent
from src.agents.cost_agent import CostAgent, ActionType, CostOption
from src.agents.path_agent import PathAgent
from src.mapf.conflict import ConflictDetector
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic
from src.robots.robot import Robot, RobotStatus


Pos = Tuple[int, int]
Path = List[Pos]


class AgenticMAPFResult:
    """Full result from one agentic coordination run."""

    def __init__(self) -> None:
        self.paths: Dict[int, Optional[Path]] = {}
        self.conflicts_before: int = 0
        self.conflicts_after: int = 0
        self.actions_taken: List[Dict[str, Any]] = []
        self.replan_count: int = 0
        self.runtime_ms: float = 0.0
        self.success_count: int = 0
        self.fail_count: int = 0
        self.total_cost: float = 0.0
        self.makespan: int = 0

    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total else 0.0


class CoordinatorAgent(BaseAgent):
    """
    Top-level coordinator that chains PathAgent → ConflictAgent → CostAgent
    and applies the best resolution action for each conflict.

    Parameters
    ----------
    heuristic : str
    move_cost : float
    wait_cost : float
    removal_cost : float
    max_rounds : int
        Maximum resolution rounds (prevents infinite loops).
    """

    def __init__(
        self,
        heuristic: str = "manhattan",
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
        removal_cost: float = 10.0,
        max_rounds: int = 15,
    ) -> None:
        super().__init__(name="CoordinatorAgent")
        self._path_agent = PathAgent(heuristic=heuristic, move_cost=move_cost)
        self._conflict_agent = ConflictAgent()
        self._cost_agent = CostAgent(
            move_cost=move_cost,
            wait_cost=wait_cost,
            removal_cost=removal_cost,
            heuristic=heuristic,
        )
        self._astar = AStarPlanner(move_cost=move_cost)
        self._h = get_heuristic(heuristic)
        self.max_rounds = max_rounds
        self.move_cost = move_cost
        self.wait_cost = wait_cost

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        return perception   # delegation to sub-agents

    def act(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return self.coordinate(
            robots=state["robots"],
            grid=state["grid"],
            heuristic=state.get("heuristic", "manhattan"),
        ).__dict__

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def coordinate(
        self,
        robots: List[Robot],
        grid: Any,
        heuristic: str = "manhattan",
    ) -> AgenticMAPFResult:
        t0 = time.perf_counter()
        result = AgenticMAPFResult()

        state: Dict[str, Any] = {
            "grid":     grid,
            "robots":   robots,
            "heuristic": heuristic,
            "forbidden": {},
            "cost_config": {
                "move": self.move_cost,
                "waiting": self.wait_cost,
                "obstacle_removal": 10.0,
            },
        }

        # Step 1: Initial path planning
        self._path_agent.run(state)

        result.paths = {rid: list(p) if p else None
                        for rid, p in state.get("paths", {}).items()}

        # Count initial conflicts
        valid = {rid: p for rid, p in result.paths.items() if p}
        result.conflicts_before = len(ConflictDetector.detect_all(valid))

        # Step 2: Iterative conflict resolution
        for _round in range(self.max_rounds):
            self._conflict_agent.run(state)

            if not state.get("has_conflicts", False):
                break

            self._cost_agent.run(state)

            conflicts = state.get("conflicts", [])
            cost_options = state.get("cost_options", {})

            if not conflicts:
                break

            # Resolve top conflict with cheapest feasible action
            top_conflict = conflicts[0]
            options = cost_options.get(0, [])

            feasible = [o for o in options if o.feasible]
            if not feasible:
                break

            best = feasible[0]
            self._apply_action(best, top_conflict, state, result)
            result.replan_count += 1

        # Final conflict check
        valid = {rid: p for rid, p in state.get("paths", {}).items() if p}
        result.conflicts_after = len(ConflictDetector.detect_all(valid))

        result.paths = state.get("paths", {})

        # Compute metrics
        for robot in robots:
            path = result.paths.get(robot.id)
            if path:
                result.success_count += 1
                robot.path = list(path)
                robot.status = RobotStatus.PLANNING
                result.total_cost += (len(path) - 1) * self.move_cost
            else:
                result.fail_count += 1
                robot.status = RobotStatus.STUCK

        lengths = [len(p) for p in result.paths.values() if p]
        result.makespan = max(lengths) - 1 if lengths else 0
        result.runtime_ms = (time.perf_counter() - t0) * 1000

        return result

    # ------------------------------------------------------------------
    # Action executor
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        option: CostOption,
        conflict: Any,
        state: Dict[str, Any],
        result: AgenticMAPFResult,
    ) -> None:
        paths = state.get("paths", {})
        robots = state.get("robots", [])
        grid = state["grid"]
        robot_map = {r.id: r for r in robots}

        if option.action == ActionType.WAIT:
            path = paths.get(option.robot_id)
            if path:
                paths[option.robot_id] = [path[0]] + path
            result.actions_taken.append({
                "action": "WAIT",
                "robot": option.robot_id,
                "cost": option.estimated_cost,
            })

        elif option.action == ActionType.DETOUR:
            robot = robot_map.get(option.robot_id)
            other_id = (
                conflict.robots[1]
                if conflict.robots[0] == option.robot_id
                else conflict.robots[0]
            )
            priority_path = paths.get(other_id) or []
            forbidden = set(priority_path)
            plan = self._astar.plan(
                grid, robot.current_pos, robot.goal, self._h, forbidden
            )
            if plan.path:
                paths[option.robot_id] = plan.path
            result.actions_taken.append({
                "action": "DETOUR",
                "robot": option.robot_id,
                "cost": option.estimated_cost,
            })

        elif option.action == ActionType.REMOVE:
            # MAPF-RO: handled by the replanning module
            result.actions_taken.append({
                "action": "REMOVE",
                "robot": option.robot_id,
                "obstacle": option.target_obstacle,
                "cost": option.estimated_cost,
            })

        state["paths"] = paths
