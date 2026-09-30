"""
src/agents/cost_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Cost Evaluation Agent.

For each conflict, evaluates the cost of the three resolution options:
    WAIT    — robot idles for one or more steps
    DETOUR  — robot reroutes around the contested area
    REMOVE  — an obstacle is removed (MAPF-RO only)

Returns a ranked list of CostOption objects so the CoordinatorAgent
can choose the cheapest feasible action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.mapf.conflict import Conflict
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic


Pos = Tuple[int, int]


class ActionType(Enum):
    WAIT = auto()
    DETOUR = auto()
    REMOVE = auto()


@dataclass
class CostOption:
    action: ActionType
    robot_id: int
    estimated_cost: float
    extra_steps: int = 0
    target_obstacle: Optional[Pos] = None   # for REMOVE
    feasible: bool = True

    def __lt__(self, other: "CostOption") -> bool:
        return self.estimated_cost < other.estimated_cost


class CostAgent(BaseAgent):
    """
    Evaluates resolution costs for each conflict.

    State keys consumed
    -------------------
    - "conflicts"    : List[Conflict]
    - "paths"        : dict[robot_id → Path]
    - "grid"         : Grid
    - "robots"       : List[Robot]
    - "cost_config"  : dict with keys move, waiting, sandbag_move, obstacle_removal

    State keys produced
    -------------------
    - "cost_options" : dict[conflict_index → List[CostOption]]
    """

    def __init__(
        self,
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
        removal_cost: float = 10.0,
        heuristic: str = "manhattan",
    ) -> None:
        super().__init__(name="CostAgent")
        self.move_cost = move_cost
        self.wait_cost = wait_cost
        self.removal_cost = removal_cost
        self._astar = AStarPlanner(move_cost=move_cost)
        self._h = get_heuristic(heuristic)

    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = state.get("cost_config", {})
        return {
            "conflicts": state.get("conflicts", []),
            "paths":     state.get("paths", {}),
            "grid":      state["grid"],
            "robots":    state["robots"],
            "move_cost":    cfg.get("move", self.move_cost),
            "wait_cost":    cfg.get("waiting", self.wait_cost),
            "removal_cost": cfg.get("obstacle_removal", self.removal_cost),
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        conflicts = perception["conflicts"]
        paths = perception["paths"]
        grid = perception["grid"]
        robots_by_id = {r.id: r for r in perception["robots"]}
        wait_cost = perception["wait_cost"]
        removal_cost = perception["removal_cost"]

        cost_options: Dict[int, List[CostOption]] = {}

        for idx, conflict in enumerate(conflicts):
            options: List[CostOption] = []
            rid_a, rid_b = conflict.robots
            robot_a = robots_by_id.get(rid_a)
            robot_b = robots_by_id.get(rid_b)
            if robot_a is None or robot_b is None:
                cost_options[idx] = []
                continue

            # Lower-priority robot is the yielding candidate
            yielding = robot_a if robot_a.priority > robot_b.priority else robot_b
            priority = robot_b if robot_a.priority > robot_b.priority else robot_a

            # Option 1: WAIT
            wait_steps = 1
            wait_total = wait_cost * wait_steps
            options.append(CostOption(
                action=ActionType.WAIT,
                robot_id=yielding.id,
                estimated_cost=wait_total,
                extra_steps=wait_steps,
            ))

            # Option 2: DETOUR
            priority_path = paths.get(priority.id) or []
            forbidden = set(priority_path)
            detour_result = self._astar.plan(
                grid, yielding.current_pos, yielding.goal, self._h, forbidden
            )
            original_cost = perception["move_cost"] * max(
                0, len(paths.get(yielding.id) or []) - 1
            )
            if detour_result.path:
                extra = detour_result.cost - original_cost
                options.append(CostOption(
                    action=ActionType.DETOUR,
                    robot_id=yielding.id,
                    estimated_cost=max(0.0, extra),
                    extra_steps=max(0, len(detour_result.path) - len(paths.get(yielding.id) or [])),
                    feasible=True,
                ))
            else:
                options.append(CostOption(
                    action=ActionType.DETOUR,
                    robot_id=yielding.id,
                    estimated_cost=float("inf"),
                    feasible=False,
                ))

            # Option 3: REMOVE (only if there is a removable obstacle)
            # In base MAPF, removal cost is always infinity
            options.append(CostOption(
                action=ActionType.REMOVE,
                robot_id=yielding.id,
                estimated_cost=removal_cost,
                feasible=False,   # set True by MAPF-RO coordinator
            ))

            options.sort()
            cost_options[idx] = options

        return {"cost_options": cost_options}

    def act(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        state["cost_options"] = decision["cost_options"]
        return decision
