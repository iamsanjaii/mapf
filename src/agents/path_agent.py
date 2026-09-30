"""
src/agents/path_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Path Planning Agent.

Responsibilities:
- Generate candidate paths for each robot using A*
- Detect unreachable goals
- Propose alternative paths when the primary path is blocked
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.planning.astar import AStarPlanner
from src.planning.heuristics import get_heuristic


Pos = Tuple[int, int]


class PathAgent(BaseAgent):
    """
    Generates and maintains paths for all robots.

    State keys consumed
    -------------------
    - "grid"      : Grid
    - "robots"    : List[Robot]
    - "heuristic" : str (optional, default "manhattan")
    - "forbidden" : dict[robot_id → set[Pos]] (optional)

    State keys produced
    -------------------
    - "paths"     : dict[robot_id → List[Pos] | None]
    - "plan_costs": dict[robot_id → float]
    - "unreachable": list[robot_id]
    """

    def __init__(self, heuristic: str = "manhattan", move_cost: float = 1.0) -> None:
        super().__init__(name="PathAgent")
        self._astar = AStarPlanner(move_cost=move_cost)
        self._heuristic = get_heuristic(heuristic)

    # ------------------------------------------------------------------

    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "grid":    state["grid"],
            "robots":  state["robots"],
            "heuristic": state.get("heuristic", "manhattan"),
            "forbidden": state.get("forbidden", {}),
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        heuristic_name = perception["heuristic"]
        h = get_heuristic(heuristic_name)
        return {"heuristic_fn": h, **perception}

    def act(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        grid = decision["grid"]
        robots = decision["robots"]
        h = decision["heuristic_fn"]
        forbidden_map = decision["forbidden"]

        paths: Dict[int, Optional[List[Pos]]] = {}
        costs: Dict[int, float] = {}
        unreachable: List[int] = []

        for robot in robots:
            forbidden = forbidden_map.get(robot.id, set())
            result = self._astar.plan(grid, robot.current_pos, robot.goal, h, forbidden)

            paths[robot.id] = result.path
            costs[robot.id] = result.cost

            if not result.success:
                unreachable.append(robot.id)

        state["paths"] = paths
        state["plan_costs"] = costs
        state["unreachable"] = unreachable

        return {"paths": paths, "plan_costs": costs, "unreachable": unreachable}
