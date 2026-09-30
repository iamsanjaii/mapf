"""
src/agents/conflict_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conflict Detection Agent.

Responsibilities:
- Inspect all robot paths
- Detect and classify conflicts
- Rank conflicts by severity
- Report to the CoordinatorAgent
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.mapf.conflict import Conflict, ConflictDetector


class ConflictAgent(BaseAgent):
    """
    Detects, classifies, and ranks conflicts.

    State keys consumed
    -------------------
    - "paths" : dict[robot_id → List[Pos] | None]

    State keys produced
    -------------------
    - "conflicts"       : List[Conflict] (sorted by severity desc)
    - "conflict_counts" : dict[type_name → int]
    - "has_conflicts"   : bool
    """

    def __init__(self) -> None:
        super().__init__(name="ConflictAgent")

    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"paths": state.get("paths", {})}

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        valid_paths = {rid: p for rid, p in perception["paths"].items() if p}
        conflicts = ConflictDetector.detect_all(valid_paths)
        conflicts.sort(key=ConflictDetector.severity_score, reverse=True)
        counts = ConflictDetector.count_by_type(conflicts)
        return {
            "conflicts": conflicts,
            "conflict_counts": counts,
            "has_conflicts": len(conflicts) > 0,
        }

    def act(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        state["conflicts"] = decision["conflicts"]
        state["conflict_counts"] = decision["conflict_counts"]
        state["has_conflicts"] = decision["has_conflicts"]
        return decision
