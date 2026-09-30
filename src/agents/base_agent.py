"""
src/agents/base_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all specialised agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Every agent in the agentic MAPF architecture must implement
    perceive → decide → act.

    Agents are deterministic Python components (no LLM required).
    They exchange information through a shared `state` dictionary
    managed by the CoordinatorAgent.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read relevant parts of the shared state.
        Return the perception dict.
        """

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason over the perception.
        Return a decision dict.
        """

    @abstractmethod
    def act(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decision, updating state.
        Return any relevant output / side-effects.
        """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: perceive → decide → act in one call."""
        perception = self.perceive(state)
        decision = self.decide(perception)
        return self.act(decision, state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
