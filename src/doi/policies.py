"""Fill policies (the experimental arms), the shared per-run state, and make_policy."""
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional, Tuple

from src.doi.config import SimConfig

if TYPE_CHECKING:
    from src.doi.agent import RobotAgent, TaskPlanInfo
    from src.doi.scenarios import Scenario

Pos = Tuple[int, int]


@dataclass
class Shared:
    triggers: List[dict] = field(default_factory=list)
    engine: Any = None
    global_belief: Any = None
    agents: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class HaulProposal:
    pits: Tuple[Pos, ...]
    evidence: float
    buy: float
    buy_per_pit: Optional[Tuple[float, ...]] = None


class FillPolicy(ABC):
    name: str = "policy"
    needs_single_rents: bool = False
    uses_gossip: bool = True

    def prepare(self, scenario: "Scenario", cfg: SimConfig, shared: Shared) -> None:
        return None

    def on_task_planned(self, agent: "RobotAgent", info: "TaskPlanInfo", t: int) -> None:
        return None

    def propose(self, agent: "RobotAgent", t: int) -> Optional[HaulProposal]:
        return None

    def prefill_set(self, scenario: "Scenario", cfg: SimConfig) -> FrozenSet[Pos]:
        return frozenset()


class NeverFillPolicy(FillPolicy):
    name = "never"
    uses_gossip = False


def make_policy(cfg: SimConfig) -> FillPolicy:
    if cfg.policy == "never":
        return NeverFillPolicy()
    raise NotImplementedError(f"policy {cfg.policy!r} is added in Task 11")
