"""SimConfig: every simulator parameter and its default, validated on construction."""
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

POLICIES = frozenset({
    "never", "myopic", "eager", "rof", "rof_pit", "rof_local", "rof_w", "rof_x",
    "central", "hindsight", "free",
})


@dataclass
class SimConfig:
    seed: int = 0
    scenario: str = "single_pit"
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    n_robots: int = 12
    tasks_per_robot: int = 20
    kappa: float = 4.0
    fee: float = 1.0
    unreachable_cost: Optional[float] = None
    r_sense: int = 2
    r_comm: float = 8.0
    latency: int = 1
    loss: float = 0.0
    r_traffic: float = 8.0
    loss_traffic: float = 0.0
    gossip_period: int = 1
    intent_window: int = 6
    policy: str = "rof"
    theta: float = 1.0
    window: Optional[int] = None
    intake: str = "none"
    intake_cache: str = "experiments/results/doi/intake_cache"
    tick_seconds: float = 0.5
    p_wrong_class: float = 0.0
    gate: bool = False
    sup_latency_median: float = 30
    sup_latency_sigma: float = 0.5
    p_catch: float = 0.9
    approval_timeout: int = 60
    approval_conf: float = 0.7
    claim: bool = True
    lease_ticks: int = 8
    stagger_cap: int = 25
    max_ticks: int = 20000
    stall_ticks: int = 100
    debug_checks: bool = False

    def __post_init__(self) -> None:
        checks = [
            (self.latency >= 1, "latency must be >= 1"),
            (0 <= self.loss <= 1, "loss must be in [0, 1]"),
            (0 <= self.loss_traffic <= 1, "loss_traffic must be in [0, 1]"),
            (self.r_comm >= 0, "r_comm must be >= 0"),
            (self.r_traffic >= 0, "r_traffic must be >= 0"),
            (self.n_robots >= 1, "n_robots must be >= 1"),
            (self.tasks_per_robot >= 1, "tasks_per_robot must be >= 1"),
            (self.policy in POLICIES, f"unknown policy {self.policy!r}"),
            (self.window is None or self.window > 0, "window must be > 0 when set"),
            (self.intake in ("none", "oracle") or self.intake.startswith("llm:"),
             f"bad intake {self.intake!r}"),
            (0 <= self.p_wrong_class <= 1, "p_wrong_class must be in [0, 1]"),
            (0 <= self.p_catch <= 1, "p_catch must be in [0, 1]"),
            (self.sup_latency_median >= 1, "sup_latency_median must be >= 1"),
            (self.approval_timeout >= 1, "approval_timeout must be >= 1"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kw: Any) -> "SimConfig":
        return dataclasses.replace(self, **kw)

    def unreachable_cost_for(self, h: int, w: int) -> float:
        if self.unreachable_cost is not None:
            return float(self.unreachable_cost)
        return float(4 * (h + w))
