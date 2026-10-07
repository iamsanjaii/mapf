"""SimConfig: every simulator parameter and its default, validated on construction."""
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.doi.kinds import KINDS

_LIGHTEST = min(k.weight for k in KINDS.values() if k.slots)      # lightest kind a robot can carry

POLICIES = frozenset({"never", "myopic", "eager", "rof", "rof_local", "rof_f", "central", "hindsight", "free",
                      "rof_r", "rof_p", "rof_a"})
FORECASTERS = ("numeric", "keyword", "oracle", "inverted")
AGENT_MODES = ("tools", "single")


@dataclass
class SimConfig:
    seed: int = 0
    scenario: str = "single_block"
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    n_robots: int = 12
    tasks_per_robot: int = 20
    kappa: float = 4.0          # cost of one push step is kappa * the obstacle kind's weight
    fee: float = 1.0            # cost of each push run
    kappa_c: float = 2.0        # cost of one loaded step is kappa_c * the carried kind's weight
    pick_fee: float = 1.0       # cost of picking an obstacle up
    drop_fee: float = 1.0       # cost of dropping it into a slot or pit
    push_max: int = 6           # longest straight push a robot will plan
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
    bundle_max: int = 1
    lam: float = 0.5
    forecaster: str = "numeric"          # rof_a: numeric | keyword | oracle | inverted | llm:<model key>
    agent_max_forecasts: int = 200       # rof_a: forecasts one run may request; later requests get no answer
    agent_mode: str = "tools"            # rof_a with an llm forecaster: tools | single
    agent_cache: str = "experiments/results/doi/agent_cache"     # where model calls are stored and replayed from
    agent_live: bool = False             # allow a miss to call the model (spends money); off: a miss is an error
    intake: str = "none"
    intake_cache: str = "experiments/results/doi/intake_cache"
    tick_seconds: float = 0.5
    p_wrong_class: float = 0.0
    max_ticks: int = 20000
    stall_ticks: int = 100
    debug_checks: bool = False
    horizon: Optional[int] = None
    record_epoch: Optional[int] = None
    delta_gossip: bool = False
    full_sync_period: int = 20
    plan_window: Optional[int] = None
    map_path: Optional[str] = None

    def __post_init__(self) -> None:
        checks = [
            (self.latency >= 1, "latency must be >= 1"),
            (0 <= self.loss <= 1, "loss must be in [0, 1]"),
            (0 <= self.loss_traffic <= 1, "loss_traffic must be in [0, 1]"),
            (self.r_comm >= 0, "r_comm must be >= 0"),
            (self.r_traffic >= 0, "r_traffic must be >= 0"),
            (self.n_robots >= 1, "n_robots must be >= 1"),
            (self.tasks_per_robot >= 1, "tasks_per_robot must be >= 1"),
            (self.push_max >= 1, "push_max must be >= 1"),
            (self.kappa_c * _LIGHTEST >= 1.0, "kappa_c too small: a loaded step must cost at least an empty one"),
            (self.pick_fee >= 0 and self.drop_fee >= 0, "pick_fee and drop_fee must be >= 0"),
            (self.policy in POLICIES, f"unknown policy {self.policy!r}"),
            (self.bundle_max in (1, 2), "bundle_max must be 1 or 2"),
            (0 < self.lam <= 1, "lam must be in (0, 1]"),
            (self.forecaster in FORECASTERS or self.forecaster.startswith("llm:"),
             f"bad forecaster {self.forecaster!r}"),
            (self.agent_max_forecasts >= 0, "agent_max_forecasts must be >= 0"),
            (self.agent_mode in AGENT_MODES, f"bad agent_mode {self.agent_mode!r}: choose tools or single"),
            (self.intake in ("none", "oracle") or self.intake.startswith("llm:"),
             f"bad intake {self.intake!r}"),
            (0 <= self.p_wrong_class <= 1, "p_wrong_class must be in [0, 1]"),
            (self.horizon is None or self.horizon >= 1, "horizon must be >= 1 when set"),
            (self.record_epoch is None or self.record_epoch >= 1, "record_epoch must be >= 1 when set"),
            (self.full_sync_period >= 1, "full_sync_period must be >= 1"),
            (self.plan_window is None or self.plan_window >= 2, "plan_window must be >= 2 when set"),
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
