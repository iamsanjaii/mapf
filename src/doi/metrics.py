"""RunResult and the per-run cost accounting."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

Pos = Tuple[int, int]


@dataclass
class RunResult:
    cfg: Dict[str, Any]
    policy: str
    J: float
    J_censored: float
    delay: float
    throughput: float
    moves: int
    waits: int
    carried_steps: int
    fills: int
    fee_total: float
    unfinished_tasks: int
    stalled: bool
    ticks: int
    messages: Dict[str, int]
    traffic_messages: Dict[str, int]
    overrides: int
    fill_ticks: List[int] = field(default_factory=list)
    triggers: List[dict] = field(default_factory=list)
    edits: List[dict] = field(default_factory=list)
    plan_infos: List[Any] = field(default_factory=list)
    filled_at: Dict[Pos, int] = field(default_factory=dict)
    appeared_at: Dict[int, int] = field(default_factory=dict)
    wasted_haul_cost: float = 0.0
    wrong_class_attempts: int = 0
    unconfirmed_hauls: int = 0
    false_report_hauls: int = 0
    claims: Dict[str, int] = field(default_factory=lambda: {"issued": 0, "lost": 0, "aborts": 0})
    approvals: Dict[str, int] = field(default_factory=lambda: {
        "requested": 0, "approved": 0, "vetoed": 0, "timeout_approved": 0, "deferred": 0})
    intake: Dict[str, int] = field(default_factory=lambda: {"reports": 0, "records": 0, "rejected": 0})
    final_stock: Dict[Pos, int] = field(default_factory=dict)
    hindsight_buy: float = 0.0
    runtime_ms: float = 0.0
    trajectory: Dict[int, List[Pos]] = field(default_factory=dict)
