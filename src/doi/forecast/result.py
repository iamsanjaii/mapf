"""ForecastResult: what any forecaster returns for one case."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForecastResult:
    answer: Optional[bool]          # None: the forecast failed, so the guard keeps the classical threshold
    confidence: Optional[float]
    reason: str
    failed: str                     # "" on success, else a failure code
    latency_s: float
    calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    model: str
