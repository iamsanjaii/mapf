"""ForecastResult: what any forecaster returns for one case."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForecastResult:
    answer: Optional[bool]          # None: the forecast failed, so the guard keeps the classical threshold
    confidence: Optional[float]
    reason: str
    failed: str                     # "" on success, else a failure code
    latency_s: float                # seconds until the answer is visible; the guard keeps the classical rule till then
