"""Approval request drafter: numbers come from the ledger; the LLM only words the reason."""
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.doi.crdt import ObstructionRecord
from src.doi.llm.client import LLMClient

Pos = Tuple[int, int]
REASON_SYSTEM = ("Explain in one short sentence, for a warehouse supervisor, what is blocking the lane. "
                 "Use only the notes provided. Do not write any numbers.")


@dataclass(frozen=True)
class ApprovalRequest:
    pits: Tuple[Pos, ...]
    evidence: float
    buy: float
    contributors: int
    census: int
    report_ids: Tuple[str, ...]
    reason: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\d", "", text).strip()[:200]


def draft_request(pits, evidence: float, buy: float, contributors: int, census: int,
                  records: Sequence[ObstructionRecord], client: Optional[LLMClient] = None) -> ApprovalRequest:
    notes = "; ".join(r.rationale for r in records)
    if client is not None and records:
        reason = _clean(client.complete(REASON_SYSTEM, notes, max_tokens=60, temperature=0.0).text)
    else:
        reason = _clean(notes)
    ids = tuple(sorted(r.report_id for r in records))
    cells = " ".join(str(tuple(p)) for p in pits)
    text = (f"Edit {cells}: fleet detour cost so far {evidence:.1f}, estimated edit cost {buy:.1f}, "
            f"from {contributors} of {census} robots, reports {', '.join(ids)}. Reason: {reason}")
    return ApprovalRequest(tuple(pits), float(evidence), float(buy), contributors, census, ids, reason, text)
