"""Exception intake: free-text incident report -> validated ObstructionRecord, with an on-disk cache."""
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

from src.doi.crdt import ObstructionRecord
from src.doi.incidents import locate
from src.doi.llm.client import LLMClient
from src.doi.scenarios import Scenario

PROMPT_VERSION = "intake-v1"
KINDS = ("pallet", "spill", "debris", "rack_damage", "other")
CLASSES = ("robot_clearable", "needs_human")

SYSTEM_PROMPT = """You convert one warehouse incident report into JSON.
Use only a location from the list the user provides. If the report does not name exactly one listed \
location, or is not about an obstruction, answer {"reject": "<short reason>"}.
Otherwise answer a JSON object with exactly these fields:
"location": one name from the list,
"kind": one of pallet, spill, debris, rack_damage, other,
"class": robot_clearable or needs_human,
"est_kits": an integer from 1 to 5 (how many clean-up kits are needed),
"confidence": a number from 0 to 1,
"rationale": at most 20 words.
Use needs_human for structural damage, hazardous material, injury, or anything the report says robots must avoid.
Output JSON only."""


def build_user_prompt(text: str, names: Sequence[str]) -> str:
    return "Locations:\n" + "\n".join(names) + "\n\nReport:\n" + text


@dataclass(frozen=True)
class IntakeResult:
    key: str
    ok: bool
    location: Optional[str]
    kind: Optional[str]
    cls: Optional[str]
    est_kits: Optional[int]
    confidence: Optional[float]
    rationale: str
    reject_reason: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    raw: str
    model: str = ""


def cache_key(text: str, names: Sequence[str]) -> str:
    return hashlib.sha256((PROMPT_VERSION + "\n" + "\n".join(names) + "\n" + text).encode()).hexdigest()


def _fail(reason: str, rationale: str = "") -> Dict[str, Any]:
    return {"ok": False, "location": None, "kind": None, "cls": None, "est_kits": None,
            "confidence": None, "rationale": rationale, "reject_reason": reason}


def parse_and_validate(raw: str, names: Sequence[str]) -> Dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return _fail("not_json")
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return _fail("not_json")
    if not isinstance(obj, dict):
        return _fail("not_json")
    if "reject" in obj:
        return _fail(f"model_reject: {obj['reject']}")
    for field in ("location", "kind", "class", "est_kits", "confidence", "rationale"):
        if field not in obj:
            return _fail(f"missing:{field}")
    kits, conf = obj["est_kits"], obj["confidence"]
    checks = [
        (obj["location"] in names, "bad_location"),
        (obj["kind"] in KINDS, "bad_kind"),
        (obj["class"] in CLASSES, "bad_class"),
        (isinstance(kits, int) and not isinstance(kits, bool) and 1 <= kits <= 5, "bad_kits"),
        (isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 <= conf <= 1, "bad_confidence"),
    ]
    for ok, reason in checks:
        if not ok:
            return _fail(reason)
    return {"ok": True, "location": obj["location"], "kind": obj["kind"], "cls": obj["class"],
            "est_kits": kits, "confidence": float(conf), "rationale": str(obj["rationale"])[:200],
            "reject_reason": ""}


def run_intake(text: str, names: Sequence[str], client: LLMClient) -> IntakeResult:
    resp = client.complete(SYSTEM_PROMPT, build_user_prompt(text, names), max_tokens=200, temperature=0.0)
    parsed = parse_and_validate(resp.text, names)
    return IntakeResult(key=cache_key(text, names), latency_s=resp.latency_s,
                        prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                        raw=resp.text, model=resp.model, **parsed)


class IntakeCache:
    def __init__(self, root: str, model_key: str) -> None:
        self.path = os.path.join(root, model_key, "cache.jsonl")
        self._d: Dict[str, IntakeResult] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        res = IntakeResult(**json.loads(line))
                        self._d[res.key] = res

    def get(self, key: str) -> Optional[IntakeResult]:
        return self._d.get(key)

    def put(self, res: IntakeResult) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(res)) + "\n")
        self._d[res.key] = res


def to_record(res: IntakeResult, report_id: str, node: int, scenario: Scenario) -> ObstructionRecord:
    if not res.ok:
        raise ValueError("cannot build a record from a rejected intake result")
    return ObstructionRecord(report_id, node, res.location, locate(scenario, res.location), res.kind,
                             res.cls, res.est_kits, res.confidence, res.rationale)
