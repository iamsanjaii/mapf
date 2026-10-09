"""The tools of the forecast agent. Each read-only tool is a pure function of a ForecastCase; none writes anything.

`answer` ends a forecast and is handled by the agent loop, not here.
"""
from typing import Any, Dict, List

from src.doi.forecast.case import ForecastCase


def _tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties, "required": required,
                                        "additionalProperties": False}}}


READ_NOTICES = _tool("read_notices", "The messages this robot has received about future work. They can be wrong "
                     "or irrelevant.", {}, [])
LEDGER_SUMMARY = _tool("ledger_summary", "The traffic recorded so far, how many trips are still to come, the price "
                       "of the move and the saving known so far.", {}, [])
TRIP_SAVING = _tool("trip_saving", "How much one trip from one zone to another would save if the obstacle were "
                    "moved: the mean saving, the share of trips that save anything, and the pairs sampled.",
                    {"from_zone": {"type": "string", "description": "a zone name from the list you were given"},
                     "to_zone": {"type": "string", "description": "a zone name from the list you were given"}},
                    ["from_zone", "to_zone"])
ANSWER = _tool("answer", "Give the forecast and end.",
               {"will_pay": {"type": "boolean", "description": "true if the fleet's saving will reach the price"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                               "description": "how likely your answer is to be right, from 0 to 1"},
                "reason": {"type": "string", "description": "one short sentence"}},
               ["will_pay", "confidence", "reason"])
TOOLS = [READ_NOTICES, LEDGER_SUMMARY, TRIP_SAVING, ANSWER]
READ_ONLY = ("read_notices", "ledger_summary", "trip_saving")


def _error(reason: str) -> Dict[str, str]:
    return {"error": reason}


def call_tool(case: ForecastCase, name: Any, args: Any) -> Dict[str, Any]:
    """Run one read-only tool. A bad name or bad arguments give {"error": ...}, which goes back to the model."""
    if name == "read_notices":
        return {"notices": [{"id": nid, "tick": tick, "text": text} for nid, tick, text in case.notices]}
    if name == "ledger_summary":
        return {**case.ledger, "price": case.price, "known_saving": case.known_saving}
    if name == "trip_saving":
        if not isinstance(args, dict):
            return _error("trip_saving needs an object with from_zone and to_zone")
        origin, dest = args.get("from_zone"), args.get("to_zone")
        if not case.zones:
            return _error("this case has no zones, so trip_saving has nothing to report")
        for zone in (origin, dest):
            if not isinstance(zone, str) or zone not in case.zones:
                return _error(f"unknown zone {zone!r}: choose from {', '.join(sorted(case.zones))}")
        row = case.trip_saving.get(f"{origin}|{dest}")
        return dict(row) if row is not None else _error(f"no trip data for {origin}|{dest}")
    return _error(f"unknown tool {name!r}: choose from {', '.join(READ_ONLY)} or answer")
