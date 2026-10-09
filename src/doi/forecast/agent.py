"""The forecast agent: a bounded, tool-using conversation that ends in one yes/no forecast.

The agent sees a ForecastCase and nothing else, through `chat`: the real client, the fake, or the record-and-replay
wrapper. It can read; it cannot move a robot or an obstacle or change a threshold. Any failure gives answer=None and
the guard keeps the classical rule. Raise PROMPT_VERSION whenever the prompts or a tool schema change.
"""
import json
from typing import Any, Optional, Tuple

from src.doi.forecast.case import ForecastCase
from src.doi.forecast.result import ForecastResult
from src.doi.forecast.tools import ANSWER, TOOLS, call_tool
from src.doi.llm.chatcache import ReplayMiss
from src.doi.llm.client import Chat, ChatResponse

PROMPT_VERSION = "forecast-v2"
MAX_TURNS = 4               # model responses per forecast
MAX_TOOL_CALLS = 8          # tool calls per forecast, not counting `answer`
MAX_TOKENS = 400
REASON_CHARS = 200
AGENT_MODES = ("tools", "single")
FORCE_ANSWER = {"type": "function", "function": {"name": "answer"}}

_QUESTION = """You advise one warehouse robot. The robot can move an obstacle out of the way now, which costs a one-off
price, or keep walking round it. Give one forecast: over the whole shift, will the fleet's total travel saving from
moving this obstacle reach the price?

The total is the saving counted so far plus the saving on the trips still to come. The saving counted so far is only
a part of it and can be far below the price while the total still reaches it. Work it out in three steps:
1. How many trips are still to come (trips expected minus trips recorded), and how many robots will make them.
2. Where those trips will go. Recorded traffic shows where trips went so far; a notice that work is moving to other
   zones means the trips still to come follow the notice, not the record. A notice that does not change where robots
   travel (a printer, a rota) should change nothing.
3. The saving per trip for those zone pairs (trip_saving). Multiply by the trips still to come, add the saving
   counted so far, and compare the sum with the price.
Answer yes when the sum reaches the price, no when it falls short. Do not default to no."""

SYSTEM_TOOLS = _QUESTION + """

You know only what this robot knows. Use the tools.
- read_notices: messages the robot has received about future work. They can be wrong or irrelevant.
- ledger_summary: the traffic recorded so far and how many trips are still to come.
- trip_saving: how much one trip between two zones would save if the obstacle were moved.
When you are ready, call answer. You have 4 turns."""

SYSTEM_SINGLE = _QUESTION + """

You know only what this robot knows, which is given below. Notices can be wrong or irrelevant. Call answer."""


def describe(case: ForecastCase) -> str:
    """The candidate action and the zones. Hidden fields (robot, tick, truth, numeric forecast) are not here."""
    lines = [f"Candidate action: {case.mode} the {case.kind} at {tuple(case.obstacle)} so that it ends at "
             f"{tuple(case.landing)}.",
             f"Price of the move: {case.price:g}.",
             f"Saving counted so far (part of the total, not the total): {case.known_saving:g}."]
    if case.zones:
        lines.append("Zones:")
        for name in sorted(case.zones):
            box = case.zones[name]
            lines.append(f"- {name}: rows {box['rows'][0]} to {box['rows'][1]}, "
                         f"columns {box['cols'][0]} to {box['cols'][1]}")
    else:
        lines.append("There are no named zones.")
    return "\n".join(lines)


def single_prompt(case: ForecastCase) -> str:
    """Mode `single`: the action, the notices, the ledger summary and the whole trip table in one message."""
    return "\n\n".join([
        describe(case),
        "Notices received:\n" + json.dumps(call_tool(case, "read_notices", {}), sort_keys=True),
        "Ledger summary:\n" + json.dumps(call_tool(case, "ledger_summary", {}), sort_keys=True),
        "Trip saving by zone pair (from|to):\n" + json.dumps(case.trip_saving, sort_keys=True)])


def _parse_args(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _checked_answer(args: Optional[dict]) -> Optional[Tuple[bool, float, str]]:
    if args is None:
        return None
    will_pay, conf = args.get("will_pay"), args.get("confidence")
    if not isinstance(will_pay, bool):
        return None
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None
    return will_pay, float(conf), str(args.get("reason", ""))[:REASON_CHARS]


class _Tally:
    def __init__(self, chat: Chat) -> None:
        self.latency_s, self.calls, self.prompt, self.completion = 0.0, 0, 0, 0
        self.model = getattr(chat, "model_id", "")

    def add(self, resp: ChatResponse) -> None:
        self.latency_s += resp.latency_s
        self.calls += 1
        self.prompt += resp.prompt_tokens
        self.completion += resp.completion_tokens
        self.model = resp.model or self.model

    def result(self, answer: Optional[bool], confidence: Optional[float], reason: str, failed: str,
               tool_calls: int) -> ForecastResult:
        return ForecastResult(answer, confidence, reason, failed, self.latency_s, self.calls, tool_calls,
                              self.prompt, self.completion, self.model)


def run_agent(case: ForecastCase, chat: Chat, mode: str = "tools") -> ForecastResult:
    """Run the model on one case for at most MAX_TURNS responses (one in mode `single`)."""
    if mode not in AGENT_MODES:
        raise ValueError(f"unknown agent mode {mode!r}: choose from {', '.join(AGENT_MODES)}")
    single = mode == "single"
    tally = _Tally(chat)
    messages = [{"role": "system", "content": SYSTEM_SINGLE if single else SYSTEM_TOOLS},
                {"role": "user", "content": single_prompt(case) if single else describe(case)}]
    tools = [ANSWER] if single else TOOLS
    used = 0
    for turn in range(1, (1 if single else MAX_TURNS) + 1):
        forced = single or turn == MAX_TURNS or used >= MAX_TOOL_CALLS
        try:
            resp = chat.chat(messages, tools, FORCE_ANSWER if forced else "required", MAX_TOKENS)
        except ReplayMiss:
            raise                       # a missing store is the owner's to fix, not a model that failed
        except Exception as exc:        # the guard keeps the classical rule; the message says why the model failed
            return tally.result(None, None, f"{type(exc).__name__}: {exc}"[:REASON_CHARS], "client_error", used)
        tally.add(resp)
        messages.append(resp.message)
        calls = resp.message.get("tool_calls") or []
        if not calls:
            return tally.result(None, None, "", "no_tool_call", used)
        for k, one in enumerate(calls):
            fn = one.get("function") or {}
            name, args = fn.get("name"), _parse_args(fn.get("arguments"))
            if name == "answer":
                checked = _checked_answer(args)
                if checked is None:
                    return tally.result(None, None, "", "bad_answer", used)
                return tally.result(checked[0], checked[1], checked[2], "", used)
            used += 1
            if used > MAX_TOOL_CALLS:
                out = {"error": "tool-call budget used up: call answer"}
            elif args is None:
                out = {"error": "arguments are not a valid JSON object"}
            else:
                out = call_tool(case, name, args)
            messages.append({"role": "tool", "tool_call_id": one.get("id") or f"call_{turn}_{k}",
                             "content": json.dumps(out, sort_keys=True)})
    return tally.result(None, None, "", "no_answer", used)
