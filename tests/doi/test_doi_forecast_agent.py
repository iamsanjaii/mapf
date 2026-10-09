import dataclasses
import json

import pytest
from src.doi.forecast.agent import (AGENT_MODES, FORCE_ANSWER, MAX_TOOL_CALLS, MAX_TURNS, PROMPT_VERSION, describe,
                                    run_agent)
from src.doi.forecast.case import ForecastCase
from src.doi.forecast.tools import ANSWER
from src.doi.llm.chatcache import ReplayMiss
from src.doi.llm.client import FakeChatClient

CASE = ForecastCase(
    case_id="x", robot=3, tick=41, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0,
    notices=(("n0", 5, "wave 2 is all in the south bays"),),
    ledger={"robots_known": 8.0, "tasks_per_robot": 20.0, "trips_recorded": 10.0, "trips_expected": 160.0,
            "saving_on_recorded_trips": 6.0, "saving_on_my_current_trip": 0.0, "my_tasks_done": 1.0,
            "my_tasks_left": 19.0},
    zones={"north": {"rows": (0, 6), "cols": (0, 9)}, "south": {"rows": (8, 14), "cols": (0, 9)}},
    trip_saving={"north|south": {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0}},
    numeric_forecast=False, truth=True)


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def msg(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def yes(**over):
    return call("answer", {"will_pay": True, "confidence": 0.8, "reason": "south work", **over})


def script(*replies, latency_s=0.5):
    it = iter(replies)
    return FakeChatClient(lambda messages, tools: next(it), latency_s=latency_s)


def test_a_model_that_reads_the_notices_then_answers():
    client = script(msg(call("read_notices", {})), msg(yes()))
    res = run_agent(CASE, client)
    assert (res.answer, res.confidence, res.reason, res.failed) == (True, 0.8, "south work", "")
    assert (res.calls, res.tool_calls, res.latency_s, res.model) == (2, 1, 1.0, "fake")
    assert res.prompt_tokens > 0 and res.completion_tokens > 0
    second = client.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert "wave 2 is all in the south bays" in second[-1]["content"]
    assert [r["tool_choice"] for r in client.requests] == ["required", "required"]
    assert PROMPT_VERSION == "forecast-v2" and AGENT_MODES == ("tools", "single")


def test_the_last_turn_forces_answer_and_a_model_that_never_answers_is_no_answer():
    client = script(*[msg(call("read_notices", {}))] * MAX_TURNS)
    res = run_agent(CASE, client)
    assert [r["tool_choice"] for r in client.requests] == ["required"] * (MAX_TURNS - 1) + [FORCE_ANSWER]
    assert (res.answer, res.failed, res.calls, res.tool_calls) == (None, "no_answer", MAX_TURNS, MAX_TURNS)


def test_text_only_reply_is_no_tool_call():
    res = run_agent(CASE, script({"role": "assistant", "content": "I think yes"}))
    assert (res.answer, res.failed, res.calls, res.tool_calls) == (None, "no_tool_call", 1, 0)


@pytest.mark.parametrize("args", [
    {"will_pay": "yes", "confidence": 0.5, "reason": "x"}, {"will_pay": 1, "confidence": 0.5, "reason": "x"},
    {"will_pay": True, "confidence": 1.5, "reason": "x"}, {"will_pay": True, "confidence": -0.1, "reason": "x"},
    {"will_pay": True, "confidence": "high", "reason": "x"}, {"will_pay": True, "confidence": True, "reason": "x"},
    {"will_pay": True, "reason": "x"}, {"confidence": 0.5}, "not json at all", "[1, 2]",
    '{"will_pay": true, "confidence": NaN}'])
def test_a_malformed_answer_is_bad_answer(args):
    res = run_agent(CASE, script(msg(call("answer", args))))
    assert (res.answer, res.failed) == (None, "bad_answer")


def test_the_reason_is_cut_and_a_missing_reason_is_empty():
    long = run_agent(CASE, script(msg(yes(reason="r" * 500))))
    assert len(long.reason) == 200
    bare = run_agent(CASE, script(msg(call("answer", {"will_pay": False, "confidence": 1}))))
    assert (bare.answer, bare.confidence, bare.reason, bare.failed) == (False, 1.0, "", "")


class _Boom:
    model_id = "boom"

    def __init__(self, good_first=False):
        self.good_first, self.n = good_first, 0

    def chat(self, messages, tools, tool_choice, max_tokens):
        self.n += 1
        if self.good_first and self.n == 1:
            return FakeChatClient(lambda m, t: msg(call("read_notices", {}))).chat(messages, tools, tool_choice, 1)
        raise TimeoutError("read timed out")


def test_a_client_that_raises_is_client_error_and_keeps_what_was_spent():
    res = run_agent(CASE, _Boom())
    assert (res.answer, res.failed, res.calls, res.model) == (None, "client_error", 0, "boom")
    later = run_agent(CASE, _Boom(good_first=True))
    assert (later.failed, later.calls, later.tool_calls) == ("client_error", 1, 1)
    assert res.reason == "TimeoutError: read timed out"      # the run's log says why the model failed


def test_a_replay_miss_is_not_a_client_error():
    class Miss:
        model_id = "m"

        def chat(self, *a):
            raise ReplayMiss("no stored response")
    with pytest.raises(KeyError, match="no stored response"):
        run_agent(CASE, Miss())


def test_more_than_the_tool_budget_is_refused_and_the_next_turn_is_forced():
    five = msg(*[call("read_notices", {}, cid=f"c{i}") for i in range(5)])
    client = script(five, five, msg(yes()))
    res = run_agent(CASE, client)
    assert res.answer is True and res.tool_calls == 10 and res.calls == 3
    assert client.requests[2]["tool_choice"] == FORCE_ANSWER                  # 10 calls used, 8 allowed
    tool_msgs = [m for m in client.requests[2]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 10 and "budget" in tool_msgs[-1]["content"] and "budget" in tool_msgs[-2]["content"]
    assert "budget" not in tool_msgs[7]["content"] and MAX_TOOL_CALLS == 8


def test_an_unknown_tool_and_unreadable_arguments_are_errors_the_model_sees():
    client = script(msg(call("teleport", {}, "a"), call("trip_saving", "{{", "b"),
                        call("trip_saving", {"from_zone": "attic", "to_zone": "south"}, "c")), msg(yes()))
    res = run_agent(CASE, client)
    assert res.answer is True and res.tool_calls == 3
    contents = [m["content"] for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert "unknown tool" in contents[0] and "valid JSON" in contents[1] and "attic" in contents[2]


def test_the_first_answer_in_a_response_ends_the_forecast():
    after_tool = run_agent(CASE, script(msg(call("read_notices", {}, "a"), yes(), call("read_notices", {}, "z"))))
    assert (after_tool.answer, after_tool.tool_calls, after_tool.calls) == (True, 1, 1)
    twice = run_agent(CASE, script(msg(yes(), call("answer", {"will_pay": False, "confidence": 0.1, "reason": "n"}, "c2"))))
    assert (twice.answer, twice.confidence) == (True, 0.8)


def test_single_mode_makes_exactly_one_call_with_everything_in_the_prompt():
    client = script(msg(yes()))
    res = run_agent(CASE, client, mode="single")
    assert (res.answer, res.calls, res.tool_calls) == (True, 1, 0) and len(client.requests) == 1
    request = client.requests[0]
    assert request["tools"] == [ANSWER] and request["tool_choice"] == FORCE_ANSWER
    user = request["messages"][1]["content"]
    assert "wave 2 is all in the south bays" in user and "trips_expected" in user and "north|south" in user
    assert "use the tools" not in request["messages"][0]["content"].lower()
    stray = run_agent(CASE, script(msg(call("read_notices", {}))), mode="single")
    assert (stray.answer, stray.failed, stray.calls) == (None, "no_answer", 1)
    with pytest.raises(ValueError, match="mode"):
        run_agent(CASE, script(), mode="chat")


def test_hidden_fields_never_reach_a_prompt():
    other = dataclasses.replace(CASE, robot=7, tick=999, truth=False, numeric_forecast=True)
    a, b = script(msg(call("read_notices", {})), msg(yes())), script(msg(call("read_notices", {})), msg(yes()))
    run_agent(CASE, a)
    run_agent(other, b)
    assert a.requests == b.requests
    text = json.dumps(a.requests)
    assert "truth" not in text and "numeric_forecast" not in text and "tick" not in describe(CASE).lower()
    s, t = script(msg(yes())), script(msg(yes()))
    run_agent(CASE, s, "single")
    run_agent(other, t, "single")
    assert s.requests == t.requests


def test_the_first_message_describes_the_action_and_lists_the_zones():
    text = describe(CASE)
    assert "(10, 10)" in text and "(10, 11)" in text and "10" in text and "6" in text
    assert "north: rows 0 to 6, columns 0 to 9" in text and "south: rows 8 to 14, columns 0 to 9" in text
    assert "no named zones" in describe(dataclasses.replace(CASE, zones={}))
