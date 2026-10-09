import json

import pytest
from src.doi.llm.chatcache import CachedChat, ReplayMiss, request_key
from src.doi.llm.client import FakeChatClient

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "SECRET-PROMPT-MARKER"}]
TOOLS = [{"type": "function", "function": {"name": "answer"}}]
REPLY = {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "answer", "arguments": "{}"}}]}


def fake():
    return FakeChatClient(lambda messages, tools: REPLY, latency_s=0.7)


def test_a_live_call_is_stored_then_replayed_with_no_client(tmp_path):
    client = fake()
    live = CachedChat(client, str(tmp_path), "small", True, "v1")
    first = live.chat(MSGS, TOOLS, "required", 100)
    assert client.calls == 1 and live.live_calls == 1 and live.hits == 0
    replay = CachedChat(None, str(tmp_path), "small", False, "v1")
    again = replay.chat(MSGS, TOOLS, "required", 100)
    assert again == first and again.latency_s == 0.7 and replay.hits == 1 and replay.live_calls == 0
    live.chat(MSGS, TOOLS, "required", 100)                      # the same request again, while live: a hit
    assert client.calls == 1 and live.hits == 1


def test_a_miss_without_live_raises_a_key_error_that_names_the_remedy(tmp_path):
    cached = CachedChat(None, str(tmp_path), "small", False, "v1")
    with pytest.raises(ReplayMiss, match="--agent-live") as err:
        cached.chat(MSGS, TOOLS, "required", 100)
    assert isinstance(err.value, KeyError) and "small" in str(err.value) and not str(err.value).startswith("'")
    spent = fake()
    with pytest.raises(ReplayMiss):                               # a client does not make a miss live
        CachedChat(spent, str(tmp_path), "small", False, "v1").chat(MSGS, TOOLS, "required", 100)
    assert spent.calls == 0


def test_the_key_depends_on_every_part_of_the_request():
    base = request_key("v1", "small", MSGS, TOOLS, "required")
    assert len(base) == 64 and request_key("v1", "small", MSGS, TOOLS, "required") == base
    forced = {"type": "function", "function": {"name": "answer"}}
    others = [request_key("v2", "small", MSGS, TOOLS, "required"), request_key("v1", "large", MSGS, TOOLS, "required"),
              request_key("v1", "small", MSGS[:1], TOOLS, "required"), request_key("v1", "small", MSGS, [], "required"),
              request_key("v1", "small", MSGS, TOOLS, forced)]
    assert len(set(others + [base])) == 6
    reordered = [{"content": "s", "role": "system"}, MSGS[1]]
    assert request_key("v1", "small", reordered, TOOLS, "required") == base       # key order in a dict is irrelevant


def test_a_line_cut_off_by_a_killed_run_is_skipped_and_the_next_append_is_safe(tmp_path):
    live = CachedChat(fake(), str(tmp_path), "small", True, "v1")
    live.chat(MSGS, TOOLS, "required", 100)
    with open(live.path, "a") as f:
        f.write('{"key": "abc", "resp')                           # no newline: the process died mid-write
    other = [{"role": "user", "content": "other"}]
    CachedChat(fake(), str(tmp_path), "small", True, "v1").chat(other, TOOLS, "required", 100)
    fresh = CachedChat(None, str(tmp_path), "small", False, "v1")
    assert fresh.chat(MSGS, TOOLS, "required", 100).latency_s == 0.7
    assert fresh.chat(other, TOOLS, "required", 100).latency_s == 0.7


def test_a_duplicate_key_keeps_the_first_response(tmp_path):
    live = CachedChat(fake(), str(tmp_path), "small", True, "v1")
    first = live.chat(MSGS, TOOLS, "required", 100)
    key = request_key("v1", "small", MSGS, TOOLS, "required")
    other = {"message": {"role": "assistant", "content": "later"}, "latency_s": 9.0, "prompt_tokens": 1,
             "completion_tokens": 1, "model": "x"}
    with open(live.path, "a") as f:
        f.write(json.dumps({"key": key, "response": other}) + "\n")
    assert CachedChat(None, str(tmp_path), "small", False, "v1").chat(MSGS, TOOLS, "required", 100) == first


def test_the_store_holds_responses_only_and_no_secret(tmp_path):
    live = CachedChat(fake(), str(tmp_path), "small", True, "v1")
    live.chat(MSGS, TOOLS, "required", 100)
    text = open(live.path).read()
    assert "tool_calls" in text and "SECRET-PROMPT-MARKER" not in text
    assert "Authorization" not in text and "sk-" not in text
    assert live.path.endswith("small/chat.jsonl") and live.model_id == "fake"
    assert CachedChat(None, str(tmp_path), "other", False, "v1").model_id == "other"
