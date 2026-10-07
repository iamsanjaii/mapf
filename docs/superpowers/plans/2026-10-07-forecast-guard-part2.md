# Forecast Guard, Part 2 (the model parts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a language model be the forecaster behind `rof_a`: chat with tools, record and replay of every call, the agent loop, the `llm:<key>` forecaster, command-line flags, the dataset builder, the scoring script and the E9 end-to-end script. No model is called by any test or by this build.

**Architecture:** The agent loop sees a `ForecastCase` only (Part 1) and talks to anything with a `chat` method. Real calls go through `CachedChat`, which stores every response under a hash of the request, so a re-run replays the same answers with the same latencies and calls nothing. A miss is an error unless `--agent-live` is set. The guard (Part 1) still decides; the model only sets the threshold.

**Tech Stack:** Python 3 in `venv/`, standard library (`urllib`), `numpy`, `pandas`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-forecast-agent-guard-design.md`. This plan implements sections 8, 9, 10 (`llm`), 11 (config, `llm/client.py`, `run_doi.py`), 12, 14.1 to 14.3, 15.4, 15.5 (items 6), 15.6 and build-order steps 6 to 9 of section 16. Part 1 is `docs/superpowers/plans/2026-10-06-forecast-guard-part1.md`; the owner decided on 2026-10-07 to build Part 2 after reading the headroom pilot (`docs/research/results.md`).

## Global Constraints

- Run Python as `venv/bin/python` (there is no `python` on PATH). Run tests with `venv/bin/python -m pytest tests/doi -q`. The suite takes about 2.5 minutes.
- Every existing test must pass unchanged. Do not edit an existing test's expected value.
- **No test and no script run in this plan calls a language model or opens a network connection.** Tests use `FakeChatClient`, scripted replies and non-model forecasters. Anything that can call a model needs `--agent-live` and says so.
- Temperature is 0. `MAX_TURNS = 4`, `MAX_TOOL_CALLS = 8`. `PROMPT_VERSION` must be raised whenever the system prompt or a tool schema changes.
- The API key is only ever placed in the `Authorization` header: never in a store file, a log line, a message, an exception text or a test output.
- Every random draw uses `src.doi.rng.stream` or `src.doi.rng.u01`. Never `random.random()`, unseeded `numpy.random` or the clock for a decision.
- Edit existing modules only where a task says so. Do not change `world.py`, `spacetime.py`, `network.py`, `pushplan.py`, `carryplan.py`, `evidence.py`, `crdt.py`, `belief.py`.
- Commit messages carry no `Co-Authored-By` line, no "Generated with" line and no mention of Claude or Anthropic.
- Never commit `toy.gif`, anything under `experiments/results/`, or the generated case files `data/forecasts/dev.jsonl`, `test.jsonl`, `human.jsonl`.
- Match the surrounding code: type hints on every function, a module docstring on every new file, no print statements in library code (scripts may print).
- `lam` is in `(0, 1]`. The threshold is `lam` on a yes, `1 / lam` on a no, exactly `1.0` with no visible answer (Part 1, unchanged).

## Departures from the spec

Each is small and deliberate. The owner approves them by approving this plan.

1. `CachedChat` takes `prompt_version` as an argument, so `src/doi/llm/` does not import `src/doi/forecast/`. The spec's key is `(PROMPT_VERSION, model_key, messages, tools, tool_choice)`; the key is the same.
2. A replay miss raises `ReplayMiss`, a `KeyError` subclass. `run_agent` lets it through; every other exception from the client becomes `client_error`. Without this a missing store would look like a model that fails, and a run would silently fall back to the classical rule.
3. `OpenAICompatClient.chat` returns the assistant message reduced to `role`, `content` and `tool_calls`, so the message is safe to send back and its cache key is stable.
4. Replay needs no credentials: the real client is built only when `agent_live` is true.
5. Mode `single` has its own system prompt (the tools text refers to tools it does not have).
6. `ForecastResult` stays in `src/doi/forecast/result.py` (Part 1); the spec puts it in `agent.py`.
7. New `src/doi/forecast/budget.py` holds the cost estimate and confirmation shared by `doi_agent_eval.py` and `doi_e9_agent.py`.
8. Config gains `agent_mode`, `agent_cache`, `agent_live` (spec) and `run_doi.py` gains `--agent-cache` besides the three spec flags.
9. "An identical `RunResult`" on replay is tested as identical `J`, `J_censored`, `ticks`, `pushes`, `forecast_log` and `forecast_cases`; `runtime_ms` differs by nature.
10. The generated case files are not committed (about 10 MB). The owner runs `doi_agent_cases.py`.
11. `doi_e9_agent.py` runs with one job when `--live` is set, so two processes never append to one store at once. Its single-call arm runs with `--live` or `--with-single` only, because replay needs that arm's own stored calls.
12. `doi_e9_headroom.usable` gains a `point` argument so E9 can reuse it.

## Review Focus

1. A replay miss in the middle of a run: the run stops with `ReplayMiss` (a `KeyError` naming `--agent-live`); it is never swallowed as `client_error` and never calls the model on its own (Tasks 4 and 5).
2. A model response with several tool calls, with `answer` in the middle or twice: the first `answer` ends the forecast and later calls are ignored (Task 4).
3. Tool arguments that are not JSON, not an object, or name an unknown zone or tool: an `{"error": ...}` goes back to the model, the call counts against the budget, nothing crashes (Tasks 3 and 4).
4. Hidden fields (`robot`, `tick`, `truth`, `numeric_forecast`) never reach a prompt: two cases that differ only in them give identical requests and so identical store keys (Task 4).
5. The store: a last line cut off by a killed run is skipped and the next append is still readable; a duplicate key keeps the first response; no request text and no key is written (Task 2). The `human` split with no `human_notices.jsonl` is skipped with a message and exit 0 (Task 7).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/doi/llm/client.py` | modify | `ChatResponse`, `Chat`, `OpenAICompatClient.chat`, `FakeChatClient`, the key rule |
| `src/doi/llm/chatcache.py` | create | `request_key`, `ReplayMiss`, `CachedChat` |
| `src/doi/forecast/tools.py` | create | tool schemas and `call_tool` |
| `src/doi/forecast/agent.py` | create | prompts, `run_agent` |
| `src/doi/forecast/forecasters.py` | modify | `LlmForecaster`, `make_forecaster(name, cfg)` |
| `src/doi/forecast/budget.py` | create | `call_budget`, `confirm_calls` |
| `src/doi/config.py` | modify | `agent_mode`, `agent_cache`, `agent_live` |
| `src/doi/policies.py` | modify | pass `cfg` to `make_forecaster` |
| `run_doi.py`, `src/doi/narrate.py` | modify | flags and their guide text |
| `experiments/doi_agent_cases.py` | create | collect forecast cases into `dev`, `test`, `human` files |
| `experiments/doi_agent_eval.py` | create | score forecasters on a case file |
| `experiments/doi_e9_agent.py` | create | the end-to-end experiment |
| `experiments/doi_e9_headroom.py` | modify | `usable(df, point=POINT)` |
| `src/doi/README.md`, `data/forecasts/README.md`, `docs/research/results.md`, `docs/research/demo-guide.md`, the spec header, `.gitignore` | modify | documents |

---

### Task 1: Chat with tools, a fake, and the key rule

**Files:**
- Modify: `src/doi/llm/client.py`
- Test: `tests/doi/test_doi_chat_client.py`

**Interfaces:**
- Consumes: existing `OpenAICompatClient`, `client_from_env` in `src/doi/llm/client.py`.
- Produces: `ChatResponse(message: dict, latency_s: float, prompt_tokens: int, completion_tokens: int, model: str)`; `Chat` protocol (`model_id: str`, `chat(messages, tools, tool_choice, max_tokens) -> ChatResponse`); `OpenAICompatClient.chat(messages: list, tools: list, tool_choice, max_tokens: int) -> ChatResponse`; `FakeChatClient(reply: Callable[[list, list], dict], latency_s=0.1, model_id="fake")` with `.calls: int` and `.requests: List[dict]` (each `{"messages", "tools", "tool_choice"}` as deep copies); the rule that `OPENAI_API_KEY` is the fallback for key `hosted` and for any model key whose URL starts with `https://api.openai.com/`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_chat_client.py`:

```python
import io
import json
import urllib.error

import pytest
from src.doi.llm.client import ChatResponse, FakeChatClient, OpenAICompatClient, client_from_env

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "read_notices", "arguments": "{}"}}
PAYLOAD = {"model": "m-2026", "usage": {"prompt_tokens": 12, "completion_tokens": 5},
           "choices": [{"message": {"role": "assistant", "content": None, "refusal": None,
                                    "tool_calls": [TOOL_CALL]}}]}
TOOLS = [{"type": "function", "function": {"name": "read_notices"}}]


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _patch(monkeypatch, payload, seen):
    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["auth"] = req.get_header("Authorization")
        return _Resp(payload)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_chat_request_shape_and_reduced_response(monkeypatch):
    seen = {}
    _patch(monkeypatch, PAYLOAD, seen)
    client = OpenAICompatClient("http://localhost:11434/v1", "small-model", api_key="sk-test")
    resp = client.chat([{"role": "user", "content": "hi"}], TOOLS, "required", 300)
    body = seen["body"]
    assert seen["url"] == "http://localhost:11434/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert body["model"] == "small-model" and body["tools"] == TOOLS and body["tool_choice"] == "required"
    assert body["max_tokens"] == 300 and body["temperature"] == 0.0 and body["messages"][0]["content"] == "hi"
    assert resp.message == {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}     # `refusal` dropped
    assert (resp.prompt_tokens, resp.completion_tokens, resp.model) == (12, 5, "m-2026") and resp.latency_s >= 0


def test_chat_forced_choice_token_param_and_no_temperature(monkeypatch):
    seen = {}
    _patch(monkeypatch, PAYLOAD, seen)
    client = OpenAICompatClient("http://x/v1", "m", token_param="max_completion_tokens", send_temperature=False)
    forced = {"type": "function", "function": {"name": "answer"}}
    client.chat([], TOOLS, forced, 50)
    assert seen["body"]["tool_choice"] == forced and seen["body"]["max_completion_tokens"] == 50
    assert "temperature" not in seen["body"] and seen["auth"] is None


def test_chat_message_without_tool_calls_keeps_only_role_and_content(monkeypatch):
    seen = {}
    payload = {"choices": [{"message": {"role": "assistant", "content": "I think yes", "tool_calls": None}}]}
    _patch(monkeypatch, payload, seen)
    resp = OpenAICompatClient("http://x/v1", "m").chat([], TOOLS, "required", 50)
    assert resp.message == {"role": "assistant", "content": "I think yes"} and resp.prompt_tokens == 0


def test_chat_http_400_explains_and_hides_nothing_secret(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"unsupported tool_choice"))
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="HTTP 400") as err:
        OpenAICompatClient("http://x/v1", "m", api_key="sk-secret").chat([], TOOLS, "required", 50)
    assert "unsupported tool_choice" in str(err.value) and "sk-secret" not in str(err.value)


def test_fake_chat_client_counts_and_records_deep_copies():
    messages = [{"role": "user", "content": "a"}]
    fake = FakeChatClient(lambda m, t: {"role": "assistant", "content": "ok"}, latency_s=0.4, model_id="f")
    resp = fake.chat(messages, TOOLS, "required", 10)
    messages.append({"role": "user", "content": "later"})
    assert isinstance(resp, ChatResponse) and resp.latency_s == 0.4 and resp.model == "f" and fake.calls == 1
    assert fake.requests == [{"messages": [{"role": "user", "content": "a"}], "tools": TOOLS,
                              "tool_choice": "required"}]


def _env(monkeypatch, key, url):
    monkeypatch.setenv(f"DOI_LLM_{key}_URL", url)
    monkeypatch.setenv(f"DOI_LLM_{key}_MODEL", "m")
    monkeypatch.delenv(f"DOI_LLM_{key}_API_KEY", raising=False)


def test_openai_key_fallback_is_for_hosted_and_openai_urls_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    _env(monkeypatch, "SMALL", "https://api.openai.com/v1")
    assert client_from_env("small")._api_key == "sk-shared"
    _env(monkeypatch, "HOSTED", "http://localhost:1/v1")
    assert client_from_env("hosted")._api_key == "sk-shared"                     # the existing rule stays
    _env(monkeypatch, "LARGE", "https://api.openai.com.evil.example/v1")
    assert client_from_env("large")._api_key is None
    _env(monkeypatch, "LOCAL", "http://localhost:11434/v1")
    assert client_from_env("local")._api_key is None
    monkeypatch.setenv("DOI_LLM_LOCAL_API_KEY", "own")
    assert client_from_env("local")._api_key == "own"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_chat_client.py -q`
Expected: collection error, `ImportError: cannot import name 'ChatResponse'`.

- [ ] **Step 3: Implement**

In `src/doi/llm/client.py`:

Change the typing import to:

```python
from typing import Any, Callable, List, Optional, Protocol, Tuple
```

Add directly after the `LLMResponse` dataclass:

```python
@dataclass(frozen=True)
class ChatResponse:
    message: dict            # the assistant message: role, content and (if any) tool_calls
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model: str


class Chat(Protocol):
    """Anything the forecast agent can talk to: the real client, the fake, or the record-and-replay wrapper."""
    model_id: str

    def chat(self, messages: list, tools: list, tool_choice: Any, max_tokens: int) -> ChatResponse: ...
```

Add these two methods to `OpenAICompatClient`, directly before `class FakeLLMClient:` (that is, after `complete`); `complete` itself is not touched:

```python
    def _post(self, body: dict) -> Tuple[dict, float]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 400:
                detail = e.read().decode(errors="replace")[:500]
                raise RuntimeError(f"HTTP 400 from {self.base_url}: {detail}") from None
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}") from None
        return json.loads(raw.decode()), time.perf_counter() - start

    def chat(self, messages: list, tools: list, tool_choice: Any, max_tokens: int) -> ChatResponse:
        """One turn of a tool-calling conversation. `tool_choice` is "required" or a forced function."""
        body = {"model": self.model_id, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                self.token_param: max_tokens}
        if self.send_temperature:
            body["temperature"] = self.temperature
        data, latency = self._post(body)
        usage = data.get("usage") or {}
        raw = data["choices"][0]["message"]
        message = {"role": "assistant", "content": raw.get("content")}
        if raw.get("tool_calls"):
            message["tool_calls"] = raw["tool_calls"]
        return ChatResponse(message=message, latency_s=latency, prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                            model=data.get("model", self.model_id))
```

Add directly before `def _need(var: str) -> str:`:

```python
class FakeChatClient:
    """A scripted chat client for tests: `reply(messages, tools)` returns the assistant message."""

    def __init__(self, reply: Callable[[list, list], dict], latency_s: float = 0.1, model_id: str = "fake") -> None:
        self.reply = reply
        self.latency_s = latency_s
        self.model_id = model_id
        self.calls = 0
        self.requests: List[dict] = []

    def chat(self, messages: list, tools: list, tool_choice: Any, max_tokens: int) -> ChatResponse:
        self.calls += 1
        self.requests.append(json.loads(json.dumps({"messages": messages, "tools": tools,
                                                    "tool_choice": tool_choice})))
        message = self.reply(messages, tools)
        return ChatResponse(message, self.latency_s, len(json.dumps(messages).split()),
                            len(json.dumps(message).split()), self.model_id)


OPENAI_URL_PREFIX = "https://api.openai.com/"


def _api_key(k: str, url: str) -> Optional[str]:
    """DOI_LLM_<KEY>_API_KEY, else OPENAI_API_KEY for the key `hosted` and for any key whose URL is OpenAI's."""
    own = os.environ.get(f"DOI_LLM_{k}_API_KEY")
    if own:
        return own
    if k == "HOSTED" or url.startswith(OPENAI_URL_PREFIX):
        return os.environ.get("OPENAI_API_KEY") or None
    return None
```

In `client_from_env`, replace the line

```python
    key = os.environ.get(f"DOI_LLM_{k}_API_KEY") or (os.environ.get("OPENAI_API_KEY") if k == "HOSTED" else None)
```

with:

```python
    key = _api_key(k, url)
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_chat_client.py tests/doi/test_doi_llm.py -q`
Expected: all pass (`test_doi_llm.py` is the existing client test and must pass unchanged).

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/llm/client.py tests/doi/test_doi_chat_client.py
git commit -m "feat(doi): chat with tools for the OpenAI-compatible client, a fake, and the key rule for api.openai.com"
```

---

### Task 2: Record and replay

**Files:**
- Create: `src/doi/llm/chatcache.py`
- Test: `tests/doi/test_doi_chatcache.py`

**Interfaces:**
- Consumes: `ChatResponse`, `Chat`, `FakeChatClient` (Task 1).
- Produces: `request_key(prompt_version, model_key, messages, tools, tool_choice) -> str` (64 hex); `ReplayMiss(KeyError)`; `CachedChat(client, root, model_key, live, prompt_version)` with `chat(messages, tools, tool_choice, max_tokens) -> ChatResponse`, `.path`, `.model_id`, `.hits: int`, `.live_calls: int`, `.client`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_chatcache.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_chatcache.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.doi.llm.chatcache'`.

- [ ] **Step 3: Implement**

Create `src/doi/llm/chatcache.py`:

```python
"""Record and replay for chat calls: a run never calls a model unless it was told to, and a re-run is identical.

The key of a call is the sha256 of the canonical JSON of (prompt version, model key, messages, tools, tool_choice).
A hit returns the stored response, with its stored latency, so a re-run makes the same decisions at the same
ticks. A miss raises ReplayMiss unless the cache is live, in which case the client is called and the response is
appended to <root>/<model key>/chat.jsonl. Only responses are stored: never a request, never a key.
"""
import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.doi.llm.client import Chat, ChatResponse


class ReplayMiss(KeyError):
    """A call that is not in the store, with replay only. A KeyError, but its message prints plainly."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def request_key(prompt_version: str, model_key: str, messages: list, tools: list, tool_choice: Any) -> str:
    blob = json.dumps([prompt_version, model_key, messages, tools, tool_choice], sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedChat:
    def __init__(self, client: Optional[Chat], root: str, model_key: str, live: bool, prompt_version: str) -> None:
        self.client, self.model_key, self.live, self.prompt_version = client, model_key, live, prompt_version
        self.path = os.path.join(root, model_key, "chat.jsonl")
        self.model_id = getattr(client, "model_id", model_key)
        self.hits = 0
        self.live_calls = 0
        self._d: Dict[str, ChatResponse] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self._d.setdefault(row["key"], ChatResponse(**row["response"]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue                                  # a blank or cut-off line

    def chat(self, messages: list, tools: list, tool_choice: Any, max_tokens: int) -> ChatResponse:
        key = request_key(self.prompt_version, self.model_key, messages, tools, tool_choice)
        stored = self._d.get(key)
        if stored is not None:
            self.hits += 1
            return stored
        if not self.live or self.client is None:
            raise ReplayMiss(f"no stored response for this model call (model key {self.model_key!r}, request "
                             f"{key[:12]}). Run the same command again with --agent-live to fill the store; that "
                             f"calls the model, so check the spend first")
        response = self.client.chat(messages, tools, tool_choice, max_tokens)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        lead = ""
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                lead = "" if f.read(1) == b"\n" else "\n"        # do not glue onto a cut-off line
        with open(self.path, "a") as f:
            f.write(lead + json.dumps({"key": key, "response": asdict(response)}) + "\n")
        self._d[key] = response
        self.live_calls += 1
        return response
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_chatcache.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/llm/chatcache.py tests/doi/test_doi_chatcache.py
git commit -m "feat(doi): record and replay for chat calls, keyed by prompt version, model, messages and tools"
```

---

### Task 3: The tools

**Files:**
- Create: `src/doi/forecast/tools.py`
- Test: `tests/doi/test_doi_forecast_tools.py`

**Interfaces:**
- Consumes: `ForecastCase` (Part 1, `src/doi/forecast/case.py`).
- Produces: `READ_NOTICES`, `LEDGER_SUMMARY`, `TRIP_SAVING`, `ANSWER` (OpenAI tool schemas, dicts), `TOOLS = [READ_NOTICES, LEDGER_SUMMARY, TRIP_SAVING, ANSWER]`, `call_tool(case: ForecastCase, name, args) -> dict` for the three read-only tools (any other name, including `answer`, gives an error dict).

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_forecast_tools.py`:

```python
import json

from src.doi.forecast.case import ForecastCase
from src.doi.forecast.tools import ANSWER, LEDGER_SUMMARY, READ_NOTICES, TOOLS, TRIP_SAVING, call_tool

LEDGER = {"robots_known": 8.0, "tasks_per_robot": 20.0, "trips_recorded": 10.0, "trips_expected": 160.0,
          "saving_on_recorded_trips": 6.0, "saving_on_my_current_trip": 0.0, "my_tasks_done": 1.0,
          "my_tasks_left": 19.0}
CASE = ForecastCase(
    case_id="x", robot=0, tick=9, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0,
    notices=(("n0", 5, "wave 2 is all in the south bays"), ("d0", 3, "the label printer is out of ribbon")),
    ledger=LEDGER,
    zones={"north": {"rows": (0, 6), "cols": (0, 9)}, "south": {"rows": (8, 14), "cols": (0, 9)}},
    trip_saving={"north|south": {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0},
                 "south|north": {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0},
                 "north|north": {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 20.0},
                 "south|south": {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 20.0}},
    numeric_forecast=False, truth=True)


def test_schemas_are_json_and_name_the_four_tools():
    assert [t["function"]["name"] for t in TOOLS] == ["read_notices", "ledger_summary", "trip_saving", "answer"]
    assert json.loads(json.dumps(TOOLS)) == TOOLS
    assert all(t["type"] == "function" for t in TOOLS)
    assert TRIP_SAVING["function"]["parameters"]["required"] == ["from_zone", "to_zone"]
    assert ANSWER["function"]["parameters"]["required"] == ["will_pay", "confidence", "reason"]
    assert ANSWER["function"]["parameters"]["properties"]["will_pay"]["type"] == "boolean"
    assert READ_NOTICES["function"]["parameters"]["properties"] == {} == LEDGER_SUMMARY["function"]["parameters"]["properties"]


def test_read_notices_returns_what_the_robot_received():
    out = call_tool(CASE, "read_notices", {})
    assert out == {"notices": [{"id": "n0", "tick": 5, "text": "wave 2 is all in the south bays"},
                               {"id": "d0", "tick": 3, "text": "the label printer is out of ribbon"}]}
    assert call_tool(CASE, "read_notices", None) == out                      # no arguments are needed


def test_ledger_summary_adds_price_and_known_saving():
    assert call_tool(CASE, "ledger_summary", {}) == {**LEDGER, "price": 10.0, "known_saving": 6.0}


def test_trip_saving_by_zone_pair_and_its_errors():
    assert call_tool(CASE, "trip_saving", {"from_zone": "north", "to_zone": "south"}) == \
        {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0}
    unknown = call_tool(CASE, "trip_saving", {"from_zone": "attic", "to_zone": "south"})
    assert "attic" in unknown["error"] and "north" in unknown["error"] and "south" in unknown["error"]
    assert "error" in call_tool(CASE, "trip_saving", {"from_zone": "north"})            # a missing argument
    assert "error" in call_tool(CASE, "trip_saving", {"from_zone": 3, "to_zone": ["south"]})
    assert "error" in call_tool(CASE, "trip_saving", None)
    assert "error" in call_tool(CASE, "trip_saving", ["north", "south"])
    nozones = ForecastCase(**{**CASE.__dict__, "zones": {}, "trip_saving": {}})
    assert "no zones" in call_tool(nozones, "trip_saving", {"from_zone": "north", "to_zone": "south"})["error"]


def test_unknown_tools_and_answer_are_errors_here():
    for name in ("teleport", "answer", None, 7):
        out = call_tool(CASE, name, {})
        assert list(out) == ["error"] and "read_notices" in out["error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_tools.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.doi.forecast.tools'`.

- [ ] **Step 3: Implement**

Create `src/doi/forecast/tools.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_tools.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/forecast/tools.py tests/doi/test_doi_forecast_tools.py
git commit -m "feat(doi): read-only tools of the forecast agent and their schemas"
```

---

### Task 4: The agent loop

**Files:**
- Create: `src/doi/forecast/agent.py`
- Test: `tests/doi/test_doi_forecast_agent.py`

**Interfaces:**
- Consumes: `ForecastCase`; `ForecastResult` (`src/doi/forecast/result.py`); `TOOLS`, `ANSWER`, `call_tool` (Task 3); `Chat`, `ChatResponse`, `FakeChatClient` (Task 1); `ReplayMiss` (Task 2).
- Produces: `PROMPT_VERSION = "forecast-v1"`, `MAX_TURNS = 4`, `MAX_TOOL_CALLS = 8`, `MAX_TOKENS = 400`, `FORCE_ANSWER`, `AGENT_MODES = ("tools", "single")`, `describe(case) -> str`, `single_prompt(case) -> str`, `SYSTEM_TOOLS`, `SYSTEM_SINGLE`, `run_agent(case: ForecastCase, chat: Chat, mode: str = "tools") -> ForecastResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_forecast_agent.py`:

```python
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
    assert PROMPT_VERSION == "forecast-v1" and AGENT_MODES == ("tools", "single")


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_agent.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.doi.forecast.agent'`.

- [ ] **Step 3: Implement**

Create `src/doi/forecast/agent.py`:

```python
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

PROMPT_VERSION = "forecast-v1"
MAX_TURNS = 4               # model responses per forecast
MAX_TOOL_CALLS = 8          # tool calls per forecast, not counting `answer`
MAX_TOKENS = 400
REASON_CHARS = 200
AGENT_MODES = ("tools", "single")
FORCE_ANSWER = {"type": "function", "function": {"name": "answer"}}

SYSTEM_TOOLS = """You advise one warehouse robot. The robot can move an obstacle out of the way now, which costs a one-off
price, or keep walking round it. Give one forecast: over the rest of this shift, will the whole fleet's
travel saving from moving this obstacle reach the price?

You know only what this robot knows. Use the tools.
- read_notices: messages the robot has received about future work. They can be wrong or irrelevant.
- ledger_summary: the traffic recorded so far and how many trips are still to come.
- trip_saving: how much one trip between two zones would save if the obstacle were moved.

Past traffic may not continue if a notice says the work is moving. A notice that does not change where
robots travel should not change your forecast. When you are ready, call answer. You have 4 turns."""

SYSTEM_SINGLE = """You advise one warehouse robot. The robot can move an obstacle out of the way now, which costs a one-off
price, or keep walking round it. Give one forecast: over the rest of this shift, will the whole fleet's
travel saving from moving this obstacle reach the price?

You know only what this robot knows, which is given below. Notices can be wrong or irrelevant.

Past traffic may not continue if a notice says the work is moving. A notice that does not change where
robots travel should not change your forecast. Call answer."""


def describe(case: ForecastCase) -> str:
    """The candidate action and the zones. Hidden fields (robot, tick, truth, numeric forecast) are not here."""
    lines = [f"Candidate action: {case.mode} the {case.kind} at {tuple(case.obstacle)} so that it ends at "
             f"{tuple(case.landing)}.",
             f"Price of the move: {case.price:g}.",
             f"Saving known to this robot so far: {case.known_saving:g}."]
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
        except Exception:
            return tally.result(None, None, "", "client_error", used)
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
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_agent.py -q`
Expected: `23 passed` (12 plain tests and 11 cases of the malformed-answer test).

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/forecast/agent.py tests/doi/test_doi_forecast_agent.py
git commit -m "feat(doi): the forecast agent loop with a turn and tool budget and typed failures"
```

---

### Task 5: The model forecaster, config, and a full run with record and replay

**Files:**
- Modify: `src/doi/forecast/forecasters.py`
- Modify: `src/doi/config.py`
- Modify: `src/doi/policies.py` (one line)
- Test: `tests/doi/test_doi_llm_forecaster.py`

**Interfaces:**
- Consumes: `run_agent`, `PROMPT_VERSION`, `AGENT_MODES` (Task 4); `CachedChat` (Task 2); `client_from_env` (Task 1); `SimConfig` (Part 1: `forecaster`, `agent_max_forecasts`); `GuardedPolicy(lam, forecaster=None)` (Part 1).
- Produces: `LlmForecaster(model_key: str, chat: Chat, mode: str = "tools")` with `name = "llm:<key>"`, `chat`, `mode`, `forecast(case)`; `make_forecaster(name: str, cfg: Optional[SimConfig] = None)` (the `llm:<key>` branch needs `cfg`); `SimConfig.agent_mode: str = "tools"`, `agent_cache: str = "experiments/results/doi/agent_cache"`, `agent_live: bool = False`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_llm_forecaster.py`:

```python
import json

import pytest
from src.doi.config import SimConfig
from src.doi.forecast.agent import PROMPT_VERSION
from src.doi.forecast.forecasters import LlmForecaster, make_forecaster
from src.doi.llm.chatcache import CachedChat, ReplayMiss
from src.doi.llm.client import FakeChatClient
from src.doi.policies import GuardedPolicy
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(messages, tools):
    """Read the notices, then say yes if the robot has any."""
    last = messages[-1]
    if last["role"] != "tool":
        return {"role": "assistant", "content": None, "tool_calls": [call("read_notices", {})]}
    has_notices = bool(json.loads(last["content"])["notices"])
    return {"role": "assistant", "content": None,
            "tool_calls": [call("answer", {"will_pay": has_notices, "confidence": 0.7, "reason": "notices"})]}


def fingerprint(res):
    return (res.J, res.J_censored, res.ticks, res.pushes, res.forecast_log, res.forecast_cases)


def episode(tmp_path, chat, mode="true", scenario="shift_notice", **cfg_kw):
    cfg = SimConfig(scenario=scenario, n_robots=6, tasks_per_robot=12, seed=3, policy="rof_a",
                    forecaster="llm:fake", agent_cache=str(tmp_path), scenario_params={"notice_mode": mode}, **cfg_kw)
    if scenario != "shift_notice":
        cfg = cfg.replace(scenario_params={}, tasks_per_robot=6, n_robots=6, seed=2)
    policy = GuardedPolicy(0.5, LlmForecaster("fake", chat, "tools"))
    return run_episode(cfg, scenario=build_scenario(cfg), policy=policy)


def test_config_knows_the_agent_settings():
    cfg = SimConfig()
    assert (cfg.agent_mode, cfg.agent_cache, cfg.agent_live) == ("tools", "experiments/results/doi/agent_cache", False)
    assert SimConfig(agent_mode="single", agent_live=True).agent_mode == "single"
    with pytest.raises(ValueError, match="agent_mode"):
        SimConfig(agent_mode="chat")


def test_make_forecaster_builds_the_llm_forecaster_without_credentials_for_replay(tmp_path, monkeypatch):
    for var in ("DOI_LLM_SMALL_URL", "DOI_LLM_SMALL_MODEL", "OPENAI_API_KEY", "DOI_LLM_SMALL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = SimConfig(forecaster="llm:small", agent_cache=str(tmp_path), agent_mode="single")
    forecaster = make_forecaster("llm:small", cfg)
    assert forecaster.name == "llm:small" and forecaster.mode == "single"
    assert forecaster.chat.client is None and forecaster.chat.live is False and forecaster.chat.prompt_version == PROMPT_VERSION
    with pytest.raises(ValueError, match="llm"):
        make_forecaster("llm:small")                                  # no config: nothing says where the store is
    with pytest.raises(ValueError, match="crystal"):
        make_forecaster("crystal_ball", cfg)


def test_make_forecaster_live_needs_the_environment(tmp_path, monkeypatch):
    cfg = SimConfig(forecaster="llm:small", agent_cache=str(tmp_path), agent_live=True)
    monkeypatch.delenv("DOI_LLM_SMALL_URL", raising=False)
    with pytest.raises(RuntimeError, match="DOI_LLM_SMALL_URL"):
        make_forecaster("llm:small", cfg)
    monkeypatch.setenv("DOI_LLM_SMALL_URL", "http://localhost:1/v1")
    monkeypatch.setenv("DOI_LLM_SMALL_MODEL", "tiny")
    forecaster = make_forecaster("llm:small", cfg)
    assert forecaster.chat.client.model_id == "tiny" and forecaster.chat.live is True


def test_a_live_run_then_the_same_run_replayed_gives_the_same_result_and_no_calls(tmp_path):
    client = FakeChatClient(reply, latency_s=1.0)
    live = CachedChat(client, str(tmp_path), "fake", True, PROMPT_VERSION)
    first = episode(tmp_path, live)
    assert client.calls > 0 and client.calls == live.live_calls and first.forecast_log
    answered = [r for r in first.forecast_log if r["answer"] is not None]
    assert answered and all(r["due"] > r["tick"] for r in answered)             # 2 s of latency is 4 ticks
    assert all(r["calls"] == 2 and r["tool_calls"] == 1 and r["forecaster"] == "llm:fake" for r in answered)
    replay = CachedChat(None, str(tmp_path), "fake", False, PROMPT_VERSION)
    second = episode(tmp_path, replay)
    assert fingerprint(second) == fingerprint(first)
    assert replay.live_calls == 0 and replay.hits > 0 and client.calls == live.live_calls


def test_a_miss_in_the_middle_of_a_run_stops_it(tmp_path):
    with pytest.raises(ReplayMiss, match="--agent-live"):
        episode(tmp_path, CachedChat(None, str(tmp_path), "fake", False, PROMPT_VERSION))
    live = CachedChat(FakeChatClient(reply, latency_s=1.0), str(tmp_path), "fake", True, PROMPT_VERSION)
    episode(tmp_path, live)
    lines = open(live.path).read().splitlines()
    with open(live.path, "w") as f:                                  # lose the last stored call
        f.write("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ReplayMiss):
        episode(tmp_path, CachedChat(None, str(tmp_path), "fake", False, PROMPT_VERSION))


def test_a_model_that_always_fails_leaves_the_classical_rule(tmp_path):
    class Down:
        model_id = "down"

        def chat(self, *a):
            raise TimeoutError("down")
    rof = run_episode(SimConfig(scenario="single_block", n_robots=6, tasks_per_robot=6, seed=2, policy="rof"))
    out = episode(tmp_path, Down(), scenario="single_block")
    assert out.J == rof.J == 519.0 and out.pushes == rof.pushes
    assert out.forecast_log and all(r["failed"] == "client_error" and r["answer"] is None for r in out.forecast_log)
    assert not out.stalled and out.unfinished_tasks == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_llm_forecaster.py -q`
Expected: failures, `ImportError: cannot import name 'LlmForecaster'`.

- [ ] **Step 3: Implement**

In `src/doi/config.py`:

Directly after the `FORECASTERS = (...)` line add:

```python
AGENT_MODES = ("tools", "single")
```

Add these three fields directly after the `agent_max_forecasts` field:

```python
    agent_mode: str = "tools"            # rof_a with an llm forecaster: tools | single
    agent_cache: str = "experiments/results/doi/agent_cache"     # where model calls are stored and replayed from
    agent_live: bool = False             # allow a miss to call the model (spends money); off: a miss is an error
```

Add this check directly after the `agent_max_forecasts` check:

```python
            (self.agent_mode in AGENT_MODES, f"bad agent_mode {self.agent_mode!r}: choose tools or single"),
```

In `src/doi/forecast/forecasters.py`:

Add to the imports (below the existing `from typing import Optional`, merging with it):

```python
from typing import TYPE_CHECKING, Optional

from src.doi.forecast.agent import PROMPT_VERSION, run_agent
from src.doi.llm.chatcache import CachedChat
from src.doi.llm.client import Chat
```

and, after the existing imports:

```python
if TYPE_CHECKING:
    from src.doi.config import SimConfig
```

Add this class directly above `FORECASTERS = {...}`:

```python
class LlmForecaster:
    """A language model behind the agent loop. `chat` is a CachedChat in a run, a fake in a test."""

    def __init__(self, model_key: str, chat: Chat, mode: str = "tools") -> None:
        self.name = f"llm:{model_key}"
        self.chat = chat
        self.mode = mode

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return run_agent(case, self.chat, self.mode)
```

Replace the function `make_forecaster` with:

```python
def make_forecaster(name: str, cfg: "Optional[SimConfig]" = None):
    """A forecaster by name. `llm:<key>` needs the run's config: it says where the store is, whether a miss may
    call the model, and the agent mode. Replay needs no credentials; only a live cache builds the real client."""
    if name.startswith("llm:"):
        if cfg is None:
            raise ValueError(f"the llm forecaster {name!r} needs the run's SimConfig (store, mode, live)")
        key = name.split(":", 1)[1]
        client = None
        if cfg.agent_live:
            from src.doi.llm.client import client_from_env
            client = client_from_env(key)
        return LlmForecaster(key, CachedChat(client, cfg.agent_cache, key, cfg.agent_live, PROMPT_VERSION),
                             cfg.agent_mode)
    if name not in FORECASTERS:
        raise ValueError(f"unknown forecaster {name!r}: choose from {', '.join(sorted(FORECASTERS))} or llm:<key>")
    return FORECASTERS[name]()
```

In `src/doi/policies.py`, in `GuardedPolicy.prepare`, replace

```python
        self.forecaster = self._given or make_forecaster(cfg.forecaster)
```

with:

```python
        self.forecaster = self._given or make_forecaster(cfg.forecaster, cfg)
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_llm_forecaster.py tests/doi/test_doi_forecasters.py tests/doi/test_doi_guarded_policy.py -q`
Expected: all pass (the two Part 1 files must pass unchanged; `make_forecaster("llm:small")` with no config still raises `ValueError` matching `llm`).

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/forecast/forecasters.py src/doi/config.py src/doi/policies.py tests/doi/test_doi_llm_forecaster.py
git commit -m "feat(doi): the llm forecaster for rof_a, with agent settings and a replayable run"
```

---

### Task 6: Command-line flags

**Files:**
- Modify: `run_doi.py`
- Modify: `src/doi/narrate.py`
- Test: `tests/doi/test_doi_cli_agent.py`

**Interfaces:**
- Consumes: `SimConfig.forecaster`, `agent_mode`, `agent_cache`, `agent_live` (Tasks 5 and Part 1); `ReplayMiss` (Task 2).
- Produces: flags `--forecaster NAME` (default `numeric`), `--agent-mode tools|single` (default `tools`), `--agent-cache DIR`, `--agent-live`; `run_doi.main` returns 2 and prints the miss to stderr on a `ReplayMiss`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_cli_agent.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import run_doi

BASE = ["--scenario", "shift_notice", "--robots", "6", "--tasks", "12", "--seed", "3", "--no-show"]


def test_rof_a_runs_from_the_command_line_with_a_named_forecaster(capsys):
    assert run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "oracle"]) == 0
    out = capsys.readouterr().out
    assert "rof_a" in out and "STALLED" not in out


def test_agent_mode_is_accepted_and_a_bad_one_is_refused(capsys):
    assert run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "keyword", "--agent-mode", "single"]) == 0
    capsys.readouterr()
    try:
        run_doi.main(BASE + ["--policy", "rof_a", "--agent-mode", "chat"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("a bad --agent-mode should exit with a usage error")


def test_an_empty_store_stops_the_run_with_the_remedy_and_no_traceback(capsys, tmp_path):
    code = run_doi.main(BASE + ["--policy", "rof_a", "--forecaster", "llm:small", "--agent-cache", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2 and "--agent-live" in captured.err and "Traceback" not in captured.err


def test_the_new_flags_are_in_the_guide(capsys):
    assert run_doi.main(["--guide"]) == 0
    out = capsys.readouterr().out
    for flag in ("--forecaster", "--agent-mode", "--agent-live", "--agent-cache"):
        assert flag in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_cli_agent.py -q`
Expected: failures, `error: unrecognized arguments: --forecaster oracle` (a `SystemExit`), and the guide test fails on the missing flags.

- [ ] **Step 3: Implement**

In `run_doi.py`:

Add to the imports, below `from src.doi.runner import run_episode`:

```python
from src.doi.llm.chatcache import ReplayMiss
```

In `build(args)`, replace the end of the main `SimConfig(...)` call (the line `bundle_max=getattr(args, "bundle_max", 1), lam=getattr(args, "lam", 0.5))` that is followed by `if args.demo == "fleet":`) with:

```python
        bundle_max=getattr(args, "bundle_max", 1), lam=getattr(args, "lam", 0.5),
        forecaster=getattr(args, "forecaster", "numeric"), agent_mode=getattr(args, "agent_mode", "tools"),
        agent_cache=getattr(args, "agent_cache", None) or SimConfig().agent_cache,
        agent_live=getattr(args, "agent_live", False))
```

Add these four arguments directly after the `--lam` argument:

```python
    ap.add_argument("--forecaster", default="numeric",
                    help="rof_a: numeric | keyword | oracle | inverted | llm:<model key>")
    ap.add_argument("--agent-mode", choices=["tools", "single"], default="tools",
                    help="rof_a with an llm forecaster: tools (the model asks for what it needs) or single (one call)")
    ap.add_argument("--agent-cache", default=None, metavar="DIR",
                    help="where model calls are stored and replayed from (default experiments/results/doi/agent_cache)")
    ap.add_argument("--agent-live", action="store_true",
                    help="let a call that is not in the store reach the model (spends money; default: an error)")
```

Replace the arm loop

```python
    for name in names:
        t0 = time.time()
        runs[name] = run_episode(cfg.replace(policy=name), scenario=scenario)
        runs[name].wall_s = time.time() - t0
```

with:

```python
    for name in names:
        t0 = time.time()
        try:
            runs[name] = run_episode(cfg.replace(policy=name), scenario=scenario)
        except ReplayMiss as miss:
            print(f"error: {miss}", file=sys.stderr)
            return 2
        runs[name].wall_s = time.time() - t0
```

In `src/doi/narrate.py`, in `FLAG_GUIDE`, directly after the `--lam X` entry (the one ending `"The multiplier is 1/X when the forecast says it will not; 1 makes rof_p behave like rof."),`) add:

```python
        ("--forecaster NAME", "rof_a only: who answers \"will this move pay?\": numeric, keyword, oracle, inverted "
                              "or llm:<model key>.",
         "oracle and inverted are the best and worst a forecaster can be; an llm forecaster reads the notices."),
        ("--agent-mode tools|single", "rof_a with an llm forecaster: the model looks things up with tools (default) "
                                      "or gets everything in one call.",
         "single is the ablation that shows what the tools add."),
        ("--agent-cache DIR", "Where model calls are stored and replayed from.",
         "A run replays stored calls and makes none; a call that is not stored is an error."),
        ("--agent-live", "Let a call that is not stored reach the model.",
         "The only flag that spends money. Check the estimate the experiment scripts print first."),
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_cli_agent.py tests/doi/test_doi_cli.py -q`
Expected: all pass.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add run_doi.py src/doi/narrate.py tests/doi/test_doi_cli_agent.py
git commit -m "feat(doi): --forecaster, --agent-mode, --agent-cache and --agent-live"
```

---

### Task 7: The case collector

**Files:**
- Create: `experiments/doi_agent_cases.py`
- Modify: `.gitignore`
- Test: `tests/doi/test_doi_agent_cases.py`

**Interfaces:**
- Consumes: `run_episode`, `SimConfig` with `forecaster="oracle"`, scenario `shift_notice` (Part 1); `case_to_dict`, `case_from_dict` (Part 1).
- Produces: `experiments/doi_agent_cases.py` with `SPLITS`, `MODES`, `collect(task) -> List[dict]`, `main(argv=None) -> int`; files `<out>/dev.jsonl`, `test.jsonl`, `human.jsonl`, each line `{"meta": {"split", "seed", "mode", "bank", "stalled"}, "case": <case_to_dict>}` with distinct `case_id`s.

- [ ] **Step 1: Write the failing test**

Create `tests/doi/test_doi_agent_cases.py`:

```python
import json
import os
import subprocess
import sys

from src.doi.forecast.case import case_from_dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "experiments", "doi_agent_cases.py")


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=ROOT)


def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_quick_dev_collection_writes_labelled_distinct_cases(tmp_path):
    proc = _run("--quick", "--splits", "dev", "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    rows = _rows(tmp_path / "dev.jsonl")
    assert rows and all(r["meta"]["split"] == "dev" and r["meta"]["bank"] == "dev" for r in rows)
    assert {r["meta"]["mode"] for r in rows} == {"true", "false", "missing", "quiet"}
    assert {r["meta"]["seed"] for r in rows} <= {0, 1}
    cases = [case_from_dict(r["case"]) for r in rows]
    assert all(c.truth is not None and c.zones and c.ledger for c in cases)
    assert len({c.case_id for c in cases}) == len(cases)
    assert any(c.notices for c in cases) and any(not c.notices for c in cases)       # notices in some modes only
    assert "dev:" in proc.stdout and "no model was called" in proc.stdout


def test_the_human_split_is_skipped_without_the_notice_file(tmp_path):
    proc = _run("--quick", "--splits", "human", "--out", str(tmp_path),
                "--human-notices", str(tmp_path / "absent.jsonl"))
    assert proc.returncode == 0, proc.stderr
    assert "human split skipped" in proc.stdout and "level 2" in proc.stdout
    assert not (tmp_path / "human.jsonl").exists()


def test_the_human_split_reads_the_people_written_texts(tmp_path):
    texts = tmp_path / "human_notices.jsonl"
    rows = [{"kind": "surge", "group": "south bays", "text": "orders are all piling into the south side"},
            {"kind": "surge", "group": "north bays", "text": "orders are all piling into the north side"},
            {"kind": "drop", "group": "south bays", "text": "south side is done for the day"},
            {"kind": "drop", "group": "north bays", "text": "north side is done for the day"},
            {"kind": "distractor", "group": "", "text": "someone left a jacket in the canteen"}]
    texts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = _run("--quick", "--splits", "human", "--out", str(tmp_path), "--human-notices", str(texts))
    assert proc.returncode == 0, proc.stderr
    cases = [case_from_dict(r["case"]) for r in _rows(tmp_path / "human.jsonl")]
    said = " ".join(text for c in cases for _, _, text in c.notices)
    assert "piling into" in said or "done for the day" in said
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/doi/test_doi_agent_cases.py -q`
Expected: FAIL, the script does not exist (`returncode` 2).

- [ ] **Step 3: Write the script**

Create `experiments/doi_agent_cases.py`:

```python
"""Collect forecast cases: every forecast `rof_a` requests on `shift_notice`, with the truth label. Calls no model.

The oracle forecaster plays the run, so the cases are the ones a good policy meets. One file per split:

  dev    seeds 0..19      notice bank dev     prompt development only
  test   seeds 100..149   notice bank test    held-out scoring
  human  seeds 300..319   notice bank human   held-out scoring, always reported apart

Seeds 200..229 are reserved for E9 and appear in no file. The `human` split needs data/forecasts/human_notices.jsonl
(see data/forecasts/README.md); without it the split is skipped and level 2 is not done.

Usage: venv/bin/python experiments/doi_agent_cases.py [--quick] [--splits dev,test,human] [--out DIR]
                                                      [--human-notices PATH] [--jobs 4]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.forecast.case import case_to_dict
from src.doi.runner import run_episode

SPLITS = {"dev": (range(0, 20), "dev"), "test": (range(100, 150), "test"), "human": (range(300, 320), "human")}
MODES = ("true", "false", "missing", "quiet")
DEFAULT_OUT = os.path.join(ROOT, "data", "forecasts")
DEFAULT_HUMAN = os.path.join(ROOT, "data", "forecasts", "human_notices.jsonl")


def collect(task) -> List[dict]:
    """One run of rof_a with the oracle forecaster; one row per forecast it requested."""
    split, seed, mode, bank, human = task
    cfg = SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, lam=0.5, bundle_max=1,
                    policy="rof_a", forecaster="oracle",
                    scenario_params={"notice_mode": mode, "notice_bank": bank, "human_notices": human})
    res = run_episode(cfg)
    meta = {"split": split, "seed": seed, "mode": mode, "bank": bank, "stalled": bool(res.stalled)}
    return [{"meta": meta, "case": case_to_dict(c)} for c in res.forecast_cases]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the first 2 seeds of each split")
    ap.add_argument("--splits", default="dev,test,human")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--human-notices", default=DEFAULT_HUMAN)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLITS:
            ap.error(f"unknown split {split!r}: choose from {', '.join(SPLITS)}")
        seeds, bank = SPLITS[split]
        if bank == "human" and not os.path.exists(args.human_notices):
            print(f"human split skipped: {args.human_notices} not found (see data/forecasts/README.md); "
                  f"level 2 is not done")
            continue
        seeds = list(seeds)[:2] if args.quick else list(seeds)
        tasks = [(split, seed, mode, bank, args.human_notices) for seed in seeds for mode in MODES]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                batches = list(pool.map(collect, tasks))
        else:
            batches = [collect(t) for t in tasks]
        seen, rows = set(), []
        for row in (r for batch in batches for r in batch):
            if row["case"]["case_id"] not in seen:
                seen.add(row["case"]["case_id"])
                rows.append(row)
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        total = sum(len(b) for b in batches)
        share = sum(1 for r in rows if r["case"]["truth"]) / max(1, len(rows))
        print(f"{split}: {len(rows)} distinct cases from {total} requests in {len(tasks)} runs; "
              f"truth is yes in {share:.2f} of them; wrote {path}")
    print("no model was called")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Append these lines to `.gitignore`:

```
data/forecasts/dev.jsonl
data/forecasts/test.jsonl
data/forecasts/human.jsonl
```

- [ ] **Step 4: Run the test**

Run: `venv/bin/python -m pytest tests/doi/test_doi_agent_cases.py -q`
Expected: `3 passed` (about a minute: 24 simulation runs).

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add experiments/doi_agent_cases.py tests/doi/test_doi_agent_cases.py .gitignore
git commit -m "feat(doi): collect forecast cases with their truth label into dev, test and human files"
```

---

### Task 8: The cost estimate and the scoring script

**Files:**
- Create: `src/doi/forecast/budget.py`
- Create: `experiments/doi_agent_eval.py`
- Test: `tests/doi/test_doi_agent_eval.py`

**Interfaces:**
- Consumes: `case_from_dict` and `ForecastCase` (Part 1); `make_forecaster`, `LlmForecaster` (Task 5); `MAX_TURNS` (Task 4); `ReplayMiss` (Task 2).
- Produces: `call_budget(requests, calls_per_forecast) -> dict`, `confirm_calls(requests, calls_per_forecast, yes=False, confirm=input, say=print) -> bool` in `src/doi/forecast/budget.py`; in `doi_agent_eval.py`: `load_cases(path) -> List[Tuple[dict, ForecastCase]]`, `ece(pairs, bins=10) -> float`, `evaluate(name, forecaster, cases, tick_seconds) -> dict`, `main(argv=None, confirm=input) -> int`; file `<out>/eval.csv`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_agent_eval.py`:

```python
import dataclasses
import importlib.util
import json
import math
import os

import pytest
from src.doi.forecast.budget import call_budget, confirm_calls
from src.doi.forecast.case import ForecastCase, case_to_dict
from src.doi.forecast.forecasters import LlmForecaster, NumericForecaster, OracleForecaster, InvertedForecaster
from src.doi.llm.client import FakeChatClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _module():
    spec = importlib.util.spec_from_file_location("doi_agent_eval", os.path.join(ROOT, "experiments", "doi_agent_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = ForecastCase(case_id="x", robot=0, tick=1, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
                    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0, notices=(), ledger={},
                    zones={"west north bays": {"rows": (0, 6), "cols": (0, 9)},
                           "west south bays": {"rows": (8, 14), "cols": (0, 9)}},
                    trip_saving={}, numeric_forecast=False, truth=True)


def _cases(path):
    # truth yes, yes, no, no; the numeric forecast says yes, no, no, yes: two right, one wrong yes, one wrong no
    spec = [(True, True), (True, False), (False, False), (False, True)]
    with open(path, "w") as f:
        for k, (truth, numeric) in enumerate(spec):
            case = dataclasses.replace(BASE, case_id=f"c{k}", truth=truth, numeric_forecast=numeric)
            f.write(json.dumps({"meta": {"split": "dev", "seed": k, "mode": "true", "bank": "dev", "stalled": False},
                                "case": case_to_dict(case)}) + "\n")


def test_budget_prints_the_bound_and_asks_unless_yes():
    assert call_budget(10, 9) == {"requests": 10, "calls": 90, "prompt_tokens": 90 * 1200, "completion_tokens": 90 * 80}
    said = []
    assert confirm_calls(10, 9, yes=True, say=said.append) is True and "at most 90 model calls" in said[0]
    assert confirm_calls(10, 9, confirm=lambda prompt: "y", say=said.append) is True
    assert confirm_calls(10, 9, confirm=lambda prompt: "n", say=said.append) is False
    asked = []
    assert confirm_calls(0, 9, confirm=lambda prompt: asked.append(prompt) or "n", say=said.append) is True
    assert asked == []                                               # nothing to spend, nothing to ask


def test_ece_by_hand():
    mod = _module()
    pairs = [(0.9, True), (0.9, False), (0.6, True)]
    assert mod.ece(pairs) == pytest.approx(0.4)                      # (2/3)*|0.5-0.9| + (1/3)*|1-0.6|
    assert math.isnan(mod.ece([]))
    assert mod.ece([(1.0, True), (1.0, True)]) == 0.0                # confidence 1.0 lands in the last bin


def test_evaluate_plain_forecasters_by_hand(tmp_path):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    cases = mod.load_cases(str(tmp_path / "cases.jsonl"))
    assert len(cases) == 4 and cases[0][0]["seed"] == 0
    numeric = mod.evaluate("numeric", NumericForecaster(), cases, 0.5)
    assert (numeric["n"], numeric["answered"], numeric["truth_share"]) == (4, 4, 0.5)
    assert (numeric["accuracy"], numeric["wrong_yes_rate"], numeric["wrong_no_rate"]) == (0.5, 0.25, 0.25)
    assert numeric["failure_rate"] == 0.0 and math.isnan(numeric["ece"]) and numeric["calls_per_forecast"] == 0.0
    assert mod.evaluate("oracle", OracleForecaster(), cases, 0.5)["accuracy"] == 1.0
    assert mod.evaluate("inverted", InvertedForecaster(), cases, 0.5)["accuracy"] == 0.0


def _call(name, args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_evaluate_a_model_counts_calls_latency_failures_and_calibration(tmp_path):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    cases = mod.load_cases(str(tmp_path / "cases.jsonl"))
    seen = {"n": 0}

    def reply(messages, tools):
        seen["n"] += 1
        if messages[-1]["role"] != "tool":
            return {"role": "assistant", "content": None, "tool_calls": [_call("read_notices", {})]}
        case_no = (seen["n"] // 2) - 1                               # replies alternate: tool call, then answer
        if case_no == 3:                                             # the fourth case: a malformed answer
            return {"role": "assistant", "content": None, "tool_calls": [_call("answer", {"will_pay": "maybe"})]}
        return {"role": "assistant", "content": None,
                "tool_calls": [_call("answer", {"will_pay": case_no < 3, "confidence": 0.9, "reason": "r"})]}

    model = LlmForecaster("fake", FakeChatClient(reply, latency_s=1.0), "tools")
    row = mod.evaluate("llm:fake", model, cases, 0.5)
    assert (row["n"], row["answered"], row["failure_rate"], row["fail_bad_answer"]) == (4, 3, 0.25, 1)
    assert row["accuracy"] == pytest.approx(2 / 3) and row["wrong_yes_rate"] == pytest.approx(1 / 3)
    assert row["ece"] == pytest.approx(abs(2 / 3 - 0.9))
    assert row["calls_per_forecast"] == 2.0 and row["tool_calls_per_forecast"] == 1.0
    assert row["latency_p50_s"] == 2.0 and row["latency_p50_ticks"] == 4.0 and row["prompt_tokens_per_forecast"] > 0


def test_quick_run_scores_the_baselines_and_writes_a_file(tmp_path):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    out = tmp_path / "out"
    assert mod.main(["--quick", "--cases", str(tmp_path / "cases.jsonl"), "--out", str(out)]) == 0
    import pandas as pd
    table = pd.read_csv(out / "eval.csv")
    assert list(table.forecaster) == ["numeric", "keyword", "oracle", "inverted"]
    assert table.set_index("forecaster").loc["oracle", "accuracy"] == 1.0


def test_an_unfilled_store_stops_the_scoring_with_the_remedy(tmp_path, capsys):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    code = mod.main(["--cases", str(tmp_path / "cases.jsonl"), "--out", str(tmp_path / "o"), "--forecasters",
                     "numeric,llm:small", "--cache", str(tmp_path / "store")])
    assert code == 2 and "--agent-live" in capsys.readouterr().err


def test_live_scoring_asks_first_and_an_answer_of_no_stops_it(tmp_path, capsys, monkeypatch):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    monkeypatch.setattr("src.doi.llm.client.client_from_env",
                        lambda key: (_ for _ in ()).throw(AssertionError("the client must not be built")))
    code = mod.main(["--cases", str(tmp_path / "cases.jsonl"), "--out", str(tmp_path / "o"), "--forecasters",
                     "llm:small", "--cache", str(tmp_path / "store"), "--live"], confirm=lambda prompt: "n")
    assert code == 1 and "at most 16 model calls" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_agent_eval.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.doi.forecast.budget'`.

- [ ] **Step 3: Implement**

Create `src/doi/forecast/budget.py`:

```python
"""A bound on what a script that can call a model would spend, and the question to ask before it does."""
from typing import Callable, Dict

EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS = 1200, 80       # per call; a rough upper figure for the tools prompt


def call_budget(requests: int, calls_per_forecast: int) -> Dict[str, int]:
    calls = requests * calls_per_forecast
    return {"requests": requests, "calls": calls, "prompt_tokens": calls * EST_PROMPT_TOKENS,
            "completion_tokens": calls * EST_COMPLETION_TOKENS}


def confirm_calls(requests: int, calls_per_forecast: int, yes: bool = False, confirm: Callable[[str], str] = input,
                  say: Callable[[str], None] = print) -> bool:
    """Print the upper bound and ask. Nothing to spend asks nothing; --yes skips the question."""
    b = call_budget(requests, calls_per_forecast)
    say(f"{b['requests']} forecast requests, at most {b['calls']} model calls ({calls_per_forecast} per forecast); "
        f"about {b['prompt_tokens']} prompt and {b['completion_tokens']} completion tokens at most "
        f"(multiply by your model's price)")
    if b["calls"] == 0 or yes:
        return True
    return confirm("Proceed with these calls? [y/N] ").strip().lower() == "y"
```

Create `experiments/doi_agent_eval.py`:

```python
"""Score forecasters on a file of forecast cases (levels 1 to 3 of section 14.1 of the design).

Accuracy, the two kinds of wrong answer, failures by code, calibration of the stated confidence, and what a forecast
costs in calls, tokens and latency. The non-model forecasters are scored on the same cases as baselines. A model is
scored from the store; a call that is not stored is an error unless --live, which asks before it spends.

Usage: venv/bin/python experiments/doi_agent_eval.py --cases data/forecasts/test.jsonl
           [--forecasters numeric,keyword,oracle,inverted,llm:small] [--agent-mode tools|single]
           [--cache DIR] [--live] [--yes] [--quick] [--out DIR]
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.forecast.agent import MAX_TURNS
from src.doi.forecast.budget import confirm_calls
from src.doi.forecast.case import ForecastCase, case_from_dict
from src.doi.forecast.forecasters import make_forecaster
from src.doi.llm.chatcache import ReplayMiss

DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "agent_eval")
DEFAULT_FORECASTERS = "numeric,keyword,oracle,inverted"
CODES = ("no_tool_call", "no_answer", "bad_answer", "client_error")


def load_cases(path: str) -> List[Tuple[dict, ForecastCase]]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(r["meta"], case_from_dict(r["case"])) for r in rows]


def ece(pairs, bins: int = 10) -> float:
    """Expected calibration error of (confidence, answer was right) pairs, in equal-width bins."""
    if not pairs:
        return float("nan")
    total = 0.0
    for b in range(bins):
        group = [(c, ok) for c, ok in pairs if min(int(c * bins), bins - 1) == b]
        if group:
            total += len(group) / len(pairs) * abs(sum(ok for _, ok in group) / len(group)
                                                   - sum(c for c, _ in group) / len(group))
    return total


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def evaluate(name: str, forecaster, cases, tick_seconds: float) -> dict:
    results = [forecaster.forecast(case) for _, case in cases]
    n = len(results)
    scored = [(r, c) for r, (_, c) in zip(results, cases) if r.answer is not None and c.truth is not None]
    right = [r.answer == c.truth for r, c in scored]
    fails = Counter(r.failed for r in results if r.failed)
    latency = [r.latency_s for r in results]
    p50 = float(np.percentile(latency, 50)) if latency else float("nan")
    p95 = float(np.percentile(latency, 95)) if latency else float("nan")
    row = {
        "forecaster": name, "n": n, "answered": len(scored),
        "truth_share": _mean(c.truth for _, c in cases if c.truth is not None),
        "accuracy": _mean(right),
        "wrong_yes_rate": _mean(1.0 if r.answer and not c.truth else 0.0 for r, c in scored),
        "wrong_no_rate": _mean(1.0 if not r.answer and c.truth else 0.0 for r, c in scored),
        "failure_rate": sum(fails.values()) / n if n else float("nan"),
        "ece": ece([(r.confidence, ok) for (r, _), ok in zip(scored, right) if r.confidence is not None]),
        "calls_per_forecast": _mean(r.calls for r in results),
        "tool_calls_per_forecast": _mean(r.tool_calls for r in results),
        "latency_p50_s": p50, "latency_p95_s": p95,
        "latency_p50_ticks": p50 / tick_seconds, "latency_p95_ticks": p95 / tick_seconds,
        "prompt_tokens_per_forecast": _mean(r.prompt_tokens for r in results),
        "completion_tokens_per_forecast": _mean(r.completion_tokens for r in results),
    }
    for code in CODES:
        row[f"fail_{code}"] = fails[code]
    return row


def main(argv=None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--forecasters", default=DEFAULT_FORECASTERS)
    ap.add_argument("--agent-mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--cache", default=None, help="the model-call store (default: the SimConfig default)")
    ap.add_argument("--live", action="store_true", help="let a call that is not stored reach the model")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--quick", action="store_true", help="the first 20 cases; a model only if its store has calls")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    if args.quick:
        cases = cases[:20]
    names = [n.strip() for n in args.forecasters.split(",") if n.strip()]
    cache = args.cache or SimConfig().agent_cache
    if args.quick:
        names = [n for n in names if not n.startswith("llm:")
                 or os.path.exists(os.path.join(cache, n.split(":", 1)[1], "chat.jsonl"))]
    models = [n for n in names if n.startswith("llm:")]
    if args.live and models:
        per = MAX_TURNS if args.agent_mode == "tools" else 1
        if not confirm_calls(len(cases) * len(models), per, args.yes, confirm):
            print("aborted")
            return 1
    rows = []
    for name in names:
        cfg = SimConfig(forecaster=name, agent_mode=args.agent_mode, agent_cache=cache, agent_live=args.live)
        try:
            rows.append(evaluate(name, make_forecaster(name, cfg), cases, cfg.tick_seconds))
        except ReplayMiss as miss:
            print(f"error: {miss}", file=sys.stderr)
            return 2
    os.makedirs(args.out, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "eval.csv"), index=False)
    shown = ["forecaster", "n", "answered", "truth_share", "accuracy", "wrong_yes_rate", "wrong_no_rate",
             "failure_rate", "ece", "calls_per_forecast", "latency_p50_ticks"]
    print(f"scored {len(cases)} cases from {args.cases}" + ("" if models else "; no model was called"))
    print(table[shown].round(3).to_string(index=False))
    print(f"wrote {os.path.join(args.out, 'eval.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_agent_eval.py -q`
Expected: `7 passed`.

If `test_evaluate_a_model_counts_calls_latency_failures_and_calibration` fails on the case numbering, the replies alternate tool call then answer per case, so case `k` is answered on reply `2k + 2`; the formula `seen["n"] // 2 - 1` gives `k` there. Fix the test's arithmetic, not the script.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/forecast/budget.py experiments/doi_agent_eval.py tests/doi/test_doi_agent_eval.py
git commit -m "feat(doi): scoring of forecasters on cases, and the call estimate that asks before spending"
```

---

### Task 9: The end-to-end experiment E9

**Files:**
- Create: `experiments/doi_e9_agent.py`
- Modify: `experiments/doi_e9_headroom.py` (`usable`)
- Test: `tests/doi/test_doi_e9_agent.py`

**Interfaces:**
- Consumes: `run_episode`, `build_scenario`, `SimConfig`, `summary_row` (existing); `GuardedPolicy`, `LlmForecaster`, `CachedChat`, `PROMPT_VERSION` (earlier tasks); `confirm_calls`, `MAX_TURNS` (Tasks 8, 4); `usable` (Part 1, extended here); `bootstrap_ci` (existing).
- Produces: `experiments/doi_e9_agent.py` with `make_cfg(seed, mode, lam, notice_tick, **kw) -> SimConfig`, `NON_MODEL_ARMS`, `model_arms(keys) -> list`, `store_has_calls(cache, key) -> bool`, `select_model_arms(keys, cache, live) -> (arms, notes)`, `run_point(task) -> List[dict]`, `summarise(df) -> pd.DataFrame`, `main(argv=None, confirm=input) -> int`; files `runs.csv`, `summary.csv`; `usable(df, point=POINT)` in `doi_e9_headroom.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_e9_agent.py`:

```python
import importlib.util
import json
import os
import sys

import pandas as pd
from src.doi.forecast.agent import PROMPT_VERSION
from src.doi.forecast.forecasters import LlmForecaster
from src.doi.llm.chatcache import CachedChat
from src.doi.llm.client import FakeChatClient
from src.doi.policies import GuardedPolicy
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(name, args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _reply(messages, tools):
    if messages[-1]["role"] != "tool":
        return {"role": "assistant", "content": None, "tool_calls": [_call("read_notices", {})]}
    has = bool(json.loads(messages[-1]["content"])["notices"])
    return {"role": "assistant", "content": None,
            "tool_calls": [_call("answer", {"will_pay": has, "confidence": 0.7, "reason": "r"})]}


def test_arms_follow_the_store_and_the_live_flag(tmp_path):
    e9 = _load("doi_e9_agent")
    assert [a[0] for a in e9.NON_MODEL_ARMS] == ["never", "rof", "rof_p", "numeric", "keyword", "oracle", "inverted"]
    arms = e9.model_arms(["small", "large"])
    assert [a[0] for a in arms] == ["llm:small", "llm:large", "llm:small single"]
    assert [a[3] for a in arms] == ["tools", "tools", "single"]
    chosen, notes = e9.select_model_arms(["small", "large"], str(tmp_path), live=False)
    assert chosen == [] and any("small" in n for n in notes)
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "chat.jsonl").write_text('{"key": "k"}\n')
    assert e9.store_has_calls(str(tmp_path), "small") and not e9.store_has_calls(str(tmp_path), "large")
    chosen, _ = e9.select_model_arms(["small", "large"], str(tmp_path), live=False)
    assert [a[0] for a in chosen] == ["llm:small"]                    # the single-call arm needs its own stored calls
    assert [a[0] for a in e9.select_model_arms(["small"], str(tmp_path), live=False, single=True)[0]] == \
        ["llm:small", "llm:small single"]
    assert len(e9.select_model_arms(["small", "large"], str(tmp_path), live=True)[0]) == 3


def test_usable_takes_the_columns_that_identify_a_point():
    headroom = _load("doi_e9_headroom")
    df = pd.DataFrame({"seed": [0, 0, 1, 1], "lam": [0.5, 0.5, 0.5, 0.5], "stalled": [False, True, False, False]})
    kept, dropped = headroom.usable(df, point=["seed", "lam"])
    assert dropped == 1 and list(kept.seed) == [1, 1]


def test_quick_run_without_a_model_writes_runs_and_a_summary(tmp_path, capsys):
    e9 = _load("doi_e9_agent")
    out = tmp_path / "e9"
    assert e9.main(["--quick", "--modes", "true", "--out", str(out), "--cache", str(tmp_path / "store")]) == 0
    runs = pd.read_csv(out / "runs.csv", dtype={"mode": str})
    assert sorted(runs.arm.unique()) == sorted(a[0] for a in e9.NON_MODEL_ARMS)
    assert sorted(runs.seed.unique()) == [200, 201, 202] and set(runs["mode"]) == {"true"}
    assert (runs.avoidable == runs.J - runs.J_free).all() and len(runs) == 3 * 7
    assert {"mode", "lam", "notice_tick", "compare", "median_diff", "lo", "hi", "n"} <= set(pd.read_csv(out / "summary.csv").columns)
    printed = capsys.readouterr().out
    assert "no model was called" in printed and "no stored calls for llm:small" in printed


def test_a_stored_model_arm_is_replayed_into_the_quick_run(tmp_path):
    e9 = _load("doi_e9_agent")
    store = str(tmp_path / "store")
    client = FakeChatClient(_reply, latency_s=1.0)
    chat = CachedChat(client, store, "fake", True, PROMPT_VERSION)
    for seed in (200, 201, 202):                                     # fill the store the way a live run would
        cfg = e9.make_cfg(seed, "true", 0.5, 5).replace(policy="rof_a", forecaster="llm:fake")
        run_episode(cfg, scenario=build_scenario(cfg), policy=GuardedPolicy(0.5, LlmForecaster("fake", chat, "tools")))
    spent = client.calls
    assert spent > 0
    out = tmp_path / "e9"
    assert e9.main(["--quick", "--modes", "true", "--model-keys", "fake", "--out", str(out), "--cache", store]) == 0
    runs = pd.read_csv(out / "runs.csv", dtype={"mode": str})
    assert set(runs.arm) - {a[0] for a in e9.NON_MODEL_ARMS} == {"llm:fake"}   # no single-call arm: nothing stored for it
    assert client.calls == spent                                                # replayed: the fake was not asked again
    assert (runs[runs.arm == "llm:fake"].forecasts > 0).all()


def test_live_asks_first_and_no_stops_it_before_any_client_is_built(tmp_path, capsys, monkeypatch):
    e9 = _load("doi_e9_agent")
    monkeypatch.setattr("src.doi.llm.client.client_from_env",
                        lambda key: (_ for _ in ()).throw(AssertionError("the client must not be built")))
    code = e9.main(["--quick", "--modes", "true", "--live", "--model-keys", "small", "--out", str(tmp_path / "o"),
                    "--cache", str(tmp_path / "store")], confirm=lambda prompt: "n")
    assert code == 1 and "model calls" in capsys.readouterr().out and not (tmp_path / "o" / "runs.csv").exists()


def test_a_store_with_a_gap_stops_the_run_with_the_remedy(tmp_path, capsys):
    e9 = _load("doi_e9_agent")
    (tmp_path / "store" / "small").mkdir(parents=True)
    (tmp_path / "store" / "small" / "chat.jsonl").write_text("{}\n")                # exists, but holds nothing usable
    code = e9.main(["--quick", "--modes", "true", "--model-keys", "small", "--out", str(tmp_path / "o"),
                    "--cache", str(tmp_path / "store")])
    assert code == 2 and "--agent-live" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_e9_agent.py -q`
Expected: failures, the script does not exist (`FileNotFoundError` from the loader).

- [ ] **Step 3: Implement**

In `experiments/doi_e9_headroom.py` replace the `usable` function with:

```python
def usable(df: pd.DataFrame, point: List[str] = POINT) -> Tuple[pd.DataFrame, int]:
    """The runs of every point (seed, mode, cost setting) in which no arm stalled, and how many points were dropped.
    `point` names the columns that identify a point."""
    bad = df.groupby(point)["stalled"].transform("any").astype(bool)
    return df[~bad], int(df[bad].groupby(point).ngroups)
```

Create `experiments/doi_e9_agent.py`:

```python
"""E9: does a forecast agent behind the guard lower fleet cost on shift_notice, and how bad is it when the notices
are wrong? (Section 14.3 of the design.)

Avoidable cost is J_censored minus the `free` arm's on the same seed. A point (seed, mode, lam, notice_tick) in
which any arm stalled is left out of every summary, as in the headroom pilot. Model arms replay stored calls and
call nothing; a call that is not stored is an error unless --live, which prints an upper bound and asks first.

Hypotheses (fixed before a full run; the margin of H9c is fixed after the --quick pilot):
  H9a  true notices:  rof_a with a model costs less than rof_a with numeric
  H9b  false notices: rof_a with a model costs no more than rof_a with inverted; its cost against rof is reported
  H9c  no notices (missing, quiet): rof_a with a model is within a fixed margin of rof_a with numeric
  H9d  is scored by doi_agent_eval.py on the human cases, not here

Usage: venv/bin/python experiments/doi_e9_agent.py [--quick] [--seeds 100] [--model-seeds 30] [--jobs 4]
           [--modes true,false,missing,quiet] [--model-keys small,large] [--with-single] [--cache DIR] [--live] [--yes]
           [--out DIR]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doi_e9_headroom import usable
from src.doi.config import SimConfig
from src.doi.forecast.agent import MAX_TURNS
from src.doi.forecast.budget import confirm_calls
from src.doi.llm.chatcache import ReplayMiss
from src.doi.metrics import summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario
from src.doi.stats import bootstrap_ci

NON_MODEL_ARMS = [("never", "never", "numeric", "tools"), ("rof", "rof", "numeric", "tools"),
                  ("rof_p", "rof_p", "numeric", "tools"), ("numeric", "rof_a", "numeric", "tools"),
                  ("keyword", "rof_a", "keyword", "tools"), ("oracle", "rof_a", "oracle", "tools"),
                  ("inverted", "rof_a", "inverted", "tools")]          # (label, policy, forecaster, agent mode)
MODES = ["true", "false", "missing", "quiet"]
PRIMARY = (0.5, 5)                                                       # (lam, notice_tick)
SECONDARY = [(0.25, 5), (1.0, 5), (0.5, 150)]                            # on the true and false modes only
POINT = ["seed", "mode", "lam", "notice_tick"]
FIRST_SEED = 200
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "e9_agent")


def make_cfg(seed: int, mode: str, lam: float, notice_tick: int, **kw) -> SimConfig:
    return SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, lam=lam, bundle_max=1,
                     scenario_params={"notice_mode": mode, "notice_tick": notice_tick}, **kw)


def model_arms(keys: List[str], single: bool = True) -> list:
    """llm:<key> with tools for every key, and (if `single`) the first key again as a single call."""
    arms = [(f"llm:{k}", "rof_a", f"llm:{k}", "tools") for k in keys]
    if single:
        arms.append((f"llm:{keys[0]} single", "rof_a", f"llm:{keys[0]}", "single"))
    return arms


def store_has_calls(cache: str, key: str) -> bool:
    path = os.path.join(cache, key, "chat.jsonl")
    return os.path.exists(path) and os.path.getsize(path) > 0


def select_model_arms(keys: List[str], cache: str, live: bool, single: bool = False) -> Tuple[list, List[str]]:
    """Model arms that can run: all of them when live, else only those whose key has stored calls. The single-call
    arm needs its own stored calls, so it runs when live or when `single` says they are there."""
    notes, usable_keys = [], []
    for k in keys:
        if live or store_has_calls(cache, k):
            usable_keys.append(k)
        else:
            notes.append(f"no stored calls for llm:{k}: its arms are left out (fill the store with --live)")
    if not usable_keys:
        return [], notes
    return model_arms(usable_keys, single=live or single), notes


def run_point(task) -> List[dict]:
    seed, mode, lam, notice_tick, arms, cache, live = task
    cfg = make_cfg(seed, mode, lam, notice_tick, agent_cache=cache, agent_live=live)
    scenario = build_scenario(cfg)
    free = run_episode(cfg.replace(policy="free"), scenario=scenario).J_censored
    rows = []
    for label, policy, forecaster, agent_mode in arms:
        res = run_episode(cfg.replace(policy=policy, forecaster=forecaster, agent_mode=agent_mode), scenario=scenario)
        cols = summary_row(res, cheap=True)
        rows.append({"seed": seed, "mode": mode, "lam": lam, "notice_tick": notice_tick, "arm": label,
                     "J": res.J_censored, "J_free": free, "avoidable": res.J_censored - free,
                     "removals": res.removals, "stalled": res.stalled, "forecasts": cols["forecasts"],
                     "forecast_acc": cols["forecast_acc"], "wrong_yes": cols["wrong_yes"],
                     "wrong_no": cols["wrong_no"], "agent_calls": cols["agent_calls"],
                     "agent_latency_s": cols["agent_latency_s"], "agent_prompt_tokens": cols["agent_prompt_tokens"],
                     "agent_completion_tokens": cols["agent_completion_tokens"]})
    return rows


def hypotheses(labels) -> list:
    """(hypothesis, mode, arm A, arm B): the difference A - B in avoidable cost; positive means B is cheaper."""
    out = []
    for model in sorted(l for l in labels if l.startswith("llm:") and "single" not in l):
        out += [("H9a", "true", "numeric", model), ("H9b", "false", "inverted", model), ("H9b", "false", "rof", model),
                ("H9c", "missing", "numeric", model), ("H9c", "quiet", "numeric", model)]
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Paired differences in avoidable cost for the hypotheses and for the headroom references."""
    kept, _ = usable(df, POINT)
    labels = set(kept.arm)
    pairs = [(h, m, a, b) for h, m, a, b in hypotheses(labels)]
    pairs += [("ref", m, a, b) for m in MODES for a, b in (("numeric", "oracle"), ("inverted", "oracle"),
                                                         ("rof", "numeric"), ("never", "rof"))]
    out = []
    for (lam, tick, mode), g in kept.groupby(["lam", "notice_tick", "mode"]):
        wide = g.pivot(index="seed", columns="arm", values="avoidable")
        for hyp, m, a, b in pairs:
            if m != mode or a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna().to_numpy(dtype=float)
            if len(d) == 0:
                continue
            med, lo, hi = bootstrap_ci(d, n=2000, seed=0)
            out.append({"lam": lam, "notice_tick": tick, "mode": mode, "hypothesis": hyp, "compare": f"{a} - {b}",
                        "mean_diff": float(d.mean()), "median_diff": med, "lo": lo, "hi": hi,
                        "share_positive": float((d > 0).mean()), "n": len(d)})
    return pd.DataFrame(out)


def count_requests(seeds, points, modes, cache) -> int:
    """Forecast requests of a run with the numeric forecaster: the number a model arm would make, roughly."""
    total = 0
    for seed in seeds:
        for mode, lam, tick in points(modes):
            cfg = make_cfg(seed, mode, lam, tick, policy="rof_a", forecaster="numeric", agent_cache=cache)
            res = run_episode(cfg, scenario=build_scenario(cfg))
            total += sum(1 for r in res.forecast_log if not r["skipped"])
    return total


def main(argv=None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="seeds 200..202, no secondary sweeps")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--model-seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--model-keys", default="small,large")
    ap.add_argument("--with-single", action="store_true",
                    help="also run the single-call arm from the store (it is always run with --live)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--live", action="store_true", help="let a call that is not stored reach the model")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cache = args.cache or SimConfig().agent_cache
    out = args.out or (DEFAULT_OUT + ("_quick" if args.quick else ""))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = list(range(FIRST_SEED, FIRST_SEED + (3 if args.quick else args.seeds)))
    model_seeds = set(seeds[:(3 if args.quick else args.model_seeds)])

    def points(selected):
        base = [(m, *PRIMARY) for m in selected]
        extra = [] if args.quick else [(m, lam, tick) for m in selected if m in ("true", "false")
                                       for lam, tick in SECONDARY]
        return base + extra

    keys = [k.strip() for k in args.model_keys.split(",") if k.strip()]
    model, notes = select_model_arms(keys, cache, args.live, args.with_single)
    for note in notes:
        print("NOTE:", note)
    jobs = args.jobs
    if args.live and model:
        per = MAX_TURNS * sum(1 for a in model if a[3] == "tools") + sum(1 for a in model if a[3] == "single")
        requests = count_requests(sorted(model_seeds), points, modes, cache)
        if not confirm_calls(requests, per, args.yes, confirm):
            print("aborted")
            return 1
        jobs = 1                                    # one process appends to the store at a time
    tasks = [(seed, m, lam, tick, NON_MODEL_ARMS + (model if seed in model_seeds else []), cache, args.live)
             for seed in seeds for m, lam, tick in points(modes)]
    try:
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                batches = list(pool.map(run_point, tasks))
        else:
            batches = [run_point(t) for t in tasks]
    except ReplayMiss as miss:
        print(f"error: {miss}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame([row for batch in batches for row in batch])
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    kept, dropped = usable(df, POINT)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    called = bool(args.live and model)
    print(f"E9: shift_notice, 8 robots, 20 tasks, seeds {seeds[0]}..{seeds[-1]}, model arms on "
          f"{len(model_seeds)} seeds; " + ("model calls were allowed" if called else "no model was called"))
    print(f"points with a stalled run are left out of every summary: dropped {dropped} of "
          f"{df.groupby(POINT).ngroups}")
    means = kept.groupby(["lam", "notice_tick", "mode", "arm"])["avoidable"].mean().unstack("arm").round(1)
    print("\nmean avoidable cost (J_censored minus the free arm's):")
    print(means.to_string())
    print("\npaired differences in avoidable cost, A - B (positive: B is cheaper); median, 95% interval, share above 0:")
    for _, r in summary.iterrows():
        print(f"  lam {r.lam:g} tick {int(r.notice_tick):3d} {r['mode']:8s} {r.hypothesis:3s} {r['compare']:26s} "
              f"median {r.median_diff:7.1f} [{r.lo:7.1f}, {r.hi:7.1f}] above 0: {r.share_positive:.2f}  n {int(r.n)}")
    print(f"\nstalled runs (all of them, kept in runs.csv): {int(df.stalled.sum())} of {len(df)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_e9_agent.py tests/doi/test_doi_e9_headroom.py -q`
Expected: all pass (about two minutes). The headroom test file must pass unchanged.

`test_a_store_with_a_gap_stops_the_run_with_the_remedy` relies on a store file that exists but holds nothing usable: the replay must hit a miss on the first call and `main` must return 2.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add experiments/doi_e9_agent.py experiments/doi_e9_headroom.py tests/doi/test_doi_e9_agent.py
git commit -m "feat(doi): E9 end-to-end experiment for the forecast agent, replay-only unless --live"
```

---

### Task 10: Documents

**Files:**
- Modify: `src/doi/README.md`, `data/forecasts/README.md`, `docs/research/results.md`, `docs/research/demo-guide.md`, `docs/superpowers/specs/2026-10-06-forecast-agent-guard-design.md` (status lines only)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: README of the simulator**

In `src/doi/README.md`:

(a) Replace the sentence `The language model is never involved in planning, pushing or traffic.` (in the section `### incidents.py, llm/ — The Language Layer`) with:

```markdown
The language model is never involved in planning, pushing or traffic. It can now do one other thing: as the forecaster behind the arm `rof_a` it answers one yes/no question per candidate move ("will the fleet's saving reach the price?"), and the predicted-threshold rule guards the answer (a yes lowers the threshold, a no raises it, no answer leaves the classical rule). It can only read; every call is stored and replayed, and a run never calls a model unless `--agent-live` is given.
```

(b) In the project-structure tree, replace the line `│   ├── llm/                         # Report intake (offline)` with:

```text
│   ├── llm/                         # Report intake and the chat client; stored, replayable model calls
│   ├── forecast/                    # Forecast cases, tools, the agent loop, the forecasters behind rof_a
│   ├── notices.py                   # Text notices about future traffic (wording banks, delivery)
```

and replace the line `│   └── doi_e2 / doi_e7 *.py         # Information and intake experiments` with:

```text
│   ├── doi_e2 / doi_e7 *.py         # Information and intake experiments
│   └── doi_e9_*.py, doi_agent_*.py  # Forecast guard: headroom pilot, end-to-end, cases, scoring
```

(c) In the experiments table (section 9) add these rows after the `doi_e7_intake.py` row:

```markdown
| `doi_e9_headroom.py` | How much can any forecaster change fleet cost on `shift_notice`? (no model) |
| `doi_agent_cases.py` | Collect forecast cases with their truth label into `data/forecasts/` (no model) |
| `doi_agent_eval.py` | Score forecasters, models included, on those cases: accuracy, calibration, failures, cost |
| `doi_e9_agent.py` | Does the forecast agent lower fleet cost, and what happens when its notices are wrong? |
```

and after the paragraph beginning `Each has a --quick pilot mode` add:

```markdown
The three scripts that can reach a model (`doi_agent_eval.py`, `doi_e9_agent.py`, and `run_doi.py` with `--agent-live`) replay stored calls by default. A call that is not stored is an error; `--agent-live` lets it reach the model, and the two experiment scripts then print an upper bound on calls and tokens and ask before spending. Model keys come from `DOI_LLM_<KEY>_URL` and `DOI_LLM_<KEY>_MODEL`; the experiments use `small` and `large`, and one exported `OPENAI_API_KEY` serves both when their URL is `https://api.openai.com/...`.
```

(d) In the arms table, replace the `rof_a` row with:

```markdown
| `rof_a` | Rent-or-Fill guarded: a forecaster (`--forecaster numeric`, `keyword`, `oracle`, `inverted` or `llm:<key>`) says whether the move will pay; the threshold is lowered on a yes, raised on a no, and stays at the price while there is no answer |
```

- [ ] **Step 2: Data README**

In `data/forecasts/README.md` replace the sentence beginning ``dev.jsonl``, ``test.jsonl`` and ``human.jsonl`` with:

```markdown
`dev.jsonl`, `test.jsonl` and `human.jsonl` (forecast cases) are produced by `experiments/doi_agent_cases.py`. They are generated, not committed (about 10 MB for `test`); each line is `{"meta": {split, seed, mode, bank, stalled}, "case": {...}}`. `dev` (seeds 0..19) is for prompt development only; `test` (seeds 100..149) and `human` (seeds 300..319) are held-out. Seeds 200..229 belong to E9. Score them with `experiments/doi_agent_eval.py`. Without `human_notices.jsonl` the `human` split is skipped and level 2 is reported as not done.
```

- [ ] **Step 3: Results status and demo guide**

In `docs/research/results.md`, under `## What exists now`, add these bullets after the `doi_common.py` bullet:

```markdown
* `experiments/doi_e9_headroom.py`: how much any forecaster can change cost on `shift_notice` (output below, no model).
* `experiments/doi_agent_cases.py`, `doi_agent_eval.py`, `doi_e9_agent.py`: the case collector, the forecaster scoring and the end-to-end experiment for the forecast agent. Built; **no model call has been made, no case file has been generated and E9 has not been run in full.** The hypotheses H9a to H9d are in the design spec; the H9c margin is to be fixed after the `--quick` pilot, with the tag `prereg-agent-v1`.
```

In `docs/research/demo-guide.md`, in `## Flags in one line each`, after the sentence ending `` `--p-wrong-class`. `` add:

```markdown
Forecast agent (arm `rof_a`): `--forecaster numeric|keyword|oracle|inverted|llm:<key>`, `--agent-mode tools|single`,
`--agent-cache`, `--agent-live` (the only flag that spends money; without it stored model calls are replayed).
```

- [ ] **Step 4: Spec header**

In `docs/superpowers/specs/2026-10-06-forecast-agent-guard-design.md` replace the two status lines at the top (`Status: design approved in conversation ...` and `No code has been written. ...`) with:

```markdown
Status: design approved by the owner on 2026-10-06. Part 1 (everything without a model) was built and measured on
2026-10-07; Part 2 (the model parts: sections 8, 9, 12, 14.1 to 14.3) is built to
`docs/superpowers/plans/2026-10-07-forecast-guard-part2.md`. No model call has been made.
```

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/README.md data/forecasts/README.md docs/research/results.md docs/research/demo-guide.md docs/superpowers/specs/2026-10-06-forecast-agent-guard-design.md
git commit -m "docs(doi): the forecast agent, its flags and scripts, and what has and has not been run"
```

- [ ] **Step 6: Stop**

End the work here. Report to the owner: the list of commits, the test count, and what is left for the owner: the human-written notices (`data/forecasts/human_notices.jsonl`), the `small` and `large` model names and a spending limit, running `doi_agent_cases.py`, the `--quick` pilot of E9 and the H9c margin with the tag `prereg-agent-v1`, then the full runs. Make no model call.

---

## Self-Review Notes

- Spec sections covered: 8 (Task 3), 9 (Task 4), 10 llm backend (Task 5), 11 edits to `config.py`, `llm/client.py`, `run_doi.py`, `narrate.py` (Tasks 1, 5, 6), 12.1 to 12.4 (Tasks 1, 2, 8, 9), 13 (already logged by Part 1; the new columns are read in Tasks 8 and 9), 14.1 to 14.3 (Tasks 7 to 9), 15.4 (Task 4), 15.5 item 6 (Task 5), 15.6 (Tasks 7 to 9), 16 steps 6 to 9, 19 (Task 10 step 6).
- Left out by design: the human-written notices, the preregistration tag, any full experiment, any model call, the wizard prompts for the new settings, and a `--notice-mode` flag (the scenario's default `true` is what the command line uses; the experiments set the mode through `scenario_params`).
- The Part 1 pinned costs (`test_doi_guard_neutral.py`) and Part 1's `test_doi_forecasters.py` / `test_doi_guarded_policy.py` are not edited; Task 5 keeps `make_forecaster("llm:small")` raising `ValueError` with "llm" in the message when no config is given.
