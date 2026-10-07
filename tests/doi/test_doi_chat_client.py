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
