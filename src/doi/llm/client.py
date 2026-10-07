"""LLM clients: an OpenAI-compatible HTTP client (stdlib only), a fake for tests, and env configuration."""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model: str


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


class LLMClient(Protocol):
    model_id: str

    def complete(self, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> LLMResponse: ...


class OpenAICompatClient:
    """POST {base_url}/chat/completions. The API key is only ever placed in the Authorization header."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout_s: float = 60.0,
                 token_param: str = "max_tokens", send_temperature: bool = True, json_mode: bool = False,
                 temperature: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.token_param = token_param
        self.send_temperature = send_temperature
        self.json_mode = json_mode
        self.temperature = temperature

    def __repr__(self) -> str:
        return f"OpenAICompatClient({self.base_url!r}, {self.model_id!r})"

    def complete(self, system: str, user: str, max_tokens: int, temperature: Optional[float] = None) -> LLMResponse:
        body = {"model": self.model_id,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.token_param: max_tokens}
        if self.send_temperature:
            body["temperature"] = self.temperature if temperature is None else temperature
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
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
        latency = time.perf_counter() - start
        data = json.loads(raw.decode())
        usage = data.get("usage") or {}
        return LLMResponse(text=data["choices"][0]["message"]["content"], latency_s=latency,
                           prompt_tokens=int(usage.get("prompt_tokens", 0)),
                           completion_tokens=int(usage.get("completion_tokens", 0)),
                           model=data.get("model", self.model_id))

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


class FakeLLMClient:
    def __init__(self, reply: Callable[[str, str], str], latency_s: float = 0.1, model_id: str = "fake") -> None:
        self.reply = reply
        self.latency_s = latency_s
        self.model_id = model_id

    def complete(self, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> LLMResponse:
        text = self.reply(system, user)
        return LLMResponse(text, self.latency_s, len((system + user).split()), len(text.split()), self.model_id)


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


def _need(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"environment variable {var} is not set")
    return value


def client_from_env(model_key: str) -> OpenAICompatClient:
    k = model_key.upper()
    url = _need(f"DOI_LLM_{k}_URL")
    model = _need(f"DOI_LLM_{k}_MODEL")
    key = _api_key(k, url)
    temp = os.environ.get(f"DOI_LLM_{k}_TEMPERATURE", "0")
    return OpenAICompatClient(
        url, model, api_key=key or None,
        token_param=os.environ.get(f"DOI_LLM_{k}_TOKEN_PARAM", "max_tokens"),
        send_temperature=temp.lower() != "none",
        temperature=0.0 if temp.lower() == "none" else float(temp),
        json_mode=os.environ.get(f"DOI_LLM_{k}_JSON_MODE", "") == "1")
