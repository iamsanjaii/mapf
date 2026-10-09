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
