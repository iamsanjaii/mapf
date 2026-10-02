import json
import pytest
from src.doi.config import SimConfig
from src.doi.crdt import ObstructionRecord
from src.doi.scenarios import build_scenario
from src.doi.llm.client import FakeLLMClient, OpenAICompatClient
from src.doi.llm.intake import (parse_and_validate, run_intake, cache_key, IntakeCache, to_record,
                                PROMPT_VERSION)

NAMES = ("aisle 2 bay 3", "north door")
GOOD = json.dumps({"location": "aisle 2 bay 3", "kind": "pallet", "class": "robot_clearable",
                   "est_kits": 1, "confidence": 0.8, "rationale": "fallen pallet"})


def test_parse_valid_with_surrounding_text():
    out = parse_and_validate("Sure:\n" + GOOD + "\nDone.", NAMES)
    assert out["ok"] and out["location"] == "aisle 2 bay 3" and out["cls"] == "robot_clearable"


@pytest.mark.parametrize("raw,reason", [
    ("no json here", "not_json"),
    (json.dumps({"location": "aisle 9 bay 9", "kind": "pallet", "class": "robot_clearable",
                 "est_kits": 1, "confidence": 0.5, "rationale": "x"}), "bad_location"),
    (json.dumps({"location": "north door", "kind": "fire", "class": "robot_clearable",
                 "est_kits": 1, "confidence": 0.5, "rationale": "x"}), "bad_kind"),
    (json.dumps({"location": "north door", "kind": "spill", "class": "robot_clearable",
                 "est_kits": 9, "confidence": 0.5, "rationale": "x"}), "bad_kits"),
    (json.dumps({"location": "north door", "kind": "spill", "class": "robot_clearable",
                 "est_kits": 1, "rationale": "x"}), "missing:confidence"),
])
def test_parse_rejects(raw, reason):
    out = parse_and_validate(raw, NAMES)
    assert not out["ok"] and out["reject_reason"] == reason


def test_model_reject():
    out = parse_and_validate('{"reject": "no bay given"}', NAMES)
    assert not out["ok"] and out["reject_reason"].startswith("model_reject")


def test_run_intake_with_fake_client_and_cache(tmp_path):
    client = FakeLLMClient(lambda system, user: GOOD, latency_s=0.7)
    res = run_intake("pallet down at aisle 2 bay 3", NAMES, client)
    assert res.ok and res.latency_s == 0.7 and res.key == cache_key("pallet down at aisle 2 bay 3", NAMES)
    cache = IntakeCache(str(tmp_path), "fake")
    cache.put(res)
    assert IntakeCache(str(tmp_path), "fake").get(res.key) == res
    assert cache_key("a", NAMES) != cache_key("b", NAMES)
    assert PROMPT_VERSION


def test_to_record_locates_cells():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=2, tasks_per_robot=2))
    client = FakeLLMClient(lambda system, user: GOOD)
    rec = to_record(run_intake("x", NAMES, client), "r0", 3, s)
    assert rec.cells == ((3, 2),) and rec.node == 3 and rec.cls == "robot_clearable"


def test_openai_compat_request_shape(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": GOOD}}],
                               "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatClient("http://localhost:11434/v1", "small-model")
    r = c.complete("sys", "user", max_tokens=50)
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["model"] == "small-model" and seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["max_tokens"] == 50 and seen["body"]["temperature"] == 0.0
    assert seen["auth"] is None and r.text == GOOD and r.prompt_tokens == 11
    h = OpenAICompatClient("https://api.openai.com/v1", "hosted-model", api_key="sk-test",
                           token_param="max_completion_tokens", send_temperature=False, json_mode=True)
    h.complete("sys", "user", max_tokens=50)
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["max_completion_tokens"] == 50 and "max_tokens" not in seen["body"]
    assert "temperature" not in seen["body"] and seen["body"]["response_format"] == {"type": "json_object"}


def test_client_from_env_never_leaks_key(monkeypatch):
    from src.doi.llm.client import client_from_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    monkeypatch.delenv("DOI_LLM_HOSTED_URL", raising=False)
    with pytest.raises(RuntimeError) as e:
        client_from_env("hosted")
    assert "DOI_LLM_HOSTED_URL" in str(e.value) and "sk-secret-value" not in str(e.value)


def test_http_400_error_body_is_surfaced_without_the_key(monkeypatch):
    import io
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"error": "unsupported parameter: max_tokens"}'))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    c = OpenAICompatClient("https://api.openai.com/v1", "m", api_key="sk-secret-value")
    with pytest.raises(RuntimeError) as e:
        c.complete("s", "u", max_tokens=5)
    assert "unsupported parameter" in str(e.value) and "sk-secret-value" not in str(e.value)


def test_cache_last_write_wins_and_missing_is_none(tmp_path):
    client = FakeLLMClient(lambda s, u: GOOD)
    res = run_intake("x", NAMES, client)
    cache = IntakeCache(str(tmp_path), "m")
    assert cache.get(res.key) is None
    cache.put(res)
    newer = IntakeResult_replace(res, rationale="newer")
    cache.put(newer)
    assert IntakeCache(str(tmp_path), "m").get(res.key).rationale == "newer"


def IntakeResult_replace(res, **kw):
    import dataclasses
    return dataclasses.replace(res, **kw)
