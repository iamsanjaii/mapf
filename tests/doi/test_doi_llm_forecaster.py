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
