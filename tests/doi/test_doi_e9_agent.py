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
    assert [a[0] for a in e9.NON_MODEL_ARMS] == ["never", "rof", "rof_p", "numeric", "keyword", "ledger", "oracle", "inverted"]
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
    assert (runs.avoidable == runs.J - runs.J_free).all() and len(runs) == 3 * 8
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
