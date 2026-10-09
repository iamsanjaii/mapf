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
                    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0, notices=(), ledger={"saving_on_recorded_trips": 0.0},
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
    assert list(table.forecaster) == ["numeric", "keyword", "ledger", "oracle", "inverted"]
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


def test_quick_with_live_keeps_a_model_whose_store_is_empty(tmp_path, capsys, monkeypatch):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    monkeypatch.setattr("src.doi.llm.client.client_from_env",
                        lambda key: (_ for _ in ()).throw(AssertionError("the client must not be built")))
    code = mod.main(["--quick", "--cases", str(tmp_path / "cases.jsonl"), "--out", str(tmp_path / "o"),
                     "--forecasters", "llm:small", "--cache", str(tmp_path / "store"), "--live"],
                    confirm=lambda prompt: "n")
    assert code == 1 and "at most 16 model calls" in capsys.readouterr().out


def test_sample_takes_evenly_spaced_cases_from_the_whole_file(tmp_path):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    assert mod.main(["--sample", "2", "--cases", str(tmp_path / "cases.jsonl"), "--out", str(tmp_path / "o")]) == 0
    import pandas as pd
    assert list(pd.read_csv(tmp_path / "o" / "eval.csv").n) == [2, 2, 2, 2, 2]
