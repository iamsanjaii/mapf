import dataclasses
import importlib.util
import json
import math
import os

import pytest
from src.doi.forecast.case import ForecastCase, case_to_dict
from src.doi.forecast.forecasters import NumericForecaster, OracleForecaster, InvertedForecaster

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
    assert numeric["failure_rate"] == 0.0 and math.isnan(numeric["ece"])
    assert mod.evaluate("oracle", OracleForecaster(), cases, 0.5)["accuracy"] == 1.0
    assert mod.evaluate("inverted", InvertedForecaster(), cases, 0.5)["accuracy"] == 0.0


def test_quick_run_scores_the_baselines_and_writes_a_file(tmp_path):
    mod = _module()
    _cases(tmp_path / "cases.jsonl")
    out = tmp_path / "out"
    assert mod.main(["--quick", "--cases", str(tmp_path / "cases.jsonl"), "--out", str(out)]) == 0
    import pandas as pd
    table = pd.read_csv(out / "eval.csv")
    assert list(table.forecaster) == ["numeric", "keyword", "ledger", "oracle", "inverted"]
    assert table.set_index("forecaster").loc["oracle", "accuracy"] == 1.0
