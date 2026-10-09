import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))

import doi_intake_eval
import doi_intake_run
from src.doi.incidents import read_jsonl
from src.doi.llm.client import FakeLLMClient


def oracle_reply(item_by_text):
    def reply(system, user):
        text = user.split("Report:\n", 1)[1]
        it = item_by_text[text]
        if it.truth_location is None:
            return json.dumps({"reject": "ambiguous"})
        return json.dumps({"location": it.truth_location, "kind": it.truth_kind, "class": it.truth_cls,
                           "est_kits": it.truth_kits, "confidence": 0.9, "rationale": "ok"})
    return reply


def test_run_then_eval_with_a_perfect_fake_model(tmp_path, capsys):
    items = read_jsonl(os.path.join(doi_intake_run.ROOT, "data", "incidents", "dev.jsonl"))[:30]
    by_text = {it.text: it for it in items}
    assert len(by_text) == len(items)
    todo_texts = set(by_text)
    orig = doi_intake_run.split_items
    doi_intake_run.split_items = lambda split: [(it.text, it.location_names) for it in items]
    try:
        client = FakeLLMClient(oracle_reply(by_text), latency_s=0.5)
        rc = doi_intake_run.main(["--model-key", "fake", "--split", "dev", "--cache-dir", str(tmp_path), "--yes"],
                                 client=client)
        assert rc == 0
        assert doi_intake_run.main(["--model-key", "fake", "--split", "dev", "--cache-dir", str(tmp_path), "--yes"],
                                   client=client) == 0
    finally:
        doi_intake_run.split_items = orig
    cache = doi_intake_eval.IntakeCache(str(tmp_path), "fake")
    m = doi_intake_eval.score(items, cache)
    assert m["scored"] == 30 and m["missing"] == 0
    assert m["location_acc"] == 1.0 and m["class_acc"] == 1.0 and m["kind_acc"] == 1.0
    assert m["latency_p50"] == 0.5 and m["schema_failure_rate"] == 0.0
    assert len(todo_texts) == 30


def test_run_asks_for_confirmation_and_can_abort(tmp_path):
    items = read_jsonl(os.path.join(doi_intake_run.ROOT, "data", "incidents", "dev.jsonl"))[:3]
    orig = doi_intake_run.split_items
    doi_intake_run.split_items = lambda split: [(it.text, it.location_names) for it in items]
    try:
        client = FakeLLMClient(lambda s, u: "{}")
        rc = doi_intake_run.main(["--model-key", "fake", "--split", "dev", "--cache-dir", str(tmp_path)],
                                 client=client, confirm=lambda prompt: "n")
        assert rc == 1
        assert not os.path.exists(os.path.join(str(tmp_path), "fake", "cache.jsonl"))
    finally:
        doi_intake_run.split_items = orig
