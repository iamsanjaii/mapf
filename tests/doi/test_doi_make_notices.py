import importlib.util
import json
import os

import pytest
from src.doi.llm.client import FakeLLMClient
from src.doi.notices import render_text
from src.doi.rng import stream

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _module():
    spec = importlib.util.spec_from_file_location("doi_make_notices", os.path.join(ROOT, "experiments", "doi_make_notices.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _field(user: str, name: str) -> str:
    return next(line.split(":", 1)[1].strip() for line in user.splitlines() if line.startswith(f"{name}:"))


def good(system, user):
    """A model that follows the brief, with a different text every call."""
    kind, group = _field(user, "Kind"), _field(user, "Group")
    voice = _field(user, "Voice")
    side = group.split()[0] if group else ""
    n = good.n = getattr(good, "n", 0) + 1
    return {"surge": f"{voice}: heads up #{n}, the next orders all land in the {side} bays",
            "drop": f"{voice}: #{n} nothing more is coming for the {side} bays",
            "distractor": f"{voice}: #{n} the label printer needs ribbon"}[kind]


def _clients(reply=good):
    return {"small": FakeLLMClient(reply, model_id="gpt-4o-mini"), "large": FakeLLMClient(reply, model_id="gpt-4o")}


def test_plan_asks_for_more_than_the_protocol_minimum():
    mod = _module()
    plan = mod.make_plan(1.25)
    kinds = [k for k, _ in plan]
    assert kinds.count("surge") >= 50 and kinds.count("drop") >= 50 and kinds.count("distractor") >= 25
    assert {g for k, g in plan if k == "surge"} == {"north bays", "south bays"}
    assert {g for k, g in plan if k == "distractor"} == {""}


def test_clean_keeps_a_valid_text_and_rejects_the_rest():
    mod = _module()
    assert mod.clean('"All orders go to the south bays after lunch."\n', "surge", "south bays") == \
        "All orders go to the south bays after lunch."
    assert mod.clean("south bays are done", "drop", "south bays") == "south bays are done"
    assert mod.clean("Orders go to the north bays", "surge", "south bays") is None          # wrong side
    assert mod.clean("Orders are coming", "surge", "south bays") is None                      # names no side
    assert mod.clean("Meeting moved, north door is locked", "distractor", "") is None          # a distractor names a side
    assert mod.clean("   ", "distractor", "") is None
    assert mod.clean("x " * 200, "distractor", "") is None                                      # too long
    assert mod.clean("Line one\nline two", "distractor", "") == "Line one line two"


def test_a_run_writes_rows_the_human_bank_can_read(tmp_path):
    mod = _module()
    out = tmp_path / "notices.jsonl"
    assert mod.main(["--yes", "--out", str(out)], clients=_clients()) == 0
    rows = [json.loads(line) for line in open(out)]
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in ("surge", "drop", "distractor")}
    assert counts["surge"] >= 40 and counts["drop"] >= 40 and counts["distractor"] >= 20
    assert {r["source"] for r in rows} == {"gpt-4o-mini", "gpt-4o"}
    assert len({r["text"].lower() for r in rows}) == len(rows)
    assert all(r["group"] in ("north bays", "south bays") for r in rows if r["kind"] != "distractor")
    assert all(r["group"] == "" for r in rows if r["kind"] == "distractor")
    text = render_text("surge", "south bays", "north bays", "human", stream(0, "x"), str(out))
    assert "south" in text                                               # the scenario can use the file as it is


def test_bad_outputs_are_dropped_duplicates_are_removed_and_a_shortfall_fails(tmp_path):
    mod = _module()
    out = tmp_path / "notices.jsonl"
    code = mod.main(["--yes", "--out", str(out)], clients=_clients(lambda s, u: "Everything is fine."))
    assert code == 1 and not out.exists()                                  # nothing valid: no file, and a clear failure
    same = _clients(lambda s, u: "All orders go to the south bays" if "surge" in u else "Nothing for the south bays")
    assert mod.main(["--yes", "--out", str(out)], clients=same) == 1       # identical texts collapse below the minimum


def test_the_estimate_is_shown_and_no_stops_the_run(tmp_path, capsys):
    mod = _module()
    asked = []
    clients = _clients(lambda s, u: asked.append(u) or good(s, u))
    code = mod.main(["--out", str(tmp_path / "n.jsonl")], clients=clients, confirm=lambda prompt: "n")
    assert code == 1 and "model calls" in capsys.readouterr().out
    assert asked == [] and not (tmp_path / "n.jsonl").exists()


def test_an_existing_file_is_not_overwritten_without_force(tmp_path):
    mod = _module()
    out = tmp_path / "n.jsonl"
    out.write_text("keep me\n")
    with pytest.raises(SystemExit):
        mod.main(["--yes", "--out", str(out)], clients=_clients())
    assert out.read_text() == "keep me\n"
    assert mod.main(["--yes", "--force", "--out", str(out)], clients=_clients()) == 0


def test_clients_built_from_the_environment_never_use_json_mode(monkeypatch, tmp_path):
    """The texts are plain prose; a JSON_MODE left over from the intake work must not reach the request."""
    mod = _module()
    for key in ("SMALL", "LARGE"):
        monkeypatch.setenv(f"DOI_LLM_{key}_URL", "https://api.openai.com/v1")
        monkeypatch.setenv(f"DOI_LLM_{key}_MODEL", "m")
        monkeypatch.setenv(f"DOI_LLM_{key}_JSON_MODE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    bodies = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "All orders go to the south bays"}}]}).encode()

    def fake_urlopen(req, timeout):
        bodies.append(json.loads(req.data.decode()))
        return Resp()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    mod.main(["--yes", "--extra", "0.1", "--out", str(tmp_path / "n.jsonl")])
    assert len(bodies) == 10 and all("response_format" not in b for b in bodies)
    assert {b["model"] for b in bodies} == {"m"}


def test_missing_model_settings_are_named_before_anything_is_asked(monkeypatch, tmp_path, capsys):
    mod = _module()
    for var in ("DOI_LLM_SMALL_URL", "DOI_LLM_SMALL_MODEL", "DOI_LLM_LARGE_URL", "DOI_LLM_LARGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOI_LLM_SMALL_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("DOI_LLM_SMALL_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    asked = []
    code = mod.main(["--out", str(tmp_path / "n.jsonl")], confirm=lambda prompt: asked.append(prompt) or "y")
    err = capsys.readouterr().err
    assert code == 2 and asked == []
    assert "DOI_LLM_LARGE_URL" in err and "DOI_LLM_LARGE_MODEL" in err and "DOI_LLM_SMALL_URL" not in err
    assert "export" in err and "--models small" in err                      # how to fix it, or to use one model only
