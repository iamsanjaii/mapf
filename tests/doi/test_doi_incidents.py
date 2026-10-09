import random
import pytest
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario
from src.doi.incidents import location_names, locate, render_report, report_rng, build_items, write_jsonl, read_jsonl


def aisles(seed=1, **params):
    return build_scenario(SimConfig(scenario="incidents_aisles", n_robots=4, tasks_per_robot=3, seed=seed,
                                    scenario_params=params))


def test_locate():
    s = aisles()
    assert locate(s, "aisle 2 bay 3") == ((3, 2),)
    assert "aisle 2 bay 3" in location_names(s)
    with pytest.raises(ValueError):
        locate(s, "aisle 99 bay 1")


def test_render_is_deterministic_per_report():
    s = aisles()
    r = s.reports[0]
    assert render_report(r, report_rng(s, r)) == render_report(r, report_rng(s, r))


def test_class_cues_only_for_needs_human():
    s = aisles(seed=4, p_human=1.0)
    for r in s.reports:
        text, _, _ = render_report(r, report_rng(s, r))
        assert text
    items = build_items(s, "dev")
    assert all(it.truth_cls == "needs_human" for it in items)


def test_ambiguous_rate_and_truth():
    texts = []
    for seed in range(40):
        s = aisles(seed=seed)
        texts += build_items(s, "dev")
    amb = [it for it in texts if it.truth_location is None]
    assert 0.03 < len(amb) / len(texts) < 0.2
    assert all(it.truth_location in it.location_names for it in texts if it.truth_location)


def test_false_reports_flagged():
    s = aisles(seed=3, p_false=1.0)
    items = build_items(s, "dev")
    assert sum(it.is_false for it in items) == 4


def test_jsonl_roundtrip(tmp_path):
    items = build_items(aisles(), "dev")
    write_jsonl(items, str(tmp_path / "x.jsonl"))
    assert read_jsonl(str(tmp_path / "x.jsonl")) == items


def test_robot_clearable_text_never_contains_a_class_cue():
    from src.doi.incidents import CLASS_CUES
    for seed in range(30):
        s = aisles(seed=seed, p_human=0.0)
        for it in build_items(s, "dev"):
            assert not any(cue in it.text.lower() for cue in CLASS_CUES)


def test_kits_follow_multi_phrases_and_templates_are_varied():
    from src.doi.incidents import TEMPLATES
    assert len(TEMPLATES) >= 12
    texts = set()
    for seed in range(20):
        for it in build_items(aisles(seed=seed), "dev"):
            texts.add(it.text)
            assert it.truth_kits in (1, 2)
    assert len(texts) > 30
