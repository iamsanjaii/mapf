import json
import pytest
from src.doi.notices import BANKS, DROP_CUES, KINDS, SURGE_CUES, load_human, render_text
from src.doi.rng import stream

CUES = SURGE_CUES + DROP_CUES


def _has(text, cues):
    return any(c in text.lower() for c in cues)


def test_banks_share_no_template_and_test_bank_has_no_dev_cue():
    for kind in KINDS:
        assert len(BANKS["dev"][kind]) == 4 and len(BANKS["test"][kind]) == 4
        assert not set(BANKS["dev"][kind]) & set(BANKS["test"][kind])
        for template in BANKS["test"][kind]:
            assert not _has(template, CUES), template


def test_dev_bank_cues_match_their_kind():
    for template in BANKS["dev"]["surge"]:
        assert _has(template, SURGE_CUES) and not _has(template, DROP_CUES)
    for template in BANKS["dev"]["drop"]:
        assert _has(template, DROP_CUES) and not _has(template, SURGE_CUES)
    for template in BANKS["dev"]["distractor"]:
        assert not _has(template, CUES)


def test_render_is_deterministic_and_names_the_group():
    a = render_text("surge", "south bays", "north bays", "dev", stream(7, "notice-n0"))
    b = render_text("surge", "south bays", "north bays", "dev", stream(7, "notice-n0"))
    assert a == b and "south bays" in a and "north" not in a
    assert "north bays" in render_text("drop", "north bays", "south bays", "test", stream(7, "notice-n1"))


def test_render_rejects_unknown_kind_and_bank():
    with pytest.raises(ValueError):
        render_text("rumour", "south bays", "north bays", "dev", stream(0, "x"))
    with pytest.raises(ValueError):
        render_text("surge", "south bays", "north bays", "prod", stream(0, "x"))


def test_human_bank_reads_the_file_and_explains_a_missing_one(tmp_path):
    with pytest.raises(ValueError, match="data/forecasts/README.md"):
        load_human(str(tmp_path / "absent.jsonl"))
    path = tmp_path / "human.jsonl"
    rows = [{"kind": "surge", "group": "south bays", "text": "everything's heading south after lunch"},
            {"kind": "drop", "group": "north bays", "text": "north side is wrapped up"},
            {"kind": "distractor", "group": "", "text": "who left the pallet jack on charge"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert render_text("surge", "south bays", "north bays", "human", stream(0, "x"), str(path)) == rows[0]["text"]
    assert render_text("distractor", "north bays", "south bays", "human", stream(0, "x"), str(path)) == rows[2]["text"]
    with pytest.raises(ValueError, match="no human notice"):
        render_text("surge", "north bays", "south bays", "human", stream(0, "x"), str(path))
