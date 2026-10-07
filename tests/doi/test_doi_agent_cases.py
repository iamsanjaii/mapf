import json
import os
import subprocess
import sys

from src.doi.forecast.case import case_from_dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "experiments", "doi_agent_cases.py")


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=ROOT)


def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_quick_dev_collection_writes_labelled_distinct_cases(tmp_path):
    proc = _run("--quick", "--splits", "dev", "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    rows = _rows(tmp_path / "dev.jsonl")
    assert rows and all(r["meta"]["split"] == "dev" and r["meta"]["bank"] == "dev" for r in rows)
    assert {r["meta"]["mode"] for r in rows} == {"true", "false", "missing", "quiet"}
    assert {r["meta"]["seed"] for r in rows} <= {0, 1}
    cases = [case_from_dict(r["case"]) for r in rows]
    assert all(c.truth is not None and c.zones and c.ledger for c in cases)
    assert len({c.case_id for c in cases}) == len(cases)
    assert any(c.notices for c in cases) and any(not c.notices for c in cases)       # notices in some modes only
    assert "dev:" in proc.stdout and "no model was called" in proc.stdout


def test_the_human_split_is_skipped_without_the_notice_file(tmp_path):
    proc = _run("--quick", "--splits", "human", "--out", str(tmp_path),
                "--human-notices", str(tmp_path / "absent.jsonl"))
    assert proc.returncode == 0, proc.stderr
    assert "human split skipped" in proc.stdout and "level 2" in proc.stdout
    assert not (tmp_path / "human.jsonl").exists()


def test_the_human_split_reads_the_people_written_texts(tmp_path):
    texts = tmp_path / "human_notices.jsonl"
    rows = [{"kind": "surge", "group": "south bays", "text": "orders are all piling into the south side"},
            {"kind": "surge", "group": "north bays", "text": "orders are all piling into the north side"},
            {"kind": "drop", "group": "south bays", "text": "south side is done for the day"},
            {"kind": "drop", "group": "north bays", "text": "north side is done for the day"},
            {"kind": "distractor", "group": "", "text": "someone left a jacket in the canteen"}]
    texts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = _run("--quick", "--splits", "human", "--out", str(tmp_path), "--human-notices", str(texts))
    assert proc.returncode == 0, proc.stderr
    cases = [case_from_dict(r["case"]) for r in _rows(tmp_path / "human.jsonl")]
    said = " ".join(text for c in cases for _, _, text in c.notices)
    assert "piling into" in said or "done for the day" in said
