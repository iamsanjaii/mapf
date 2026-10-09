"""Incident report text rendering, location grounding and the labelled intake dataset."""
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from src.doi.rng import stream
from src.doi.scenarios import Report, Scenario

Pos = Tuple[int, int]

TEMPLATES = (
    "{what} at {loc}",
    "{loc}: {what}",
    "heads up, {what} near {loc}",
    "{what} reported at {loc}, please check",
    "can someone look at {loc}? {what}",
    "we have {what} at {loc}",
    "{loc} is blocked, {what}",
    "robots are stuck behind {what} at {loc}",
    "FYI {what} at {loc}",
    "{loc} - {what}, path blocked",
    "supervisor note: {what} at {loc}",
    "radio: {loc} has {what}",
    "got a report of {what} at {loc}",
    "{what} spotted around {loc}",
)

KIND_PHRASES = {
    "pallet": ["a fallen pallet", "a pallet in the way", "a pallet dropped across the lane", "a tipped pallet"],
    "spill": ["a spill on the floor", "a puddle", "liquid on the ground", "a wet patch across the lane"],
    "debris": ["debris on the floor", "some broken packaging", "boxes scattered", "cardboard everywhere"],
    "rack_damage": ["a damaged rack", "a bent shelf beam", "a rack that has shifted", "a collapsed shelf section"],
}
MULTI_PHRASES = {
    "pallet": ["two pallets", "a couple of fallen pallets"],
    "debris": ["a lot of debris", "debris piled up across the lane"],
    "spill": ["a big spill", "a large puddle"],
}
GENERAL_CUES = ("keep robots away", "someone is hurt", "do not send robots here")
KIND_CUES = {
    "rack_damage": ("rack upright looks bent",),
    "spill": ("chemical smell", "smells like solvent"),
    "pallet": (),
    "debris": (),
}
CLASS_CUES = GENERAL_CUES + tuple(c for cues in KIND_CUES.values() for c in cues)
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten", 11: "eleven"}
DOOR_ABBR = {"north door": "N door", "middle door": "mid door", "south door": "S door"}
AISLE_BAY = re.compile(r"^aisle (\d+) bay (\d+)$")


@dataclass(frozen=True)
class IncidentItem:
    item_id: str
    split: str
    scenario: str
    seed: int
    report_id: str
    text: str
    source: str
    truth_location: Optional[str]
    truth_kind: str
    truth_cls: str
    truth_kits: int
    is_false: bool
    location_names: Tuple[str, ...]


def location_names(scenario: Scenario) -> Tuple[str, ...]:
    return tuple(sorted(scenario.locations))


def locate(scenario: Scenario, name: str) -> Tuple[Pos, ...]:
    if name not in scenario.locations:
        raise ValueError(f"unknown location {name!r}")
    return scenario.locations[name]


def report_rng(scenario: Scenario, report: Report) -> random.Random:
    return stream(scenario.meta.get("seed", 0), f"report-{report.report_id}")


def _phrase_location(rng: random.Random, name: str) -> Tuple[str, Optional[str]]:
    u = rng.random()
    m = AISLE_BAY.match(name)
    if m:
        k, b = int(m.group(1)), int(m.group(2))
        if u < 0.5:
            return name, name
        if u < 0.7:
            return f"A{k} B{b}", name
        if u < 0.9:
            return f"aisle {WORDS[k]}, bay {WORDS[b]}", name
        return f"aisle {k}", None
    if name in DOOR_ABBR:
        if u < 0.5:
            return name, name
        if u < 0.7:
            return DOOR_ABBR[name], name
        if u < 0.9:
            return name, name
        return "a door on the wall", None
    return name, name


def _typo(rng: random.Random, text: str) -> str:
    idx = [i for i in range(len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha() and text[i] != text[i + 1]]
    if not idx:
        return text
    i = rng.choice(idx)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def render_report(report: Report, rng: random.Random) -> Tuple[str, Optional[str], int]:
    template = rng.choice(TEMPLATES)
    loc_text, truth_location = _phrase_location(rng, report.location)
    what = rng.choice(KIND_PHRASES.get(report.kind, ["something blocking the lane"]))
    cue = rng.choice(GENERAL_CUES + KIND_CUES.get(report.kind, ())) if report.cls == "needs_human" else ""
    kits = 1
    multi = MULTI_PHRASES.get(report.kind)
    if multi and rng.random() < 0.25:
        what, kits = rng.choice(multi), 2
    text = template.format(loc=loc_text, what=what)
    if cue:
        text = f"{text}. {cue}"
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < 0.2:
        text = _typo(rng, text)
    return text, truth_location, kits


def build_items(scenario: Scenario, split: str) -> list:
    names = location_names(scenario)
    seed = scenario.meta.get("seed", 0)
    items = []
    for report in scenario.reports:
        text, truth_location, kits = render_report(report, report_rng(scenario, report))
        items.append(IncidentItem(
            item_id=f"{scenario.name}-{seed}-{report.report_id}", split=split, scenario=scenario.name,
            seed=seed, report_id=report.report_id, text=text, source="template",
            truth_location=truth_location, truth_kind=report.kind, truth_cls=report.cls, truth_kits=kits,
            is_false=report.oid is None, location_names=names))
    return items


def write_jsonl(items: Sequence[IncidentItem], path: str) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def read_jsonl(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["location_names"] = tuple(d["location_names"])
            items.append(IncidentItem(**d))
    return items
