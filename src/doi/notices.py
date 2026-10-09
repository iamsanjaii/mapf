"""Notices: short operational messages about future traffic. Their wording, and (see NoticeFeed) their delivery.

Two wording banks, `dev` and `test`, share no template and no cue phrase. The keyword forecaster's cue lists come
from `dev` only, so a rule that matches phrases cannot look as good as a model on `test` text by construction.
A third source, `human`, reads texts written by people (see data/forecasts/README.md).
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from src.doi.crdt import NoticeRecord

KINDS = ("surge", "drop", "distractor")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANKS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "dev": {
        "surge": ("wave 2 is all in the {group}",
                  "the next batch of orders goes to the {group}",
                  "the {group} get busy after the break",
                  "picking moves to the {group} for the second half"),
        "drop": ("the {group} are finished after this wave",
                 "no more orders for the {group} after the break",
                 "the {group} go quiet for the second half",
                 "picking in the {group} ends soon"),
        "distractor": ("fire drill reminder: Thursday at 10",
                       "label printer at pack station 2 is out of ribbon",
                       "please return scanners to the charging dock",
                       "reminder: tidy the signage in the {group} this week"),
    },
    "test": {
        "surge": ("afternoon orders are concentrated on the {group}",
                  "expect most pick tasks in the {group} from now until end of shift",
                  "inbound has been re-slotted: the {group} take the remaining volume",
                  "work shifts from the {other} over to the {group}"),
        "drop": ("the {group} are done for today once current jobs clear",
                 "nothing further is scheduled in the {group}",
                 "remaining volume has been pulled out of the {group}",
                 "we are winding down the {group} and sending everything to the {other}"),
        "distractor": ("forklift training is in the yard at 3pm",
                       "the dock door sensor is being recalibrated",
                       "canteen closes early today",
                       "cycle count paperwork for the {group} is due Friday"),
    },
}
PREFIXES: Dict[str, Tuple[str, ...]] = {
    "dev": ("", "FYI: ", "supervisor: ", "radio: "),
    "test": ("", "ops update: ", "shift lead: ", "note from planning: "),
}
# Cue phrases of the keyword forecaster. Taken from the dev bank only; never tuned on test or human text.
SURGE_CUES = ("all in", "goes to", "get busy", "moves to")
DROP_CUES = ("finished", "no more", "go quiet", "ends soon")


def load_human(path: str) -> List[dict]:
    """Rows {"kind", "group", "text"} written by people. A relative path is taken from the repository root."""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        raise ValueError(f"human notice file {path} not found: see data/forecasts/README.md for how to collect it")
    with open(full) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_text(kind: str, group: str, other: str, bank: str, rng: random.Random,
                human_path: Optional[str] = None) -> str:
    """The text of one notice of `kind` about zone group `group` (`other` is the other group)."""
    if kind not in KINDS:
        raise ValueError(f"unknown notice kind {kind!r}: choose from {', '.join(KINDS)}")
    if bank == "human":
        rows = [r for r in load_human(human_path or "")
                if r["kind"] == kind and (kind == "distractor" or r["group"] == group)]
        if not rows:
            raise ValueError(f"no human notice of kind {kind!r} for group {group!r} in {human_path}")
        return rng.choice(rows)["text"]
    if bank not in BANKS:
        raise ValueError(f"unknown notice bank {bank!r}: choose dev, test or human")
    template = rng.choice(BANKS[bank][kind])
    return rng.choice(PREFIXES[bank]) + template.format(group=group, other=other)


class NoticeFeed:
    """Hands each notice, at its tick, to the live robot nearest the map centre. Gossip does the rest.

    There is no broadcast to the fleet: who knows a notice depends on range, loss and delay, as with the ledger."""

    def __init__(self, scenario) -> None:
        self.by_tick: Dict[int, list] = {}
        for n in scenario.notices:
            self.by_tick.setdefault(n.emit_tick, []).append(n)
        self.centre = (scenario.grid.height // 2, scenario.grid.width // 2)
        self.delivered = 0

    def emit(self, t: int, agents) -> None:
        for n in self.by_tick.get(t, []):
            live = [a for a in agents if not a.finished]
            if not live:
                continue
            target = min(live, key=lambda a: (abs(a.pos[0] - self.centre[0]) + abs(a.pos[1] - self.centre[1]), a.id))
            target.ingest_notice(NoticeRecord(n.notice_id, t, n.text))
            self.delivered += 1
