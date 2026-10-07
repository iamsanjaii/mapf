"""Generate notice texts with language models, for the `human` notice bank. THE TEXTS ARE NOT WRITTEN BY PEOPLE.

data/forecasts/README.md asks for notices written by people who have not seen the templates. This script is the
stand-in when that is not available: it asks language models, with no template in the prompt, for one short message
per call (kind and zone group given, in a rotating voice). The output goes to `llm_notices.jsonl`, not to
`human_notices.jsonl`, and every row records which model wrote it. A result computed on these texts must not be
reported as "human" (level 2 of the design); a model scoring the texts another model wrote is a weaker test, and
that has to be said where it is reported.

Model keys come from DOI_LLM_<KEY>_URL and DOI_LLM_<KEY>_MODEL (and OPENAI_API_KEY for api.openai.com URLs):
  export DOI_LLM_SMALL_URL=https://api.openai.com/v1 DOI_LLM_SMALL_MODEL=gpt-4o-mini
  export DOI_LLM_LARGE_URL=https://api.openai.com/v1 DOI_LLM_LARGE_MODEL=gpt-4o

Usage: venv/bin/python experiments/doi_make_notices.py [--models small,large] [--extra 1.25] [--out PATH]
                                                       [--force] [--yes]
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.forecast.budget import confirm_calls

GROUPS = ("north bays", "south bays")
MINIMUM = {"surge": 40, "drop": 40, "distractor": 20}          # the protocol in data/forecasts/README.md
VOICES = ("radio dispatcher", "shift supervisor typing in the team chat", "order picker", "forklift driver",
          "inbound planner", "night-shift lead")
SYSTEM = ("You write one short message that a person at a warehouse would send on a radio or team chat. Reply with "
          "the message only: no quotes, no explanation, at most 20 words. Never include real names, company names, "
          "sites or personal data.")
MAX_CHARS = 200
DEFAULT_OUT = os.path.join(ROOT, "data", "forecasts", "llm_notices.jsonl")


def make_plan(extra: float = 1.25) -> List[Tuple[str, str]]:
    """(kind, group) per call: the protocol's counts plus a margin for texts that are rejected or repeated."""
    per_cell = math.ceil(20 * extra)
    return ([("surge", g) for g in GROUPS for _ in range(per_cell)]
            + [("drop", g) for g in GROUPS for _ in range(per_cell)]
            + [("distractor", "") for _ in range(per_cell)])


def instruction(kind: str, group: str) -> str:
    side = group.split()[0] if group else ""
    if kind == "surge":
        return (f"Write a message saying that work (picking orders) is about to move into the {group}, so that area "
                f"is about to get busy. Say which side, using the word {side}.")
    if kind == "drop":
        return (f"Write a message saying that work in the {group} is about to stop, so that area is about to go "
                f"quiet. Say which side, using the word {side}.")
    return ("Write a message about something that does not change where anyone works or travels: equipment, a "
            "schedule, safety, the canteen, paperwork. Do not use the words north or south.")


def clean(raw: str, kind: str, group: str) -> Optional[str]:
    """The text if it can be used for (kind, group), else None."""
    text = " ".join(raw.split()).strip().strip("\"'").strip()
    if not text or len(text) > MAX_CHARS:
        return None
    lower = text.lower()
    if kind == "distractor":
        return None if ("north" in lower or "south" in lower) else text
    return text if group.split()[0] in lower else None


def main(argv=None, clients: Optional[Dict[str, object]] = None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="small,large", help="model keys, used in turn")
    ap.add_argument("--extra", type=float, default=1.25, help="margin over the protocol's counts")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        ap.error(f"{args.out} exists; use --force to overwrite it")
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    plan = make_plan(args.extra)
    if not confirm_calls(len(plan), 1, args.yes, confirm):
        print("aborted")
        return 1
    if clients is None:
        from src.doi.llm.client import client_from_env
        clients = {k: client_from_env(k) for k in keys}
    rows, seen, failures = [], set(), 0
    for i, (kind, group) in enumerate(plan):
        client = clients[keys[i % len(keys)]]
        user = (f"Kind: {kind}\nGroup: {group}\nVoice: {VOICES[i % len(VOICES)]}\n"
                f"Task: {instruction(kind, group)}")
        try:
            raw = client.complete(SYSTEM, user, max_tokens=60, temperature=1.0).text
            failures = 0
        except Exception as e:
            failures += 1
            print(f"call {i + 1} failed: {type(e).__name__}: {e}", file=sys.stderr)
            if failures >= 5:
                print("five failures in a row, stopping", file=sys.stderr)
                return 2
            continue
        text = clean(raw, kind, group)
        if text is not None and text.lower() not in seen:
            seen.add(text.lower())
            rows.append({"kind": kind, "group": group, "text": text, "source": client.model_id})
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in MINIMUM}
    print(f"kept {len(rows)} of {len(plan)}: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    short = [k for k, need in MINIMUM.items() if counts[k] < need]
    if short:
        print(f"not enough usable texts for {', '.join(short)} (need {MINIMUM}); nothing written. "
              f"Run again with a larger --extra", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.out}\nthese texts were written by language models, not people: do not report results "
          f"on them as the human set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
