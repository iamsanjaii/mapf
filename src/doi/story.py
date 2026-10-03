"""The rent-or-push story of a run in plain language: rent meters, captions and a one-line verdict.

Pure data (no matplotlib), shared by the animation window, the HTML player and the terminal summary.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Pos = Tuple[int, int]
WORDS = {(-1, 0): "up", (1, 0): "down", (0, 1): "right", (0, -1): "left"}


@dataclass
class Meter:
    """Detour cost ("rent") paid so far around one obstacle, against the price of pushing it away."""
    cell: Pos
    kind: str
    price: float
    cumulative: np.ndarray          # rent paid up to and including each tick
    pushed_tick: Optional[int]      # first tick at which a robot cleared this cell (pushed, carried, filled), if ever
    decided_tick: Optional[int]     # first tick at which a robot decided to clear it, if ever
    how: str = "pushed"             # pushed, carried or filled

    def at(self, t: int) -> float:
        return float(self.cumulative[min(t, len(self.cumulative) - 1)])

    def pushed_by(self, t: int) -> bool:
        return self.pushed_tick is not None and self.pushed_tick <= t

    def status(self, t: int) -> str:
        if self.pushed_by(t):
            return {"pushed": "pushed away", "carried": "carried away", "filled": "filled"}[self.how]
        if self.decided_tick is not None and self.decided_tick <= t:
            return {"pushed": "pushing", "carried": "carrying", "filled": "filling"}[self.how]
        return "over the price, not pushed" if self.at(t) >= self.price else ""


def cell_name(cells) -> str:
    cells = [cells] if isinstance(cells, tuple) and len(cells) == 2 and isinstance(cells[0], int) else cells
    return " + ".join(f"({r},{c})" for r, c in cells)


def kind_label(kind: str) -> str:
    return kind.replace("_", " ")


def _subject(tr: dict) -> Tuple[Pos, ...]:
    """The cells a decision is about: the pit for a fill (the debris is only the means), else the obstacle."""
    return (tr["landing"],) if tr.get("mode") == "fill" else tr["cells"]


def obstacle_prices(results: Dict[str, object]) -> Dict[Pos, Tuple[float, str]]:
    """Price (and kind) of pushing each obstacle cell, as quoted by the first arm that decided to push it."""
    prices: Dict[Pos, Tuple[float, str]] = {}
    for res in results.values():
        for tr in sorted(res.triggers, key=lambda x: x["tick"]):
            kind = "pit" if tr.get("mode") == "fill" else tr["kind"]
            for cell in _subject(tr):
                prices.setdefault(cell, (float(tr["buy"]), kind))
    return prices


def rent_meters(result, prices: Dict[Pos, Tuple[float, str]], max_meters: int = 4) -> List[Meter]:
    """One meter per obstacle that starts on the map (or appears as an incident) and has a price, biggest first.

    Cells where a pushed obstacle merely landed are not obstacles' homes, so they get no meter."""
    n = result.ticks + 1
    meters: List[Meter] = []
    homes = set(result.scenario.obstacles) | {c for inc in result.scenario.incidents for c in inc.cells}
    for cell, (price, kind) in prices.items():
        if cell not in homes:
            continue
        steps = np.zeros(n)
        for info in result.plan_infos:
            if info.rent > 0 and cell in info.bundle and info.tick < n:
                steps[info.tick] += info.rent
        cleared = [(p["start_tick"], "pushed") for p in result.pushes if p["origin"] == cell]
        cleared += [(e["tick"], "carried" if e["mode"] == "carry" else "filled") for e in result.carry_log
                    if (e["origin"] == cell and e["mode"] == "carry") or (e["target"] == cell and e["mode"] == "fill")]
        pushed, how = min(cleared, default=(None, "pushed"))
        decided = min((tr["tick"] for tr in result.triggers if cell in _subject(tr)), default=None)
        if steps.sum() > 0 or pushed is not None:
            meters.append(Meter(cell, kind, price, np.cumsum(steps), pushed, decided, how))
    meters.sort(key=lambda m: -m.cumulative[-1])
    return meters[:max_meters]


SLOT_NAME = {"rack": "rack", "dump": "dump zone"}


def _clear_decision(tr: dict, mode: str, scenario) -> str:
    """The caption for a decision to carry an obstacle to a slot or to fill a pit with it."""
    known, buy, robot = tr["known"], tr["buy"], tr["robot"]
    paid = known >= buy and buy > 0
    if mode == "fill":
        what = f"the pit at {cell_name(tr['landing'])} with the {kind_label(tr['kind'])} at {cell_name(tr['cells'])}"
        if paid:
            return f"Detours so far cost {known:.0f}, filling {what} costs {buy:.0f}: robot {robot} decides to fill it"
        return f"Robot {robot} fills {what}: it is cheaper than going round"
    slot = SLOT_NAME[scenario.slots[tr["landing"]]]
    what = f"the {kind_label(tr['kind'])} at {cell_name(tr['cells'])} to the {slot} at {cell_name(tr['landing'])}"
    if paid:
        return (f"Detours so far cost {known:.0f}, carrying {what} costs {buy:.0f}: "
                f"robot {robot} decides to carry it")
    return f"Robot {robot} carries {what}: it is cheaper than going round"


def plain_events(result, scenario) -> List[Tuple[int, str]]:
    """What happened, in sentences a reviewer can read cold."""
    out: List[Tuple[int, str]] = []
    for tr in sorted(result.triggers, key=lambda x: x["tick"]):
        mode = tr.get("mode", "push")
        if mode != "push":
            out.append((tr["tick"], _clear_decision(tr, mode, scenario)))
            continue
        what = f"the {kind_label(tr['kind'])} at {cell_name(tr['cells'])}"
        if tr["known"] >= tr["buy"] and tr["buy"] > 0:
            out.append((tr["tick"], f"Detours so far cost {tr['known']:.0f}, pushing {what} costs "
                                    f"{tr['buy']:.0f}: robot {tr['robot']} decides to push it"))
        else:
            out.append((tr["tick"], f"Robot {tr['robot']} pushes {what}: it is cheaper than going round"))
    for run in result.pushes:
        o, l, n = run["origin"], run["landing"], run["steps"]
        way = WORDS[((l[0] > o[0]) - (l[0] < o[0]), (l[1] > o[1]) - (l[1] < o[1]))]
        out.append((run["start_tick"], f"Robot {run['robot']} pushes the {kind_label(run['kind'])} at "
                                       f"{cell_name(o)} {n} cell{'s' if n != 1 else ''} {way}, to {cell_name(l)}"))
    for e in result.carry_log:
        if e["mode"] == "fill":
            out.append((e["tick"], f"Robot {e['robot']} filled the pit at {cell_name(e['target'])} with "
                                   f"{kind_label(e['kind'])} from {cell_name(e['origin'])}"))
        else:
            out.append((e["tick"], f"Robot {e['robot']} carried the {kind_label(e['kind'])} from "
                                   f"{cell_name(e['origin'])} to the {SLOT_NAME[scenario.slots[e['target']]]} at "
                                   f"{cell_name(e['target'])}"))
    for oid, tick in sorted(result.appeared_at.items()):
        inc = next(i for i in scenario.incidents if i.oid == oid)
        out.append((tick, f"{inc.kind.capitalize()} appears at {cell_name(inc.cells)}: robots must go round it "
                          f"or push it aside"))
    if result.stalled:
        out.append((result.ticks, "Stalled: no robot can make progress"))
    return sorted(out, key=lambda x: x[0])


def caption_at(events: List[Tuple[int, str]], t: int, idle: str) -> Tuple[str, str]:
    """(latest event, the one before it) as of tick t; `idle` is shown until something happens."""
    done = [text for tick, text in events if tick <= t]
    if not done:
        return idle, ""
    return done[-1], (done[-2] if len(done) > 1 else "")


def detour_paid(result) -> float:
    """Total detour (rent) the robots' routes paid because of obstacles, as each robot planned its task."""
    return float(sum(i.rent for i in result.plan_infos))


def verdict(results: Dict[str, object], main: str = "rof", baseline: str = "never") -> str:
    """One sentence on how the main arm did against the baseline."""
    if main not in results or baseline not in results:
        names = ", ".join(f"{n} {r.J:.0f}" for n, r in results.items())
        return f"total cost: {names}"
    m, b = results[main], results[baseline]
    rent = detour_paid(b)
    if m.J == b.J and m.removals == 0 and rent == 0:
        return (f"{main} matches {baseline} ({m.J:.0f}): no robot's route ran through an obstacle, so there was "
                f"nothing to push and the cost is just the walking.")
    if m.J < b.J:
        what = f"it pushed {m.removals} obstacle{'s' if m.removals != 1 else ''} aside once the detours added up"
        return f"{main} saves {b.J - m.J:.0f} ({100 * (b.J - m.J) / b.J:.0f}%) against {baseline}: {what}."
    if m.removals == 0:
        return (f"{main} matches {baseline} ({m.J:.0f}): obstacles cost the robots {rent:.0f} steps of detour in "
                f"total, but no push was cheaper than going round, so it did not push.")
    return f"{main} costs {m.J - b.J:.0f} more than {baseline} ({m.J:.0f} vs {b.J:.0f}): pushing did not pay back."
