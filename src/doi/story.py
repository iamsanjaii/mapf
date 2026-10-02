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
    pushed_tick: Optional[int]      # first tick at which a robot pushed the obstacle off this cell, if ever
    decided_tick: Optional[int]     # first tick at which a robot decided to push it, if ever

    def at(self, t: int) -> float:
        return float(self.cumulative[min(t, len(self.cumulative) - 1)])

    def pushed_by(self, t: int) -> bool:
        return self.pushed_tick is not None and self.pushed_tick <= t

    def status(self, t: int) -> str:
        if self.pushed_by(t):
            return "pushed away"
        if self.decided_tick is not None and self.decided_tick <= t:
            return "pushing"
        return "over the price, not pushed" if self.at(t) >= self.price else ""


def cell_name(cells) -> str:
    cells = [cells] if isinstance(cells, tuple) and len(cells) == 2 and isinstance(cells[0], int) else cells
    return " + ".join(f"({r},{c})" for r, c in cells)


def kind_label(kind: str) -> str:
    return kind.replace("_", " ")


def obstacle_prices(results: Dict[str, object]) -> Dict[Pos, Tuple[float, str]]:
    """Price (and kind) of pushing each obstacle cell, as quoted by the first arm that decided to push it."""
    prices: Dict[Pos, Tuple[float, str]] = {}
    for res in results.values():
        for tr in sorted(res.triggers, key=lambda x: x["tick"]):
            for cell in tr["cells"]:
                prices.setdefault(cell, (float(tr["buy"]), tr["kind"]))
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
        pushed = min((p["start_tick"] for p in result.pushes if p["origin"] == cell), default=None)
        decided = min((tr["tick"] for tr in result.triggers if cell in tr["cells"]), default=None)
        if steps.sum() > 0 or pushed is not None:
            meters.append(Meter(cell, kind, price, np.cumsum(steps), pushed, decided))
    meters.sort(key=lambda m: -m.cumulative[-1])
    return meters[:max_meters]


def plain_events(result, scenario) -> List[Tuple[int, str]]:
    """What happened, in sentences a reviewer can read cold."""
    out: List[Tuple[int, str]] = []
    for tr in sorted(result.triggers, key=lambda x: x["tick"]):
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
