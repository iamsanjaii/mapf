"""Plain-language descriptions of scenarios, arms and flags, and the event timeline of a run.

Single source for `run_doi.py --list`, `--guide`, the animation captions and the demo guide.
"""
from typing import Dict, List, Tuple

SCENARIO_NOTES: Dict[str, Tuple[str, str]] = {
    "single_pit": ("S", "Two rooms joined by a long way round (a 3-wide door at the bottom). One pit in the wall "
                        "near the top is a shortcut. The basic ski-rental question: is one fill worth it?"),
    "two_pits_parallel": ("S", "Two pits in the same wall; either one alone opens a shortcut (SUBSTITUTES). "
                               "Tests that after one fill the robots do not pay for a second pit they no longer need."),
    "series_pits": ("S", "A one-cell corridor with two pits in a row (COMPLEMENTS): filling only one saves nothing, "
                         "so a rule that looks at one pit at a time never fills. The bundle ledger does."),
    "multi_pit_wall": ("S", "Four pits spread along the wall. Several fills may pay, at different times; a good "
                            "all-round demo with 8 to 12 robots."),
    "shift": ("S", "Like multi_pit_wall, but half way through the run the busy region moves, so old evidence "
                   "becomes stale (the stretch experiment E5)."),
    "random_pits": ("S", "A random obstacle map with random pits (the legacy generator). Some goals are only "
                         "reachable through a pit, so 'never' can get stuck."),
    "incidents_room": ("D", "Two rooms, three one-cell doors, no pits. Obstructions (a pallet, a spill) appear in "
                            "the doorways at random times; at most two, so one door stays open."),
    "incidents_aisles": ("D", "A warehouse-shaped map: one-cell-wide shelf aisles and three cross aisles. An "
                              "obstruction in an aisle forces a detour. Reports are text like 'pallet down in "
                              "aisle 7 bay 3'."),
    "warehouse_pits": ("S", "Stage 1: a real benchmark map (MovingAI) with pits placed in one-cell corridors, "
                            "hundreds of robots, a fixed time horizon and throughput as the measure."),
    "warehouse_incidents": ("D", "Stage 1: the same benchmark maps with incidents appearing in corridors."),
}

ARM_NOTES: Dict[str, str] = {
    "never": "never edits the map (pays every detour)",
    "myopic": "edits only if ONE task's detour already pays for the edit",
    "eager": "edits as soon as any detour is seen (theta = 0)",
    "rof": "Rent-or-Fill: edits when the fleet's accumulated detour cost reaches the edit cost",
    "rof_pit": "Rent-or-Fill with a per-cell ledger (ablation: fails on pits in series)",
    "rof_local": "Rent-or-Fill without sharing the ledger (ablation: every robot only knows its own detours)",
    "rof_w": "Rent-or-Fill that forgets evidence older than a window (stretch, for shifting demand)",
    "rof_x": "Rent-or-Fill that extrapolates what it knows to robots it never heard from (stretch)",
    "central": "Rent-or-Fill with one omniscient ledger (the cost of NOT being decentralised)",
    "hindsight": "benchmark: knows every task in advance, opens the best pits at t=0, is charged the buy cost",
    "free": "benchmark: every blocked cell is open for free (the travel nobody can avoid)",
}

# group -> list of (flag, what it is, effect in the story)
FLAG_GUIDE: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("WHAT TO RUN", [
        ("--demo toy", "One robot, one pit, four crossings of the wall.",
         "The cleanest picture of the ski-rental rule. Start here."),
        ("--demo fleet", "8 robots, 4 pits in a wall.", "Shows sharing, claims and congestion."),
        ("--scenario NAME", "Which map and task stream (see --list).", "Everything else is a knob on top of it."),
        ("--policy a,b,c", "Which decision rules (arms) to compare, same tasks for each.",
         "never = baseline, rof = the method, central = what a boss with full information would do."),
        ("--robots N / --tasks K", "Fleet size and tasks per robot.",
         "More robots or tasks means more total detour cost, so filling pays sooner."),
        ("--seed S", "Random seed for starts, goals and incidents.", "Same seed gives the same run."),
    ]),
    ("COSTS (what 'worth it' means)", [
        ("--kappa X", "Cost per step while carrying a kit (default 4).",
         "Higher: the edit gets more expensive when the depot is far from the pit."),
        ("--fee X", "Fixed cost per edit (default 1).", "Raise it to 50 or 200 and filling stops being worth it."),
        ("--theta X", "Trigger multiplier (default 1).",
         "0 = Eager (fill at any detour), 1 = ski rental, 2 = wait for twice the evidence."),
    ]),
    ("INFORMATION (who knows what)", [
        ("--r-comm R", "How far a robot's ledger messages reach (0 = none, inf = everywhere).",
         "0 means every robot learns detour costs alone; it takes longer to reach the threshold."),
        ("--loss P", "Chance each ledger message is dropped.", "Models a bad wireless link."),
        ("--latency L", "Ticks a ledger message takes to arrive.", "Stale information delays decisions."),
        ("--claim on|off", "A lease-based lock so only one robot hauls a kit to a pit.",
         "Off: several robots may haul at once and the extras are wasted."),
    ]),
    ("LANGUAGE LAYER (incident scenarios)", [
        ("--intake none|oracle", "How incident reports reach the fleet.",
         "none: robots learn of an obstruction only by seeing it. oracle: a perfect structured report arrives "
         "(the stand-in for a language model that reads the text)."),
        ("--gate", "A human supervisor must approve each permanent edit (simulated).",
         "Adds a wait before hauling; vetoes catch edits on cells that need a human."),
        ("--p-false / --p-report", "How often a report is false / how often an incident gets reported at all.",
         "False reports cost detours until a robot sees the cell is clear."),
        ("--p-wrong-class", "Chance a report wrongly says 'robots can clear this'.",
         "With the gate on, the supervisor catches most of these."),
    ]),
    ("OUTPUT", [
        ("--replay / --replay-every N", "Print ASCII frames of one arm.", "Quick look in the terminal."),
        ("--gif PATH", "Save an animation (side by side, running cost counter).", "The demo to show."),
        ("--gif-arms a,b", "Which arms appear in the animation (default never,rof).", ""),
        ("--show", "Open the animation in a window instead of saving.", ""),
        ("--no-benchmarks", "Skip the free / hindsight reference runs.", "Faster, but no HR_av column."),
    ]),
]

COLUMN_NOTES = [
    ("J", "total cost = moves + waits + carry steps x kappa + fills x fee (lower is better)"),
    ("HR_av", "(J_arm - J_free) / (J_hindsight - J_free): how many times worse than hindsight, on the avoidable cost"),
    ("PoD", "J_arm / J_central: price of deciding from local, delayed information"),
    ("wasted", "cost of hauls that were started and then abandoned (a kit carried for nothing)"),
    ("overrides", "times the physics layer stopped two robots colliding (the safety layer, not the planner)"),
]


def events(result, scenario) -> List[Tuple[int, str]]:
    ev: List[Tuple[int, str]] = []
    for tr in result.triggers:
        rel = ">=" if tr["known"] >= tr["buy"] else "<  (theta = 0 fires on any positive evidence)"
        ev.append((tr["tick"], f"robot {tr['robot']} TRIGGER: known detour cost {tr['known']:.0f} {rel} buy cost "
                               f"{tr['buy']:.0f} for pits {list(tr['pits'])}"))
    for e in result.edits:
        wait = f", approval wait {e['approval_wait']}" if e["approval_wait"] else ""
        ev.append((e["claim_tick"], f"hauling starts for pit {e['pit']} (offered at tick {e['trigger_tick']}{wait})"))
        ev.append((e["fill_tick"], f"FILLED pit {e['pit']}: estimated cost {e['B_est']:.0f}, realised "
                                   f"{e['B_real']:.0f}"))
    for oid, tick in sorted(result.appeared_at.items()):
        inc = next(i for i in scenario.incidents if i.oid == oid)
        ev.append((tick, f"incident {oid} appears at {list(inc.cells)} ({inc.kind}, {inc.cls})"))
    if result.stalled:
        ev.append((result.ticks, "STALLED: no progress for stall_ticks"))
    return sorted(ev, key=lambda x: x[0])


def short_events(result, scenario) -> List[Tuple[int, str]]:
    """Compact captions for the animation (one line each, no duplicate triggers)."""
    out: List[Tuple[int, str]] = []
    seen_trigger = set()
    for tick, text in events(result, scenario):
        if "TRIGGER" in text:
            key = text.split("for pits")[-1]
            if key in seen_trigger:
                continue
            seen_trigger.add(key)
            out.append((tick, "fleet's detour cost has reached the edit cost" if "theta" not in text
                        else "edit triggered on first detour"))
        elif "hauling starts" in text:
            out.append((tick, "a robot fetches a kit"))
        elif "FILLED" in text:
            out.append((tick, "pit FILLED, shortcut open" + text.split("FILLED pit")[-1].split(":")[0]))
        elif "incident" in text:
            out.append((tick, text))
    return out
