"""Plain-language descriptions of scenarios, arms and flags, and the event timeline of a run.

Single source for `run_doi.py --list`, `--guide`, the animation captions and the demo guide.
"""
from typing import Dict, List, Tuple

SCENARIO_NOTES: Dict[str, Tuple[str, str]] = {
    "single_block": ("S", "Two rooms split by a barrier, joined by a long way round (a 3-wide door at the bottom). "
                          "One pallet sits in an opening of the barrier. The basic question: is pushing it away "
                          "worth it?"),
    "two_blocks_parallel": ("S", "Two obstacles in the same barrier; removing either opens a shortcut (SUBSTITUTES)."),
    "series_blocks": ("S", "A one-cell corridor with two obstacles in a row. Neither can be pushed clear (each "
                           "blocks the other), so this shows the case where pushing is impossible."),
    "multi_block_wall": ("S", "Four obstacles spread along the barrier. Several pushes may pay, at different "
                              "times; a good all-round demo with 8 to 12 robots."),
    "shift": ("S", "Like multi_block_wall, but half way through the run the busy region moves, so old evidence "
                   "becomes stale."),
    "complements": ("S", "Three rooms in a row; each wall has a doorway at the top blocked by a pallet and open "
                         "doors at the bottom. Opening one doorway saves nothing; opening both saves a lot (the "
                         "obstacles are complements)."),
    "random_blocks": ("S", "An open floor with removable obstacles (pallets, crates, shelf units) anywhere, and "
                           "optional wall strips. Isolated obstacles cost a step or two of detour, so only a "
                           "cluttered floor or a cheap push (low kappa) gives pushing anything to win."),
    "incidents_room": ("D", "Two rooms, three one-cell doors. Obstructions (a pallet, a spill) appear in the "
                            "doorways at random times; at most two, so one door stays open."),
    "incidents_aisles": ("D", "A warehouse-shaped map: one-cell-wide shelf aisles and three cross aisles. An "
                              "obstruction in an aisle forces a detour. Reports are text like 'pallet down in "
                              "aisle 7 bay 3'."),
    "warehouse_racks": ("S", "Two rooms split by a barrier; crates sit in its openings and racks stand along the "
                             "aisles. A robot can push a crate aside, or lift it and put it on a rack, which "
                             "clears it for good. Racks hold one crate each."),
    "dump_central": ("S", "The same barrier with pallets, a crate and a shelf unit in it, and one dump region (a "
                          "block of slots) in a far corner. Hauling to the dump is far, so the choice between "
                          "pushing aside and carrying away is visible; only the dump takes the shelf unit."),
    "site_pits": ("S", "A construction-site barrier: three of its openings are pits that block the way, and "
                       "pieces of debris lie about. A robot can lift debris and drop it into a pit, which fills "
                       "it for good and opens the path."),
    "mixed": ("S", "Racks, a dump region and a pit together in one barrier: a crate, a shelf unit (dump only), a "
                   "pallet and a pit, with debris to fill it. Each robot picks the cheapest of push, carry or "
                   "fill for each obstacle."),
    "warehouse_blocks": ("S", "A benchmark warehouse map (MovingAI) with obstacles placed in one-cell corridors, "
                              "many robots, a fixed time horizon and throughput as the measure."),
    "warehouse_incidents": ("D", "The same benchmark maps with incidents appearing in corridors."),
}

ARM_NOTES: Dict[str, str] = {
    "never": "never pushes (pays every detour)",
    "myopic": "pushes only if it is cheaper for ME than the detour (ignores the rest of the fleet)",
    "eager": "pushes as soon as the ledger shows any saving for the fleet (theta = 0)",
    "rof": "Rent-or-Fill: pushes when the fleet's accumulated detour cost reaches the price of the push",
    "rof_local": "Rent-or-Fill without sharing the ledger (every robot only knows its own detours)",
    "rof_f": "Rent-or-Fill that forecasts the fleet's remaining traffic from what it has seen",
    "rof_r": "Rent-or-Fill with a random threshold between 0 and 1 times the price (expected ratio e/(e-1))",
    "rof_p": "Rent-or-Fill that lowers its threshold when its forecast says the push will pay, and raises it when not",
    "central": "Rent-or-Fill with one omniscient ledger and map (the cost of NOT being decentralised)",
    "hindsight": "benchmark: knows every task in advance, removes the best obstacles at tick 0, charged the lowest "
                 "possible price",
    "free": "benchmark: every obstacle is gone at tick 0, for free (the travel nobody can avoid)",
}

# group -> list of (flag, what it is, effect in the story)
FLAG_GUIDE: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("WHAT TO RUN", [
        ("--play", "Short interactive mode: asks for the grid size, number of robots and how many L, C and S, "
                   "runs, opens the window, and offers another go.", "The quickest way to try your own numbers."),
        ("--demo scatter", "An open floor with obstacles anywhere (no barrier); each robot has one start and one "
                           "goal.", "The easiest picture of the mechanics. Uses kappa 1 and seed 8."),
        ("--sim NAME", "Simulation only: run one arm and open the window (NAME: toy, fleet, scatter or a scenario).",
         "The visual to show. The terminal prints a few lines."),
        ("--demo toy", "One robot, one shelf unit in a gap, four crossings.",
         "The cleanest picture of the rent-or-push rule. Start here."),
        ("--demo fleet", "8 robots, 4 obstacles in a barrier.", "Shows sharing, collateral and congestion."),
        ("--scenario NAME", "Which map and task stream (see --list).", "Everything else is a knob on top of it."),
        ("--policy a,b,c", "Which decision rules (arms) to compare, same tasks for each.",
         "never = baseline, rof = the method, central = what a boss with full information would do."),
        ("--robots N / --tasks K", "Fleet size and tasks per robot.",
         "More robots or tasks means more total detour cost, so pushing pays sooner."),
        ("--seed S", "Random seed for starts, goals and incidents.", "Same seed gives the same run."),
    ]),
    ("BUILD YOUR OWN MAP (asks in the terminal for anything not given; --yes accepts defaults)", [
        ("--build", "Start the interactive set-up: layout, map, fleet, costs, arms, then a map preview.",
         "Prints the equivalent one-line command at the end so the run can be repeated exactly."),
        ("--layout barrier|strips|map", "barrier: two zones split by a barrier with removable obstacles in it. "
                                        "strips: random wall strips plus obstacles anywhere. map: your own ASCII "
                                        "file.", ""),
        ("--rows / --cols", "Grid size.", ""),
        ("--blocks N / --doors N / --wall-col C", "Barrier: removable obstacles in it, door rows at the bottom, "
                                                  "its column.", "Fewer doors or more crossing trips make a push "
                                                                 "pay sooner."),
        ("--kind K", "Barrier: mixed (default), pallet, crate or shelf_unit.", "Heavier kinds cost more per push step."),
        ("--crossing Q", "Barrier: share of trips that cross it, 0-1.", "0 means nothing is worth removing."),
        ("--strips N / --strip-min A / --strip-max B", "Strips: number and length of the permanent wall strips.", ""),
        ("--pallets N / --crates N / --shelves N", "Strips: how many removable obstacles of each kind.",
         "A crate weighs 0.5, a pallet 1.0, a shelf unit 2.0."),
        ("--map FILE", "ASCII map: # wall, . free, L pallet, C crate, S shelf unit, P pit, T rack, D dump slot.",
         "Robot starts and tasks are drawn from the seed."),
    ]),
    ("COSTS (what 'worth it' means)", [
        ("--kappa X", "A push step costs kappa times the obstacle's weight (default 4).",
         "Higher: pushing gets more expensive, so it takes more detour cost to justify."),
        ("--fee X", "Fixed cost per push run (default 1).", "Raise it to 50 or 200 and pushing stops being worth it."),
        ("--kappa-c X", "A loaded step costs kappa-c times the carried kind's weight (default 2, at least 2).",
         "Higher: hauling gets more expensive, so far racks and the dump zone lose to pushing."),
        ("--pick-fee X / --drop-fee X", "Fixed cost of lifting an obstacle / of dropping it into a slot or pit "
                                        "(default 1 each).", "Raise both and carrying stops being worth it."),
        ("--push-max N", "Longest straight push a robot will plan (default 6).", ""),
        ("--theta X", "Trigger multiplier (default 1).",
         "0 = Eager (push at any saving), 1 = ski rental, 2 = wait for twice the evidence."),
        ("--bundle-max N", "1 = single pushes (default), 2 = also two-step plans.",
         "With 2, a robot can plan two pushes in a row, which pays when two obstacles must both go (complements)."),
        ("--lam X", "rof_p only: threshold multiplier when the forecast says the push will pay (default 0.5).",
         "The multiplier is 1/X when the forecast says it will not; 1 makes rof_p behave like rof."),
    ]),
    ("INFORMATION (who knows what)", [
        ("--r-comm R", "How far a robot's ledger messages reach (0 = none, inf = everywhere).",
         "0 means every robot learns detour costs alone; it takes longer to reach the threshold."),
        ("--loss P", "Chance each ledger message is dropped.", "Models a bad wireless link."),
        ("--latency L", "Ticks a ledger message takes to arrive.", "Stale information delays decisions."),
    ]),
    ("LANGUAGE LAYER (incident scenarios)", [
        ("--intake none|oracle", "How incident reports reach the fleet.",
         "none: robots learn of an obstruction only by seeing it, and never push one whose class is unknown. "
         "oracle: a perfect structured report arrives (the stand-in for a language model that reads the text)."),
        ("--p-false / --p-report", "How often a report is false / how often an incident gets reported at all.",
         "False reports cost detours until a robot sees the cell is clear."),
        ("--p-wrong-class", "Chance a report wrongly says 'robots can clear this'.",
         "The world refuses the push and the robot learns the obstacle needs a human."),
    ]),
    ("OUTPUT", [
        ("--show / --no-show", "Open (or skip) the animation window; on by default for --demo and --sim.", ""),
        ("--html PATH", "Save a self-contained player (play, pause, scrub) for any browser.", "Best way to share."),
        ("--gif PATH", "Save a GIF of the same view.", "Open in a browser; macOS Preview splits the frames."),
        ("--gif-arms a,b", "Which arms appear in the animation (default never,rof).", ""),
        ("--verbose", "Full text output: setup, event timelines, benchmarks, statistics.",
         "Hidden by default so the window is the demo."),
        ("--replay / --replay-every N", "Print ASCII frames of one arm.", "Quick look in the terminal."),
        ("--no-benchmarks", "Skip the free / hindsight reference runs (with --verbose).",
         "Faster, but no HR_av column."),
    ]),
]

COLUMN_NOTES = [
    ("J", "total cost = moves + waits + push steps x kappa x weight + pushes x fee + loaded steps x kappa-c x "
          "weight + picks x pick fee + drops x drop fee (lower is better)"),
    ("carried / filled", "obstacles taken to a rack or the dump region / pits filled with debris"),
    ("slot_conflicts", "drops refused because the slot was already full (or the pit already filled): the cost of "
                       "not knowing what the others have done"),
    ("HR_av", "(J_arm - J_free) / (J_hindsight - J_free): how many times worse than hindsight, on the avoidable cost"),
    ("PoD", "J_arm / J_central: price of deciding from local, delayed information"),
    ("collateral", "detour paid because a pushed obstacle was left where it blocks a route (measured on the true map)"),
    ("rejected", "pushes the world refused (no room, obstacle gone, or it needs a human)"),
    ("overrides", "times the physics layer stopped two robots colliding (the safety layer, not the planner)"),
]


SLOT_LABEL = {"rack": "rack", "dump": "dump zone"}


def events(result, scenario) -> List[Tuple[int, str]]:
    ev: List[Tuple[int, str]] = []
    for tr in result.triggers:
        rel = ">=" if tr["known"] >= tr["buy"] else "<  (the arm does not wait for the ledger)"
        mode = tr.get("mode", "push")
        if mode == "carry":
            what = f"to carry the {tr['kind']} at {list(tr['cells'][0])} to {list(tr['landing'])}"
        elif mode == "fill":
            what = f"to fill the pit at {list(tr['landing'])} with the {tr['kind']} at {list(tr['cells'][0])}"
        else:
            what = f"to push the {tr['kind']} at {list(tr['cells'][0])} {tr['steps']} cell(s)"
        ev.append((tr["tick"], f"robot {tr['robot']} DECIDES: saving {tr['known']:.0f} {rel} price {tr['buy']:.0f} "
                               f"{what}"))
    for e in result.carry_log:
        if e["mode"] == "fill":
            ev.append((e["tick"], f"robot {e['robot']} FILLS the pit at {list(e['target'])} with {e['kind']} from "
                                  f"{list(e['origin'])}"))
        else:
            label = SLOT_LABEL[scenario.slots[e["target"]]]
            ev.append((e["tick"], f"robot {e['robot']} CARRIES the {e['kind']} from {list(e['origin'])} to the "
                                  f"{label} at {list(e['target'])}"))
    for run in result.pushes:
        ev.append((run["start_tick"], f"robot {run['robot']} PUSHES the {run['kind']} from {list(run['origin'])} "
                                      f"to {list(run['landing'])} ({run['steps']} step(s), until tick "
                                      f"{run['end_tick']})"))
    for oid, tick in sorted(result.appeared_at.items()):
        inc = next(i for i in scenario.incidents if i.oid == oid)
        ev.append((tick, f"incident {oid} appears at {list(inc.cells)} ({inc.kind}, {inc.cls})"))
    if result.stalled:
        ev.append((result.ticks, "STALLED: no progress for stall_ticks"))
    return sorted(ev, key=lambda x: x[0])
