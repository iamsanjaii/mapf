"""Interactive set-up in the style of the legacy main.py: ask, validate, build, preview, confirm.

`configure` fills an argparse Namespace from the terminal (flags given on the command line are not asked
again), builds the scenario with src.doi.builder and returns it. Reading input uses `input`, so tests can
feed answers; end of input accepts the default.
"""
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.doi.builder import BuildError, LAYOUTS, build, check
from src.doi.config import POLICIES, SimConfig
from src.doi.kinds import KINDS

Field = Tuple[str, str, Callable, object, Callable, str]   # dest, prompt, cast, default, validate, hint

BOX = ("╔══════════════════════════════════════════╗\n"
       "║   Rent-or-Fill Simulator: Configuration  ║\n"
       "╚══════════════════════════════════════════╝")

LAYOUT_HELP = {
    "barrier": "two zones split by a barrier (a rack line, conveyor or trench) with removable obstacles in its "
               "openings; the long way round is through doors at the bottom",
    "strips": "an open floor with removable obstacles (pallets, crates, shelf units) anywhere, and optional wall strips",
    "map": "your own ASCII map file (# wall, . free, L pallet, C crate, S shelf unit)",
}


def _between(lo, hi):
    return lambda v: lo <= v <= hi


LAYOUT_FIELDS: Dict[str, List[Field]] = {
    "barrier": [
        ("rows", "Grid rows (height)", int, 15, lambda v: v >= 8, "at least 8"),
        ("cols", "Grid columns (width)", int, 21, lambda v: v >= 9, "at least 9"),
        ("wall_col", "Barrier column", int, lambda a: a.cols // 2, lambda v: v >= 2, "at least 2, leaving room"),
        ("blocks", "Removable obstacles in the barrier", int, 4, lambda v: v >= 1, "at least 1"),
        ("doors", "Door rows at the bottom (the long way round)", int, 3, lambda v: v >= 1, "at least 1"),
        ("kind", "Kind of obstacle (mixed, pallet, crate, shelf_unit)", str, "mixed",
         lambda v: v == "mixed" or v in KINDS, "mixed, pallet, crate or shelf_unit"),
        ("crossing", "Share of trips that cross the barrier 0-1", float, 0.8, _between(0, 1), "between 0 and 1"),
    ],
    "strips": [
        ("rows", "Grid rows (height)", int, 12, lambda v: v >= 5, "at least 5"),
        ("cols", "Grid columns (width)", int, 16, lambda v: v >= 5, "at least 5"),
        ("strips", "Number of permanent wall strips", int, 0, lambda v: v >= 0, "0 or more"),
        ("strip_min", "Shortest strip (cells)", int, 4, lambda v: v >= 1, "at least 1"),
        ("strip_max", "Longest strip (cells)", int, 8, lambda v: v >= 1, "at least 1"),
        ("pallets", "Pallets (weight 1.0), anywhere on the floor", int, lambda a: round(0.11 * a.rows * a.cols),
         lambda v: v >= 0, "0 or more"),
        ("crates", "Crates (light, weight 0.5)", int, lambda a: round(0.07 * a.rows * a.cols),
         lambda v: v >= 0, "0 or more"),
        ("shelves", "Shelf units (heavy, weight 2.0)", int, lambda a: round(0.04 * a.rows * a.cols),
         lambda v: v >= 0, "0 or more"),
    ],
    "map": [
        ("map", "Path to the ASCII map file", str, None, lambda v: bool(v), "a file path"),
    ],
}

COMMON_FIELDS: List[Field] = [
    ("robots", "Number of robots", int, 8, lambda v: v >= 1, "at least 1"),
    ("tasks", "Trips per robot (1 = one start and one goal)", int, 1, lambda v: v >= 1, "at least 1"),
    ("seed", "Random seed", int, 0, lambda v: True, "a whole number"),
    ("kappa", "Cost of a push step = kappa x the obstacle's weight", float, 1.0, lambda v: v >= 0, "0 or more"),
    ("fee", "Fixed cost per push run (fee)", float, 1.0, lambda v: v >= 0, "0 or more"),
    ("push_max", "Longest straight push a robot will plan (cells)", int, 6, lambda v: v >= 1, "at least 1"),
    ("r_comm", "Message radius in cells (0 = no sharing, inf = all)", float, 8.0, lambda v: v >= 0, "0 or more"),
    ("loss", "Message loss probability 0-1", float, 0.0, _between(0, 1), "between 0 and 1"),
    ("latency", "Message delay in ticks", int, 1, lambda v: v >= 1, "at least 1"),
    ("bundle_max", "Pushes planned in a row (1 = single, 2 = two-step plans)", int, 1, lambda v: v in (1, 2),
     "1 or 2"),
    ("lam", "rof_p threshold multiplier when its forecast says the push pays", float, 0.5,
     lambda v: 0 < v <= 1, "above 0 and at most 1"),
    ("policy", "Arms to compare, comma separated", str, "never,rof", lambda v: all(p in POLICIES for p in
                                                                                 v.split(",")), "names from --list"),
]


def ask(prompt: str, cast: Callable, default, validate: Callable = lambda v: True, hint: str = "",
        input_fn: Callable = input, out: Callable = print):
    while True:
        try:
            raw = input_fn(f"  {prompt} [{default}]: ").strip()
        except EOFError:
            return default
        try:
            value = cast(raw) if raw else default
            if not validate(value):
                raise ValueError
            return value
        except (ValueError, TypeError):
            out(f"    ✗  Invalid input{': ' + hint if hint else ''}. Please try again.")


def ask_yes(prompt: str, default: bool, input_fn: Callable = input) -> bool:
    try:
        raw = input_fn(f"  {prompt} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
    except EOFError:
        return default
    return default if not raw else raw.startswith("y")


def _fill(args, fields: List[Field], explicit: Set[str], yes: bool, input_fn, out) -> None:
    """Set each field from the command line, the previous answer, or the default, asking unless `yes`."""
    for dest, prompt, cast, default, validate, hint in fields:
        if dest in explicit:
            continue
        current = getattr(args, dest, None)
        if current is None:
            current = default(args) if callable(default) else default
        setattr(args, dest, current if yes else ask(prompt, cast, current, validate, hint, input_fn, out))


def equivalent_command(args, layout: str) -> str:
    parts = ["python run_doi.py", f"--layout {layout}"]
    for dest, *_ in LAYOUT_FIELDS[layout] + COMMON_FIELDS:
        v = getattr(args, dest, None)
        if v is not None:
            parts.append(f"--{dest.replace('_', '-')} {v}")
    parts.append("--yes")
    if args.no_show:
        parts.append("--no-show")
    return " ".join(parts)


def configure(args, explicit: Set[str], preview: Callable, yes: bool = False, input_fn: Callable = input,
              out: Callable = print) -> Tuple[SimConfig, object]:
    """Ask for everything not given on the command line, build the map, show it, and let the user confirm."""
    if not yes:
        out("\n" + BOX + "\n  Press Enter to keep the default value.\n")
    layout = args.layout or ("map" if args.map else None)
    if layout is None:
        if yes:
            layout = "barrier"
        else:
            for name in LAYOUTS:
                out(f"    {name:8s} {LAYOUT_HELP[name]}")
            layout = ask("Layout", str, "barrier", lambda v: v in LAYOUTS, "barrier, strips or map",
                         input_fn, out)
    args.layout = layout
    common_done, last_error = False, None
    while True:
        _fill(args, LAYOUT_FIELDS[layout], explicit, yes, input_fn, out)
        if not common_done:
            _fill(args, COMMON_FIELDS, explicit, yes, input_fn, out)
            if not yes and "no_show" not in explicit:
                args.no_show = not ask_yes("Open the animation window?", True, input_fn)
            common_done = True
        cfg = SimConfig(seed=args.seed, n_robots=args.robots, tasks_per_robot=args.tasks, kappa=args.kappa,
                        fee=args.fee, push_max=args.push_max, r_comm=args.r_comm, loss=args.loss,
                        latency=args.latency, bundle_max=args.bundle_max, lam=args.lam,
                        kappa_c=getattr(args, "kappa_c", 2.0), pick_fee=getattr(args, "pick_fee", 1.0),
                        drop_fee=getattr(args, "drop_fee", 1.0))
        answers = {k: getattr(args, k, None) for k in ("rows", "cols", "wall_col", "blocks", "doors", "kind",
                                                       "crossing", "strips", "strip_min", "strip_max", "pallets",
                                                       "crates", "shelves", "map")}
        try:
            cfg, scenario = build(layout, cfg, answers)
            break
        except BuildError as e:
            out(f"\n    ✗  {e}\n")
            if yes or str(e) == last_error:       # nothing changed since the last try (e.g. input ended)
                raise
            last_error = str(e)
            explicit = explicit - {f[0] for f in LAYOUT_FIELDS[layout]}      # re-ask the map questions
    while True:
        out("")
        preview(cfg, scenario)
        for w in check(scenario, cfg):
            out(f"  {'note' if w.startswith('info:') else 'warning'}: {w.replace('info: ', '')}")
        if yes:
            break
        choice = ask("[r]un, new [s]eed, [c]hange answers, [q]uit", str, "r", lambda v: v[:1] in "rscq" and v != "",
                     "r, s, c or q", input_fn, out).lower()[0]
        if choice == "q":
            raise SystemExit("cancelled")
        if choice == "s":
            args.seed += 1
            cfg, scenario = build(layout, cfg.replace(seed=args.seed), answers)
            continue
        if choice == "c":
            return configure(args, explicit - {f[0] for f in LAYOUT_FIELDS[layout] + COMMON_FIELDS},
                             preview, yes, input_fn, out)
        break
    out(f"\n  Repeat this run exactly with:\n    {equivalent_command(args, layout)}\n")
    return cfg, scenario
