"""run_doi.py: command-line runner for the decentralised Rent-or-Fill simulator (src/doi).

Examples
  python run_doi.py --sim toy                        # simulation only: opens a window, 3 lines of text
  python run_doi.py --sim scatter                    # obstacles anywhere on an open floor, one trip per robot
  python run_doi.py --sim fleet                      # 8 robots, 4 obstacles in a barrier
  python run_doi.py --demo toy                       # window with never vs rof side by side, short summary
  python run_doi.py --demo toy --verbose             # + the full event timeline, benchmarks and statistics
  python run_doi.py --scenario single_block --robots 12 --policy rof,never,central --seed 3
  python run_doi.py --scenario incidents_aisles --intake oracle --policy rof --robots 8
  python run_doi.py --build                          # set the map and fleet interactively, preview, then run
  python run_doi.py --demo toy --html toy.html       # save a player (play/pause/scrub) for any browser
  python run_doi.py --demo toy --gif toy.gif         # save a GIF (open in a browser: Preview shows frames apart)
  python run_doi.py --guide                          # plain-language guide to every flag
  python run_doi.py --list
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.doi.config import POLICIES, SimConfig
from src.doi.animate import animate_runs
from src.doi.metrics import collateral_cost, hindsight_ratios
from src.doi.narrate import ARM_NOTES, COLUMN_NOTES, FLAG_GUIDE, SCENARIO_NOTES, events
from src.doi.runner import run_episode
from src.doi.scenarios import DEFAULTS, build_scenario, scenario_from_ascii
from src.doi.kinds import KINDS
from src.doi.metrics import loaded_at, slots_full_at
from src.doi.story import verdict
from src.doi.builder import BuildError, build as build_layout, check
from src.doi.wizard import COMMON_FIELDS, ask, ask_yes, configure
from src.environment.grid import CellType

TOY_ROWS = ["...#...", "...S...", "...#...", "...#...", "...#...", "......."]
TOY_TASKS = [(1, 4), (1, 2), (1, 4), (1, 2)]
SCATTER = dict(H=12, W=16, strips=0, pallets=21, crates=13, shelves=8)   # a cluttered open floor, obstacles anywhere
ROBOT_GLYPHS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

LEGEND = """Legend:  # permanent wall   L pallet   C crate   S shelf unit   R debris   P pit   (removable obstacles)
         T rack / D dump slot (t, d once holding an obstacle)   0-9,a-z robots, @ a robot carrying something
         . free cell"""

def build(args):
    if args.demo == "toy":
        scenario = scenario_from_ascii(TOY_ROWS, [(1, 2)], [TOY_TASKS], name="toy")
        cfg = SimConfig(n_robots=1, tasks_per_robot=len(TOY_TASKS), kappa=args.kappa, fee=args.fee,
                        kappa_c=getattr(args, "kappa_c", 2.0), pick_fee=getattr(args, "pick_fee", 1.0),
                        drop_fee=getattr(args, "drop_fee", 1.0), push_max=args.push_max, theta=args.theta,
                        bundle_max=getattr(args, "bundle_max", 1), lam=getattr(args, "lam", 0.5),
                        debug_checks=True)
        return cfg, scenario
    params = {}
    if args.p_false is not None:
        params["p_false"] = args.p_false
    if args.p_report is not None:
        params["p_report"] = args.p_report
    cfg = SimConfig(
        scenario=args.scenario, seed=args.seed, n_robots=args.robots, tasks_per_robot=args.tasks,
        kappa=args.kappa, fee=args.fee, kappa_c=getattr(args, "kappa_c", 2.0),
        pick_fee=getattr(args, "pick_fee", 1.0), drop_fee=getattr(args, "drop_fee", 1.0), push_max=args.push_max, r_comm=args.r_comm, loss=args.loss,
        latency=args.latency, intake=args.intake, max_ticks=args.max_ticks,
        scenario_params=params, p_wrong_class=args.p_wrong_class, theta=args.theta,
        bundle_max=getattr(args, "bundle_max", 1), lam=getattr(args, "lam", 0.5))
    if args.demo == "fleet":
        cfg = cfg.replace(scenario="multi_block_wall", n_robots=8, tasks_per_robot=10, seed=args.seed)
    if args.demo == "scatter":
        cfg = cfg.replace(scenario="random_blocks", n_robots=8, tasks_per_robot=1, seed=args.seed,
                          scenario_params=SCATTER)
    return cfg, build_scenario(cfg)


def render(scenario, t, result, show_goals=False):
    g = scenario.grid
    cells = [["#" if g.get(r, c) == CellType.OBSTACLE else "." for c in range(g.width)] for r in range(g.height)]
    trace = result.obstacle_trace
    full = slots_full_at(result, t)
    for (r, c), slot in scenario.slots.items():
        glyph = "T" if slot == "rack" else "D"
        cells[r][c] = glyph.lower() if (r, c) in full else glyph
    for (r, c), kind in (trace[min(t, len(trace) - 1)] if trace else scenario.obstacles).items():
        cells[r][c] = KINDS[kind].glyph if kind in KINDS else "?"
    for i, traj in sorted(result.trajectory.items()):
        if t < len(traj):
            r, c = traj[t]
            cells[r][c] = "@" if loaded_at(result, i, t) else ROBOT_GLYPHS[i % len(ROBOT_GLYPHS)]
    return "\n".join("  " + " ".join(row) for row in cells)


def show_map(cfg, scenario):
    probe = run_episode(cfg.replace(policy="never", max_ticks=0), scenario=scenario)
    print(f"Map {scenario.grid.height}x{scenario.grid.width}, {len(scenario.starts)} robots at their start cells:")
    print(render(scenario, 0, probe))
    print(LEGEND)


def print_setup(args, cfg, scenario):
    print("=" * 78)
    print(f"scenario {scenario.name} (family {scenario.family}): {scenario.grid.height}x{scenario.grid.width} grid, "
          f"{len(scenario.starts)} robot(s), {len(scenario.tasks[0])} tasks each, seed {cfg.seed}")
    print(f"costs: a push step costs kappa x weight = {cfg.kappa:g} x (0.5 crate, 1 pallet, 2 shelf unit), a push run "
          f"costs a fee of {cfg.fee:g}, moving or waiting costs 1 per tick")
    print(f"ledger sharing: radius {cfg.r_comm:g}, loss {cfg.loss:g}, latency {cfg.latency}; intake {cfg.intake}")
    print("=" * 78)
    probe = run_episode(cfg.replace(policy="never", max_ticks=0), scenario=scenario)
    print("\nMap at tick 0 (robots shown at their start cells):")
    print(render(scenario, 0, probe))
    print(LEGEND)
    print("\nEach robot walks its own task list. It senses cells within radius", cfg.r_sense,
          "and shares what it knows only by messages.\nEvery task it plans is recorded in a ledger. When its own route "
          "runs through an obstacle it can go round (paying 'rent') or push the\nobstacle aside on its way; an arm "
          "decides when the fleet's accumulated rent justifies the push.")


def print_arms(policies, runs, scenario):
    for name in policies:
        r = runs[name]
        print("\n" + "-" * 78)
        print(f"ARM {name}: {ARM_NOTES.get(name, '')}")
        print("-" * 78)
        print(summarise(r))
        for tick, text in events(r, scenario):
            print(f"  tick {tick:5d}  {text}")
        if not events(r, scenario):
            print("  (nothing was pushed: every detour was walked)")
        print(f"  pushes {r.removals} ({r.push_steps} steps), refused pushes {r.push_rejected}; "
              f"arbiter overrides {r.overrides}; "
              f"messages {r.messages['transmissions']} ledger / {r.traffic_messages['transmissions']} traffic")


HINDSIGHT_NOTE = ("hindsight is a relaxed lower reference (obstacles vanish, no walking); it is not achievable and "
                 "not an upper bound.")


def print_summary(policies, runs):
    """The default terminal output: one table of cost and pushes, and one sentence on who won."""
    print(f"\n{'arm':10s} {'cost':>8s} {'pushes':>7s} {'carried':>8s} {'filled':>7s}")
    for name in policies:
        r = runs[name]
        cost = r.J_censored + r.hindsight_buy if name == "hindsight" else r.J_censored
        print(f"{name:10s} {cost:8.0f} {r.removals:7d} {r.carries:8d} {r.fills:7d}"
              + ("   (stalled)" if r.stalled else ""))
    if "hindsight" in policies:
        print(HINDSIGHT_NOTE)
    print("\n" + verdict({n: runs[n] for n in policies}))
    print("--verbose: event timeline, benchmarks and statistics.   --guide: every flag.   --list: scenarios and arms.")


def print_comparison(names, runs):
    print("\n" + "=" * 78)
    print("COMPARISON (same scenario, same tasks, same traffic layer for every arm)")
    print("=" * 78)
    have_bench = "free" in runs and "hindsight" in runs
    header = (f"{'arm':10s} {'J':>8s} {'pushes':>6s} {'carried':>7s} {'filled':>6s} {'waits':>6s} {'collat':>7s} "
              f"{'reject':>6s} {'stalled':>7s}")
    if have_bench:
        header += f" {'HR_av':>7s}"
    if "central" in runs:
        header += f" {'PoD':>6s}"
    print(header)
    for name in names:
        r = runs[name]
        cost = r.J_censored + r.hindsight_buy if name == "hindsight" else r.J_censored
        line = (f"{name:10s} {cost:8.0f} {r.removals:6d} {r.carries:7d} {r.fills:6d} {r.waits:6d} "
                f"{collateral_cost(r, r.scenario, SimConfig(**r.cfg)):7.0f} {r.push_rejected:6d} {str(r.stalled):>7s}")
        if have_bench:
            if name in ("free", "hindsight"):
                line += f" {'-':>7s}"
            else:
                v = hindsight_ratios(r, runs["hindsight"], runs["free"])["hr_av"]
                line += f" {v:7.2f}" if not math.isnan(v) else f" {'n/a':>7s}"
        if "central" in runs:
            line += f" {r.J_censored / runs['central'].J_censored:6.2f}" if name not in ("free", "hindsight", "central") else f" {'-':>6s}"
        print(line)
    if "hindsight" in runs:
        print(HINDSIGHT_NOTE)
    if have_bench:
        h = runs["hindsight"]
        print(f"\nfree      : every obstacle gone at no charge (J = travel nobody can avoid)          J = {runs['free'].J_censored:.0f}"
              f"\nhindsight : knows all tasks, removes the best obstacles at t=0, charged {h.hindsight_buy:.0f}   "
              f"J = {h.J_censored + h.hindsight_buy:.0f} incl. charge")
        print("HR_av = (J_arm - J_free) / (J_hindsight - J_free): 1.0 would match hindsight, ski-rental theory predicts about 2.")
    if "central" in runs:
        print("PoD = J_arm / J_central: what the arm loses by deciding from local, delayed information.")


def summarise(result):
    carry = (f" + loaded steps {result.carry_steps} at {result.carry_cost:.0f} + {result.picks} pick(s) and "
             f"{result.drops} drop(s) x fee") if result.picks else ""
    return (f"J = {result.J:.0f}  (moves {result.moves - result.push_steps - result.carry_steps} + waits "
            f"{result.waits} + push steps {result.push_steps} at {result.push_cost:.0f} + {result.removals} push "
            f"run(s) x fee{carry})   ticks = {result.ticks}   unfinished tasks = {result.unfinished_tasks}")


PLAY_ARMS = ["never", "myopic", "eager", "rof", "central"]
PLAY_FIELDS = [   # dest, prompt, cast, default, validate, hint
    ("rows", "Grid rows (height)", int, 12, lambda v: v >= 5, "at least 5"),
    ("cols", "Grid columns (width)", int, 16, lambda v: v >= 5, "at least 5"),
    ("robots", "Number of robots (each gets one start and one goal)", int, 8, lambda v: v >= 1, "at least 1"),
    ("pallets", "Pallets L (weight 1.0)", int, 21, lambda v: v >= 0, "0 or more"),
    ("crates", "Crates C (light, weight 0.5)", int, 13, lambda v: v >= 0, "0 or more"),
    ("shelves", "Shelf units S (heavy, weight 2.0)", int, 8, lambda v: v >= 0, "0 or more"),
    ("kappa", "Cost of one push step = this x the obstacle's weight", float, 1.0, lambda v: v >= 0, "0 or more"),
    ("seed", "Random seed (the same number gives the same map)", int, 8, lambda v: True, "a whole number"),
]


def play(args, input_fn=input, out=print):
    """Short interactive loop: ask for the grid, robots and obstacle counts, run, show, and offer another go."""
    vals = {f[0]: f[3] for f in PLAY_FIELDS}
    if args.robots_given:
        vals["robots"] = args.robots
    out("\n  Obstacles anywhere on an open floor; every robot has one start and one goal.\n"
        "  Press Enter to keep the number in brackets.\n")
    last_error = None
    while True:
        for dest, prompt, cast, default, validate, hint in PLAY_FIELDS:
            vals[dest] = ask(prompt, cast, vals[dest], validate, hint, input_fn, out)
        cfg = SimConfig(seed=vals["seed"], n_robots=vals["robots"], tasks_per_robot=1, kappa=vals["kappa"],
                        bundle_max=getattr(args, "bundle_max", 1), lam=getattr(args, "lam", 0.5))
        try:
            cfg, scenario = build_layout("strips", cfg, dict(
                rows=vals["rows"], cols=vals["cols"], strips=0, strip_min=1, strip_max=1,
                pallets=vals["pallets"], crates=vals["crates"], shelves=vals["shelves"]))
        except BuildError as e:
            out(f"\n    \u2717  {e}\n")
            if str(e) == last_error:
                raise
            last_error = str(e)
            continue
        last_error = None
        out("")
        show_map(cfg, scenario)
        for w in check(scenario, cfg):
            out(f"  {'note' if w.startswith('info:') else 'warning'}: {w.replace('info: ', '')}")
        runs = {p: run_episode(cfg.replace(policy=p), scenario=scenario) for p in PLAY_ARMS}
        print_summary(PLAY_ARMS, runs)
        out("\n  Repeat this run: python run_doi.py --layout strips --yes --strips 0 --tasks 1 "
            f"--rows {vals['rows']} --cols {vals['cols']} --robots {vals['robots']} --pallets {vals['pallets']} "
            f"--crates {vals['crates']} --shelves {vals['shelves']} --kappa {vals['kappa']:g} "
            f"--seed {vals['seed']} --bundle-max {cfg.bundle_max} --lam {cfg.lam:g} "
            f"--policy {','.join(PLAY_ARMS)}")
        if not args.no_show:
            out("\n  Opening the window (never vs rof). Close it to continue.")
            animate_runs(scenario, {a: runs[a] for a in ("never", "rof")}, cfg, show=True, fps=args.fps)
        if not ask_yes("Try different numbers?", False, input_fn):
            return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=["toy", "fleet", "scatter"], help="ready-made scenarios, compared in a window")
    ap.add_argument("--sim", metavar="NAME", help="simulation only: run one arm and open the window. NAME is toy, "
                    "fleet or any scenario from --list")
    ap.add_argument("--list", action="store_true", help="list scenarios and policies, then exit")
    ap.add_argument("--scenario", default="single_block")
    ap.add_argument("--play", action="store_true",
                    help="short interactive mode: asks for grid size, robots and the number of L, C and S, runs, "
                         "shows the window, and offers another go")
    ap.add_argument("--build", action="store_true",
                    help="set the map and run up interactively (asks in the terminal, then previews the map)")
    ap.add_argument("--layout", choices=["barrier", "strips", "map"], default=None,
                    help="build your own map: barrier (two zones, obstacles in its openings), strips (random wall "
                         "strips plus obstacles anywhere) or map (ASCII file); asks for anything not given")
    ap.add_argument("--rows", type=int, default=None, help="build: grid rows")
    ap.add_argument("--cols", type=int, default=None, help="build: grid columns")
    ap.add_argument("--wall-col", type=int, default=None, help="barrier: column of the barrier")
    ap.add_argument("--blocks", type=int, default=None, help="barrier: removable obstacles in the barrier")
    ap.add_argument("--doors", type=int, default=None, help="barrier: door rows at the bottom (the long way round)")
    ap.add_argument("--kind", default=None, help="barrier: mixed (default), pallet, crate or shelf_unit")
    ap.add_argument("--crossing", type=float, default=None, help="barrier: share of trips that cross, 0-1")
    ap.add_argument("--strips", type=int, default=None, help="strips: number of permanent wall strips")
    ap.add_argument("--strip-min", type=int, default=None, help="strips: shortest strip")
    ap.add_argument("--strip-max", type=int, default=None, help="strips: longest strip")
    ap.add_argument("--pallets", type=int, default=None, help="strips: number of pallets")
    ap.add_argument("--crates", type=int, default=None, help="strips: number of crates")
    ap.add_argument("--shelves", type=int, default=None, help="strips: number of shelf units")
    ap.add_argument("--map", default=None, metavar="FILE",
                    help="map: ASCII map file (# wall . free L pallet C crate S shelf unit)")
    ap.add_argument("--yes", action="store_true", help="build: accept defaults, never prompt")
    ap.add_argument("--policy", default="rof", help="one or more arms, comma separated (see --list)")
    ap.add_argument("--robots", type=int, default=12)
    ap.add_argument("--tasks", type=int, default=20, help="tasks per robot")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default 0; 8 in --demo scatter)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="a push step costs kappa x the obstacle's weight (default 4; 1 in --demo scatter)")
    ap.add_argument("--fee", type=float, default=1.0, help="cost per push run")
    ap.add_argument("--kappa-c", type=float, default=2.0,
                    help="a loaded step costs kappa-c x the carried kind's weight (default 2, at least 2)")
    ap.add_argument("--pick-fee", type=float, default=1.0, help="cost of lifting an obstacle")
    ap.add_argument("--drop-fee", type=float, default=1.0, help="cost of dropping it into a slot or pit")
    ap.add_argument("--push-max", type=int, default=6, help="longest straight push a robot will plan")
    ap.add_argument("--r-comm", type=float, default=8.0, help="ledger radius (0 = no sharing, inf = global)")
    ap.add_argument("--loss", type=float, default=0.0, help="ledger message loss probability")
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--bundle-max", type=int, choices=[1, 2], default=1,
                    help="1: single pushes; 2: also two-step plans (two obstacles moved in sequence)")
    ap.add_argument("--lam", type=float, default=0.5, help="rof_p: threshold multiplier when the forecast says yes")
    ap.add_argument("--intake", default="none", help="none | oracle   (family D scenarios)")
    ap.add_argument("--p-false", type=float, default=None)
    ap.add_argument("--p-report", type=float, default=None)
    ap.add_argument("--p-wrong-class", type=float, default=0.0)
    ap.add_argument("--max-ticks", type=int, default=20000)
    ap.add_argument("--replay", action="store_true", help="print ASCII frames of the first policy's run")
    ap.add_argument("--replay-every", type=int, default=None, help="ticks between frames")
    ap.add_argument("--live", action="store_true", help="animate the replay in the terminal")
    ap.add_argument("--replay-policy", default=None, help="arm to replay (default: rof if run, else the first)")
    ap.add_argument("--gif", default=None, metavar="PATH", help="save the animation as a GIF")
    ap.add_argument("--gif-arms", default=None, help="arms shown in the animation (default never,rof)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--html", default=None, metavar="PATH", help="save a self-contained HTML player")
    ap.add_argument("--show", action="store_true", help="open the animation window (default for --demo and --sim)")
    ap.add_argument("--no-show", action="store_true", help="do not open the window for --demo / --sim")
    ap.add_argument("--verbose", action="store_true",
                    help="full text output: setup, event timelines, benchmarks, statistics")
    ap.add_argument("--guide", action="store_true", help="explain every flag and output column, then exit")
    ap.add_argument("--no-benchmarks", action="store_true", help="skip the free/hindsight benchmark runs")
    args = ap.parse_args(argv)
    if args.play:
        if args.demo or args.sim is not None or args.build or args.layout or args.map:
            ap.error("--play asks its own questions: do not combine it with --demo, --sim, --build or --layout")
        args.robots_given = "--robots" in (argv if argv is not None else sys.argv[1:])
        return play(args)
    if args.seed is None and not (args.build or args.layout or args.map):
        args.seed = 8 if (args.demo == "scatter" or args.sim == "scatter") else 0
    if args.kappa is None and not (args.build or args.layout or args.map):
        args.kappa = 1.0 if (args.demo == "scatter" or args.sim == "scatter") else 4.0
    custom = args.build or args.layout is not None or args.map is not None
    if custom and (args.demo or args.sim is not None):
        ap.error("--build / --layout / --map build their own map: do not combine with --demo or --sim")
    sim_only = args.sim is not None or custom
    if args.sim is not None:
        if args.demo:
            ap.error("use either --sim or --demo")
        if args.sim in ("toy", "fleet", "scatter"):
            args.demo = args.sim
        elif args.sim in DEFAULTS:
            args.scenario = args.sim
        else:
            ap.error(f"unknown --sim {args.sim!r}: toy, fleet or one of {', '.join(sorted(DEFAULTS))}")

    if args.guide:
        for title, rows in FLAG_GUIDE:
            print(f"\n{title}")
            for flag, what, effect in rows:
                print(f"  {flag:28s} {what}")
                if effect:
                    print(f"  {'':28s}   -> {effect}")
        print("\nHOW TO READ THE TABLE")
        for col, text in COLUMN_NOTES:
            print(f"  {col:10s} {text}")
        return 0
    if args.list:
        print("scenarios (S = obstacles on the map from the start, D = obstructions appear during the run):")
        for name in sorted(DEFAULTS):
            fam, text = SCENARIO_NOTES.get(name, ("?", ""))
            print(f"  {name:20s} [{fam}] {text}")
        print("\narms (--policy):")
        for name in sorted(POLICIES):
            print(f"  {name:10s} {ARM_NOTES.get(name, '')}")
        return 0

    built = None
    if custom and not args.guide and not args.list:
        tokens = argv if argv is not None else sys.argv[1:]
        given = {a.dest for a in ap._actions
                 if any(t == o or t.startswith(o + "=") for t in tokens for o in a.option_strings)}
        for f in COMMON_FIELDS:
            if f[0] not in given:
                setattr(args, f[0], None)
        try:
            built = configure(args, given, show_map, yes=args.yes)
        except BuildError as e:
            ap.error(str(e))
    if args.policy == "rof" and not sim_only:
        if args.demo == "toy":
            args.policy = "never,myopic,eager,rof,central"
        elif args.demo == "fleet":
            args.policy = "never,rof,central"
        elif args.demo == "scatter":
            args.policy = "never,myopic,eager,rof,central"
    policies = [p.strip() for p in args.policy.split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            ap.error(f"unknown policy {p!r}; see --list")
    cfg, scenario = built if built else build(args)
    window = args.show or ((args.demo or sim_only) and not args.no_show)
    n_tasks = len(scenario.tasks[0])

    if args.verbose:
        print_setup(args, cfg, scenario)
    else:
        what = "simulation" if sim_only else "comparison"
        print(f"{what}: {scenario.name} ({len(scenario.starts)} robot{'s' if len(scenario.starts) != 1 else ''}, "
              f"{n_tasks} tasks each, seed {cfg.seed}); arms: {', '.join(policies)}")

    runs = {}
    names = list(policies)
    if args.verbose and not args.no_benchmarks:
        for extra in ("free", "hindsight"):
            if extra not in names and not (extra == "hindsight" and scenario.family == "D"):
                names.append(extra)
    for name in names:
        t0 = time.time()
        runs[name] = run_episode(cfg.replace(policy=name), scenario=scenario)
        runs[name].wall_s = time.time() - t0

    if args.verbose:
        print_arms(policies, runs, scenario)
        print_comparison(names, runs)
    elif sim_only and len(policies) == 1:
        r = runs[policies[0]]
        lifted = f", {r.carries} carried, {r.fills} filled" if r.drops else ""
        print(f"result: total cost {r.J_censored:.0f}, {r.removals} push run{'s' if r.removals != 1 else ''}"
              f"{lifted}, {r.ticks} ticks" + (", STALLED" if r.stalled else ""))
    else:
        print_summary(policies, runs)

    if window or args.gif or args.html:
        if args.gif_arms:
            arms = [a.strip() for a in args.gif_arms.split(",")]
        elif sim_only:
            arms = policies
        else:
            arms = [p for p in ("never", "rof") if p in runs] or policies[:2]
        missing = [a for a in arms if a not in runs]
        if missing:
            ap.error(f"--gif-arms {missing} were not run; add them to --policy")
        chosen = {a: runs[a] for a in arms}
        if window:
            print("\nopening the window: space = play/pause, slider = scrub, close it to finish "
                  "(--no-show to skip)")
        saved = animate_runs(scenario, chosen, cfg, path=args.gif, html=args.html, show=window, fps=args.fps)
        for label, target in (("GIF", args.gif), ("HTML player", args.html)):
            if target:
                print(f"{label} saved to {target}" + ("  (open it in a browser: macOS Preview shows GIF frames "
                                                       "as separate images)" if label == "GIF" else ""))

    if args.replay or args.live:
        chosen = args.replay_policy or ("rof" if "rof" in runs else policies[0])
        first = runs[chosen]
        every = args.replay_every or max(1, first.ticks // 12)
        print("\n" + "=" * 78)
        print(f"REPLAY of arm {chosen} (every {every} ticks)")
        print("=" * 78)
        for t in range(0, first.ticks + 1, every):
            if args.live:
                print("\033[2J\033[H", end="")
            print(f"\ntick {t}")
            print(render(scenario, t, first))
            if args.live:
                time.sleep(0.25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
