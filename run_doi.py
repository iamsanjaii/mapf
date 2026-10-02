"""run_doi.py: command-line runner for the decentralised Rent-or-Fill simulator (src/doi).

Examples
  python run_doi.py --demo toy                       # one robot, one pit: the ski-rental rule step by step
  python run_doi.py --demo fleet --replay            # 8 robots, 4 pits in a wall, with an ASCII replay
  python run_doi.py --scenario single_pit --robots 12 --policy rof,never,central --seed 3
  python run_doi.py --scenario incidents_aisles --intake oracle --gate --policy rof --robots 8
  python run_doi.py --demo toy --gif toy.gif         # save a side-by-side animation (never vs rof)
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
from src.doi.metrics import hindsight_ratios
from src.doi.narrate import ARM_NOTES, COLUMN_NOTES, FLAG_GUIDE, SCENARIO_NOTES, events
from src.doi.runner import run_episode
from src.doi.scenarios import DEFAULTS, build_scenario, scenario_from_ascii
from src.environment.grid import CellType

TOY_ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
TOY_TASKS = [(1, 4), (1, 2), (1, 4), (1, 2)]
ROBOT_GLYPHS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

LEGEND = """Legend:  # wall   P pit (blocked until filled)   o filled pit   X obstruction (incident)
         D depot (holds kits)   0-9,a-z robots   . free cell"""

def build(args):
    if args.demo == "toy":
        scenario = scenario_from_ascii(TOY_ROWS, [(1, 2)], [TOY_TASKS], depot_stock=2, name="toy")
        cfg = SimConfig(n_robots=1, tasks_per_robot=len(TOY_TASKS), kappa=args.kappa, fee=args.fee,
                        claim=(args.claim == "on"), theta=args.theta, debug_checks=True)
        return cfg, scenario
    params = {}
    if args.p_false is not None:
        params["p_false"] = args.p_false
    if args.p_report is not None:
        params["p_report"] = args.p_report
    cfg = SimConfig(
        scenario=args.scenario, seed=args.seed, n_robots=args.robots, tasks_per_robot=args.tasks,
        kappa=args.kappa, fee=args.fee, claim=(args.claim == "on"), r_comm=args.r_comm, loss=args.loss,
        latency=args.latency, intake=args.intake, gate=args.gate, max_ticks=args.max_ticks,
        scenario_params=params, p_wrong_class=args.p_wrong_class, theta=args.theta)
    if args.demo == "fleet":
        cfg = cfg.replace(scenario="multi_pit_wall", n_robots=8, tasks_per_robot=10, seed=args.seed)
    return cfg, build_scenario(cfg)


def render(scenario, t, result, show_goals=False):
    g = scenario.grid
    cells = [["." for _ in range(g.width)] for _ in range(g.height)]
    for r in range(g.height):
        for c in range(g.width):
            ct = g.get(r, c)
            cells[r][c] = {CellType.OBSTACLE: "#", CellType.PIT: "P", CellType.SANDBAG: "D"}.get(ct, ".")
    for pit, ft in result.filled_at.items():
        if ft <= t and 0 <= pit[0] < g.height:
            cells[pit[0]][pit[1]] = "o"
    for inc in scenario.incidents:
        at = result.appeared_at.get(inc.oid)
        for cell in inc.cells:
            filled = result.filled_at.get(cell)
            if at is not None and at <= t and not (filled is not None and filled <= t):
                cells[cell[0]][cell[1]] = "X"
    for i, traj in sorted(result.trajectory.items()):
        if t < len(traj):
            r, c = traj[t]
            cells[r][c] = ROBOT_GLYPHS[i % len(ROBOT_GLYPHS)]
    return "\n".join("  " + " ".join(row) for row in cells)


def summarise(result):
    return (f"J = {result.J:.0f}  (moves {result.moves - result.carried_steps} + waits {result.waits} + "
            f"carried {result.carried_steps} x kappa + fills {result.fills} x fee)   ticks = {result.ticks}   "
            f"unfinished tasks = {result.unfinished_tasks}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=["toy", "fleet"], help="ready-made scenarios for explaining the system")
    ap.add_argument("--list", action="store_true", help="list scenarios and policies, then exit")
    ap.add_argument("--scenario", default="single_pit")
    ap.add_argument("--policy", default="rof", help="one or more arms, comma separated (see --list)")
    ap.add_argument("--robots", type=int, default=12)
    ap.add_argument("--tasks", type=int, default=20, help="tasks per robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=4.0, help="cost per carried step")
    ap.add_argument("--fee", type=float, default=1.0, help="cost per edit")
    ap.add_argument("--claim", choices=["on", "off"], default=None,
                    help="default: on, except off in --demo toy")
    ap.add_argument("--r-comm", type=float, default=8.0, help="ledger radius (0 = no sharing, inf = global)")
    ap.add_argument("--loss", type=float, default=0.0, help="ledger message loss probability")
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--intake", default="none", help="none | oracle   (family D scenarios)")
    ap.add_argument("--gate", action="store_true", help="human-on-the-loop approval gate (family D)")
    ap.add_argument("--p-false", type=float, default=None)
    ap.add_argument("--p-report", type=float, default=None)
    ap.add_argument("--p-wrong-class", type=float, default=0.0)
    ap.add_argument("--max-ticks", type=int, default=20000)
    ap.add_argument("--replay", action="store_true", help="print ASCII frames of the first policy's run")
    ap.add_argument("--replay-every", type=int, default=None, help="ticks between frames")
    ap.add_argument("--live", action="store_true", help="animate the replay in the terminal")
    ap.add_argument("--replay-policy", default=None, help="arm to replay (default: rof if run, else the first)")
    ap.add_argument("--gif", default=None, metavar="PATH", help="save a side-by-side animation as a GIF")
    ap.add_argument("--gif-arms", default=None, help="arms shown in the animation (default never,rof)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--show", action="store_true", help="show the animation in a window")
    ap.add_argument("--guide", action="store_true", help="explain every flag and output column, then exit")
    ap.add_argument("--no-benchmarks", action="store_true", help="skip the free/hindsight benchmark runs")
    args = ap.parse_args(argv)
    if args.claim is None:
        args.claim = "off" if args.demo == "toy" else "on"

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
        print("scenarios (S = pits known from the start, D = obstructions appear during the run):")
        for name in sorted(DEFAULTS):
            fam, text = SCENARIO_NOTES.get(name, ("?", ""))
            print(f"  {name:20s} [{fam}] {text}")
        print("\narms (--policy):")
        for name in sorted(POLICIES):
            print(f"  {name:10s} {ARM_NOTES.get(name, '')}")
        return 0

    if args.demo == "toy" and args.policy == "rof":
        args.policy = "never,myopic,eager,rof,central"
    if args.demo == "fleet" and args.policy == "rof":
        args.policy = "never,rof,central"
    policies = [p.strip() for p in args.policy.split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            ap.error(f"unknown policy {p!r}; see --list")
    cfg, scenario = build(args)

    print("=" * 78)
    print(f"scenario {scenario.name} (family {scenario.family}): {scenario.grid.height}x{scenario.grid.width} grid, "
          f"{len(scenario.starts)} robot(s), {len(scenario.tasks[0])} tasks each, seed {cfg.seed}")
    print(f"costs: carrying a kit costs kappa = {cfg.kappa:g} per step, an edit costs {cfg.fee:g}, "
          f"moving or waiting costs 1 per tick")
    print(f"ledger sharing: radius {cfg.r_comm:g}, loss {cfg.loss:g}, latency {cfg.latency}; "
          f"claims {'on' if cfg.claim else 'off'}; intake {cfg.intake}; gate {'on' if cfg.gate else 'off'}")
    print("=" * 78)
    probe = run_episode(cfg.replace(policy="never", max_ticks=0), scenario=scenario)
    print("\nMap at tick 0 (robots shown at their start cells):")
    print(render(scenario, 0, probe))
    print(LEGEND)
    print("\nEach robot walks its own task list. It senses cells within radius", cfg.r_sense,
          "and shares what it knows only by messages.\nEvery time it plans a task it records the extra steps a "
          "blocked cell costs it (the 'rent'). An arm decides when the\nfleet's accumulated rent justifies "
          "paying for the edit (fetching a kit from a depot, carrying it, applying it).")

    runs = {}
    names = list(policies)
    if not args.no_benchmarks:
        for extra in ("free", "hindsight"):
            if extra not in names and not (extra == "hindsight" and scenario.family == "D"):
                names.append(extra)
    for name in names:
        t0 = time.time()
        runs[name] = run_episode(cfg.replace(policy=name), scenario=scenario)
        runs[name].wall_s = time.time() - t0

    for name in policies:
        r = runs[name]
        print("\n" + "-" * 78)
        print(f"ARM {name}: {ARM_NOTES.get(name, '')}")
        print("-" * 78)
        print(summarise(r))
        for tick, text in events(r, scenario):
            print(f"  tick {tick:5d}  {text}")
        if not events(r, scenario):
            print("  (no edits: every detour was walked)")
        print(f"  claims issued {r.claims['issued']}, lost {r.claims['lost']}, aborted hauls {r.claims['aborts']}, "
              f"wasted haul cost {r.wasted_haul_cost:.0f}; arbiter overrides {r.overrides}; "
              f"messages {r.messages['transmissions']} ledger / {r.traffic_messages['transmissions']} traffic")

    print("\n" + "=" * 78)
    print("COMPARISON (same scenario, same tasks, same traffic layer for every arm)")
    print("=" * 78)
    have_bench = "free" in runs and "hindsight" in runs
    header = f"{'arm':10s} {'J':>8s} {'fills':>5s} {'waits':>6s} {'wasted':>7s} {'stalled':>7s}"
    if have_bench:
        header += f" {'HR_av':>7s}"
    if "central" in runs:
        header += f" {'PoD':>6s}"
    print(header)
    for name in names:
        r = runs[name]
        line = f"{name:10s} {r.J_censored:8.0f} {r.fills:5d} {r.waits:6d} {r.wasted_haul_cost:7.0f} {str(r.stalled):>7s}"
        if have_bench:
            if name in ("free", "hindsight"):
                line += f" {'-':>7s}"
            else:
                v = hindsight_ratios(r, runs["hindsight"], runs["free"])["hr_av"]
                line += f" {v:7.2f}" if not math.isnan(v) else f" {'n/a':>7s}"
        if "central" in runs:
            line += f" {r.J_censored / runs['central'].J_censored:6.2f}" if name not in ("free", "hindsight", "central") else f" {'-':>6s}"
        print(line)
    if have_bench:
        h = runs["hindsight"]
        print(f"\nfree      : every blocked cell open at no charge (J = travel nobody can avoid)       J = {runs['free'].J_censored:.0f}"
              f"\nhindsight : knows all tasks, opens the best pits at t=0, charged {h.hindsight_buy:.0f} to buy   "
              f"J = {h.J_censored + h.hindsight_buy:.0f} incl. buy")
        print("HR_av = (J_arm - J_free) / (J_hindsight - J_free): 1.0 would match hindsight, ski-rental theory predicts about 2.")
    if "central" in runs:
        print("PoD = J_arm / J_central: what the arm loses by deciding from local, delayed information.")

    if args.gif or args.show:
        arms = [a.strip() for a in (args.gif_arms.split(",") if args.gif_arms else
                                    [p for p in ("never", "rof") if p in runs] or policies[:2])]
        missing = [a for a in arms if a not in runs]
        if missing:
            ap.error(f"--gif-arms {missing} were not run; add them to --policy")
        saved = animate_runs(scenario, {a: runs[a] for a in arms}, cfg, path=args.gif, show=args.show, fps=args.fps)
        if saved:
            print(f"\nanimation saved to {saved}")

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
