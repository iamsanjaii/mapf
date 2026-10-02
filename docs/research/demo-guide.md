# Demo guide: explaining `run_doi.py` to a professor

`python run_doi.py --guide` prints every flag in plain language, `--list` describes every scenario and
arm. This page gives the story, the order to show things in, and numbers that were actually produced.

## The story in four sentences
Warehouse robots sometimes meet an obstacle in their way: a pallet in a gap, a shelf unit in an aisle. They
can detour (pay "rent" every trip) or push the obstacle aside on their way through (pay once). This is ski
rental: push when the rent paid so far equals the price. The difficulty is that no robot sees the whole
fleet's detours, so robots share a small ledger by messages, and a pushed obstacle is still an obstacle where
it lands. We measure how much that decentralisation and missing information costs against a central planner
and a hindsight benchmark.

## What the map shows (say this first)
Dark cells are permanent walls: nothing ever goes through them. Coloured letters are removable obstacles
(`L` pallet, `C` crate, `S` shelf unit); they can be anywhere. Removing one empties its cell. A robot removes
one by **pushing** it a cell at a time along its own route, so a push never takes it off its way. A heavier
kind costs more per step. The barrier with obstacles in its openings is a convenient shape, not a claim about
real warehouses: it makes one removal create a clear shortcut.

## Order to show things
The window is the demo; the terminal is a few lines of footnote. `--verbose` brings back the full text dump.
00. **Try your own numbers**: `python run_doi.py --play` asks for the grid size, the number of robots, and how many
   pallets (L), crates (C) and shelf units (S), plus the cost of a push step and a seed. It shows the map and any
   warnings, prints the comparison table, opens the window (never vs rof), then asks whether to go again with
   different numbers. Every robot gets one start and one goal.
0. **Obstacles anywhere, one trip each**: `python run_doi.py --demo scatter` is the easiest picture of the
   mechanics. A 12x16 open floor, 42 obstacles scattered at random (no barrier), 8 robots each with one start
   (hollow square) and one goal (star), and a cheap push (kappa 1). Seed 8 gives never 81, myopic 64, eager 73,
   rof 64, central 64. **That seed was picked because it shows the effect.** Over seeds 0 to 11 the totals are
   never 1168, rof 1114, eager 1136, myopic 1135, central 1138: pushing saves about 5% overall, and on a few
   seeds (1 and 10) it costs more than it saves because a pushed obstacle lands in someone's way. Isolated
   obstacles on open floor cost a step or two of detour, so pushing only pays when the floor is cluttered or
   pushing is cheap; with the default kappa of 4 it almost never does.
1. **Simulation only**: `python run_doi.py --sim toy` opens one window with a single arm (rof) and prints two
   lines. Controls: space or the Pause button, Restart, a speed slider and a tick scrubber. The **rent meter**
   under the map is the idea made visible: a bar of detour cost paid so far climbs towards a black line, the
   price of pushing. When a robot decides to push, a black ring appears round it, the obstacle slides, and the
   bar turns green. The caption says each step in words. `--sim fleet` does the same with 8 robots.
2. **Comparison**: `python run_doi.py --demo toy` opens never and rof side by side and prints the table:
   never 40, myopic 40, eager 23, rof 31, central 31. Each trip pays 8 rent; pushing the shelf two cells costs
   17. Myopic compares 17 with the 10-step detour and never pushes; eager pushes at once and wins here because
   all four trips come; rof waits until two trips have paid 16 against a price of 15. `--verbose` adds the
   benchmarks (free 8, hindsight 8 + 9 = 17) and timelines.
3. **Knowing when not to push**: `python run_doi.py --demo toy --fee 100 --policy never,rof` gives 40 and 40:
   rof makes no push when the price is never repaid.
4. **Fleet**: `python run_doi.py --demo fleet` (8 robots, 4 obstacles in a barrier): never 1552, rof 1233,
   central 1244 on seed 0 (the barrier holds pallets, a crate and a shelf unit). Several meters, one per obstacle; `--verbose` shows each push and its landing.
5. **Build your own map**: `python run_doi.py --build` asks for the layout, sizes, kinds, costs and arms,
   prints the map with warnings, then runs it and prints the one-line command that repeats the run.
6. **Sharing the result**: `--html toy.html` writes a self-contained player (play, pause, scrub) that opens in
   any browser; send that rather than a GIF. `--gif` still works, but macOS Preview shows the frames of a
   GIF as separate images; open it in a browser or Quick Look. Add `--no-show` to render files without opening
   the window.

## What each scenario shows
| Scenario | Plain meaning | Demo status |
|---|---|---|
| `scatter` (demo) | open floor, obstacles anywhere, one trip per robot | the plainest view; see item 0 above |
| `toy` (demo) | one robot, one shelf unit in a gap, four crossings | clean |
| `single_block` | two rooms, one pallet in the barrier, long way round at the bottom | with 12 robots congestion dominates and pushing can lose |
| `multi_block_wall` | four obstacles in the barrier (pallets, a crate, a shelf unit) | works: 8 robots, 10 tasks, seed 0: never 1552, rof 1233 |
| `series_blocks` | two obstacles in a one-cell corridor | neither can be pushed clear: shows "no room" |
| `complements` | two doorways in series | only a two-step plan sees the saving |
| `shift` | demand moves half way through | not demo material yet |
| `random_blocks` | open floor with obstacles anywhere (optional wall strips) | what `scatter` uses; often little to gain unless cluttered |
| `incidents_room`, `incidents_aisles` | obstructions appear during the run; reports arrive as text | use to explain reports and the safety rule |
| `warehouse_blocks`, `warehouse_incidents` | benchmark maps | not demo material |

## Flags in one line each
Setup: `--sim`, `--demo`, `--scenario`, `--policy a,b,c` (arms compared on identical tasks), `--robots`,
`--tasks`, `--seed`. Costs: `--kappa` (push step = kappa x weight), `--fee` (cost per push run),
`--push-max`, `--theta` (0 eager, 1 ski rental). Information: `--r-comm` (message reach, 0 none), `--loss`,
`--latency`. Map: `--build`, `--layout barrier|strips|map`, `--rows`, `--cols`, `--blocks`, `--doors`,
`--kind`, `--crossing`, `--strips`, `--pallets`, `--crates`, `--shelves`, `--map`, `--yes`. Language layer:
`--intake none|oracle`, `--p-false`, `--p-report`, `--p-wrong-class`. Output: `--html`, `--gif`, `--gif-arms`,
`--show`, `--no-show`, `--verbose`, `--replay`, `--no-benchmarks`.
Table columns: J total cost; HR_av times worse than hindsight on avoidable cost; PoD cost relative to the
central arm; collateral detour caused by a parked obstacle.

## Honest caveats to say out loud
* Single runs are anecdotes. The claims are about aggregates over many seeds, and that experiment has not been
  run on this model yet (see `results.md`).
* `eager` can beat `rof` when traffic keeps coming (the toy), and loses when it stops. The ski-rental rule's
  value is that it is never far from the best either way, not that it wins every run.
* A pushed obstacle can block someone else. The ledger's evidence includes that, but only from what the
  robot knows; the `collateral` column measures what it really cost on the true map.
* The barrier maps are built to make pushing worthwhile. `--crossing 0` or a high `--fee` shows when it is not.
* The language model is simulated: `--intake oracle` stands in for it, and no real model has been run.
* Only pushing exists. Carrying obstacles to a dump zone, claim locks and detour pushes for others' benefit
  are later stages (see the design in `docs/superpowers/specs`).
