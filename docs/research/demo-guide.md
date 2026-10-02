# Demo guide: explaining `run_doi.py` to a professor

`python run_doi.py --guide` prints every flag in plain language, `--list` describes every scenario and
arm. This page gives the story, the order to show things in, and numbers that were actually produced.

## The story in four sentences
Warehouse robots sometimes meet a blocked cell: a gap they cannot cross, or a pallet that fell in an aisle.
They can detour (pay "rent" every trip) or pay once to fix the cell (fetch a kit from a depot, carry it,
apply it). This is ski rental: fix when the rent paid so far equals the price. The difficulty is that no
robot sees the whole fleet's detours, so robots share a small ledger by messages and a lock stops two robots
fixing the same cell. We measure how much that decentralisation and missing information costs against a
central planner and a hindsight benchmark.

## Order to show things
1. **Animation** (the visual): `python run_doi.py --demo toy --gif toy.gif` (add `--show` to open a window).
   Left panel never fixes the pit and keeps walking the long way round; right panel (rof) waits until
   detours add up, fetches a kit (black ring plus orange square) and fills the pit (red to green). The
   running cost counter ends at 40 (never) and 29 (rof).
2. **Terminal story**: `python run_doi.py --demo toy` prints the table:
   never 40, myopic 40, eager 30, rof 29, central 31, free 8, hindsight 8 + 9 = 17.
   One crossing saves 8 but the fix costs 9, so myopic never fills; eager fills too early; rof waits for
   two crossings (16 >= 9).
3. **Knowing when not to buy**: `python run_doi.py --demo toy --fee 100 --policy never,rof` gives 40 and 40:
   rof makes no edit when the price is never repaid.
4. **Fleet**: `python run_doi.py --demo fleet --gif fleet.gif` (8 robots, 4 pits in a wall). Many robots
   print TRIGGER at nearly the same tick because each decides locally; the claim (lock) lets one haul.

## What each scenario shows
| Scenario | Plain meaning | Demo status |
|---|---|---|
| `toy` (demo) | one robot, one pit, four crossings | clean |
| `single_pit` | two rooms, one shortcut pit near the top, long way round at the bottom | avoid with 12 robots: the pit saves little on average and funnels traffic |
| `multi_pit_wall` | four pits in a wall | works: 8 robots, 10 tasks, seed 2: never 1600, rof 1421, central 1314 |
| `series_pits` | two pits in a one-cell corridor; fixing one saves nothing | works: 4 robots, 12 tasks, seed 2: never 1108, rof_pit 1108 (never fills), rof 970 |
| `two_pits_parallel` | two pits, either one is a shortcut | both pits were filled in the seeds tried, so it does not yet show "no second fill" |
| `incidents_room`, `incidents_aisles` | obstructions appear during the run; reports arrive as text | see the caveat below |
| `shift`, `random_pits`, `warehouse_*` | demand shift, random maps, benchmark maps | not demo material yet |

## Flags in one line each
Setup: `--demo`, `--scenario`, `--policy a,b,c` (arms compared on identical tasks), `--robots`, `--tasks`,
`--seed`. Costs: `--kappa` (cost per carried step), `--fee` (cost per fix), `--theta` (0 eager, 1 ski rental).
Information: `--r-comm` (message reach, 0 none), `--loss`, `--latency`, `--claim on|off` (the lock).
Language layer: `--intake none|oracle`, `--gate` (simulated human approval), `--p-false`, `--p-report`,
`--p-wrong-class`. Output: `--gif`, `--gif-arms`, `--show`, `--replay`, `--no-benchmarks`.
Table columns: J total cost; HR_av times worse than hindsight on avoidable cost; PoD cost relative to the
central arm; wasted cost of abandoned hauls.

## Honest caveats to say out loud
* Single runs are anecdotes. On several larger seeds `eager` or `rof_local` matched or beat `rof`; the
  claims are about aggregates over 30 seeds, and that experiment has not been run yet (see `results.md`).
* The effect of sharing information (`--r-comm 0` against `inf`) is not visible in one small run.
* The incident demo is not yet persuasive. `incidents_aisles`, 8 robots, 8 tasks, seed 202: never 999;
  rof without reports 995 (no fills); rof with `--intake oracle` 1022 (one fill, 30 realised cost, little
  left to save); with `--gate` and a 50% wrong-class rate 1084 (the gate adds a 30-tick wait). Use it only to
  explain how reports, the gate and the safety rule work, not to claim a saving.
* The language model is simulated: `--intake oracle` stands in for it, and no real model has been run.
* The supervisor is simulated.
