# Rent-or-Fill, Stage 1: removable obstacles that robots push

Status: approved in conversation on 2026-10-02; implementation started immediately at the owner's request.
Pits, kits, depots and sandbags are dropped. The pit-based simulator is recoverable from git (commit `29fdf45`).

## Intent

A reviewer watches a warehouse map in which obstacles can be removed, and the fleet decides, from partial
information, when removing one beats walking round it. Robots remove an obstacle by physically pushing it.
The objective is the total cost of the whole fleet (`J`), not path feasibility at any price. A task that cannot
be completed is charged the large unreachable penalty, so infeasibility is expensive but allowed.

## Model

**Cells.** `.` free, `#` permanent wall (never removable), and removable obstacles of named kinds placed
anywhere on the grid: pallet `L` (weight 1.0), crate `C` (0.5), shelf unit `S` (2.0). Kinds are a table in
`src/doi/kinds.py` (glyph, colour, weight). Permanent walls may form continuous strips.

**Push.** A robot standing next to an obstacle steps into its cell. The obstacle slides one cell onward if the
cell beyond is free (no wall, obstacle or robot). If the robot's route continues straight it keeps pushing.
A push step costs `kappa * weight` instead of 1; one push run (consecutive pushes of one obstacle by one
robot in one direction) also costs one `fee`. A pushed obstacle is an ordinary obstacle where it lands.

**Own path only.** A robot never leaves its route to push. For the task in hand it compares:
- the detour `d_block` (shortest route avoiding every believed-blocked cell), and
- the best push plan: walk to an approach cell, push `k` steps along a straight line, then walk to the goal
  with the obstacle at its landing cell. `total = d(pos, approach) + k*kappa*w + fee + d(end, goal)`.
Candidate plans consider each believed obstacle, 4 directions and `k` from 1 to `push_max` (default 6).

**Rent and price.** Rent for a task is `d_block - d_open`, as before (`d_open` opens every believed removable
obstacle). The price of a push plan is `max(0, total - d_open)`, the premium paid over the ideal route.

**Ledger.** Every task a robot plans is recorded as an origin/destination pair (not only tasks with rent),
so the fleet can also see traffic that a relocated obstacle would start to block. Evidence for a plan is the
net reduction in recorded travel distance if the push had been done earlier:
`E = sum over records of d(origin, dest | B) - d(origin, dest | B')`, with `B'` = believed blocked cells after
the push. Collateral at the landing cell therefore lowers `E` automatically. Records spread by gossip within
`r_comm` (loss and latency apply), as before.

**Arms.**

| Arm | Push when |
|---|---|
| `never` | never |
| `myopic` | `total <= d_block` (cheaper for me; ignores the ledger) |
| `eager` | `E > 0` |
| `rof` | `E >= theta * price` (the method; gossip ledger) |
| `rof_local` | same, ledger holds only the robot's own records |
| `rof_f` | forecast: `E * (remaining tasks / completed tasks) >= price` |
| `central` | same rule, one omniscient ledger refreshed from the true map |
| `free` | benchmark: all obstacles removed at tick 0, free |
| `hindsight` | benchmark: best subset removed at tick 0, charged `fee + kappa*w` each (lower bound; obstacles vanish) |

**Belief.** The shared belief keeps the existing last-seen-blocked and last-seen-free tick registers (merge by
max, so robots still converge in any order), a per-cell kind register (tick-encoded), the rent/traffic records
and the census. There is no grow-only "filled" set: a moved obstacle is learned from sensing (`r_sense`) or
gossip. `editable()` is the set of believed-blocked cells that are not `needs_human`.

**Cost.** `J = (moves - push_steps) + waits + sum(kappa*w per push step) + fee * push_runs`, tracked per tick
(`tick_cost`) so the animation's running counter equals `J` at the end.

**Metrics added.** `removals` (push runs), `push_steps`, `push_rejected` (by reason), `collateral_cost`
(rent paid in the true map because a relocated obstacle sat on a route, computed post hoc), plus the
existing `J`, `HR_av`, `PoD`, `stalled`.

## Scenarios

Same shapes, new names: `single_block`, `two_blocks_parallel`, `series_blocks`, `multi_block_wall`, `shift`
(solid barrier with removable obstacles in it, doors at the bottom as the long way round);
`random_blocks` (random continuous wall strips plus removable obstacles of chosen kinds);
`incidents_room`, `incidents_aisles` (obstructions appear mid-run; non-`needs_human` ones are pushable, the
kind sets the weight; reports and intake unchanged); `warehouse_blocks`, `warehouse_incidents`
(benchmark maps, obstacles in one-cell corridors). Tasks and starts never sit on an obstacle cell.

## Builder, window, CLI

`--build` / `--layout barrier|strips|map`. Barrier: rows, columns, barrier column, number of removable
obstacles in it (`--blocks`), door rows, `--kind`, `--crossing`. Strips: random wall strips (`--strips`,
`--strip-min`, `--strip-max`) plus counts per kind (`--pallets`, `--crates`, `--shelves`). Map: ASCII file using `# . L C S`. Checks warn about obstacles with no
push room, nothing worth removing, and unreachable goals. The window draws each kind with its colour and
letter, the rent meter has one bar per obstacle, captions describe pushes, and a collateral counter is shown.
`--pits`, `--stock`, `--depot-dist`, `--claim`, `--gate`, `--p-wrong-class` and `--stagger` go away.

## Removed in Stage 1

Pit cells, kits, depots, sandbags, hauling, claim leases (`ClaimSet`, `PNStock`), the approval gate and
simulated supervisor, `rof_pit`, `rof_w`, `rof_x`, experiments E1, E3, E4, E6, E8 and their pilot results
(re-expressed in Stage 3). The legacy `src/` outside `src/doi` is untouched.

## Later stages

2. Carry: dump zones `Z`, pick-up and drop actions, dump-slot capacity, per-kind choice of mode.
3. Detour pushes justified by the ledger, claim lock, window features, E1-E8 re-expressed and re-piloted,
   theory wording (collateral adds a term to the bound).

## Testing

TDD per module. Hand-checked cases: a 1-cell gap with one obstacle (push through, then free), a corridor
with no room (no push), collateral when a landing cell blocks a recorded route, `J == sum(tick_cost)`.
Determinism for a seed, and the existing motion, network, rng, spacetime, stats, incident and intake tests
stay green.

## Implementation notes (decisions made while building, 2026-10-02)

* **Ledger endpoints.** A past record's start and goal cells are treated as open when summing travel (a finished
  task cannot be blocked at its goal). The task in hand is counted separately from the robot's current position,
  with a blocked goal really unreachable; in aggregated-record mode one traversal is taken off the matching entry.
* **Eligibility.** A robot only plans a push on an obstacle it has confirmed by sight whose class is
  `robot_clearable`. Scenario obstacles are visibly plain; an incident obstruction is classed only by a report,
  and once a robot has pushed one it is plain too. Unknown class means never pushed.
* **Dead cells.** A plan never lands an obstacle where it could never be pushed again (no axis with free floor on
  both sides); otherwise obstacles end up in corners on goal cells for good.
* **Goal cells.** A robot whose goal cell is believed blocked walks up to it (to see what is on it) and waits
  there instead of bumping into it.
* **Forecast arm.** `rof_f` scales the saving on all recorded traffic by the tasks still to come and adds the
  task in hand in full, using the known task counts (an optimistic information assumption).
* **Livelock.** A run is marked stalled if nothing is completed or pushed for `20 * (H + W)` ticks, because
  wandering robots otherwise look like progress.
* **Toy.** With a shelf unit (weight 2.0) the toy gives never 40, myopic 40, eager 23, rof 31, central 31,
  free 8, hindsight 17: steady traffic favours pushing early, which ski-rental theory allows.
* **Deferred.** Incident scenarios keep working (non-`needs_human` obstructions are pushable, the kind sets
  the weight); the approval gate, claims, `rof_pit/w/x` and experiments E1, E3 to E6, E8 are removed.
