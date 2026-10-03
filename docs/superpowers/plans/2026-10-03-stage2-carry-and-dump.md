# Stage 2 plan: carry obstacles to a dump zone, rack or pit

Status: built 2026-10-03 (all six build-order steps; 397 tests pass, 52 legacy + 345 in `tests/doi`). Written from the
professor's brief on obstacles. The design below is the original plan; where the build differs, the section
"Implementation notes" at the end is the record. Stage 1 behaviour is unchanged when a map has no slots or pits.

Naming note: `2026-10-02-stage2-twin-outline.md` also says "Stage 2" (digital twin). That one is a separate
track and does not depend on this one; in the push-model numbering used here it comes after Stage 3.

## The professor's brief

- **Warehouse.** An obstacle can sit in the middle of a pathway, and a crate there has to be kept on a rack.
  There can also be a central location: a **dump zone**, held back for this stage.
- **Construction or disaster site.** A **pit** can be filled by materials or debris, which opens a path.

## Why Stage 1 is not enough

A pushed obstacle only moves one cell at a time and stays where it lands. Collateral cost (a pushed obstacle
blocking another route) is the price of that. The brief asks for a place where removed obstacles end up, so
the cost of a removal has to include getting the obstacle there, and the cleared cell stays clear.

## Model

**Three removal modes**, chosen per obstacle by the robot from its own beliefs:

| Mode | What happens | Where the obstacle ends up | Cost |
|---|---|---|---|
| push (Stage 1) | slide along a line | the landing cell, still an obstacle | `kappa*w` per step + fee |
| carry | pick up, walk to a slot, drop | a slot (rack cell or dump-zone cell), off the route | pick-up fee + `kappa_c*w` per step walked + drop fee |
| fill | carry to a pit and drop in | consumed; the pit becomes floor | as carry, with the pit as the slot |

**Cells and slots.** New cell types: `D` dump-zone cell and `T` rack cell (both passable floor when empty), and
`P` pit (impassable until filled). A *slot* is a dump or rack cell with capacity 1; a pit has capacity 1 and is
consumed when filled. Capacity is what makes the choice interesting: a full rack no longer helps.
A filled pit stays filled for good (decided 2026-10-03): it becomes ordinary floor and cannot reappear.

**Rack versus dump zone.** Same mechanism, different geometry. Racks are many small slots spread through the
map (near where crates are found); the dump zone is a region of many slots (decided 2026-10-03): a block of `D` cells, each holding one obstacle,
so it fills up cell by cell and there is no entrance queue. Both are slots; a scenario
may use either or both. Kinds get an allowed-slot set (a crate may go on a rack, a shelf unit only to the dump
zone, debris to a pit or the dump zone).

**Actions.** `Pick(cell)` (robot becomes loaded, one obstacle at a time, its speed unchanged in the first
version), `Drop(cell)`. A loaded robot cannot pick or push. A robot blocked while loaded drops at the nearest
free slot or, failing that, keeps carrying (it is charged per tick as usual).

**Plan and price.** For a believed obstacle `o` and slot `s`:
`total = d(pos, adj(o)) + pick_fee + kappa_c*w*d(adj(o), s) + drop_fee + d(s, goal)`
(with the obstacle's cell open and the slot now occupied). Price is `max(0, total - d_open)`, as in Stage 1.
Evidence `E` and the arm rules (`rof`, `central`, `myopic`, ...) are unchanged; only the price set grows.
For each obstacle the robot picks the cheapest of push, carry and fill, then applies the arm's rule.

**Belief.** One more component for slot occupancy (see implementation note 2: a grow-only set). A filled pit
never reverts (see note 1). Robots can disagree about a slot: two may head for the same one.
That is accepted in the first version (the second robot finds it full and re-plans, which is a measured cost),
because a claim lock is a Stage 3 item.

**Cost.** `J` gains pick fees, drop fees and carry steps; `tick_cost` still sums to `J`. New metrics:
`carries`, `fills`, `slot_conflicts` (arrived and found it full), `slot_utilisation`, and the dump-zone
traffic jam (waits within 2 cells of a slot).

## Scenarios

- `warehouse_racks`: aisle map with crates blocking aisles and a few rack slots; crates must go to a rack.
- `dump_central`: one dump zone at the edge of the map; far crates make carrying expensive, so the arm's choice
  between pushing aside and hauling away is visible.
- `site_pits`: a barrier of pits across the route, debris kinds lying nearby; filling opens the short route.
- `mixed`: racks, a dump zone and pits together, with a shelf unit that only the dump zone accepts.

## Builder, window, CLI

Map files gain `D T P`. `--build` takes `--slots`, `--slot-layout rack|dump`, `--pits`, `--debris`. The window
draws slots and pits, shows a load marker on carrying robots, and the caption says "carried crate to rack 3".
`--pits` and `--stock` flags return in a new form; the old sandbag-station and kit logic stays removed. The
pit simulator at commit `29fdf45` is a reference for the pit cell only.

## Build order

1. Cell types and slot data (`kinds.py` allowed-slot sets, `world.py`, `maps.py`), with the map parser.
2. `Pick` and `Drop` actions in `world.py`, loaded-robot state, hand-checked tests.
3. Carry and fill plans in `pushplan.py` (rename to a removal planner), cost bookkeeping.
4. Slot occupancy register in `belief.py` and CRDT merge test.
5. Arm wiring in `policies.py`, then scenarios, builder and window.
6. Experiments: choice of mode versus slot capacity and distance.

## Tests (hand-checked)

- A crate one step from a rack: carry costs `pick_fee + kappa_c*w + drop_fee`, matching a manual count.
- A full slot is not chosen.
- A pit filled by debris turns to floor and stays floor, and the route through it is open for every robot that
  learns of it (gossip merge of the filled set is a union).
- A dump region of `n` slots accepts exactly `n` obstacles; the `n+1`-th robot finds it full and re-plans.
- Two robots targeting one slot: the second re-plans and `slot_conflicts == 1`.
- `J == sum(tick_cost)` with carrying; determinism for a seed; Stage 1 tests stay green with no slots on the map.

## Decisions

Settled (2026-10-03):

1. A filled pit stays filled permanently (grow-only "filled" set in the belief).
2. The central dump zone is a region of many one-obstacle slots, not a single entrance with a queue.
   Congestion near the dump then comes from the layout and slot count, and is measured by the near-slot waits metric.
3. A loaded robot moves at normal speed in the simulation. A real loaded robot would be slower from the extra
   weight, but that is not modelled now; the carry cost `kappa_c*w` per step already stands in for the load.
   If it is added later, it is a per-kind slowdown on loaded steps (a loaded step takes more than one tick).

4. Filling pits with debris is part of Stage 2, not a later experiment. The `fill` mode, the `P` cell, the
   `site_pits` and `mixed` scenarios and the pit tests above all ship in this stage.

Still open: none.

## Implementation notes (built 2026-10-03)

Where the build departs from the design above, and what it found.

**Departures from the design**

1. **A pit is an obstacle of kind `pit`, not a new cell type.** It is unpushable and uncarriable. Filling it deletes
   the obstacle, so every existing mechanism (belief registers, rent, evidence, gossip) handles it unchanged. Permanence
   needed no "filled" set: the world never recreates a pit, and the belief's blocked/free tick registers merge by max,
   so a robot that saw the fill beats any stale gossip (pinned by `test_a_filled_pit_stays_filled_when_a_stale_belief_gossips_it`).
2. **Slot occupancy is a grow-only set** (`BeliefState.slots_full`), not a tick register, because a slot is never
   emptied. So "seeing a slot freed makes it choosable again" does not arise and has no test.
3. **Rack cells are fixtures (not walkable); dump cells are walkable floor.** A stored item is not an obstacle, so it
   never blocks a route, and a dump region cannot seal itself off as it fills. Pick and drop act on an *adjacent*
   cell, as push does.
4. **`kappa_c` must be at least 2** (validated), so that a loaded step never costs less than an empty one: the lightest
   carriable kind weighs 0.5. Defaults: `kappa_c` 2, `pick_fee` 1, `drop_fee` 1.
5. **Fill sources are any debris on the map**, not only debris on the robot's route: the pit is on the route, the
   debris is the means. Carry sources are on the route, as push candidates are. The decision's subject for meters and
   captions is the pit for a fill, the obstacle for a carry.
6. **New modules instead of renaming `pushplan.py`:** `carryplan.py` (plans), `carrier.py` (state machine, like
   `pusher.py`). The existing push planner is untouched except for a `key` property.
7. **A loaded robot never abandons its load.** If its target turns out full it retargets to the cheapest free slot or
   pit; if none exists it carries on with its task loaded and looks again whenever its belief changes.
8. **Builder:** a fourth layout `site` (`--layout site --pits --debris --racks --dump-rows --dump-cols`) replaces the
   planned `--slots/--slot-layout` flags. Map files accept `P`, `T`, `D`.
9. **Hindsight lower bound** (`oracle.buy_lb`) is now the cheaper of the push bound and the carry bound (pick plus
   drop fee, if some slot or pit can take the kind), and a pit is priced as a pick plus a drop if any debris exists.

**What it found (quick runs, direction only)**

* On all four scenarios every arm finishes and `J == sum(tick_cost)`: 864 runs (4 scenarios x 6 seeds x 2 fleet sizes x
  2 radii x 9 arms) and 250 more with one rack, one dump cell, or fewer debris than pits, with no stalls or crashes.
* At default costs pushing beats hauling on `dump_central`: `rof` carries nothing and pushes. Carrying starts to win
  when pushing gets dear (`doi_s2_modes.py`: at kappa 16 a near dump gets 2 carries and no pushes, a far dump 1 carry and
  2 pushes) and as racks are added (Spearman of racks against carries 0.89).
* `free` is not a lower bound on `J` once robots crowd a doorway: on `site_pits` (8 robots, seed 2) `free` cost 1244
  against 1209 for `central`, because opening every obstacle at tick 0 sends everyone through the same gap.
* Two robots after one slot: the second finds it full (or sees it full on approach) and carries its load on to its
  goal; the cost shows up as `slot_conflicts` or as a wasted pick-up.

**Full S2 run (30 seeds, 12 robots x 12 tasks, `--jobs 8`, 3 minutes, no run stalled; results in the git-ignored
`experiments/results/doi/s2_modes`)**

* `rof` beats `never` in every seed at every rack count, by a median of 676 to 718 on a cost of about 2,900 (min 442).
* Racks change the mix, not the total. Median `rof` cost is 2,225 with no rack and 2,235 with eight, while pushes fall
  from 7 to 1 and carries rise from 0 to 4 (Spearman of racks against carries 0.94). At default costs a crate
  pushed aside costs about what a crate carried away does, and a parked crate rarely blocks anyone, so a rack buys
  no saving here. The case for carrying is collateral, which this map barely has.
* `rof` and `central` are within 0.1% (PoD 0.999 to 1.000) and median `slot_conflicts` is 0 with full sharing.
* Haul distance and push cost (`dump_central`): with a near dump, median carries rise 1, 2, 3, 3 as kappa goes
  4, 8, 16, 32 and pushes fall 5, 2, 0, 0. With a far dump the same sweep gives carries 0, 0, 1, 3 and pushes
  7, 5, 2, 0. Carrying costs more than pushing at every kappa with the far dump (median J 2,431 against 2,312 near,
  at kappa 32), which is the price of the haul.

**Not done**: a claim lock on slots (Stage 3), loaded-robot slowdown (decided out of scope), and a scenario where
parked obstacles would otherwise cause collateral (needed to show carrying paying for itself).
