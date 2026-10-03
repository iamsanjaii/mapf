# Stage 2 plan: carry obstacles to a dump zone, rack or pit

Status: plan only, written 2026-10-03 from the professor's brief on obstacles. Nothing here is built.
Stage 1 (push model, `2026-10-02-push-obstacles-design.md`) stays unchanged until this stage starts.

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

**Belief.** One more tick-stamped register for slot occupancy (set when seen full or dropped into, cleared when
seen empty), merged by max like the others. Pits use a grow-only "filled" set instead, since a filled pit never
reverts (merge by union, so robots converge in any order). Robots can disagree about a slot: two may head for the same one.
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
- A full slot is not chosen; seeing it freed makes it choosable again.
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
