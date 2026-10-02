# Rent-or-Fill: Decentralised Online Decisions on Permanent Layout Edits for Warehouse Robot Fleets, with LLM Exception Intake and Human-on-the-Loop Approval

Status: DESIGN SPEC v2 (no code written). Date: 2026-10-02. Target: IEEE T-ASE Special Issue "Agentic
Intelligence for Materials Handling, Warehousing, and Logistics 5.0" (deadline 15 Feb 2027).
Companion plan: `docs/superpowers/plans/2026-10-02-rent-or-fill.md` (v2).
Prior-art notes: `docs/prior_art.md`. Review that produced v2: `docs/research/review-2026-10-02.md`.

This document is the thing to approve or reject. The plan only executes it.

### What changed from v1

| Change | Section | Review item |
|---|---|---|
| Option A chosen: LLM layer for exception intake and human-on-the-loop approval; LLM kept out of the decision rule | 3, 5 | review §3 |
| Warehouse mapping: blocked cell = pit or incident obstruction; edit = fill or clear | 2.1 | review §3 |
| Dynamic incident scenarios added next to the static pit scenarios | 2, 7.1 | new with option A |
| Safety invariant: no haul toward a permanent edit without sensor confirmation by a robot | 5.4 | new with option A |
| Ledger stores task records, not bundle totals; evidence recomputed under the current filled set | 4.3 | review 5.3 |
| Optional time window on evidence (RoF-W) | 4.3, 4.6 | review 5.8 |
| Traffic channel separated from ledger channel | 2.4 | review 2.1 |
| Arbiter override logging | 7.4 | review 2.2 |
| Ratio defined on avoidable cost (HR_av); "competitive ratio" renamed "hindsight ratio" | 2.6 | review 5.1, 5.6 |
| E1 sweeps fee and kappa; pilot feasibility check before preregistration | 7.3, 8 | review 5.2 |
| Realised buy cost tracked and used in P1/P2 | 4.4, 6 | review 5.4 |
| Rent counted at planning time made explicit in P1 | 6 | review 5.5 |
| "Lifelong" dropped from the title; throughput added | title, 7.4 | review 5.7 |
| Prior art extended | 1 | review §4 |
| Validation ladder from toy grid to digital twin | 9 | review §3, §6 |
| Scope trimmed: RoF-X and E5 are stretch goals | 4.6, 7.3 | review 5.9 |

---

## 0. Corrections to earlier analysis (read first)

Facts verified by reading the repo on 2026-10-02. They change what the project can honestly claim.

| # | Earlier claim | Verified fact | Consequence |
|---|---|---|---|
| 1 | "Robots carry sandbags and pay" | `SandbagManager.deploy` (`src/mapf_ro/sandbag.py`) teleports the fill and charges Manhattan distance x 4. No robot moves, carries or is delayed. | "Who hauls" is not modelled today. It must be built as a physical, time-extended action. |
| 2 | "Removal runs in the agentic loop" | It does not. `CostAgent` marks REMOVE `feasible=False`; `CoordinatorAgent._apply_action` REMOVE branch only appends a log entry. | Agentic MAPF and MAPF-RO are two disconnected programs. |
| 3 | "MAPF-RO handles conflicts" | `MAPFROPlanner.plan` runs independent A* per robot. No time dimension, no conflicts. Experiment E (`experiments/mapf_ro.py`) therefore measures a no-interaction setting. | Any claim about removal inside multi-robot traffic needs a new simulator. |
| 4 | "Traffic index drives the decision" | README says `removal_cost = base_cost / traffic_index`. Code uses `sandbag_travel + fill_cost`; `traffic_index` is recorded only. | Do not cite the README formula. Fix README (plan task 0). |
| 5 | "`prioritized.py:226-228` edge check is a bug" (earlier remark) | Re-derived: the check rejects any move that leaves a cell another robot enters at t+1. That catches every swap and also rejects some legal "follow" moves. Sound, conservative, not a bug. | Do NOT change `prioritized.py`. Retracted. |
| 6 | Execution model | There is none. Paths are precomputed; no tick loop moves robots. | The new work needs a tick-based world. |
| 7 | "The repo is agentic" | `src/agents/` is a rule-based, centralised coordinator (Path/Conflict/Cost agents under one `CoordinatorAgent`). No LLM or foundation model anywhere. | "Agentic" in the CFP's sense (LLM / foundation model) is new work in v2, not existing work. |

Baseline: `pytest tests -q` gives 52 passed (2026-10-02, Python 3.14.6). The refactor-free
rule in the plan means this must stay at 52 passed until the final task.

---

## 1. Positioning

Searches run 2026-10-02 on top of `docs/prior_art.md`. Abstract level only; section 12 lists the reading
that must happen before any paper claim. Items marked (check) were added from memory in the review and are
unverified.

### 1.1 Environment modification in MAPF

| Work | Setting | Why it is not this project |
|---|---|---|
| Multi-Agent Terraforming / tMAPF (Vainshtein, Solovey, Salzman 2022) | Some agents move obstacles; CBS/PBS extensions | Centralised, offline, all tasks known, jointly optimised |
| Destructible obstacles (Andreychuk, Yakovlev 2018) | Agents destroy obstacles inside Theta* | Centralised, offline |
| M-PAMO (CBS/PP, 2025) | MAPF among movable obstacles | Centralised, offline |
| Guidance-graph optimisation for lifelong MAPF (Zhang et al. 2024) (check); warehouse layout co-design (2023) | Shape the environment for throughput | Offline, centralised, design-time |
| Lifelong / online MAPF (Li et al. 2020; Švancara et al. AAAI 2019 (check)) | Task streams | Static obstacles; no environment decisions |

### 1.2 Rent-or-buy theory

| Work | Setting | Difference |
|---|---|---|
| Ski rental (Karlin et al. 1988) | Single agent, single purchase | P1 is this result; included only as an anchor |
| Multi-Agent Ski-Rental (arXiv 2507.15727, 2025) | Agents rent, buy individually or buy a group pass | No space, routing, communication limits or physical execution |
| Online Facility Location (Meyerson FOCS 2001); parking permit (Meyerson 2005) (check) | Open a facility vs. pay connection cost per spatial request | **Closest theory.** Centralised online algorithm with full information; no robots, no execution, no information delay. Must be distinguished explicitly. |
| Online Steiner forest, buy-at-bulk | Centralised online network design | No decentralised information |

### 1.3 Decentralised coordination and task allocation

| Work | Setting | Difference |
|---|---|---|
| Contract Net (Smith 1980); MRTA taxonomy (Gerkey & Matarić 2004); market-based MRTA (Dias et al. 2006) (check) | Allocate known tasks to robots | Task value is known up front. Here the value of the edit is unknown and accumulates online from team rent. |
| CBBA (Choi, Brunet, How, T-RO 2009) (check) | Decentralised bundle auctions with consensus | **Reviewers will ask about this.** CBBA allocates tasks of known reward. RoF decides *whether* a shared task exists at all, from distributed, partial evidence; the hauler choice (section 4.5) is a much simpler protocol than CBBA and is not claimed as new. |
| Decentralised MAPF: DMAPF, token negotiation (2024), PRISM (2025), Karma (2026) | Decentralised path conflicts | Static environment; the decision variable is the path, never the map |

### 1.4 LLMs in multi-robot systems

| Work | Setting | Difference |
|---|---|---|
| LLM multi-robot planners, e.g. SMART-LLM (Kannan et al. 2023), RoCo (Mandi et al. 2023) (check) | LLM decomposes or negotiates tasks in the planning loop | The LLM sits on the decision path. Here it is deliberately kept off it (section 5.1). |
| LLM / VLM for warehouse exception handling | To be searched (section 12) | Unknown. Must be searched before any claim about intake. |

Do NOT claim any of these as new: removable or movable obstacles in MAPF; trading removal cost against
detour cost; marginal-cost pricing; negotiation between path-planning agents; karma or credits; CRDTs in
multi-robot systems in general; LLM-to-structured-record extraction in general.

### 1.5 The research gap in one sentence

When obstructions appear unpredictably, no robot knows the future task stream, and no robot holds the
global picture, how should a warehouse fleet decide whether, when and by whom a costly, permanent layout
edit is made, how much is lost to missing information compared with a central planner and a hindsight
benchmark, and how can a language-model layer turn unstructured incident reports into inputs for that
decision without ever being able to trigger a permanent edit on its own?

### 1.6 Contributions claimed (provisional, pending section 12)

1. A decentralised online decision rule for shared permanent edits (RoF) with a measured cost of
   decentralisation (PoD) and of missing information (coverage at trigger), under range-limited, lossy
   communication. **Main contribution.**
2. A record-based CRDT ledger whose evidence is recomputed under the current edit set, which handles both
   complementary and substitute edits (section 4.3).
3. An agentic architecture in which an LLM layer does exception intake and drafts approval requests, with a
   safety invariant that keeps it off the permanent-action path, and a measurement of its accuracy,
   latency and end-to-end cost (sections 5, 8).
4. Propositions P1 to P4 as support (section 6). Not a headline.

### 1.7 Why decentralise in a warehouse

A warehouse execution system usually sees everything, so the paper must say why it should not decide
alone. Three cases, each a CFP topic:

* **Mixed-vendor fleets.** AMRs, forklifts and shuttles from different vendors often have separate fleet
  managers; no single controller holds all detour costs.
* **Network constraints.** Yards, docks, cold stores and high-bay racking have weak or patchy wireless.
  ("communication-efficient coordination under industrial network constraints")
* **Graceful degradation.** When the central link fails, the fleet should keep making sensible decisions.
  PoD is read as the cost of losing the central link. ("reliability and graceful-degradation strategies")

---

## 2. Problem definition: DOI-MAPF

Decentralised Online Infrastructure MAPF.

### 2.1 Warehouse mapping

| Simulator term | Warehouse meaning |
|---|---|
| Blocked cell | A cell no robot can pass: a static gap (`PIT`) or an incident obstruction (fallen pallet, spill, debris, damaged floor plate) |
| Edit | One carried trip of a kit (bridging plate, clean-up kit, ramp) from a station to the site, then applying it. After the edit the cell is passable for the rest of the horizon. |
| Station | Equipment or kit station with finite stock (`SANDBAG` cell type in the grid) |
| Permanent | Once cleared or bridged, a cell stays passable. A new incident at the same cell is a new obstruction with a new id. Undoing an edit is never needed or allowed. This is why ski-rental applies. |
| Robot-clearable vs. needs-human | Some obstructions (rack damage, hazardous spill) cannot be edited by robots. Their rent is still recorded and reported to the supervisor as "cost of not clearing" (section 5.3); robots never edit them. |

The simulator keeps the v1 mechanics (fetch from station, carry, apply), so `PIT`/`SANDBAG` remain the
code names.

### 2.2 World

H x W grid, 4-connected. Cell types: FREE, OBSTACLE (permanent), PIT (impassable until filled),
SANDBAG (passable; a station holding stock). Two scenario families:

* **Family S (static).** Pits and stations are known to all robots a priori. Pit status (unfilled/filled)
  is dynamic and learned by sensing or messages. Used for H1 to H4 (theory checks).
* **Family D (dynamic incidents).** Pits do not exist at t=0. Obstruction `o` appears at tick `t_o` on a
  cell set `C_o` with class `robot_clearable` or `needs_human`. A free-text incident report about `o` is
  emitted at `t_o + delta_o` at a reporting point (section 5.2); with probability `p_report` there is no
  report and robots learn of `o` only by sensing. False reports (no obstruction behind them) are emitted at
  rate `p_false`. Used for H7 and H8.

### 2.3 Robots

N robots, synchronous global tick clock. Each robot owns a fixed list of K tasks (task k = go to goal g_k,
then immediately start k+1). Robots know only their own list. Tasks and incidents are generated before the
run from a seed, so every arm sees identical streams. At the scaled stage (section 9) a fixed-horizon,
endless-stream variant is used instead, and throughput is the primary metric.

### 2.4 Information

Per tick a robot (a) senses cells within Manhattan radius `r_sense`, (b) exchanges messages on two
separate channels:

* **Traffic channel** (INTENT messages for collision avoidance): radius `r_traffic`, loss `loss_traffic`.
  Held fixed (`r_traffic = 8`, `loss_traffic = 0`) in every experiment except E6 so that sweeps of the
  ledger channel do not change traffic quality.
* **Ledger channel** (STATE messages: ledger, edit status, stock, claims, obstruction records, approvals):
  radius `r_comm`, latency, loss. This is the channel swept in E2.

`r_comm = 0` means no ledger messages; `r_comm = inf` means global broadcast (still subject to loss).
No other channel exists between robots. The supervisor link is described in section 5.3.

### 2.5 Objective

Minimise total cost `J = sum over robots (moves + waits) + kappa * carried_steps + F * edits` over a fixed
workload (every robot completes all K tasks). Secondary: sum of completion times (delay); at the scaled
stage, throughput (tasks per 1000 ticks).

### 2.6 Benchmarks and ratios

* `FreeOpen`: every blocked cell is open from t=0 at no charge, same traffic layer. `J_free` is the travel
  that every arm pays anyway.
* `HindsightFill`: knows every task in advance, picks the edit set S* minimising static cost plus buy
  cost, pre-opens S* at t=0, and is charged `buy_lb(S*)`. Same traffic layer. Family S only.
* `CentralOnline`: RoF with a single global ledger, instant knowledge, omniscient hauler choice, no
  approval gate. Isolates the cost of decentralisation.
* `OPT_LB`: closed-form lower bound without congestion (section 7.2). Sanity check only.

Ratios:

* **Avoidable hindsight ratio** `HR_av = (J_alg - J_free) / (J_hind - J_free)`. This is the ski-rental
  quantity: it removes the travel that no arm can avoid, which in v1 compressed every ratio toward 1.
  Runs with `J_hind - J_free < 1` are excluded and their count reported.
* `HR = J_alg / J_hind` is reported alongside but not tested.
* **Price of decentralisation** `PoD = J_RoF / J_CentralOnline`.

`HindsightFill` uses static costs and a lower-bound buy cost, so it is neither a true optimum nor a bound
under congestion. The paper calls these "hindsight ratios", never "competitive ratios".

---

## 3. Architecture

```
            supervisor console (human-on-the-loop)            incident reports (text)
                       ^  approve / veto                              |
                       |                                              v
 L2  LLM layer   [ approval drafter ]                       [ exception intake ]
     (event-driven, edge, off the tick loop)                         |
                       ^ justification request                       | ObstructionRecord
                       |                                             v
 L1  RoF core    per robot: ledger CRDT -> trigger -> claim -> hauler FSM      (every tick, numeric)
                       |
 L0  safety      world arbitration of physical moves (vertex/swap/cycle), sensing
```

* **L0** is physics. Its interventions are logged (section 7.4) so the coordination cannot be centralised
  in secret.
* **L1** is the decision core: decentralised, numeric, deterministic, every tick. Sections 4 and 6.
* **L2** handles what is language-shaped: turning reports into records and turning ledger evidence into a
  sentence a supervisor can approve. Event-driven, never per tick. Section 5.

---

## 4. The decision core: Rent-or-Fill (RoF)

### 4.1 Rent: what a detour costs

When robot i plans task tau (start s, goal g) under its belief about which cells are blocked
(unfilled pits, plus obstructions it believes present, section 5.4):

* `d_block` = shortest path length treating every believed-blocked editable cell as blocked.
  If unreachable, use `U = unreachable_cost` (default 4 * (H + W)).
* `d_open` = shortest path length treating every believed-blocked **robot-clearable** cell as free.
  Canonical tie-break (all robots must agree): lexicographic (steps, cells_used, path-as-sequence).
* `bundle(tau)` = the set of blocked cells on that canonical open path.
* `rent(tau) = max(0, d_block - d_open)`.

Rent is counted when the task is planned, before the detour is walked: it is the extra steps the robot is
about to walk because those cells are closed. This is a one-task look-ahead and is accounted for in P1.

### 4.2 Why bundles

Counterfactual savings per cell fail for complements. Two pits in series in a one-cell corridor give a
single-pit saving of zero each, so a per-pit ledger never accumulates. Bundles credit the whole minimal
set. Substitutes (parallel pits) need more than bundles; see 4.3. Hypothesis H3 tests both.

### 4.3 Ledger as a CRDT of task records

v1 stored rent per bundle. That never invalidates evidence: after pit A is filled, rent recorded under the
parallel pit B still counts toward B even though A now serves those tasks. v2 stores records and
recomputes.

* `RentRecord = (robot, task_idx, origin, dest, tick, rent_counted)`, immutable.
  `Ledger = GSet[RentRecord]` (keyed by `(robot, task_idx)`), merge = union.
* Edit status per blocked cell: lattice UNFILLED < FILLED (merge = max). Edits are permanent, so staleness
  only ever means "I still think it is unfilled".
* Station stock: PN-counter per station.
* Claims, obstruction records and approvals: sections 4.5, 5.2, 5.3.

**Evidence under the current edit set F.** For each record `r`, compute
`d = dream_path(F, r.origin, r.dest)` (cached per `(origin, dest, F)`), residual `T_r = d.bundle`, and
contribution `c_r = min(r.rent_counted, d.rent)`. Then

```
evidence(T) = sum of c_r over records with T_r == T  (and r.tick >= now - W for RoF-W)
```

Effects: a record served by a substitute edit has an empty residual and contributes nothing; a series
record with one pit filled moves to the remaining pit with its full rent; old records drop out under a
window `W` without breaking merge (the window is a pure function of record ticks).

Cost: records grow as N * K. Toy stage: full-state gossip. Scaled stage: delta-state gossip and record
compaction (section 9).

State travels by gossip on the ledger channel every `gossip_period` ticks; information also spreads by
robot motion.

### 4.4 Trigger

For each non-empty residual set T:

```
buy_est(T) = sum over p in T of ( fee + kappa * dist(nearest station with stock, cell adjacent to p) )
fire when    evidence(T) >= theta * buy_est(T)            # theta = 1.0 default
```

Only cells whose status is CONFIRMED and class `robot_clearable` (section 5.4) may appear in a firing T.
Choose the firing T with the largest `evidence - buy_est`; ties by smaller |T|, then lexicographic.
With theta = 1 and one pit this is the deterministic ski-rental rule (anchor for H1).

`buy_est` excludes the hauler's unloaded walk to the station and the delay to its own task. Every edit
logs `B_est` and the realised cost `B_real` = fee + kappa * carried steps + unloaded steps of the haul
(including pickup and apply ticks). The approval wait is logged separately (`approval_wait`): the robot
keeps working its task while it waits, so the wait is part of the latency L in P2, not of `B_real`.
P1 and P2 are stated in terms of `B_real`.

### 4.5 Who hauls: claim, stagger, yield

1. A robot that sees T fire and no live claim on any cell of T waits `stagger = floor(h_i / kappa)` ticks,
   capped at `stagger_cap`, where `h_i = dist(pos_i, station) + kappa * dist(station, site_adj)`. Cheap
   haulers go first. It stands down if a live claim for T arrives meanwhile.
2. It then requests approval if the gate is on (section 5.3), and on approval issues claim
   `ticket = (lamport, robot_id)` with lease `lease_ticks`, renewing every tick.
3. Claims are a map ticket -> expiry (max-merge). `effective_claim(p, now)` = smallest ticket with
   expiry > now. A robot holding a claim that learns of a smaller effective ticket aborts.
4. Hauler FSM: GO_PICKUP -> CARRY -> APPLY -> RESUME. Before pickup and before apply it re-validates: cell
   still blocked (belief plus sensing) and own claim effective. On abort while carrying it returns the kit
   (`wasted_haul_cost`).
5. The world, not the robots, makes a double edit impossible: the second apply on an open cell is rejected.

This is a lease-based distributed lock with Lamport ordering. It is standard machinery and is not claimed
as a contribution; H4 measures it.

Variant `claim=off`: any robot whose trigger fires hauls immediately (H4).

### 4.6 Arms

| Arm | Rule | Status |
|---|---|---|
| `NeverFill` | Never edits. | core |
| `Myopic` | Edits when the rent of a single task already exceeds `buy_est(T)`. | core |
| `Eager` | theta = 0: any positive evidence fires. | core |
| `RoF` | Sections 4.3 to 4.5. Main method. | core |
| `RoF-Pit` | Per-cell counterfactual rent per record (ablation for H3). | core |
| `RoF-Local` | Ledger never gossiped (ledger channel off; traffic channel unchanged). | core |
| `CentralOnline` | Section 2.6. | core |
| `HindsightFill`, `FreeOpen` | Section 2.6. | core |
| `RoF-W` | Evidence window W. | stretch (E5) |
| `RoF-X` | Extrapolates evidence by census / contributors. | stretch (E5) |

The L2 settings (`intake`, `gate`) are orthogonal to the arms and listed in section 5.5.

---

## 5. The LLM layer (L2)

### 5.1 What the LLM does and does not do

The fill decision is a numeric threshold rule evaluated every tick. An LLM inside it would add latency,
cost and nondeterminism and contribute nothing a sum and a comparison do not already do. The LLM is
therefore restricted to two language-shaped jobs:

| LLM does | LLM never does |
|---|---|
| Turn free-text incident reports into structured obstruction records (5.2) | Compute rent, evidence or buy cost |
| Draft a one-paragraph approval request from ledger evidence for a supervisor (5.3) | Fire a trigger, issue a claim, choose a hauler |
| | Plan paths or resolve traffic |
| | Start a haul toward a permanent edit without physical confirmation (5.4) |

The paper states this boundary explicitly as a design result: the agentic layer sits at the
human/language interface, and the control loop stays verifiable.

### 5.2 Exception intake

**Input.** A free-text incident report: an operator message ("pallet down in aisle 7 by bay 12, blocking
the lane"), a WMS/WES event line, or (Stage 2 only) a VLM caption of a robot camera frame.

**Grounding.** The prompt lists the site's location names (for example "aisle 3 bay 5", "north door"),
retrieved from the map. The model must answer with one of those names or reject the report; code maps the
name to cells with `locate(name)`. The model never produces coordinates, so a location can be wrong but
never off-map. (Tool calling is a possible later variant; the closed list keeps Stage 0 provider-neutral
and deterministic to validate.)

**Output (JSON schema, validated).**

```
ObstructionRecord = {
  report_id, location: <name from the list>, cells: locate(location), kind: pallet|spill|debris|rack_damage|other,
  class: robot_clearable|needs_human, est_kits: int, confidence: float in [0,1], rationale: str }
```

Records that fail schema or map checks (cells off-map, on walls, empty) go to the supervisor queue and
never enter the ledger.

**Placement and decentralisation.** The robot or edge node that receives the report calls a hosted model
(`gpt-4o-mini` through the OpenAI API). The LLM is the one cloud dependency in the system, and it is
deliberately optional: it is event-driven, off the decision path, and cannot start a haul (5.4). If the
cloud link is down, intake degrades to `none` (obstructions learned by sensing only); E7 measures the
cost of that degradation. On-device inference is not claimed (section 11). The record enters the ledger
CRDT as an element of
`ObstructionSet = GSet` keyed by `report_id`. If two nodes process the same report, merge keeps the record
from the smaller node id (deterministic). Different reports of one incident produce separate records; cell
status is per cell, so they combine naturally.

**Determinism in experiments.** LLM calls are made once per incident text per model, offline, and their
outputs and measured latencies are cached. The simulator replays the cached record after the measured
latency. This keeps runs deterministic and the LLM cost bounded.

### 5.3 Human-on-the-loop approval gate

**When.** Before issuing a claim on a permanent edit (step 2 of 4.5), if `gate=on`.

**What.** The robot's L2 drafter produces a request from structured fields only: residual set T, evidence,
`buy_est`, number of distinct contributing robots, coverage proxy (contributors / census), records age,
obstruction rationale. The LLM writes the sentence; the numbers are inserted from the ledger, not
generated, so they cannot be hallucinated.

**Supervisor link.** The supervisor is a deliberate central authority for permanent edits only, off the
per-tick path. Approval or veto enters the CRDT as `Approvals: Map[(T, ticket) -> approve|veto]`
(veto dominates). If the supervisor is unreachable or does not answer within `approval_timeout`, a
timeout policy applies: `approve` for robot-clearable records with confidence >= c, else `defer` (re-ask
later). The cost of the gate is the approval wait, which enters the latency term L of P2 (the robot keeps
working its task while it waits).

**Needs-human obstructions.** Robots never edit them. Their evidence is reported to the supervisor every
`report_period` as "fleet cost of this obstruction so far" so humans can prioritise. Not simulated beyond
reporting.

**Supervisor model in simulation.** A stochastic responder: response latency from a log-normal fit
(parameters swept), approves when evidence >= buy and the record is robot-clearable, vetoes injected
wrong-class records with probability `p_catch`. The value of the drafted justification to real humans
cannot be measured in simulation; section 8 says what is measured.

### 5.4 Safety invariant and belief rules

Obstruction status per cell is three max-registers (each merges by max, so the product is a CRDT):
`report_tick` (latest tick a record named the cell), `blocked_tick` (latest tick a robot sensed it
blocked), `free_tick` (latest tick a robot sensed it free). Static pits start with `blocked_tick = 0`.
Edited cells are in the grow-only `filled` set and are passable from then on; the generator never places
an incident on a cell that can be edited earlier, so `filled` never needs retracting.

* REPORTED: `report_tick > max(blocked_tick, free_tick)`.
* CONFIRMED: `blocked_tick >= free_tick` and `blocked_tick >= 0`.
* REFUTED: `free_tick > blocked_tick` and `free_tick >= report_tick`.

Latest observation wins, so a false report is refuted once sensed, and a real incident that appears later
on a refuted cell becomes CONFIRMED again when sensed.

* **Planning:** REPORTED and CONFIRMED cells are treated as blocked (conservative). A hallucinated report
  therefore costs detours until some robot senses the cell; this cost is measured (`false_report_cost`).
* **Triggering:** only CONFIRMED, robot-clearable cells may appear in a firing T (4.4).
* **Invariant (tested):** no haul toward a permanent edit starts unless the target cell is CONFIRMED in
  the hauler's belief at the moment it issues its claim (`unconfirmed_hauls = 0`). An LLM output alone can
  never start a haul. (Applying an edit already requires being adjacent, and therefore sensing, so the
  invariant is checked at the decision, where it is not automatic.)
* **Class:** per cell, lattice `unknown < robot_clearable < needs_human` (join = max). A record sets the
  class; a rejected apply (`needs_human` from the world) sets `needs_human`. Robots never trigger on
  `needs_human` cells. A robot that attempts one because a record was wrong produces a
  `wrong_class_attempt` (a wasted haul), never an edit.

### 5.5 L2 settings

| Setting | Values | Meaning |
|---|---|---|
| `intake` | `none` | No reports; obstructions learned by sensing only (degraded mode) |
| | `oracle` | Ground-truth structured record delivered at report time |
| | `llm:<model>` | Cached LLM output delivered after measured latency |
| `gate` | `off`, `on` | Approval gate (5.3) |
| `model` | `hosted`: `gpt-4o-mini` through the OpenAI API. Optional extra (only if time allows): `small`, a local open-weight model (for example `qwen2.5:7b` via Ollama) for an on-device comparison | Resolved snapshot recorded in `results.md` |

### 5.6 Incident dataset

* Generated from family-D ground truth: each obstruction gets 1 to 3 report texts from templates with
  aisle/bay names, then paraphrased.
* A held-out subset of at least 100 reports is written by people who did not see the templates (lab
  members), to limit the risk of evaluating an LLM on LLM-like text.
* Includes ambiguous, partial, wrong-location and false reports at known rates.
* Split: dev (prompt design, pilot thresholds) and test (frozen before the preregistration tag).

---

## 6. Propositions (support, not headline)

Single edit, rent r_t in [0, r_max] per task, realised purchase cost `B` (= `B_real`), total rent R.
Hindsight cost `OPT = min(B, R)`. Rent is counted at planning time, one task ahead.

* **P1 (full information).** RoF with theta = 1 fires when counted rent reaches `B_est`. Paid rent at that
  point is at most `B_est`, since the last counted task's rent is never walked. Cost <= `B_est + B + r_max`
  if it buys, `R` if not. State the ratio with the gap `B - B_est` explicit; it reduces to the classical
  `2 + r_max/B` when `B_est = B`.
* **P2 (information delay).** If the firing robot's known rent lags true rent by at most D at purchase and
  the haul plus approval takes L ticks with rent accruing at most lambda per tick, the ratio grows by at
  most `(D + lambda*L) / B`. Coverage rho = known/true at trigger and L are logged per edit, so P2 is
  checkable.
* **P3 (isolation).** n robots, equal rent rate, no ledger channel: before the first edit the team pays
  about n*B. Ratio approaches n + 1. Check whether arXiv 2507.15727 already contains this.
* **P4 (complements and substitutes).** A per-cell ledger on a series corridor never fires. The record
  ledger restores P1 on series corridors and, on k parallel substitutes, makes at most one edit per
  disjoint group of served records (no double purchase from stale evidence).

Proof sketches are a plan task. Where a proof does not close, report the empirical law and the gap.

---

## 7. Experimental protocol (Stage 0, toy grid)

### 7.1 Scenarios

Family S (v1 geometry, unchanged):
* `single_pit`: two rooms, one pit in the wall, a door of width 3 at the far end.
* `two_pits_parallel`: two pits in the same wall (substitutes).
* `series_pits`: one-cell corridor with two pits in a row (complements).
* `multi_pit_wall`: 4 pits in a wall.
* `shift` (stretch): hotspot moves mid-run.

Family D (new):
* `incidents_room`: two-room map without pits and three one-cell doors; obstructions appear in the
  doorways at uniformly random ticks (at most two, so one door stays open); mixed classes; reports with
  `delta_o`, `p_report`, `p_false` as parameters.
* `incidents_aisles`: a 13x21 aisle layout (one-cell shelves, one-cell aisles, three cross aisles) where an
  obstruction in one aisle forces a detour through a cross aisle. Locations are "aisle k bay b". This is the
  warehouse-shaped toy map.

### 7.2 OPT_LB

As v1: for edit subset S, `LB(S) = sum over tasks dist_S(origin, dest) + sum over p in S (fee + kappa *
dist(station, site_adj))`. Exhaustive over relevant cells (at most 10), else greedy plus swap (flagged).

### 7.3 Experiments

| ID | Question | Sweep | Status |
|---|---|---|---|
| E1 | H1 validity | `single_pit`; `depot_dist` in {1, 2, 4, 8}; `fee` in {1, 10, 50, 200}; `kappa` in {1, 4, 16}; K in {2, 5, 10, 20, 40}; N in {4, 12}. Report the R/B values actually reached. | core |
| E2 | H2 price of information | `single_pit`, `multi_pit_wall`; `r_comm` (ledger) in {0, 2, 4, 8, 16, inf}; loss in {0, 0.2, 0.5}; latency in {1, 3}; N in {6, 12, 24}; traffic channel fixed | core |
| E3 | H3 complements and substitutes | `series_pits`, `two_pits_parallel`; RoF, RoF-Pit, Myopic, Eager, NeverFill | core |
| E4 | H4 commitment | `single_pit`; loss in {0, 0.1, 0.3, 0.5}; claim on/off; lease in {4, 8, 16} | core |
| E7 | H7 intake | Family D; intake in {none, oracle, llm:hosted}; `p_report` in {0.5, 0.9}; `p_false` in {0, 0.1, 0.2} | core |
| E8 | H8 gate | Family D; gate on/off; supervisor latency median in {5, 30, 120} ticks; `p_catch` in {0.5, 0.9}; injected wrong-class rate 0.1 | core |
| E5 | H5 non-stationarity | `shift`; RoF, RoF-W, RoF-X | stretch |
| E6 | Scaling | Stage 1 maps (section 9) | Stage 1 |

Seeds 0..29 per cell (n = 30). Every run logs to JSONL and one CSV row with the git commit hash and, for
LLM arms, the model id and cache hash.

### 7.4 Metrics per run

v1 metrics: `J`, `delay`, `fills`, `fill_ticks`, `carried_steps`, `wasted_haul_cost`, `messages_sent`,
`message_units`, `stale_detour_cost`, `rent_paid_before_first_fill`, per-edit `known_rent_at_trigger`,
`true_rent_at_trigger`, `claims_issued`, `claims_lost`, `aborts`, `stalled`, `runtime_ms`.

New in v2:
* `HR_av`, `HR`, `PoD`; per-edit `B_est`, `B_real`, `approval_wait`.
* `arbiter_overrides` per 1000 ticks (L0 interventions on agent moves), per arm.
* `throughput` (tasks per 1000 ticks).
* Intake: per-record cell-localisation exact match and cell F1, class accuracy, `est_kits` absolute error,
  schema-reject rate, latency (p50/p95), energy or token cost per report.
* `false_report_cost` (extra steps caused by REPORTED cells that were never blocked).
* `unconfirmed_hauls` (hauls started on a cell not CONFIRMED in the hauler's belief): must be 0.
* `wrong_class_attempts` (hauls ending in a rejected apply on a needs-human cell) and their wasted cost.
* `false_report_hauls` (hauls toward cells named only by false reports): must be 0.

---

## 8. Hypotheses (preregistered; frozen at tag `prereg-v1`)

**Pilot before the tag.** Run `--quick` sweeps of E1 to E4, E7, E8 on dev seeds and the dev incident split.
For each threshold below, confirm that the sweep can reach the region it refers to. If not, change the
*sweep*, not the threshold. Thresholds marked [pilot] are set from the dev split, written here, then
frozen. All comparisons paired by seed: median, bootstrap 95% CI, sign-flip permutation p-value. Negative
results are reported.

* **H1 Validity.** `single_pit`, full comm, theta = 1, loss 0, gate off:
  `HR_av(RoF) <= 2 + r_max/B_real + 0.15`. `HR_av(Eager) > 3` somewhere with R/B <= 0.3;
  `HR_av(NeverFill) > 3` where R/B >= 5.
* **H2 Price of information.** Traffic channel fixed. `HR_av(RoF)` is non-increasing in ledger `r_comm`
  (Spearman, per scenario). At `r_comm = 0` it is within 25% of `n_beneficiaries + 1` in the equal-rent
  case. At `r_comm = inf` `PoD <= 1.05`. Toy maps only test the direction; the size of the effect is a
  Stage 1 result.
* **H3 Complements and substitutes.** On `series_pits`, `RoF-Pit` makes no edit and `RoF` makes two with
  `HR_av <= 2.5`. On `two_pits_parallel`, `RoF` never makes the second edit when the first serves every
  recorded task.
* **H4 Commitment.** At loss >= 0.3, `claim=off` has at least 2x the `wasted_haul_cost` of `claim=on`;
  zero double edits in both.
* **H7 Intake.** (a) `unconfirmed_hauls = 0` and `false_report_hauls = 0` in every run, including `p_false = 0.2`. (b) On the test split,
  `llm:hosted` cell exact-match accuracy >= [pilot]. (c) On family D, `J(llm:hosted)` is within [pilot]% of
  `J(oracle)` and lower than `J(none)` at `p_report = 0.9`. (d) `false_report_cost` grows at most
  linearly in `p_false`.
* **H8 Gate.** (a) The increase in `J` from `gate=on` matches the P2 latency term within 25% (measured
  against predicted from logged `approval_wait`). (b) `wrong_class_attempts` with `gate=on` is at most
  `(1 - p_catch)` times that with `gate=off`. H8 measures cost and catch rate of the gate in simulation
  only; the usefulness of the drafted text to real supervisors is not claimed unless a user study is run.
* **H5 (stretch).** Under a hotspot shift, `RoF-W` has lower regret than `RoF`; `RoF-X` lower than `RoF`
  when stationary. Report as negative if no difference.
* **H6 (Stage 1).** There is r* <= 0.4 * max(H, W) with `PoD(r*) <= 1.15` on warehouse benchmark maps.
  Not tested on the toy maps, where it holds trivially.

---

## 9. Validation ladder (toy to real scale)

| Stage | World | What it adds | Required for submission |
|---|---|---|---|
| 0 | Toy grids of section 7.1, about 15x21, N <= 24 | All core hypotheses; L2 with cached LLM outputs | Yes |
| 1 | Warehouse benchmark maps (MovingAI warehouse set, League of Robot Runners maps), N = 100 to 500, endless task stream, fixed horizon | Throughput, H2 effect size, H6, scaling; delta-state gossip; windowed space-time planning | Yes |
| 2 | Digital-twin-in-the-loop: Gazebo or Isaac Sim warehouse, ROS 2 with DDS as the ledger transport (real loss and latency), live LLM inference on edge hardware, optional VLM intake from rendered frames | The CFP's "digital-twin-in-the-loop" bar; real latency and energy for L2 | Yes, at least one scenario |
| 3 | Small physical demo (3 to 6 robots, mock obstructions) | Hardware evidence | Optional |

Scaling work required before Stage 1: delta-state CRDT gossip; record compaction (merge old records of
the same OD into one summary record with a max-merge counter); cached station BFS maps; a window bound on
space-time A*.

## 10. Timeline to 15 Feb 2027

| Window | Work |
|---|---|
| Oct 2026 | Plan tasks 0 to 9 (sim core, traffic, NeverFill); incident dataset templates; human-written report subset |
| Nov 2026 | Hauler, claims, policies, record ledger; L2 intake and gate with cached outputs; pilot and `prereg-v1` |
| Dec 2026 | E1 to E4, E7, E8 on Stage 0; theory write-up |
| Jan 2027 | Stage 1 runs; Stage 2 minimal twin (one scenario) |
| Feb 2027 | Writing; prior-art verification closed before submission |

If January slips, Stage 2 shrinks to one twin scenario with live LLM latency only. Stretch items (E5,
RoF-X, RoF-W, Stage 3) are dropped first.

## 11. Threats to validity

* The traffic layer is simple (local intents plus priority arbitration). It is held fixed in every arm;
  `stalled` and `arbiter_overrides` are reported per arm; no stalled run is discarded silently.
* `HindsightFill` uses static costs and a lower-bound buy cost; it is not a true optimum under congestion.
* Gossip by proximity ties information spread to robot mobility; results depend on layout.
* Synchronous ticks and a shared clock are assumed; clock skew is out of scope.
* Rent is counted at planning time, a proxy for the realised detour; E1 reports realised against counted.
* Incident reports are partly synthetic; the human-written subset is reported separately. LLM outputs are
  cached once per model, so run-to-run LLM variance is measured only on the intake set, not inside runs.
* The supervisor is simulated; H8 claims nothing about human usefulness.
* The LLM is a hosted cloud model; the paper does not claim on-device or on-edge inference. Its latency
  includes the public internet. Loss of the cloud link is modelled as `intake = none`. A local
  open-weight comparison is an optional extra. Stage 2 re-measures on target hardware. Temperature 0 does not guarantee
  identical outputs, so intake run-to-run agreement is reported.
* One kit per robot; station scarcity is not in the main experiments.

## 12. Verification before any claim is written

1. Read in full: Terraforming MAPF (arXiv 2203.10540); Andreychuk & Yakovlev (arXiv 1807.00771). Confirm
   centralised, offline.
2. Read in full: Multi-Agent Ski-Rental (arXiv 2507.15727). List which of P1 to P4 it already contains.
3. Read: Meyerson, Online Facility Location (FOCS 2001), and the parking-permit paper. Write the precise
   difference.
4. Read: CBBA (Choi, Brunet, How 2009). Write the precise difference from 4.5 and from the trigger.
5. Read: the 2024 token-based decentralised MAPF paper (Autonomous Agents and Multi-Agent Systems).
6. Search: "ski rental" with "multi-robot", "warehouse", "MAPF"; "online" with "movable obstacles";
   "CRDT" with "multi-robot"; "LLM" or "vision-language" with "warehouse exception" or "incident";
   "LLM" with "multi-robot task allocation"; "human-on-the-loop" with "AMR".
7. Record each result in `docs/research/prior-art-verification.md` with date and verdict.

**Pivot rules.**
* If a paper does decentralised online permanent map edits for robots: re-scope to P3, the record ledger
  and the L2 safety architecture.
* If a paper already does LLM exception intake feeding a multi-robot planner with a confirmation
  invariant: keep L2 as engineering, drop it from the contributions list.
* If H2 shows no dependence on `r_comm` on Stage 1 maps: the maps are wrong, not the idea. Add maps where
  beneficiaries are spatially separated before concluding.
* If H1 fails with theta = 1: check `B_est` against `B_real` and realised against counted rent before
  blaming the algorithm.

## 13. Impact on the implementation plan

The plan `docs/superpowers/plans/2026-10-02-rent-or-fill.md` was updated to v2 on 2026-10-02. For the
record, the changes from the v1 plan were:

* Task 1 config: add `r_traffic`, `loss_traffic`, `intake`, `gate`, `approval_timeout`, `window`; add
  policy names `free`, `rof_w`.
* Task 2: add family D scenarios and the incident generator.
* Task 5: replace the bundle `GCounterMap` ledger with `GSet[RentRecord]` and recomputed evidence; add
  `ObstructionSet`, obstruction status lattice, `Approvals`.
* Task 6: two channels.
* Task 7: dynamic obstructions in `World`; `arbiter_overrides` counter.
* Tasks 10 to 12: approval step in the hauler; `B_est`/`B_real`; `HR_av`, `FreeOpen`, new metrics.
* New tasks: L2 intake (prompt, schema, tools, cache), approval drafter, supervisor model, incident
  dataset, E7 and E8 scripts, Stage 1 and Stage 2 harnesses.
* Task 15: pilot step before the tag; E1 sweep extended; E5 and RoF-X marked stretch.

## 14. Out of scope

Learned communication; credit or karma systems as a headline; an LLM in the decision, claim, planning or
traffic loop; CBS/PBS baselines (offline, cannot face an unknown stream); 8-connected grids; continuous
space at Stage 0 and 1; reversible edits.
