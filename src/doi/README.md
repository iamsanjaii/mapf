# Rent-or-Fill: Decentralised Remove-or-Detour Simulator

**Robots that decide, without a boss, when it is cheaper to push an obstacle out of the way than to keep walking round it**

This is the second half of the project. The legacy simulator (see the main [README](../../README.md)) asks *how do robots plan paths?* This one asks a different question: *when something blocks the way, should the fleet pay once to move it, or keep paying a detour on every trip?* And it asks it with no central planner: every robot only knows what it has seen and what its neighbours have told it.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Core Concepts](#2-core-concepts)
3. [Architecture Overview](#3-architecture-overview)
4. [Module-by-Module Explanation](#4-module-by-module-explanation)
5. [The Decision Rules (Arms)](#5-the-decision-rules-arms)
6. [The Robot Agent](#6-the-robot-agent)
7. [Pushing Obstacles](#7-pushing-obstacles)
8. [Metrics](#8-metrics)
9. [Experiments](#9-experiments)
10. [How to Run](#10-how-to-run)
11. [Project Structure](#11-project-structure)

---

## 1. What Problem Does This Solve?

Imagine a warehouse where a pallet has been left in a gap of a rack line, or a shelf unit stands in an aisle. Every robot that needs to get through has two choices:

- **Rent:** take the long way round, again and again, every trip
- **Push:** shove the obstacle aside on its way through, once, and every later trip is short

This is the classic **ski-rental problem**: keep renting skis, or buy a pair? The best rule when you do not know the future is simple: *buy when the rent you have paid so far equals the price.*

The goal is the **lowest total cost for the whole fleet**, not just finding a path for everyone at any price. And the hard part is that **no robot sees the whole fleet's detours**. Each robot only knows its own, plus whatever neighbours have sent it. So this project simulates the decision *without a central planner* and measures what that costs.

| Strategy | Intelligence Level | Description |
|---|---|---|
| Never push | Naïve | Always take the detour |
| Myopic / Eager | Rule-based | Push if it is cheaper for *me* / push at the first sign of a saving |
| **Rent-or-Fill (RoF)** | Decentralised | Push when the fleet's *shared* detour total reaches the price |
| Central | Omniscient | Same rule, but one boss sees every detour instantly |

It also covers **incidents**: obstructions that appear *during* the run and are reported as text ("pallet down in aisle 7"), which robots must verify and classify before they touch them.

---

## 2. Core Concepts

### What the Map Models

Robots never go through a wall. A cell is either floor, a **permanent wall** (never removable), or holds a **removable obstacle**: anything that can sit on the floor, such as a pallet, a crate or a shelf unit. Removing an obstacle empties its cell, so robots can pass. Obstacles can be anywhere on the grid; they are not tied to walls (`--demo scatter` has no wall at all).

The only way to remove one is to **move it somewhere else**. A robot does that by **pushing**: it steps into the obstacle and the obstacle slides one cell onward. A pushed obstacle is still an obstacle where it lands, so it can block someone else. In Stage 2 a robot can also **carry** an obstacle to a rack or dump slot, where it is gone for good, or **fill** a pit with debris (section 7).

| Glyph | Meaning | Push step costs |
|---|---|---|
| `.` | free floor | 1 |
| `#` | permanent wall (strips are allowed) | n/a |
| `L` | pallet | `kappa` × 1.0 |
| `C` | crate (light) | `kappa` × 0.5 |
| `S` | shelf unit (heavy) | `kappa` × 2.0 |

Other glyphs: `W` spill and `K` rack damage (incident kinds, never pushed when a report says they need a human), `R` debris (can be carried, or dropped into a pit), `P` pit (cannot be pushed; a robot fills it), `T` a rack slot (holds one pallet or crate) and `D` a dump slot (holds one obstacle of any carryable kind; a dump region is a block of them).

The barrier maps (a dark column with obstacles in its openings and doors at the bottom) are one convenient shape: removing an obstacle opens a crossing, and the long way round is through the doors. It is a simplification chosen so that one removal creates a clear shortcut. The demo maps are also built to make pushing worthwhile (most trips cross the barrier, one door is far away), and `--crossing` and `--doors` let you check when it stops paying.

### Rent

The extra steps a robot walks because obstacles are in the way. If a task takes 10 steps going round and 2 steps through the obstacle's cell, the rent for that task is 8.

### Price

What it costs this robot to push the obstacle away on its own route, over the ideal route:

```
price = (walk to the approach cell + push steps × kappa × weight + fee + walk on to the goal) − (ideal route length)
```

### The Ledger

Every task a robot plans is recorded as an origin and a destination, whether or not it paid any rent. Robots broadcast these records to nearby robots, so the ledger spreads by gossip. It is built from CRDTs, which means robots can merge ledgers in any order and always agree in the end.

### Evidence and the Trigger

The **evidence** for a push is how much shorter the recorded tasks' routes would have been if the obstacle had been moved before: the fall in total recorded travel when the obstacle goes from its cell to its landing cell. An obstacle that lands on a route someone used lowers the evidence automatically (this is the **collateral** effect). A robot pushes when `evidence ≥ theta × price`. `theta = 1` is the ski-rental rule; `theta = 0` pushes at any saving.

### Own Path Only

When it pushes, a robot never leaves its route to push something for someone else. It only pushes an obstacle that is on the route to its own goal, using steps it was going to walk anyway. (Carrying and filling do cost a detour: the robot walks to the obstacle and then to the slot, and the plan is judged on that whole cost.) The collective part is in the *decision*: it pushes when the fleet's accumulated rent justifies it, even if its own detour would have been cheaper.

### Cost (J)

How every arm is scored:

```
J = moves + waits + (push steps × kappa × weight) + (push runs × fee)
```

With carrying and filling the cost also includes `loaded steps × kappa_c × weight` and `pick_fee + drop_fee` for each carry or fill. Lower is better. A task that can never be completed is charged a large penalty, so impossible is expensive but not forbidden.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│          run_doi.py   (command line + window)        │
│  --sim / --demo / --build / --layout / --verbose     │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  scenarios.py/builder   │
          │  Map, obstacles, robot  │
          │  task lists, incidents  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐        one pass per tick:
          │        runner.py        │        1. sense nearby cells
          │      the tick loop      │───────▶2. receive messages
          └───┬─────────────────┬───┘        3. decide (push or walk)
              │                 │            4. broadcast ledger
   ┌──────────▼───┐      ┌──────▼───────┐    5. world applies moves
   │   world.py   │      │ RobotAgent   │       and pushes
   │ ground truth │      │ (one per     │──── asks a PushPolicy:
   │ + physics    │      │  robot)      │     "push, or go round?"
   └──────────────┘      └──────┬───────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
        ┌─────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
        │  belief   │    │  pushplan   │    │ network.py  │
        │ ledger +  │    │  + pusher   │    │ lossy,      │
        │ cell state│    │  (do a push)│    │ short range │
        └───────────┘    └─────────────┘    └─────────────┘
                       │
          ┌────────────▼────────────┐
          │   metrics / story.py    │
          │  cost, rent meters,     │
          │  captions, verdict      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │       animate.py        │
          │  window + GIF + HTML    │
          └─────────────────────────┘
```

The simulator is **tick-based**: time moves in whole steps, and in each step every robot senses, listens, decides and acts. `world.py` is the only place that knows the truth; robots only see what is within `r_sense` cells of them.

---

## 4. Module-by-Module Explanation

### `kinds.py` — The Obstacle Kinds

A small table: name, glyph, colour and the weight of one push step. Adding a kind is one more row.

---

### `scenarios.py` — The Maps

Builds a map, the obstacles and every robot's task list from a name and a seed. The same seed always gives the same run, so arms can be compared fairly.

```python
cfg = SimConfig(scenario="multi_block_wall", n_robots=8, tasks_per_robot=10, seed=0)
scenario = build_scenario(cfg)
```

There are two families: **S** (obstacles on the map from the start) and **D** (incidents that appear during the run). `run_doi.py --list` describes each one in plain language.

---

### `builder.py`, `wizard.py` — Building Your Own Map

`builder.py` turns sizes, counts and kinds into a validated map (`barrier`, `strips` or an ASCII `map` file) and checks it: obstacles that can never be pushed, nothing worth removing, too many obstacles for the space. Every problem comes back as a plain message. `wizard.py` is the legacy-style prompt loop on top of it: it asks, validates, previews the map and prints the one-line command that repeats the run.

---

### `world.py` — Ground Truth and Physics

Holds what is really true: where robots and obstacles are, and which incidents are active. It resolves moves so two robots can never share a cell or swap places, and it carries out pushes: the obstacle slides one cell if the cell beyond is free. If it has to stop a collision it logs an *override*; if it refuses a push (no room, obstacle gone, needs a human) the robot waits that tick.

---

### `agent.py` — One Robot

One `RobotAgent` per robot. Each tick it senses, receives, decides, acts and broadcasts. See section 6.

---

### `belief.py`, `crdt.py` — What a Robot Knows

The robot's private view: its ledger of tasks, and for every cell the last tick it was seen blocked and the last tick it was seen free (so a moved obstacle is learned from sensing or gossip), the kind of obstacle on it, and whether a robot may clear it. The data structures in `crdt.py` can be merged from any neighbour in any order.

---

### `evidence.py` — What a Push Would Have Saved

Adds up the recorded tasks' route lengths with the obstacle in its old place and in its new place, and returns the difference. The robot's own task in hand is counted separately from where it stands now.

---

### `pushplan.py`, `pusher.py` — Planning and Doing a Push

`pushplan.py` finds, for one robot and one task, the cheapest way to push an obstacle that is on its route: which approach cell, which direction, how many steps. `pusher.py` carries the plan out: walk to the approach cell, then push step by step, and give up (with a short cooldown) if the world refuses.

---

### `carryplan.py`, `carrier.py` — Carrying and Filling

`carryplan.py` plans the other two ways to remove an obstacle: `carry` lifts one on the robot's own route and puts it on a rack or dump slot that accepts its kind, and `fill` lifts debris and drops it into a pit. A plan is judged like a push plan, by what it costs this robot and by what it does to the believed map, and only the cheapest plan per obstacle is kept. `carrier.py` is the small state machine that carries one out: walk to the obstacle, haul it loaded to the slot, drop it. Once loaded, a robot does not abandon the load; if its slot turns out to be full it picks the cheapest remaining target.

---

### `abstract/` — The Abstract Model

A stripped-down version of the problem (`core`, `instance`, `offline`, `online`) with the exact offline optimum, used by experiment E1 to measure the competitive ratio of the rule against the best possible choice.

---

### `policies.py` — The Decision Rules

Each *arm* (never, myopic, eager, rof, central, ...) is a small class that answers one question: *given this push plan, push or go round?* See section 5.

---

### `network.py` — Messages

A lossy, short-range broadcast channel. Messages can be dropped, delayed and only reach robots within `r_comm` cells. Two channels run in parallel: one for traffic intentions (so robots avoid each other) and one for the ledger.

---

### `paths.py`, `spacetime.py` — Walking

`paths.py` computes shortest paths and the "dream path" (the route if the obstacles were gone) that defines rent. `spacetime.py` is the local collision-avoiding planner each robot uses to move.

---

### `incidents.py`, `notices.py`, `llm/`, `forecast/` — The Language Layer

Language models are used in two places and are never involved in planning, pushing or traffic (a test checks that the decision modules do not import `llm`).

* **Exception intake** (`incidents.py`, `llm/intake.py`). For incident scenarios, reports are rendered as text and an intake step turns the text into a structured record. `oracle` is a perfect stand-in; `llm/` can call a real model, offline and cached.
* **The forecast agent** (`notices.py`, `forecast/`). In `shift_notice`, a short text notice tells a robot that work is about to move. A *forecaster* answers one yes/no question per candidate move: "will the fleet's saving reach the price?" (`--forecaster numeric`, `keyword`, `ledger`, `oracle`, `inverted` or `llm:<key>`). For `llm:<key>` an agent loop (`forecast/agent.py`) gives the model three read-only tools (the notices, a ledger summary, the saving of one trip between two zones) and at most four turns. The arm `rof_a` uses the answer: a yes lowers the threshold, a no raises it, and no answer (the model failed, timed out or sent a bad reply) leaves the classical rule. The model cannot move a robot or change a threshold. Every call is stored and replayed, and a run never calls a model unless `--agent-live` is given.

---

### `runner.py` — The Tick Loop

Ties everything together: builds the world, the network and the agents, then runs ticks until every task is done. It stops a run as stalled if nothing is completed or pushed for a long time. Returns a `RunResult` with the cost and the full event record.

```python
result = run_episode(cfg.replace(policy="rof"), scenario=scenario)
print(result.J, result.removals)
```

---

### `story.py`, `animate.py`, `narrate.py` — Making It Readable

`story.py` turns a run into plain language: a rent meter per obstacle, captions, a one-line verdict. `animate.py` draws it (live window, GIF, HTML player). `narrate.py` holds the descriptions behind `--list` and `--guide`.

---

## 5. The Decision Rules (Arms)

Every arm runs on the **same map and same tasks**, so the only difference is the rule.

| Arm | Rule |
|---|---|
| `never` | Never pushes. Pays every detour. The baseline. |
| `myopic` | Pushes only if the push is cheaper than my own detour |
| `eager` | Pushes as soon as the ledger shows any saving (`theta = 0`) |
| **`rof`** | Pushes when the ledger's saving reaches `theta × price` (**the method**) |
| `rof_local` | The same, but the ledger holds only the robot's own tasks |
| `rof_f` | The same, but it forecasts the saving over the tasks still to come |
| `rof_r` | Rent-or-Fill with a random threshold between 0 and 1 times the price (expected ratio e/(e-1)) |
| `rof_p` | Rent-or-Fill that lowers its threshold when its forecast says the push will pay, and raises it when not |
| `rof_a` | Rent-or-Fill guarded: a forecaster (`--forecaster numeric`, `keyword`, `ledger`, `oracle`, `inverted` or `llm:<key>`) says whether the move will pay; the threshold is lowered on a yes, raised on a no, and stays at the price while there is no answer |
| `central` | The same rule with one omniscient ledger and map (what a boss would do) |
| `free` | Benchmark: every obstacle gone at tick 0, for free |
| `hindsight` | Benchmark: knows all tasks, removes the best obstacles at tick 0, charged the lowest possible price |

### Worked example (`--demo toy`)

One robot, one shelf unit in a gap of a barrier, four trips across. Going round costs 10 steps, through the gap 2, so each trip pays 8 rent. Pushing the shelf two cells costs `2 × 4 × 2.0 + 1 = 17`.

| Arm | Cost | What happened |
|---|---|---|
| never | 40 | Went round four times |
| myopic | 40 | 17 is more than the 10-step detour, so it never pushed |
| eager | 23 | Pushed straight away; the best here because all four trips come |
| **rof** | **31** | Waited until the detours (16) matched the price (15), then pushed |
| central | 31 | Same rule with global knowledge |

Notice that `eager` wins this one: traffic keeps coming. With a single trip it would lose, and `rof` is the rule that is never far from the best in either case. That robustness, not a win on every run, is what the ski-rental rule offers.

---

## 6. The Robot Agent

Every robot runs the same loop each tick:

```
1. SENSE      look at cells within r_sense (default 2)
2. RECEIVE    merge ledgers and intentions heard from neighbours
3. DECIDE     if pushing: carry on
              else: plan the next task and record it in the ledger,
                    then ask the policy "push, or go round?"
4. ACT        move, wait, or push
5. BROADCAST  send my ledger and my next few steps to neighbours
```

**Why this matters:** a robot decides with partial, possibly stale information. Lower `r_comm`, higher `loss` or higher `latency` means it takes longer to reach the trigger, and a pushed obstacle may land on a route another robot already used. The cost of that gap is exactly what the project measures.

---

## 7. Pushing Obstacles

### The Plan

For the task in hand a robot looks at each obstacle on its open route, from each of four sides, for up to `push_max` steps in a straight line. A plan is: *walk to the cell behind the obstacle, push `k` steps, walk on to the goal with the obstacle where it ended*. The cheapest plan is the candidate.

### Safety Rules

A robot only pushes an obstacle it has **seen** and knows a robot may **clear**. A report it has not verified, or an obstacle whose class needs a human, is never pushed. The world also refuses any push with no room beyond the obstacle, or one that needs a human.

### Carrying and Filling (Stage 2)

Pushing leaves the obstacle on the floor. Two other ways remove it for good:

* **Carry:** lift a pallet, crate or shelf unit and put it on a free rack slot (`T`, pallets and crates) or dump slot (`D`, any carryable kind). A slot holds one obstacle, so slots run out.
* **Fill:** lift debris and drop it into a pit (`P`). The pit and the debris are both gone, and a filled pit stays filled.

A loaded step costs `kappa_c × weight` (default `kappa_c = 2.0`), plus `pick_fee` and `drop_fee`. Each robot picks the cheapest of push, carry or fill for each obstacle on its route. Scenarios: `warehouse_racks`, `dump_central`, `site_pits`, `mixed`; the `site` layout builds one. In the full S2 run, racks changed the mix of push and carry, not the total cost. On `dump_central` the dump is far away, so at the default costs nothing is carried (`experiments/doi_s2_modes.py`).

### What Can Go Wrong

- **Collateral:** the obstacle lands where it blocks a route. The ledger-based evidence tries to avoid this, but a robot decides from its own belief. The `collateral` metric measures it on the true map.
- **Corners:** an obstacle pushed into a spot it can never be pushed out of would block that cell for ever. Robots refuse such landing cells.
- **Goal cells:** a pushed obstacle can land on a cell where some robot later needs to go. That robot walks up to it, sees what it is, and pushes it again if it may.
- **No room:** in a long one-cell corridor there may be nowhere to push to. The obstacle cannot be removed that way, and the builder warns about it.

---

## 8. Metrics

Every run records these (`--verbose` prints them):

| Metric | Description |
|---|---|
| **J** | Total cost: moves + waits + push steps × kappa × weight + push runs × fee |
| **removals** | How many push runs were made |
| **push_steps** | Total push steps |
| **push_rejected** | Pushes the world refused |
| **collateral_cost** | Detour paid because a pushed obstacle sat on a route (true map) |
| **waits** | Ticks spent waiting (congestion) |
| **stalled** | Whether the run got stuck |
| **HR_av** | `(J_arm − J_free) / (J_hindsight − J_free)`: how many times worse than hindsight, on the *avoidable* cost. Ski-rental theory predicts about 2 |
| **PoD** | `J_arm / J_central`: the price of deciding from local, delayed information |
| **overrides** | Times the physics layer had to stop a collision |
| **carries**, **fills** | Obstacles taken to a slot, and pits filled |
| **carry_cost** | Loaded steps at `kappa_c × weight` each |
| **slot_conflicts** | Drops refused because the slot was full or the pit already filled |

---

## 9. Experiments

| Script | Question |
|---|---|
| `doi_e1_ratio.py` | E1: how close is the rule to the exact optimum (competitive ratio), in the abstract model and on grids? |
| `doi_e2_information.py` | E2: how much does limited sharing (range, loss, delay) cost? |
| `doi_e7_intake.py` | E7: what does turning text reports into records cost and risk? |
| `doi_s2_modes.py` | S2: push, carry or fill? How the choice moves with slot capacity and haul distance |
| `doi_e9_headroom.py` | How much can any forecaster change fleet cost on `shift_notice`? (no model) |
| `doi_agent_cases.py` | Collect forecast cases with their truth label into `data/forecasts/` (no model) |
| `doi_agent_eval.py` | Score forecasters, models included, on those cases: accuracy, calibration, failures, cost |
| `doi_e9_agent.py` | Does the forecast agent lower fleet cost, and what happens when its notices are wrong? |
| `doi_make_notices.py` | Write stand-in notice texts with language models (kept apart from the human set) |
| `doi_build_incidents.py`, `doi_intake_run.py`, `doi_intake_eval.py` | The incident report datasets, and running and scoring the intake step |

Most have a `--quick` pilot mode, and `doi_common.py` is the shared grid runner. The experiments built on the earlier pit model (single-resource validity, complements, claims, shifting demand, warehouse scale, the approval gate) were removed with it and have not been re-expressed for pushing.

The scripts that can reach a model (`doi_agent_eval.py`, `doi_e9_agent.py`, `doi_make_notices.py`, and `run_doi.py` with `--agent-live`) replay stored calls by default. A call that is not stored is an error; `--agent-live` (`--live` in the scripts) lets it reach the model, and the scripts then print an upper bound on calls and tokens and ask before spending. Model keys come from `DOI_LLM_<KEY>_URL` and `DOI_LLM_<KEY>_MODEL`; the experiments use `small` and `large`, and one exported `OPENAI_API_KEY` serves both when their URL is `https://api.openai.com/...`. To score a model use `--sample N`, not `--quick`: the first 20 cases all come from one seed.

### What has been measured

**Does a forecast help? (E9, no model, 100 seeds, `shift_notice`, lambda = 0.5.)** Mean avoidable cost, total cost minus the `free` arm's on the same seed. Lower is better.

| Arm | Work shifts to the other bays | No shift |
|---|---|---|
| `never` | 375 | 387 |
| `rof` | 147 | 105 |
| `rof_a` + `numeric` | 130 | 75 |
| `rof_a` + `ledger` | 116 | 77 |
| `rof_a` + `oracle` (best possible) | 112 | 72 |
| `rof_a` + `inverted` (worst possible) | 178 | 142 |

The guarded rule with a plain numeric forecast beats `rof` by a median of 12 to 17 per run (95% interval above zero). A perfect forecast would add a further 4% to 14% at lambda = 0.5 (27% in the shift case at lambda = 0.25). At lambda = 1 every arm is the same as `rof`.

**Can a language model give the forecast? (accuracy on 100 dev cases spread over 20 seeds.)**

| Forecaster | Accuracy |
|---|---|
| `ledger` (yes if the saving already recorded exceeds 2; reads no notice) | 0.90 (0.88 on all dev, 0.86 on 2,949 held-out cases) |
| `gpt-4o` with tools | 0.59 |
| `gpt-4o-mini` with tools | 0.55 |
| `numeric` extrapolation | 0.48 |
| `keyword` notice reader | 0.55 (0.49 on held-out wording) |

The truth is "yes" in half of these cases, but the models answer "yes" in only about 20% (`gpt-4o-mini`) and 30% (`gpt-4o`) of them. The model arms of E9 have not been run, so there is no end-to-end cost for a model yet.

**Limits.**

* In E9 the cost is identical with a true notice and with none, and `keyword` equals `numeric`. Either notices rarely name the obstacle's region or they do not reach the robots in time. This has not been checked yet.
* There is no human-written notice set. `data/forecasts/llm_notices.jsonl` holds 124 texts written by `gpt-4o-mini` and `gpt-4o`; anything scored on it is a model scored on model-written text.
* At a high fee a pushed pallet can land on another robot's pending goal and the rule never clears it (16 of 8,000 runs in an earlier pilot, none in the 8,000 above). Stalled points are left out of the summaries.

Single runs are anecdotes.

---

## 10. How to Run

### Setup
```bash
cd MAPF
source venv/bin/activate
pip install -r requirements.txt
```

### See it (opens a window)
```bash
python run_doi.py --play          # asks grid size, robots, number of L, C and S; runs; opens the window; repeats
python run_doi.py --demo scatter  # obstacles anywhere on an open floor, one start and one goal per robot
python run_doi.py --sim toy       # simulation only: one arm, two lines of text
python run_doi.py --sim fleet     # 8 robots, 4 obstacles in a barrier
python run_doi.py --demo toy      # never vs rof side by side, short table
```

The window has Pause/Play (or space), Restart, a speed slider and a tick scrubber. Under each map, a **rent meter** shows detour cost climbing towards a black line (the price of pushing). When a robot pushes, the bar turns green.

### Build your own map (asks in the terminal)
```bash
python run_doi.py --build                      # asks layout, map, fleet, costs, arms; previews the map
python run_doi.py --layout barrier --rows 15 --cols 21 --blocks 3 --doors 2 --robots 8 --yes
python run_doi.py --layout strips --rows 20 --cols 20 --strips 3 --pallets 3 --crates 2 --shelves 1
python run_doi.py --layout map --map my_map.txt    # # wall  . free  L pallet  C crate  S shelf unit
```

Anything you pass as a flag is not asked again; `--yes` accepts every default. After the questions it prints the map and notes (for example "this shelf unit has no room to be pushed", or "nothing worth removing on this map"), then lets you run, try a new seed, change answers or quit. At the end it prints the exact one-line command to repeat the run.

### Share it
```bash
python run_doi.py --demo toy --html toy.html   # opens in any browser, with play/scrub
python run_doi.py --demo toy --gif toy.gif     # open in a browser, not Preview
```

### Custom Runs
```bash
# 12 robots, compare three rules on the same tasks, no window
python run_doi.py --scenario multi_block_wall --robots 12 --policy rof,never,central --seed 3

# make pushing expensive: rof should stop pushing
python run_doi.py --demo toy --fee 100 --policy never,rof --no-show

# full text output: event timeline, benchmarks, statistics
python run_doi.py --demo toy --verbose --no-show
```

### Stage 2: racks, a dump region, pits
```bash
python run_doi.py --scenario warehouse_racks --robots 8 --policy rof,never
python run_doi.py --scenario site_pits --robots 8 --policy rof,never
python run_doi.py --layout site --rows 15 --cols 21 --pits 3 --debris 4 --racks 4 --yes
```

### A forecaster behind `rof_a`
```bash
python run_doi.py --scenario shift_notice --robots 8 --policy rof,rof_a --forecaster ledger   # no model
python run_doi.py --scenario shift_notice --robots 8 --policy rof_a --forecaster llm:small --agent-live
```

### Help
```bash
python run_doi.py --guide     # every flag in plain language
python run_doi.py --list      # every scenario and arm
```

### Run an Experiment
```bash
python experiments/doi_e2_information.py --quick --jobs 4
python experiments/doi_agent_cases.py                       # forecast cases for dev and test (no model)
python experiments/doi_agent_eval.py --cases data/forecasts/dev.jsonl --forecasters numeric,keyword,ledger --sample 100
python experiments/doi_e9_agent.py --model-keys "" --jobs 4  # E9 with no model arms
```

### Run Tests
```bash
python -m pytest tests/doi -q      # 486 tests
python -m pytest tests -q          # with the legacy tests: 538
```

---

## 11. Project Structure

```
MAPF/
│
├── run_doi.py                       # Command-line runner + window
│
├── src/doi/
│   ├── config.py                    # SimConfig: every parameter and default
│   ├── kinds.py                     # Obstacle kinds: glyph, colour, push weight
│   ├── scenarios.py                 # Maps, obstacles, task lists, incidents
│   ├── builder.py                   # Validated custom maps (barrier / strips / ASCII file)
│   ├── wizard.py                    # Interactive prompts, map preview, repeat command
│   ├── world.py                     # Ground truth, physics, pushing, collisions
│   ├── runner.py                    # The tick loop → RunResult
│   │
│   ├── agent.py                     # RobotAgent (sense/receive/decide/act)
│   ├── pushplan.py                  # The cheapest push plan on a robot's own route
│   ├── pusher.py                    # Carries a push plan out
│   ├── carryplan.py                 # Carry and fill plans (Stage 2)
│   ├── carrier.py                   # Carries a carry or fill plan out
│   ├── policies.py                  # The arms: never / myopic / eager / rof / central ...
│   ├── belief.py                    # A robot's private view
│   ├── crdt.py                      # Mergeable data structures for the ledger
│   ├── evidence.py                  # What a push would have saved the fleet
│   ├── network.py                   # Lossy short-range messages
│   ├── paths.py                     # Shortest paths + "dream path" rent
│   ├── spacetime.py                 # Collision-aware local planner
│   │
│   ├── incidents.py                 # Incident report text
│   ├── llm/                         # Report intake and the chat client; stored, replayable model calls
│   ├── forecast/                    # Forecast cases, tools, the agent loop, the forecasters behind rof_a
│   │                                #   (numeric, keyword, ledger, oracle, inverted, llm)
│   ├── notices.py                   # Text notices about future traffic (wording banks, delivery)
│   │
│   ├── abstract/                    # The abstract model and its exact optimum (E1)
│   ├── oracle.py                    # Hindsight benchmark
│   ├── metrics.py                   # RunResult + HR_av / PoD / collateral
│   ├── stats.py                     # Bootstrap CIs, paired tests
│   ├── maps.py                      # MovingAI benchmark map loader
│   ├── rng.py                       # Deterministic randomness
│   │
│   ├── story.py                     # Rent meters, captions, verdict
│   ├── animate.py                   # Window, GIF and HTML player
│   └── narrate.py                   # Text behind --list and --guide
│
├── experiments/
│   ├── doi_common.py                # Shared grid runner
│   ├── doi_e1 / doi_e2 / doi_e7 *.py  # Ratio, information and intake experiments
│   ├── doi_s2_modes.py              # Push versus carry versus fill
│   ├── doi_e9_*.py, doi_agent_*.py  # Forecast guard: headroom pilot, end-to-end, cases, scoring
│   └── doi_make_notices.py          # Stand-in notice texts written by models
│
├── data/
│   ├── incidents/                   # Incident report datasets
│   └── forecasts/                   # Stand-in notices and the protocol for the human set
│
├── docs/research/
│   ├── demo-guide.md                # How to present the demo
│   ├── theory.md                    # The propositions (written for the earlier pit model)
│   └── results.md                   # Status of the experiments
│
└── tests/doi/                       # 486 tests, one file per area
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Arrays and statistics |
| `matplotlib` | The window, GIF and HTML player |
| `pandas` | The experiment scripts (summaries and CSV output) |
| `pytest` | Unit testing |

The simulator itself needs only `numpy` and `matplotlib`; everything else uses the Python standard library, including the HTTP calls to a model. It reuses the legacy `Grid` and `CellType` from `src/environment/` and does not modify any legacy code.
