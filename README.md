# AMR-MAPF Agent Simulator

**Agentic Multi-Robot Path Finding and Environment Modification Simulator**

A research-grade Python simulator for studying how multiple robots can navigate a shared 2D grid world — avoiding collisions, resolving conflicts, and even modifying the environment when obstacles block their paths.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Core Concepts](#2-core-concepts)
3. [Architecture Overview](#3-architecture-overview)
4. [Module-by-Module Explanation](#4-module-by-module-explanation)
5. [The Three Algorithms](#5-the-three-algorithms)
6. [The Agent System](#6-the-agent-system)
7. [MAPF-RO: Environment Modification](#7-mapf-ro-environment-modification)
8. [Metrics](#8-metrics)
9. [Experiments](#9-experiments)
10. [How to Run](#10-how-to-run)
11. [Project Structure](#11-project-structure)

---

## 1. What Problem Does This Solve?

Imagine a warehouse with dozens of robots fetching items from shelves. Each robot needs to get from point A to point B. If they plan their paths independently, they will:

- Crash into each other (vertex conflict)
- Block each other in narrow corridors (edge conflict)
- Wait forever at the same goal cell (goal conflict)

This project simulates that exact problem and compares **three progressively smarter planning strategies**:

| Strategy | Intelligence Level | Description |
|---|---|---|
| Independent A* | Naïve | Each robot plans alone, ignoring others |
| Prioritised MAPF | Rule-based | Robots plan in a fixed priority order |
| Agentic MAPF | Agent-based | Specialised software agents cooperate to resolve conflicts |

It also includes **MAPF-RO** (Multi-Agent Path Finding with Removable Obstacles), where some obstacles (pits and sandbags) can be modified at a cost to create shorter routes.

---

## 2. Core Concepts

### The Grid World

The environment is a 2D grid where each cell is one of four types:

```
FREE      (white)  — a robot can walk here
OBSTACLE  (dark)   — permanently impassable wall
PIT       (purple) — a hole that blocks movement unless filled with a sandbag
SANDBAG   (orange) — a movable object that can be used to fill a pit
```

### Robots

Each robot has:
- A **start position** (where it begins)
- A **goal position** (where it needs to reach)
- A **priority** (used in prioritised planning)
- A **status**: IDLE → PLANNING → MOVING → DONE / STUCK

### Paths

A path is a list of `(row, col)` positions from start to goal. The planner finds the shortest path through the grid.

### Conflicts

When multiple robots share the same grid, three kinds of conflicts can occur:

| Conflict Type | Description | Example |
|---|---|---|
| **Vertex** | Two robots at the same cell at the same time | R0 and R1 both at (3,4) at t=2 |
| **Edge** | Two robots swap positions (head-on collision) | R0 goes A→B while R1 goes B→A |
| **Goal** | Two robots assigned the same goal cell | R0 and R1 both targeting (5,5) |
| **Blocking** | A robot parked at a goal blocks another robot's path | R0 waiting at (5,5) while R1 needs to pass through |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   main.py / CLI                     │
│         (User chooses algorithm & parameters)       │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   EnvironmentGenerator  │
          │  Builds the grid world  │
          │  Places robots randomly │
          └────────────┬────────────┘
                       │
         ┌─────────────▼──────────────┐
         │         Algorithm          │
         │  ┌─────────────────────┐   │
         │  │   Independent A*    │   │
         │  │   Prioritised MAPF  │   │
         │  │   Agentic MAPF ◄────┼───┼── CoordinatorAgent
         │  └─────────────────────┘   │        │
         └─────────────┬──────────────┘   PathAgent
                       │                  ConflictAgent
          ┌────────────▼────────────┐     CostAgent
          │    ConflictDetector     │
          │  Finds vertex/edge/goal │
          │  conflicts in all paths │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      MetricsCollector   │
          │  Records M1–M8 metrics  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │       Visualiser        │
          │  Static PNG + Live GIF  │
          └─────────────────────────┘
```

---

## 4. Module-by-Module Explanation

### `src/environment/grid.py` — The Grid

The foundation of the simulator. Stores the 2D world as a NumPy array.

**Key responsibilities:**
- Stores cell types (FREE, OBSTACLE, PIT, SANDBAG) in a NumPy matrix
- `neighbours(r, c)` — returns passable adjacent cells (up/down/left/right)
- `in_bounds(r, c)` — checks if a coordinate exists
- `is_passable(r, c)` — checks if a robot can step on a cell
- `free_positions()` / `obstacle_positions()` — lists all cells of a type

```python
grid = Grid(20, 20)
grid.set(5, 3, CellType.OBSTACLE)
neighbours = grid.neighbours(5, 2, passable_only=True)  # [(4,2),(6,2),(5,1)]
```

---

### `src/environment/generator.py` — Random World Builder

Creates reproducible random environments. Given the same `seed`, it always produces the same grid, robot starts, and robot goals — critical for fair experiment comparison.

**What it does:**
1. Fills the grid with obstacles based on `obstacle_density` (e.g. 0.20 = 20% of cells)
2. Optionally places pits and sandbags
3. Randomly places robot start and goal positions on FREE cells
4. Ensures starts ≠ goals and they don't overlap each other

```python
gen = EnvironmentGenerator(width=20, height=20, obstacle_density=0.20, seed=42)
grid, starts, goals, registry = gen.generate(num_robots=5)
```

---

### `src/planning/heuristics.py` — Distance Estimates

A* needs a **heuristic** — an estimate of how far a cell is from the goal. Four heuristics are implemented:

| Heuristic | Formula | Best for |
|---|---|---|
| **Manhattan** | `│Δr│ + │Δc│` | 4-directional movement (this project) |
| **Euclidean** | `√(Δr² + Δc²)` | Continuous space |
| **Chebyshev** | `max(│Δr│, │Δc│)` | 8-directional movement |
| **Octile** | Diagonal-aware | 8-directional with different diagonal cost |

Manhattan is theoretically optimal for 4-directional grids and the experiments confirm it expands the fewest nodes.

---

### `src/planning/astar.py` — The Core Pathfinder

A standard **A* search algorithm** that finds the shortest path from a start cell to a goal cell.

**How A* works:**
1. Start at the start cell with cost 0
2. At each step, pick the cell with the lowest `f = g + h` score
   - `g` = actual cost to reach this cell
   - `h` = heuristic estimate to goal
3. Expand neighbours, update costs
4. Stop when the goal is reached

**Extra features in this implementation:**
- `forbidden` set — cells the planner must avoid (used by prioritised MAPF)
- Returns a `PlanResult` with: path, cost, nodes expanded, runtime

```python
planner = AStarPlanner()
result = planner.plan(grid, start=(0,0), goal=(19,19), heuristic=manhattan)
print(result.path)   # [(0,0), (0,1), ..., (19,19)]
print(result.cost)   # 38.0
```

---

### `src/mapf/reservation.py` — Space-Time Reservation Table

When multiple robots plan sequentially, each robot must know which cells are **already reserved** by previously-planned robots.

The reservation table stores:
```
(position, timestep) → robot_id
```

So if Robot 0 is at cell (3,4) at time t=2, no other robot can go there at t=2.

**Goal locking:** Once a robot reaches its goal, it stays there permanently. The table locks the goal cell for all future timesteps.

---

### `src/mapf/conflict.py` — Conflict Detector

Scans all robot paths and finds every conflict.

```python
conflicts = ConflictDetector.detect_all(paths)
# Returns a list of Conflict objects, each with:
# - conflict_type: VERTEX / EDGE / GOAL / BLOCKING
# - robots: (robot_id_a, robot_id_b)
# - location: (row, col)
# - timestep: when it happens
```

Also provides:
- `count_by_type(conflicts)` — breakdown by conflict type
- `severity_score(conflict)` — GOAL > VERTEX > EDGE (for prioritisation)

---

### `src/mapf/prioritized.py` — Prioritised Planner (Baseline 2)

Robots plan one at a time in **priority order**. Each robot uses **space-time A*** — an extension of A* that expands nodes as `(position, timestep)` pairs.

**Why space-time A*?**
Normal A* finds the spatially shortest path but ignores time. Space-time A* can route around cells that are occupied by *other robots at that specific timestep*, allowing paths to cross the same cell at different times.

**Planning loop:**
```
For each robot (in priority order):
    1. Run space-time A* with all previously-reserved cells as forbidden
    2. If a path is found → reserve it in the table
    3. Next robot plans around the reserved cells
```

Higher-priority robots get shorter paths. Lower-priority robots may need to detour or wait.

---

### `src/mapf/coordinator.py` — Rule-Based Coordinator (Baseline 2b)

A post-planning conflict resolver. After all paths are planned, it detects remaining conflicts and applies simple rules:

- **WAIT**: Make a lower-priority robot wait one timestep (inserts a duplicate position into its path)
- **REROUTE**: Replan the lower-priority robot's path entirely

---

## 5. The Three Algorithms

### Algorithm 1: Independent A* (Naïve Baseline)

Each robot plans its path as if it's the only robot in the world. Paths are computed independently using standard A*.

**Result:** Fast to compute, but produces many conflicts. On a 20×20 grid with 10 robots, you'll typically see 5–15 vertex conflicts.

```python
python main.py --algo independent --robots 10
```

---

### Algorithm 2: Prioritised MAPF

Robots plan in order of priority (typically, shortest Manhattan distance to goal = higher priority). Each robot uses space-time A* to avoid cells already reserved by higher-priority robots.

**Result:** Significantly fewer conflicts than independent A*. Occasionally a lower-priority robot cannot find a path if the grid is heavily blocked.

```python
python main.py --algo prioritized --robots 10
```

---

### Algorithm 3: Agentic MAPF (Proposed)

Four specialised **software agents** cooperate to plan, detect, and resolve conflicts:

```python
python main.py --algo agentic --robots 10
```

See the next section for a detailed breakdown.

---

## 6. The Agent System

The agentic approach decomposes the planning problem into specialised roles. This mirrors real-world multi-agent systems where different components handle different concerns.

### Agent 1: `PathAgent`

**Role:** Plans paths for all robots using prioritised space-time A*.

**Perceives:** The current grid, robot states, and the reservation table.  
**Decides:** Which heuristic to use, what priority order to assign.  
**Acts:** Updates robot paths and the reservation table.

---

### Agent 2: `ConflictAgent`

**Role:** Scans all current paths and identifies every conflict.

**Perceives:** All robot paths and the reservation table.  
**Decides:** Which conflicts are highest severity (GOAL > VERTEX > EDGE).  
**Acts:** Produces a ranked list of conflicts for the coordinator.

---

### Agent 3: `CostAgent`

**Role:** For each conflict, evaluates the cheapest resolution strategy.

Three options are evaluated:

| Option | Description | Cost |
|---|---|---|
| **WAIT** | Insert a wait step (robot pauses one timestep) | `wait_cost` (1 unit) |
| **DETOUR** | Replan with a forbidden zone around the conflict | `detour_cost` (extra path length) |
| **REMOVE** | Fill a pit with a sandbag to open a new route | `removal_cost` (configurable, default 10) |

The agent computes the expected total cost of each option and recommends the cheapest one.

---

### Agent 4: `CoordinatorAgent`

**Role:** The orchestrator. Runs the planning-detection-resolution loop.

```
Loop (up to max_iterations):
    1. PathAgent     → plan paths for all robots
    2. ConflictAgent → detect all conflicts
    3. If no conflicts → DONE ✓
    4. CostAgent     → evaluate resolution options for top conflict
    5. Apply resolution (WAIT / DETOUR / REMOVE)
    6. Repeat
```

**Why this is better than rule-based:** The cost-aware approach avoids unnecessary detours (if waiting is cheaper) and avoids unnecessary waits (if a short detour saves multiple future conflicts).

---

## 7. MAPF-RO: Environment Modification

**MAPF with Removable Obstacles** extends the simulator to handle dynamic environments where some obstacles can be removed or modified.

### Pits (`src/mapf_ro/pit.py`)

- Pits are holes in the ground — robots cannot cross them by default
- A pit can be **filled** by moving a sandbag into it
- Once filled, the cell becomes FREE and all robots can pass through it
- A filled pit counts as a permanent environmental change for this run

### Sandbags (`src/mapf_ro/sandbag.py`)

- Sandbags are movable objects sitting on the grid
- Sandbag transport is accounted as a cost (Manhattan distance x per-step cost); robots do not physically carry them in the legacy code
- Moving a sandbag costs extra energy (configurable, default 4 units per step)

### Traffic Index (`src/mapf_ro/removal.py`)

Before deciding to fill a pit, the system calculates a **Traffic Index**:
- How many robots would benefit from this pit being filled?
- If 5 robots all need to cross the same pit, the cost is shared across all 5

### Decision Logic

Current behaviour (as implemented in `src/mapf_ro/removal.py`):

```
removal_cost = sandbag_travel_cost + fill_cost
net_benefit  = detour_cost - removal_cost
fill the pit when net_benefit > 0
```

The traffic index is recorded but does not affect the decision. Sandbag
transport is a cost estimate (Manhattan distance x per-step cost); no robot
physically carries a sandbag in the legacy MAPF-RO code. The decentralised
simulator in `src/doi/` models carrying physically.

### Replanning Loop (`src/mapf_ro/replanning.py`)

After a pit is filled, all affected robots must **replan** their paths because the grid has changed. The replanning loop:
1. Fills the pit
2. Updates the grid
3. Triggers a full replan for all robots whose paths cross the modified area
4. Repeats until no more modifications are needed

---

## 8. Metrics

The simulator tracks **8 metrics (M1–M8)** for every run:

| ID | Metric | Description |
|---|---|---|
| M1 | **Success Rate** | % of robots that reached their goal |
| M2 | **Total Path Cost** | Sum of all robots' path lengths |
| M3 | **Makespan** | Time until the last robot reaches its goal |
| M4 | **Conflict Count** | Number of remaining conflicts in final paths |
| M5 | **Replan Count** | How many times paths were recalculated |
| M6 | **Runtime (ms)** | Wall-clock time for planning |
| M7 | **Modification Cost** | Energy spent filling pits and moving sandbags |
| M8 | **Total Energy** | M2 + M7 (complete system energy budget) |

These are stored in a `RunMetrics` dataclass and serialised to CSV for analysis.

---

## 9. Experiments

Five experiment scripts test different research questions:

### Experiment A — Heuristic Comparison
```bash
python experiments/q1_heuristics.py --seed 42 --runs 10
```
**Question:** Which A* heuristic is fastest and expands the fewest nodes?  
**Variables:** Manhattan vs Euclidean vs Chebyshev, across grid sizes 20/30/40 and densities 10/20/30%  
**Finding:** Manhattan is fastest on 4-directional grids (fewer nodes expanded, lower runtime)

---

### Experiments B/C/D/F — Multi-Robot Scaling
```bash
python experiments/q2_mapf.py --seed 42 --runs 5
```
**Question:** How does each algorithm scale with more robots and larger grids?  
**Variables:** 2/5/10/15 robots, grid sizes 20/30/40, obstacle densities 10/20/30%  
**Findings:**
- Independent A* conflict count grows with robots²
- Prioritised MAPF significantly reduces conflicts
- Agentic MAPF further reduces conflicts and adapts better to high density

---

### Experiment E — MAPF-RO vs Detour
```bash
python experiments/mapf_ro.py --seed 42 --runs 5
```
**Question:** When is filling a pit cheaper than routing around it?  
**Variables:** Pit densities 5/10%, robot counts 5/10  
**Finding:** With high traffic near pits, removal cost is justified and reduces total energy

---

## 10. How to Run

### Setup
```bash
git clone <your-repo-url>
cd MAPF
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Live Simulation (interactive window)
```bash
python main.py --algo agentic --live
```

### Custom Parameters
```bash
# 30 rows × 40 columns, 10 robots, 25% obstacles
python main.py --algo agentic --rows 30 --cols 40 --robots 10 --density 0.25 --live

# Save as GIF instead of live window
python main.py --algo agentic --rows 20 --cols 20 --robots 8 --animate
```

### Run All Experiments
```bash
python experiments/q1_heuristics.py --seed 42 --runs 10
python experiments/q2_mapf.py       --seed 42 --runs 5
python experiments/mapf_ro.py       --seed 42 --runs 5
```

### Run Tests
```bash
pytest tests/ -v
pytest tests/ -v --cov=src     # with coverage report
```

---

## 11. Project Structure

```
MAPF/
│
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── config/
│   └── default.yaml                 # All parameters (no hardcoding)
│
├── src/
│   ├── environment/
│   │   ├── grid.py                  # 2D grid + CellType enum
│   │   ├── generator.py             # Random reproducible world builder
│   │   └── obstacles.py             # Obstacle/Pit/Sandbag dataclasses
│   │
│   ├── planning/
│   │   ├── astar.py                 # A* pathfinder → PlanResult
│   │   ├── heuristics.py            # Manhattan/Euclidean/Chebyshev/Octile
│   │   └── planner.py               # Planner facade
│   │
│   ├── robots/
│   │   ├── robot.py                 # Robot dataclass + RobotStatus enum
│   │   └── manager.py               # RobotManager (spawn, step, priority)
│   │
│   ├── mapf/
│   │   ├── conflict.py              # Vertex/Edge/Goal/Blocking detection
│   │   ├── reservation.py           # Space-time reservation table
│   │   ├── prioritized.py           # Prioritised MAPF with space-time A*
│   │   └── coordinator.py           # Rule-based conflict coordinator
│   │
│   ├── agents/
│   │   ├── base_agent.py            # Abstract agent (perceive/decide/act)
│   │   ├── path_agent.py            # Plans paths for all robots
│   │   ├── conflict_agent.py        # Detects and ranks conflicts
│   │   ├── cost_agent.py            # Evaluates WAIT vs DETOUR vs REMOVE
│   │   └── coordinator_agent.py     # Orchestrates all agents in a loop
│   │
│   ├── mapf_ro/
│   │   ├── pit.py                   # Pit semantics and filling logic
│   │   ├── sandbag.py               # Movable sandbag manager
│   │   ├── removal.py               # Traffic index + removal cost model
│   │   └── replanning.py            # MAPF-RO iterative replanning loop
│   │
│   ├── metrics/
│   │   ├── metrics.py               # RunMetrics dataclass (M1–M8)
│   │   └── experiments.py           # ExperimentManager → CSV output
│   │
│   └── visualization/
│       ├── renderer.py              # Static Matplotlib grid renderer
│       └── animation.py             # FuncAnimation live/GIF animator
│
├── experiments/
│   ├── q1_heuristics.py             # Experiment A: heuristic comparison
│   ├── q2_mapf.py                   # Experiments B/C/D/F: multi-robot
│   ├── mapf_ro.py                   # Experiment E: removal vs detour
│   └── results/                     # All CSV + PNG outputs saved here
│
└── tests/
    ├── test_astar.py                # 22 A* unit tests
    ├── test_conflicts.py            # 9 conflict detection tests
    └── test_mapf.py                 # 21 MAPF + RobotManager tests
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Grid storage and array operations |
| `matplotlib` | Visualisation (static images + animations) |
| `pandas` | Experiment result storage and analysis |
| `pyyaml` | Reading the config file |
| `pytest` | Unit testing |
| `pytest-cov` | Coverage reporting |

---

## Configuration (`config/default.yaml`)

All parameters are centralised — no values are hardcoded in the source.

```yaml
grid:
  width: 20
  height: 20
  obstacle_density: 0.20

robots:
  count: 5

planning:
  algorithm: agentic        # independent | prioritized | agentic
  heuristic: manhattan      # manhattan | euclidean | chebyshev | octile
  max_timesteps: 200

cost:
  movement: 1
  waiting: 1
  sandbag_move: 4
  obstacle_removal: 10

experiment:
  seed: 42
  runs_per_config: 10
```

---

## Decentralised DOI-MAPF simulator (`src/doi/`)

**Plain-language explanation of this part: [src/doi/README.md](src/doi/README.md).**

A tick-based, decentralised multi-robot simulator for the Rent-or-Push decision rule: robots decide whether,
when and by whom a removable obstacle (a pallet, a crate, a shelf unit) is pushed out of the way rather than
walked round, from partial and delayed information. It sits beside the legacy code and does not modify it.

Three layers:

* **L0** `world.py`: ground truth, move arbitration (vertex, swap and cycle conflicts), pushing and an override log.
* **L1** `agent.py`, `pusher.py`, `policies.py`: one agent per robot with its own CRDT belief
  (`crdt.py`, `belief.py`, `evidence.py`), talking only through a lossy, range-limited two-channel
  `network.py`. The push decision is a numeric threshold rule.
* **L2** `llm/`: exception intake, run offline; the simulator replays cached outputs. The LLM is never on the
  decision, planning or traffic path (a test checks the imports).

See it run (opens a window with play/pause, restart, speed and scrub controls and a rent meter that shows
detour cost climbing towards the price of pushing), or build your own map:

```bash
python run_doi.py --play                      # type the grid size, robots and number of L, C, S
python run_doi.py --demo scatter              # obstacles anywhere, one start and one goal per robot
python run_doi.py --sim toy                  # simulation only, a few lines of text
python run_doi.py --sim fleet                # 8 robots, 4 obstacles in a barrier
python run_doi.py --demo toy                 # never vs rof side by side, short table
python run_doi.py --build                    # set the map and fleet interactively, preview, then run
python run_doi.py --demo toy --html toy.html # self-contained player for any browser
python run_doi.py --guide                    # every flag in plain language (--verbose: full text output)
```

Run the tests:

```bash
python -m pytest tests/doi -q
```

Run one experiment with a small pilot configuration:

```bash
python experiments/doi_e2_information.py --quick --jobs 4
```

Results land in `experiments/results/doi/` (git-ignored). The other script is `doi_e7_intake.py`.
The earlier experiments on the pit model were removed with it; see `docs/research/results.md`.

LLM intake is configured through environment variables only; keys are never written to files:

```bash
export OPENAI_API_KEY=...                     # kept in your shell profile
export DOI_LLM_HOSTED_URL=https://api.openai.com/v1
export DOI_LLM_HOSTED_MODEL=gpt-4o-mini
export DOI_LLM_HOSTED_JSON_MODE=1
python experiments/doi_intake_run.py --model-key hosted --split dev   # prints a cost estimate and asks first
```

The design, hypotheses and status are in `docs/research/research-design.md`, `docs/research/theory.md`,
`docs/research/results.md` and `docs/research/prior-art-verification.md`.
