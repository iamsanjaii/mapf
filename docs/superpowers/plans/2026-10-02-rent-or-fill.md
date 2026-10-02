# Rent-or-Fill (DOI-MAPF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan version:** v2 (2026-10-02), matching spec v2 (option A: LLM exception intake and human-on-the-loop approval). Review that produced v2: `docs/research/review-2026-10-02.md`.

**Goal:** Build a tick-based, decentralised multi-robot simulator, the Rent-or-Fill decision core, the LLM layer (exception intake, approval drafting) with a simulated supervisor, and the baselines, metrics and experiments needed to test hypotheses H1 to H4, H7 and H8 on the toy grid (Stage 0), then H2 effect size and H6 on warehouse benchmark maps (Stage 1). H5 is a stretch goal. Stage 2 (digital twin) gets its own plan after Stage 1 results.

**Architecture:** A new package `src/doi/` sits beside the existing code and does not modify it. Three layers (spec section 3): L0 a ground-truth `World` that arbitrates physical moves and logs every override; L1 one `RobotAgent` per robot holding only its own CRDT belief (task-record ledger, edit and obstruction status, station stock, claims, approvals) and talking only through a two-channel, lossy, range-limited `Network` (traffic channel and ledger channel); L2 an LLM layer in `src/doi/llm/` that runs offline over incident reports, whose cached outputs the simulator replays after their measured latency. Fill decisions are made by pluggable `FillPolicy` objects so every arm shares the same traffic layer. The LLM never sits on the decision, claim, planning or traffic path.

**Tech Stack:** Python 3.14 (existing `venv/`), numpy, pandas, matplotlib, pyyaml, pytest. No new dependencies (scipy is NOT installed and must not be added). LLM access uses only the standard library (`urllib.request`) against OpenAI-compatible `/v1/chat/completions` endpoints: the **OpenAI API** with `gpt-4o-mini` (`hosted`), the only model required. The client is provider-neutral, so an optional local model (`small`, for example `qwen2.5:7b` via Ollama) can be added later without code changes. Endpoint URL, model id and API key come from environment variables and are never committed, printed or logged.

**Spec:** `docs/research/research-design.md` v2 (read it fully before Task 1; section numbers below refer to it). Prior art: `docs/prior_art.md`.

## Global Constraints

Every task's requirements include this section.

- All new code lives in `src/doi/`, tests in `tests/doi/test_doi_*.py`, experiment scripts in `experiments/doi_*.py`, incident data in `data/incidents/`. Do not modify any existing file except in Task 0 and Task 20.
- `pytest tests -q` must report the existing **52 passed** plus the new tests at every commit. Never edit or delete an existing test.
- Positions are `(row, col)` tuples, `Pos = Tuple[int, int]`. Grid size is `H` rows by `W` cols. Movement is 4-connected with neighbour order `[(-1,0),(1,0),(0,1),(0,-1)]` everywhere.
- **Determinism:** never call `hash()`, the global `random` module, `np.random.seed`, or time-based seeds. All randomness comes from `src/doi/rng.py`. Wherever iteration order can change behaviour, iterate over `sorted(...)`.
- Cost model constants: `kappa = 4.0` (cost per carried step), `fee = 1.0` (cost per fill), moving or waiting costs 1 per tick per active robot.
- Finished robots (all K tasks done) are removed from the world and cost nothing afterwards.
- Message latency must be at least 1 tick. A message sent at tick t is never visible before tick t+1.
- Agents never read `World` ground truth except through `World.observe(...)` and the `ActionResult` returned for their own action. Metrics code reads ground truth; agent code does not. The `central` arm and the simulated supervisor are the only declared exceptions (spec 2.6, 5.3).
- **No network calls in tests or in the simulator.** Tests use `FakeLLMClient`. The simulator reads LLM outputs only from the intake cache written by `experiments/doi_intake_run.py` (Task 14). A missing cache entry is an error, never a live call.
- **The LLM boundary is enforced in code.** Nothing under `src/doi/llm/` may be imported by `policies.py`, `hauler.py`, `agent.py`, `spacetime.py`, `world.py`, `evidence.py` or `belief.py`. Task 15 adds a test that checks this by reading the import lines.
- Every module starts with a short docstring stating its purpose. No comments that narrate what the next line does. Type hints on all public functions.
- Git: one commit per task, imperative lower-case subject. **Do not add `Co-Authored-By` lines, "Generated with" lines, or any mention of Claude or Anthropic to any commit message.** Do not push.
- Naming: code keeps `pit` / `fill` / `depot` / `SANDBAG` from v1. In the spec these are "blocked cell", "edit" and "station" (spec 2.1). Do not rename existing identifiers.
- Numbers in tests were verified by hand against the grid geometry in Task 2 and Task 3. If a test and the rules disagree, stop and report; do not silently change either.

## Review Focus

Failure modes the spec implies that no single task's happy path exercises. Each has a pinned test in the task that owns it.

1. A robot that learns a pit was filled must replan immediately and must not keep detouring (Task 10, `test_other_robot_replans_when_fill_learned`).
2. `loss = 1.0` with `r_comm = inf` must behave exactly like `r_comm = 0` for ledger content and must not crash (Task 6, `test_full_loss_equals_no_comm`; Task 11).
3. A pit that never pays back must never be filled, and `RoF` must then produce exactly the `NeverFill` cost (Task 11, `test_rof_equals_neverfill_when_unprofitable`).
4. A hauler that finishes its tasks, is despawned, or loses a claim mid-haul must not leave a pit permanently claimed or a sandbag lost (Task 10, `test_claim_lease_expires_and_reclaims`, `test_abort_returns_bag`).
5. Two robots dropping on the same pit in the same tick must produce exactly one fill and one rejected drop with the bag retained (Task 7, `test_double_drop_one_fill`).
6. Filling one of two substitute pits must remove the evidence of records that the first fill now serves (Task 5, `test_substitute_fill_invalidates_evidence`).
7. A false report must cost detours but must never start a haul (Task 15, `test_false_report_never_hauled`).
8. A wrong-class record must never produce an edit on a needs-human cell; with the gate on and `p_catch = 1` it must not even produce an attempt (Task 15, `test_gate_vetoes_wrong_class`).
9. Changing ledger `r_comm` must not change traffic messages (Task 6, `test_intent_uses_traffic_channel`).
10. An LLM answer cannot change the numbers in an approval request (Task 14, `test_drafter_numbers_are_not_generated`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/doi/__init__.py` | Empty. |
| `src/doi/rng.py` | Counter-based deterministic randomness. |
| `src/doi/config.py` | `SimConfig` dataclass: every parameter and default. |
| `src/doi/scenarios.py` | `Scenario`, `build_scenario`, `scenario_from_ascii`. |
| `src/doi/paths.py` | Passability, shortest path with canonical tie-break, `dream_path`, `single_pit_rents`. |
| `src/doi/oracle.py` | Hindsight-optimal fill set and `OPT_LB`. |
| `src/doi/crdt.py` | `GSet`, `MaxRegisterMap`, `PNStock`, `ClaimSet`, `RecordSet`, `ApprovalSet`. |
| `src/doi/belief.py` | `BeliefState` composing the CRDTs; cell status and class queries. |
| `src/doi/evidence.py` | `EvidenceEngine`: residual evidence recomputed from task records under the current edit set. |
| `src/doi/network.py` | `Message`, `Network` (traffic and ledger channels; range, latency, loss). |
| `src/doi/world.py` | Ground truth, incidents, actions, arbitration, override log, sensing. |
| `src/doi/spacetime.py` | Local space-time A* used by agents. |
| `src/doi/agent.py` | `RobotAgent`: sense, receive, decide, act, broadcast. |
| `src/doi/hauler.py` | Hauling finite-state machine and claim handling. |
| `src/doi/policies.py` | `FillPolicy` and the arms (spec 4.6). |
| `src/doi/incidents.py` | Location names, `locate`, report text rendering, dataset records. |
| `src/doi/llm/client.py` | `LLMClient` protocol, `OpenAICompatClient` (stdlib HTTP), `FakeLLMClient`. |
| `src/doi/llm/intake.py` | Prompt, output validation, `IntakeResult`, cache read/write. |
| `src/doi/llm/drafter.py` | Approval request: numbers from the ledger, one LLM-written reason sentence. |
| `src/doi/supervisor.py` | Simulated supervisor (latency, approve/veto, timeout policy). |
| `src/doi/maps.py` | MovingAI `.map` loader and endless task streams (Stage 1). |
| `src/doi/metrics.py` | `RunResult`, post-hoc metrics. |
| `src/doi/runner.py` | `run_episode`, `run_arms`, competitive ratio. |
| `src/doi/stats.py` | Bootstrap CI, sign-flip test, Spearman. |
| `experiments/doi_common.py` | Grid-of-configs runner with process pool and CSV output. |
| `experiments/doi_build_incidents.py` | Writes `data/incidents/{dev,test}.jsonl`. |
| `experiments/doi_intake_run.py` | Runs a model over the incident sets, writes the cache. |
| `experiments/doi_intake_eval.py` | Intake accuracy, latency, cost tables. |
| `experiments/doi_e1_validity.py` ... `doi_e8_gate.py`, `doi_e6_scaling.py` | One script per experiment. |
| `docs/research/theory.md` | Propositions P1 to P4 with proof sketches. |
| `docs/research/prior-art-verification.md` | Results of the spec's section 12 reading tasks. |

Task order matters: each task consumes interfaces from earlier ones.

---

### Task 0: Hygiene and baseline

**Files:**
- Create: `.gitignore`
- Modify: `README.md` (the "Decision Logic" and "Sandbags" passages only)
- Create: `docs/research/` is already present (contains `research-design.md`).

**Interfaces:** none.

- [ ] **Step 1: Record the baseline**

Run: `source venv/bin/activate && python -m pytest tests -q -p no:cacheprovider`
Expected: `52 passed`.

- [ ] **Step 2: Create `.gitignore`** with exactly these lines:

```
__pycache__/
*.pyc
.idea/
.DS_Store
.coverage
.pytest_cache/
venv/
experiments/results/doi/
.env
.env.*
```

- [ ] **Step 3: Untrack committed junk**

Run: `git rm --cached .DS_Store .coverage` then `git status --short`.
Expected: both listed as `D` (staged deletions); no `__pycache__` lines remain in `git status`.

- [ ] **Step 4: Correct the README**

In the README section "Traffic Index" through "Decision Logic", replace the pseudo-code block that says `removal_cost = base_cost / traffic_index` with this text (keep surrounding headings):

```
Current behaviour (as implemented in `src/mapf_ro/removal.py`):

    removal_cost = sandbag_travel_cost + fill_cost
    net_benefit  = detour_cost - removal_cost
    fill the pit when net_benefit > 0

The traffic index is recorded but does not affect the decision. Sandbag
transport is a cost estimate (Manhattan distance x per-step cost); no robot
physically carries a sandbag in the legacy MAPF-RO code. The decentralised
simulator in `src/doi/` models carrying physically.
```

In the "Sandbags" bullet list, change "A robot can **carry** a sandbag and **deploy** it into an adjacent pit" to "Sandbag transport is accounted as a cost (Manhattan distance x per-step cost); robots do not physically carry them in the legacy code".

- [ ] **Step 5: Run tests, commit**

Run: `python -m pytest tests -q -p no:cacheprovider` -> `52 passed`.

```bash
git add .gitignore README.md docs/
git commit -m "chore: add gitignore, untrack junk, correct README removal-cost description"
```

---

### Task 1: Deterministic randomness and configuration

**Files:**
- Create: `src/doi/__init__.py` (empty), `src/doi/rng.py`, `src/doi/config.py`
- Create: `tests/doi/conftest.py`, `tests/doi/test_doi_rng.py`

**Interfaces:**
- Produces:
  - `splitmix64(x: int) -> int`
  - `mix(seed: int, *keys: int) -> int`
  - `u01(seed: int, *keys: int) -> float`  (uniform in [0, 1))
  - `stream(seed: int, name: str) -> random.Random`
  - `SimConfig` dataclass (fields below) with `to_dict() -> dict`, `replace(**kw) -> SimConfig`, `unreachable_cost_for(h: int, w: int) -> float`.

`tests/doi/conftest.py` must contain exactly:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

**Rules for `rng.py`:**
- `MASK = (1 << 64) - 1`.
- `splitmix64(x)`: `x = (x + 0x9E3779B97F4A7C15) & MASK; z = x; z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK; z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK; return z ^ (z >> 31)`.
- `mix(seed, *keys)`: `h = splitmix64(seed & MASK)`; for each key `h = splitmix64((h ^ (key & MASK)) & MASK)`; return `h`. Negative keys are masked by `& MASK` (two's complement); this is intended.
- `u01(seed, *keys) = (mix(seed, *keys) >> 11) / float(1 << 53)`.
- `stream(seed, name)`: `key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")`; return `random.Random(mix(seed, key))`.

**Rules for `SimConfig`** (non-frozen dataclass, all fields with these defaults; `scenario_params: dict` uses `field(default_factory=dict)`):

| field | default | meaning |
|---|---|---|
| `seed` | 0 | master seed |
| `scenario` | `"single_pit"` | scenario name |
| `scenario_params` | `{}` | overrides of scenario defaults |
| `n_robots` | 12 | |
| `tasks_per_robot` | 20 | K |
| `kappa` | 4.0 | cost per carried step |
| `fee` | 1.0 | cost per fill |
| `unreachable_cost` | `None` | `None` means `4 * (H + W)` |
| `r_sense` | 2 | Manhattan sensing radius |
| `r_comm` | 8.0 | ledger channel: Manhattan radius, `float("inf")` allowed, 0 means no STATE messages |
| `latency` | 1 | ledger channel latency in ticks, must be >= 1 |
| `loss` | 0.0 | ledger channel per-receiver drop probability |
| `r_traffic` | 8.0 | traffic channel (INTENT) radius; held fixed in every experiment except E6 |
| `loss_traffic` | 0.0 | traffic channel drop probability |
| `gossip_period` | 1 | STATE broadcast every this many ticks |
| `intent_window` | 6 | W |
| `policy` | `"rof"` | one of `never, myopic, eager, rof, rof_pit, rof_local, rof_w, rof_x, central, hindsight, free` |
| `theta` | 1.0 | trigger multiplier |
| `window` | `None` | evidence window in ticks for `rof_w` (`None` = no window); must be > 0 if set |
| `intake` | `"none"` | `none`, `oracle`, or `llm:<model_key>` (spec 5.5) |
| `intake_cache` | `"experiments/results/doi/intake_cache"` | directory of cached LLM intake outputs |
| `tick_seconds` | 0.5 | wall-clock seconds per tick, converts LLM latency to ticks |
| `p_wrong_class` | 0.0 | probability that a delivered record says `robot_clearable` for a `needs_human` incident |
| `gate` | `False` | human-on-the-loop approval gate |
| `sup_latency_median` | 30 | supervisor response latency median, ticks |
| `sup_latency_sigma` | 0.5 | log-normal sigma; 0 means fixed latency |
| `p_catch` | 0.9 | probability the supervisor vetoes a request whose record class is wrong |
| `approval_timeout` | 60 | ticks before the timeout policy applies |
| `approval_conf` | 0.7 | timeout policy approves only records with confidence >= this |
| `claim` | `True` | use claim protocol |
| `lease_ticks` | 8 | |
| `stagger_cap` | 25 | |
| `max_ticks` | 20000 | |
| `stall_ticks` | 100 | |
| `debug_checks` | `False` | when True, World asserts invariants each tick |

`__post_init__` validates: `latency >= 1`, `0 <= loss <= 1`, `0 <= loss_traffic <= 1`, `r_comm >= 0`, `r_traffic >= 0`, `n_robots >= 1`, `tasks_per_robot >= 1`, `policy` in the allowed set, `window is None or window > 0`, `intake` is `none`, `oracle` or starts with `llm:`, `0 <= p_wrong_class <= 1`, `0 <= p_catch <= 1`, `sup_latency_median >= 1`, `approval_timeout >= 1`; raise `ValueError` otherwise.

- [ ] **Step 1: Write the failing tests** in `tests/doi/test_doi_rng.py`

```python
import pytest
from src.doi.rng import splitmix64, mix, u01, stream
from src.doi.config import SimConfig


def test_splitmix64_reference_vector():
    assert splitmix64(0) == 0xE220A8397B1DCDAF


def test_u01_deterministic_and_in_range():
    a = u01(7, 1, 2, 3)
    assert a == u01(7, 1, 2, 3)
    assert 0.0 <= a < 1.0
    assert u01(7, 1, 2, 3) != u01(7, 1, 2, 4)
    assert u01(7, 1, 2, 3) != u01(8, 1, 2, 3)


def test_u01_roughly_uniform():
    xs = [u01(1, i) for i in range(100000)]
    assert abs(sum(xs) / len(xs) - 0.5) < 0.01
    assert abs(sum(1 for x in xs if x < 0.3) / len(xs) - 0.3) < 0.01


def test_stream_reproducible_and_named():
    a = [stream(5, "tasks-0").random() for _ in range(3)]
    b = [stream(5, "tasks-0").random() for _ in range(3)]
    c = [stream(5, "tasks-1").random() for _ in range(3)]
    assert a == b and a != c


def test_config_validation():
    SimConfig()
    with pytest.raises(ValueError):
        SimConfig(latency=0)
    with pytest.raises(ValueError):
        SimConfig(loss=1.5)
    with pytest.raises(ValueError):
        SimConfig(policy="nope")
    with pytest.raises(ValueError):
        SimConfig(intake="gpt")
    with pytest.raises(ValueError):
        SimConfig(window=0)
    SimConfig(intake="llm:hosted", policy="free", window=50)
    assert SimConfig().replace(seed=3).seed == 3
    assert SimConfig().unreachable_cost_for(15, 21) == 144.0
    assert SimConfig(unreachable_cost=50.0).unreachable_cost_for(15, 21) == 50.0
```

- [ ] **Step 2: Run, expect failure**

Run: `python -m pytest tests/doi/test_doi_rng.py -q -p no:cacheprovider`
Expected: collection errors (`ModuleNotFoundError: src.doi`).

- [ ] **Step 3: Implement** `rng.py` and `config.py` per the rules above.
- [ ] **Step 4: Run, expect pass** (same command).
- [ ] **Step 5: Run the full suite** `python -m pytest tests -q -p no:cacheprovider` -> `57 passed` (52 + 5).
- [ ] **Step 6: Commit** `git add src/doi tests/doi && git commit -m "feat(doi): counter-based rng and SimConfig"`

---

### Task 2: Scenarios

**Files:**
- Create: `src/doi/scenarios.py`, `tests/doi/test_doi_scenarios.py`

**Interfaces:**
- Consumes: `SimConfig`, `stream` (Task 1); `Grid`, `CellType` from `src/environment/grid.py`; `EnvironmentGenerator` from `src/environment/generator.py`.
- Produces:

```python
@dataclass(frozen=True)
class Incident:                      # family D ground truth (spec 2.2)
    oid: int
    cells: Tuple[Pos, ...]           # sorted; one cell in every Stage 0 scenario
    appear_tick: int
    kind: str                        # pallet | spill | debris | rack_damage
    cls: str                         # robot_clearable | needs_human

@dataclass(frozen=True)
class Report:
    report_id: str                   # "r0", "r1", ... in order of (emit_tick, location)
    oid: Optional[int]               # None for a false report
    location: str                    # key of Scenario.locations
    emit_tick: int
    kind: str
    cls: str

@dataclass
class Scenario:
    name: str
    family: str                      # "S", "D" or "ascii"
    grid: Grid                       # static map: pits are CellType.PIT, depots CellType.SANDBAG; never mutated.
                                     # Incident cells are FREE in this grid; they become blocked in World at appear_tick.
    pits: List[Pos]                  # sorted; static pits only (empty in family D)
    depots: Dict[Pos, int]           # depot (station) cell -> initial stock
    starts: List[Pos]                # one per robot, distinct
    tasks: List[List[Pos]]           # tasks[i][k] = goal of robot i's k-th task; len == tasks_per_robot
    incidents: List[Incident]        # sorted by oid; empty in family S
    reports: List[Report]            # sorted by (emit_tick, location); empty in family S
    locations: Dict[str, Tuple[Pos, ...]]   # location name -> cells; empty in family S
    meta: Dict[str, Any]

DEFAULTS: Dict[str, Dict[str, Any]]            # per-scenario default parameters (below)
def build_scenario(cfg: SimConfig) -> Scenario
def scenario_from_ascii(rows: List[str], starts: List[Pos], tasks: List[List[Pos]],
                        depot_stock: int = 2, name: str = "ascii",
                        incidents: Sequence[Incident] = (), reports: Sequence[Report] = (),
                        locations: Optional[Dict[str, Tuple[Pos, ...]]] = None) -> Scenario
```

**Rules: `scenario_from_ascii`.** Characters: `.` FREE, `#` OBSTACLE, `P` PIT, `D` depot (grid cell type SANDBAG, stock `depot_stock`). All rows must have equal length, else `ValueError`.

**Rules: two-room family** (`single_pit`, `two_pits_parallel`, `series_pits`, `multi_pit_wall`, `shift` share one generator `_two_room(params, cfg)`). Parameters and defaults:

| scenario | H | W | wall_col | wall_thick | pit cells | door_rows | depot | other |
|---|---|---|---|---|---|---|---|---|
| `single_pit` | 15 | 21 | 10 | 1 | `[(3,10)]` | `(12,13,14)` | `(3, e+1+depot_dist)` | `depot_dist=2`, `q_cross=0.8`, `stock=2` |
| `two_pits_parallel` | 15 | 21 | 10 | 1 | `[(3,10),(5,10)]` | `(12,13,14)` | `(4, e+1+depot_dist)` | same |
| `series_pits` | 15 | 21 | 10 | 5 | `[(3,11),(3,13)]` | `(12,13,14)` | `(3, e+1+depot_dist)` | corridor row 3, same |
| `multi_pit_wall` | 15 | 21 | 10 | 1 | `[(2,10),(4,10),(6,10),(8,10)]` | `(12,13,14)` | `(4, e+1+depot_dist)` | `stock=len(pits)` |
| `shift` | 15 | 21 | 10 | 1 | `[(2,10),(4,10),(10,10),(12,10)]` | `(6,7,8)` | `(7, e+1+depot_dist)` | `shift_after_task=10`, `hot_before=(0,6)`, `hot_after=(8,14)`, `q_hot=0.8`, `stock=4` |

where `e = wall_col + wall_thick - 1` (east-most wall column). Pit cells and the pit-cell defaults are overridable through `cfg.scenario_params` keys `H, W, wall_col, wall_thick, pit_cells, door_rows, depot_dist, depot_row, stock, q_cross`. Unknown keys raise `ValueError`.

Geometry construction order: all cells FREE; wall columns `wall_col .. wall_col+wall_thick-1` set OBSTACLE for every row; door rows set FREE across those columns; for `series_pits` the corridor row `r=3` is set FREE across those columns; pit cells set PIT; depot cell set SANDBAG (depot rows/cols must be inside the grid and not a wall; raise `ValueError` otherwise).

Rooms: west = columns `0 .. wall_col-1`; east = columns `e+1 .. W-1`. Room free cells exclude pits and the depot cell.

Starts: robots `0 .. n//2 - 1` start in the west room, the rest in the east room (so with `n=1` the robot starts west). For robot `i` use its own stream `stream(seed, f"start-{i}")` and draw `rng.choice(sorted_room_free_cells)` repeatedly until the cell is not already used by a lower-id robot (at most 1000 tries, then `ValueError`). Per-robot streams make robot `i`'s start and tasks independent of `n_robots`.

Tasks: for robot `i` use `rng = stream(seed, f"tasks-{i}")`. Keep `room` = room of the previous goal (initially the start room). For task `k`:
- Non-shift scenarios: with probability `q_cross` pick the other room, else the same room; goal is `rng.choice(sorted_room_free_cells)` excluding the previous position (retry until different).
- `shift`: as above, but every goal (not only cross-wall goals) is drawn as follows: take the band `hot_before` (inclusive row range) when `k < shift_after_task`, else `hot_after`; build the list of chosen-room free cells whose row lies in the band; with probability `q_hot` choose from that list (if non-empty), else from all cells of the chosen room.

**Rules: `random_pits`.** Parameters `H=20, W=20, obstacle_density=0.10, pit_density=0.04, sandbag_count=2`. Use `EnvironmentGenerator(width=W, height=H, obstacle_density=..., pit_density=..., sandbag_count=..., seed=cfg.seed).generate(num_robots=cfg.n_robots)`; depots = SANDBAG cells with stock 1 each. Tasks: for each robot sample goals uniformly (stream `tasks-i`) from free cells (FREE or SANDBAG) that are in the same connected component as the robot's previous position when all pits are open (BFS); retry up to 200 times; fall back to the previous position's neighbour list if exhausted (raise `ValueError` if none). Add this scenario's name to `DEFAULTS`.

**Rules: family D** (spec 2.2, 7.1). Shared parameters and defaults: `n_incidents`, `appear_max=150`, `p_human=0.25`, `p_report=0.9`, `delta_report=5`, `p_false=0.0`, `stock=3`. Only these keys (plus `q_cross` for `incidents_room`) are accepted in `scenario_params`; others raise `ValueError`.

* `incidents_room`: H=15, W=21, wall column 10 is OBSTACLE except three one-cell doors `(2,10)`, `(7,10)`, `(12,10)` (FREE). Depots `(7,8)` and `(7,12)`, stock `stock` each. Locations: `"north door" -> ((2,10),)`, `"middle door" -> ((7,10),)`, `"south door" -> ((12,10),)`. Incident candidates: the three door cells. `n_incidents` default 2, must be <= 2 (`ValueError` otherwise) so one door always stays open. Starts and tasks as `single_pit` (two rooms, `q_cross=0.8`), with room free cells excluding depots.
* `incidents_aisles`: H=13, W=21. Rows 0, 6, 12 are FREE cross aisles. In rows 1..5 and 7..11, odd columns 1, 3, ..., 19 are OBSTACLE (shelves) and even columns 0, 2, ..., 20 are FREE (aisles). Depots `(6,0)` and `(6,20)`, stock `stock` each. Locations: for aisle `k` in 1..11 (column `2(k-1)`) and bay `b` in 1..10 (row `b` for b <= 5, row `b+1` for b >= 6): `f"aisle {k} bay {b}" -> ((row, col),)`; plus `"west station" -> ((6,0),)`, `"east station" -> ((6,20),)`. Incident candidates: aisle-bay cells in aisles 2..10 (columns 0 and 20 stay clear so the map stays connected). `n_incidents` default 4. Goal cells: all aisle-bay cells not in any incident. Starts: robot `i` draws from cross-aisle cells of rows 0 and 12 (stream `start-i`, distinct as in the two-room rule). Tasks: goal drawn uniformly from goal cells (stream `tasks-i`), different from the previous position.

Incident generation (both, stream `stream(seed, "incidents")`): `cells = rng.sample(sorted(candidates), n_incidents)`; for `oid` in order: `appear_tick = rng.randint(0, appear_max)`; `cls = "needs_human"` if `rng.random() < p_human` else `"robot_clearable"`; `kind = "rack_damage"` if `needs_human` (aisles) or `"spill"` (room), else `rng.choice(["pallet", "spill", "debris"])`. Reports (stream `"reports"`): for each incident in oid order, if `rng.random() < p_report`, a true report at `emit_tick = appear_tick + delta_report` with the incident's kind and class. False reports: `n_false = sum(rng.random() < p_false for _ in range(n_incidents))`; each at a location whose cell is a candidate but not an incident cell (`rng.choice` over sorted names), `emit_tick = rng.randint(0, appear_max + delta_report)`, kind and class drawn as for incidents, `oid=None`. Sort all reports by `(emit_tick, location)` and number them `r0, r1, ...`. Task goals never equal an incident cell (filter before drawing). Incident cells are FREE in `grid`.

`meta` must contain: `rooms` (for two-room family: dict with `west_cols`, `east_cols` ranges), `params` (resolved parameters), `seed`. Family S scenarios have `family="S"`, `incidents=[]`, `reports=[]`, `locations={}`; `scenario_from_ascii` sets `family="ascii"` and passes its optional arguments through.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_scenarios.py`

```python
import pytest
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario, scenario_from_ascii
from src.environment.grid import CellType


def test_single_pit_geometry():
    s = build_scenario(SimConfig(scenario="single_pit", n_robots=4, tasks_per_robot=5, seed=1))
    g = s.grid
    assert (g.height, g.width) == (15, 21)
    assert s.pits == [(3, 10)]
    assert g.get(3, 10) == CellType.PIT
    assert g.get(0, 10) == CellType.OBSTACLE and g.get(11, 10) == CellType.OBSTACLE
    assert all(g.get(r, 10) == CellType.FREE for r in (12, 13, 14))
    assert s.depots == {(3, 13): 2}
    assert g.get(3, 13) == CellType.SANDBAG
    assert len(s.starts) == 4 and len(set(s.starts)) == 4
    assert all(len(t) == 5 for t in s.tasks)
    assert all(c[1] < 10 for c in s.starts[:2]) and all(c[1] > 10 for c in s.starts[2:])


def test_series_pits_geometry():
    s = build_scenario(SimConfig(scenario="series_pits", n_robots=2, tasks_per_robot=2))
    g = s.grid
    assert s.pits == [(3, 11), (3, 13)]
    assert [g.get(3, c) for c in range(10, 15)] == [
        CellType.FREE, CellType.PIT, CellType.FREE, CellType.PIT, CellType.FREE]
    assert g.get(2, 12) == CellType.OBSTACLE and g.get(4, 12) == CellType.OBSTACLE
    assert s.depots == {(3, 17): 2}


def test_build_is_deterministic_and_seed_sensitive():
    a = build_scenario(SimConfig(seed=3, n_robots=3, tasks_per_robot=4))
    b = build_scenario(SimConfig(seed=3, n_robots=3, tasks_per_robot=4))
    c = build_scenario(SimConfig(seed=4, n_robots=3, tasks_per_robot=4))
    assert (a.starts, a.tasks) == (b.starts, b.tasks)
    assert (a.starts, a.tasks) != (c.starts, c.tasks)


def test_tasks_do_not_depend_on_other_robot_count():
    a = build_scenario(SimConfig(seed=3, n_robots=4, tasks_per_robot=4))
    b = build_scenario(SimConfig(seed=3, n_robots=6, tasks_per_robot=4))
    assert a.tasks[0] == b.tasks[0]


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        build_scenario(SimConfig(scenario_params={"bogus": 1}))


def test_ascii_scenario():
    rows = ["...#...", "...P..D", "...#..."]
    s = scenario_from_ascii(rows, starts=[(1, 2)], tasks=[[(1, 4)]], depot_stock=3)
    assert s.pits == [(1, 3)] and s.depots == {(1, 6): 3}
    with pytest.raises(ValueError):
        scenario_from_ascii(["..", "..."], [(0, 0)], [[(0, 1)]])


def test_shift_goals_follow_band():
    cfg = SimConfig(scenario="shift", n_robots=6, tasks_per_robot=20, seed=2)
    s = build_scenario(cfg)
    before = [t[k] for t in s.tasks for k in range(0, 10) if t[k][1] > 10]
    after = [t[k] for t in s.tasks for k in range(10, 20) if t[k][1] > 10]
    assert sum(1 for p in before if p[0] <= 6) / max(1, len(before)) > 0.6
    assert sum(1 for p in after if p[0] >= 8) / max(1, len(after)) > 0.6


def test_aisles_geometry_and_locations():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=4, tasks_per_robot=3, seed=1))
    g = s.grid
    assert (g.height, g.width) == (13, 21) and s.family == "D" and s.pits == []
    assert g.get(1, 1) == CellType.OBSTACLE and g.get(0, 1) == CellType.FREE and g.get(6, 1) == CellType.FREE
    assert s.locations["aisle 2 bay 3"] == ((3, 2),)
    assert s.locations["aisle 2 bay 6"] == ((7, 2),)
    assert s.depots == {(6, 0): 3, (6, 20): 3}


def test_incidents_generated_and_goals_avoid_them():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=6, tasks_per_robot=5, seed=1))
    cells = [c for inc in s.incidents for c in inc.cells]
    assert len(s.incidents) == 4 and len(set(cells)) == 4
    assert all(c[1] not in (0, 20) for c in cells)
    assert all(s.grid.get(*c) == CellType.FREE for c in cells)
    assert all(inc.cls in ("robot_clearable", "needs_human") for inc in s.incidents)
    assert not any(goal in cells for t in s.tasks for goal in t)
    assert all(r.oid is not None for r in s.reports) and len(s.reports) <= 4


def test_false_reports_point_at_clear_cells():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=2, tasks_per_robot=2, seed=3,
                                 scenario_params={"p_false": 1.0}))
    false = [r for r in s.reports if r.oid is None]
    cells = {c for inc in s.incidents for c in inc.cells}
    assert len(false) == 4
    assert all(s.locations[r.location][0] not in cells for r in false)
    assert [r.report_id for r in s.reports] == [f"r{i}" for i in range(len(s.reports))]


def test_room_doors():
    s = build_scenario(SimConfig(scenario="incidents_room", n_robots=4, tasks_per_robot=3, seed=1))
    assert s.grid.get(2, 10) == CellType.FREE and s.grid.get(3, 10) == CellType.OBSTACLE
    assert s.locations["north door"] == ((2, 10),)
    with pytest.raises(ValueError):
        build_scenario(SimConfig(scenario="incidents_room", scenario_params={"n_incidents": 3}))
```

- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError: src.doi.scenarios`).
- [ ] **Step 3: Implement** per the rules. Note `test_tasks_do_not_depend_on_other_robot_count` requires per-robot rng streams and that robot 0's start room is independent of `n` (it is west in both).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): scenario generators and ascii builder"`

---

### Task 3: Path primitives and the dream path

**Files:**
- Create: `src/doi/paths.py`, `tests/doi/test_doi_paths.py`

**Interfaces:**
- Consumes: `Grid`, `CellType`; `Scenario` fields (grid, pits).
- Produces:

```python
PassFn = Callable[[Pos], bool]
def passable_fn(grid: Grid, open_pits: FrozenSet[Pos], closed: FrozenSet[Pos] = frozenset()) -> PassFn
def bfs_dist_map(passable: PassFn, source: Pos, height: int, width: int) -> Dict[Pos, int]
def shortest_path(passable: PassFn, start: Pos, goal: Pos, height: int, width: int,
                  pit_cells: FrozenSet[Pos] = frozenset()) -> Optional[List[Pos]]

@dataclass(frozen=True)
class DreamResult:
    d_block: int
    d_open: int
    bundle: FrozenSet[Pos]
    rent: int
    path_open: Optional[Tuple[Pos, ...]]

def dream_path(grid: Grid, pits: Sequence[Pos], filled: FrozenSet[Pos],
               start: Pos, goal: Pos, unreachable: float,
               hard_blocked: FrozenSet[Pos] = frozenset()) -> DreamResult
def single_pit_rents(grid: Grid, pits: Sequence[Pos], filled: FrozenSet[Pos],
                     start: Pos, goal: Pos, unreachable: float,
                     hard_blocked: FrozenSet[Pos] = frozenset()) -> Dict[Pos, int]
```

**Rules.**
- `passable_fn(grid, open_pits, closed)(p)`: in bounds and `p not in closed` and (`grid.is_passable(*p)` or (`grid.get(*p) == CellType.PIT` and `p in open_pits`)). Obstacles are never opened. `closed` is how dynamic obstructions (FREE in the static grid) are blocked.
- Meaning of the `pits` argument from here on: the *editable blocked cells the caller believes in* (static pits, plus believed-blocked robot-clearable obstruction cells in family D). `hard_blocked` are cells blocked in both the blocked and the open path (believed `needs_human` cells).
- `bfs_dist_map`: plain BFS with the global neighbour order, returns distance for every reachable cell including the source (distance 0).
- `shortest_path`: A* over lexicographic cost `(steps, pits_used)`, heap key `(g_steps + h, g_pits, counter)` where `h` is Manhattan distance and `counter` is a monotonically increasing insertion index (the final canonical tie-break). `pits_used` counts entered cells that are in `pit_cells`. Returns the list of cells including start and goal, or `None`. `start == goal` returns `[start]`. If `start` or `goal` is not passable return `None`.
- `dream_path`: `unfilled = set(pits) - filled`. Blocked path: `passable_fn(grid, filled, closed=frozenset(unfilled) | hard_blocked)`; `d_block = len(path)-1` or `unreachable` if none. Open path: `passable_fn(grid, frozenset(pits), closed=hard_blocked)` with `pit_cells = frozenset(unfilled)`; if none, return `DreamResult(d_block, unreachable, frozenset(), 0, None)`. Else `bundle = frozenset(c for c in path if c in unfilled)`, `d_open = len(path)-1`, `rent = max(0, d_block - d_open)`. If `bundle` is empty force `rent = 0`.
- `single_pit_rents`: for each `p` in `sorted(unfilled)`: `passable_fn(grid, filled | {p}, closed=(frozenset(unfilled) - {p}) | hard_blocked)`; `d_p` is the path length (or `unreachable`); rent is `max(0, d_block - d_p)`; include only entries with rent > 0.

- [ ] **Step 1: Write failing tests**

```python
from src.doi.paths import passable_fn, bfs_dist_map, shortest_path, dream_path, single_pit_rents
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario, scenario_from_ascii


def test_single_pit_dream_path_numbers():
    s = build_scenario(SimConfig(scenario="single_pit", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 11), unreachable=144.0)
    assert (r.d_block, r.d_open, r.rent) == (20, 2, 18)
    assert r.bundle == frozenset({(3, 10)})
    r2 = dream_path(s.grid, s.pits, frozenset({(3, 10)}), (3, 9), (3, 11), 144.0)
    assert (r2.d_block, r2.d_open, r2.rent, r2.bundle) == (2, 2, 0, frozenset())


def test_series_needs_bundle():
    s = build_scenario(SimConfig(scenario="series_pits", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 15), 144.0)
    assert (r.d_block, r.d_open, r.rent) == (24, 6, 18)
    assert r.bundle == frozenset({(3, 11), (3, 13)})
    assert single_pit_rents(s.grid, s.pits, frozenset(), (3, 9), (3, 15), 144.0) == {}
    r2 = dream_path(s.grid, s.pits, frozenset({(3, 13)}), (3, 9), (3, 15), 144.0)
    assert r2.bundle == frozenset({(3, 11)}) and r2.rent == 18


def test_parallel_pits_are_substitutes():
    s = build_scenario(SimConfig(scenario="two_pits_parallel", n_robots=1, tasks_per_robot=1))
    rents = single_pit_rents(s.grid, s.pits, frozenset(), (3, 9), (3, 11), 144.0)
    assert rents == {(3, 10): 18, (5, 10): 14}
    r = dream_path(s.grid, s.pits, frozenset(), (3, 9), (3, 11), 144.0)
    assert r.bundle == frozenset({(3, 10)})


def test_unreachable_uses_penalty():
    s = scenario_from_ascii(["..#..", "..#..", "..#.."], [(0, 0)], [[(0, 4)]])
    r = dream_path(s.grid, s.pits, frozenset(), (0, 0), (0, 4), unreachable=50.0)
    assert r.d_block == 50 and r.d_open == 50 and r.rent == 0 and r.bundle == frozenset()


def test_canonical_tiebreak_is_deterministic():
    s = scenario_from_ascii(["....", "....", "...."], [(0, 0)], [[(2, 3)]])
    pf = passable_fn(s.grid, frozenset())
    a = shortest_path(pf, (0, 0), (2, 3), 3, 4)
    b = shortest_path(pf, (0, 0), (2, 3), 3, 4)
    assert a == b and len(a) == 6


def test_bfs_dist_map():
    s = scenario_from_ascii(["...", ".#.", "..."], [(0, 0)], [[(2, 2)]])
    d = bfs_dist_map(passable_fn(s.grid, frozenset()), (0, 0), 3, 3)
    assert d[(0, 0)] == 0 and d[(2, 2)] == 4 and (1, 1) not in d


def test_dynamic_obstruction_and_hard_blocked():
    s = build_scenario(SimConfig(scenario="incidents_room", n_robots=1, tasks_per_robot=1))
    r = dream_path(s.grid, [(2, 10)], frozenset(), (2, 9), (2, 11), 144.0)
    assert (r.d_block, r.d_open, r.rent, r.bundle) == (12, 2, 10, frozenset({(2, 10)}))
    r2 = dream_path(s.grid, [(2, 10)], frozenset(), (2, 9), (2, 11), 144.0,
                    hard_blocked=frozenset({(7, 10)}))
    assert (r2.d_block, r2.d_open, r2.rent) == (22, 2, 20)
```

The numbers: blocked at the north door, the detour goes through the middle door (5 down, 2 across, 5 up = 12); with the middle door also hard-blocked it goes through the south door (10 + 2 + 10 = 22).

The numbers: `single_pit` rent 18 and `parallel` rent for the second pit 14 (route via row 5 pit: block 20, open one pit at row 5 gives 6) were verified by hand.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): shortest paths with canonical tie-break and dream path"`

---

### Task 4: Hindsight oracle

**Files:**
- Create: `src/doi/oracle.py`, `tests/doi/test_doi_oracle.py`

**Interfaces:**
- Consumes: `Scenario`, `SimConfig`, `passable_fn`, `bfs_dist_map`, `dream_path`.
- Produces:

```python
@dataclass(frozen=True)
class Hindsight:
    s_star: FrozenSet[Pos]
    lb_star: float
    lb_empty: float
    buy_lb: Dict[Pos, float]
    relevant: Tuple[Pos, ...]
    exhaustive: bool

def task_pairs(scenario: Scenario) -> List[Tuple[Pos, Pos]]
def buy_lb(scenario: Scenario, cfg: SimConfig) -> Dict[Pos, float]
def hindsight(scenario: Scenario, cfg: SimConfig) -> Hindsight
```

**Rules.**
- `task_pairs`: for each robot `i` in id order, pairs `(starts[i], tasks[i][0]), (tasks[i][0], tasks[i][1]), ...`.
- `buy_lb(p) = cfg.fee + cfg.kappa * d` where `d` is the minimum over depots with stock > 0 and over passable neighbours `n` of `p` (under "all pits open") of the BFS distance from the depot to `n` with all pits open. If no depot reaches `p`, `inf`.
- `relevant` = sorted union over all task pairs of `dream_path(...).bundle` (with `filled = frozenset()`).
- For a subset `S`: `lb(S) = sum over pairs of dist_S(origin, dest) + sum over p in S of buy_lb[p]`, where `dist_S` is BFS distance with open pits `S` (unreachable counts as `cfg.unreachable_cost_for(H, W)`). Subsets are only valid if `|S| <= sum(depot stock)`; invalid subsets are skipped. Cache one BFS map per unique origin per subset.
- If `len(relevant) <= 10`: evaluate all subsets, `exhaustive=True`. Otherwise greedy forward selection (add the pit with the largest decrease of `lb` until no decrease), then one pass of single-pit swap improvement, `exhaustive=False`.
- Ties: smaller `|S|`, then lexicographically smaller sorted tuple.
- `lb_empty = lb(frozenset())`, `lb_star = lb(s_star)`.
- `hindsight` and `buy_lb` are defined for family S and ascii scenarios with static pits only; for `scenario.family == "D"` raise `ValueError` (spec 2.6: no hindsight benchmark for incidents).

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.oracle import hindsight, buy_lb, task_pairs
from src.doi.scenarios import scenario_from_ascii

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
CFG = SimConfig(n_robots=1, tasks_per_robot=4)


def make(tasks):
    return scenario_from_ascii(ROWS, [(1, 2)], [tasks], depot_stock=2)


def test_buy_lb_and_pairs():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    assert buy_lb(s, CFG) == {(1, 3): 9.0}
    assert task_pairs(s)[0] == ((1, 2), (1, 4)) and len(task_pairs(s)) == 4


def test_fill_pays_off_with_four_crossings():
    s = make([(1, 4), (1, 2), (1, 4), (1, 2)])
    h = hindsight(s, CFG)
    assert h.s_star == frozenset({(1, 3)})
    assert h.lb_empty == 40.0 and h.lb_star == 17.0 and h.exhaustive


def test_fill_does_not_pay_with_one_crossing():
    s = make([(1, 4)])
    h = hindsight(s, SimConfig(n_robots=1, tasks_per_robot=1))
    assert h.s_star == frozenset() and h.lb_star == h.lb_empty == 10.0


def test_series_hindsight_needs_both_pits():
    from src.doi.scenarios import build_scenario
    cfg = SimConfig(scenario="series_pits", n_robots=2, tasks_per_robot=10, seed=1)
    s = build_scenario(cfg)
    h = hindsight(s, cfg)
    assert h.s_star in (frozenset(), frozenset(s.pits))
    assert h.lb_star <= h.lb_empty
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): hindsight oracle and OPT lower bound"`

---

### Task 5: CRDTs, belief state and evidence

**Files:**
- Create: `src/doi/crdt.py`, `src/doi/belief.py`, `src/doi/evidence.py`, `tests/doi/test_doi_crdt.py`, `tests/doi/test_doi_evidence.py`

v2 change (spec 4.3): the ledger stores immutable task records and evidence is recomputed under the current edit set, so a fill at a substitute removes evidence that it now serves. Cell status for dynamic obstructions uses three max-registers (spec 5.4).

**Interfaces:**
- Produces in `crdt.py`:

```python
Ticket = Tuple[int, int]            # (lamport, robot_id)
BundleKey = Tuple[Pos, ...]         # sorted tuple of cells

class GSet:
    def add(self, x) -> None; def __contains__(self, x) -> bool
    def items(self) -> FrozenSet; def merge(self, other: "GSet") -> bool
    def units(self) -> int; def copy(self) -> "GSet"; def canonical(self) -> Any

class MaxRegisterMap:               # key -> int; merge = max per key; missing keys read as `default`
    def __init__(self, default: int = -1)
    def raise_to(self, key, value: int) -> None
    def get(self, key) -> int
    def keys(self) -> List                         # sorted
    def merge(self, other: "MaxRegisterMap") -> bool; def units(self) -> int
    def copy(self) -> "MaxRegisterMap"; def canonical(self) -> Any

class PNStock:
    def __init__(self, initial: Dict[Pos, int])
    def take(self, depot: Pos, robot_id: int, n: int = 1) -> None
    def give_back(self, depot: Pos, robot_id: int, n: int = 1) -> None
    def mark_empty(self, depot: Pos, robot_id: int) -> None   # takes[robot] += remaining
    def remaining(self, depot: Pos) -> int
    def merge(self, other: "PNStock") -> bool; def units(self) -> int
    def copy(self) -> "PNStock"; def canonical(self) -> Any

@dataclass(frozen=True)
class Claim:
    pits: Tuple[Pos, ...]
    hauler: int

class ClaimSet:
    def issue(self, ticket: Ticket, claim: Claim, expiry: int) -> None
    def renew(self, ticket: Ticket, expiry: int) -> None        # max with existing
    def effective(self, pit: Pos, now: int) -> Optional[Tuple[Ticket, Claim]]
    def tickets_of(self, robot_id: int) -> List[Ticket]
    def merge(self, other: "ClaimSet") -> bool; def units(self) -> int
    def copy(self) -> "ClaimSet"; def canonical(self) -> Any

@dataclass(frozen=True)
class RentRecord:
    robot: int
    task_idx: int
    origin: Pos
    dest: Pos
    tick: int
    rent: float                      # rent counted at planning time

class RecordSet:                     # key (robot, task_idx) -> one immutable record; merge = union
    def add(self, rec: RentRecord) -> None
    def records(self) -> List[RentRecord]          # sorted by (robot, task_idx)
    def merge(self, other: "RecordSet") -> bool; def units(self) -> int
    def copy(self) -> "RecordSet"; def canonical(self) -> Any

@dataclass(frozen=True)
class ObstructionRecord:
    report_id: str
    node: int                        # robot / edge node that ran intake
    location: str
    cells: Tuple[Pos, ...]
    kind: str
    cls: str                         # robot_clearable | needs_human
    est_kits: int
    confidence: float
    rationale: str

class ObstructionSet:                # key report_id; same key from two nodes: keep the smaller `node`
    def add(self, rec: ObstructionRecord) -> None
    def get(self, report_id: str) -> Optional[ObstructionRecord]
    def records(self) -> List[ObstructionRecord]   # sorted by report_id
    def merge(self, other: "ObstructionSet") -> bool; def units(self) -> int
    def copy(self) -> "ObstructionSet"; def canonical(self) -> Any

ApprovalKey = Tuple[Tuple[Pos, ...], Ticket]
class ApprovalSet:                   # key -> "approve" | "veto"; merge: veto dominates
    def set(self, key: ApprovalKey, decision: str) -> None
    def get(self, key: ApprovalKey) -> Optional[str]
    def merge(self, other: "ApprovalSet") -> bool; def units(self) -> int
    def copy(self) -> "ApprovalSet"; def canonical(self) -> Any
```

- Produces in `belief.py`:

```python
CLASS_CODE = {"unknown": 0, "robot_clearable": 1, "needs_human": 2}

class BeliefState:
    def __init__(self, robot_id: int, depots: Dict[Pos, int], static_pits: Sequence[Pos])
    records: RecordSet; filled: GSet; stock: PNStock; claims: ClaimSet; census: GSet
    report_tick: MaxRegisterMap; blocked_tick: MaxRegisterMap; free_tick: MaxRegisterMap
    cls: MaxRegisterMap                            # default 0, codes from CLASS_CODE
    obstructions: ObstructionSet; approvals: ApprovalSet
    lamport: int                                   # merged by max
    def add_record(self, rec: RentRecord) -> None  # also census.add(rec.robot)
    def add_obstruction(self, rec: ObstructionRecord, t: int) -> None
    def observe_cell(self, cell: Pos, blocked: bool, t: int) -> None
    def mark_needs_human(self, cell: Pos) -> None
    def status(self, cell: Pos) -> str             # "filled" | "confirmed" | "reported" | "refuted" | "none"
    def believed_blocked(self) -> FrozenSet[Pos]   # status in {confirmed, reported}
    def editable(self) -> Tuple[Pos, ...]          # sorted: (believed_blocked minus needs_human) | filled
    def hard_blocked(self) -> FrozenSet[Pos]       # believed_blocked with class needs_human
    def next_ticket(self) -> Ticket                # lamport += 1; returns (lamport, robot_id)
    def merge(self, other: "BeliefState") -> bool
    def snapshot(self) -> "BeliefState"            # deep copy for sending
    def units(self) -> int
    def canonical(self) -> Any
```

- Produces in `evidence.py`:

```python
class EvidenceEngine:
    def __init__(self, grid: Grid, unreachable: float)
    def evidence(self, belief: BeliefState, now: int, window: Optional[int] = None,
                 extrapolate: bool = False, per_pit: bool = False) -> Dict[BundleKey, float]
    def contributors(self) -> Dict[BundleKey, FrozenSet[int]]   # from the last evidence() call
```

**Rules.**
- Merges: `GSet` union; `MaxRegisterMap` max per key; `PNStock` per-depot per-robot max for takes and returns; `ClaimSet` union of tickets with max expiry; `RecordSet` union (a key always maps to the same record, since `(robot, task_idx)` is written once); `ObstructionSet` per key keep the record with the smaller `node`; `ApprovalSet` per key `veto` beats `approve`.
- `PNStock.remaining = initial - sum(takes) + sum(returns)`, floored at 0 when read.
- `ClaimSet.effective(pit, now)`: among tickets whose claim covers `pit` and whose expiry `> now`, the smallest ticket; `None` if none.
- `BeliefState.__init__`: `blocked_tick.raise_to(p, 0)` and `cls.raise_to(p, 1)` for every static pit.
- `add_obstruction(rec, t)`: store; for each cell `report_tick.raise_to(cell, t)` and `cls.raise_to(cell, CLASS_CODE[rec.cls])`.
- `observe_cell(cell, blocked, t)`: `blocked_tick.raise_to(cell, t)` if blocked else `free_tick.raise_to(cell, t)`.
- `mark_needs_human(cell)`: `cls.raise_to(cell, 2)`.
- `status(cell)`, first match wins: `filled` if in `filled`; `confirmed` if `blocked_tick >= 0 and blocked_tick >= free_tick`; `reported` if `report_tick >= 0 and report_tick > free_tick`; `refuted` if `free_tick >= 0`; else `none`. (This is spec 5.4 with the precedence made explicit.)
- `believed_blocked()` iterates over the union of keys of the three registers.
- `merge` merges every component, `lamport = max(...)`, returns `True` if anything changed.
- `canonical()` is hashable, order-independent, covers every component plus `lamport`, and excludes `robot_id`.
- `snapshot()` deep copies. `units()` sums `units()` of every component.
- `EvidenceEngine.evidence` (spec 4.3): `pits = belief.editable()`, `F = frozenset(belief.filled.items())`, `hard = belief.hard_blocked()`, `U` the unreachable cost. For each record `r` in `belief.records.records()`: skip if `window is not None and r.tick < now - window`.
  - default: `d = dream_path(grid, pits, F, r.origin, r.dest, U, hard)`; `T = tuple(sorted(d.bundle))`; `c = min(r.rent, d.rent)`; if `T` and `c > 0`: `ev[T] += c`, add `r.robot` to `contributors[T]`.
  - `per_pit=True`: for each `(p, rent)` in `single_pit_rents(grid, pits, F, r.origin, r.dest, U, hard)`: `c = min(r.rent, rent)`; `ev[(p,)] += c`; add contributor.
  - `extrapolate=True`: afterwards multiply each `ev[T]` by `len(belief.census.items()) / max(1, len(contributors[T]))`.
  - Cache `dream_path` / `single_pit_rents` results in a dict keyed by `(origin, dest, pits, F, hard, per_pit)`.
  - Return values as floats.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_crdt.py`

```python
import random
from src.doi.belief import BeliefState
from src.doi.crdt import Claim, ClaimSet, RentRecord, ObstructionRecord

DEPOTS = {(0, 0): 3}


def obs(rid="r0", node=0, cells=((1, 1),), cls="robot_clearable"):
    return ObstructionRecord(rid, node, "x", cells, "pallet", cls, 1, 0.9, "why")


def rand_state(rid: int, rng: random.Random) -> BeliefState:
    b = BeliefState(rid, DEPOTS, [(2, 2)])
    for k in range(rng.randint(0, 4)):
        b.add_record(RentRecord(rid, k, (0, 0), (0, rng.randint(1, 5)), rng.randint(0, 9), rng.randint(1, 9)))
    if rng.random() < 0.5:
        b.filled.add((rng.randint(0, 2), 1))
    if rng.random() < 0.5:
        b.stock.take((0, 0), rid)
    if rng.random() < 0.5:
        b.claims.issue(b.next_ticket(), Claim(((1, 1),), rid), rng.randint(1, 20))
    if rng.random() < 0.5:
        b.observe_cell((1, 1), rng.random() < 0.5, rng.randint(0, 9))
    if rng.random() < 0.5:
        b.add_obstruction(obs(node=rid), rng.randint(0, 9))
    if rng.random() < 0.3:
        b.approvals.set(((((1, 1),), (1, rid))), rng.choice(["approve", "veto"]))
    return b


def merged(a, b):
    x = a.snapshot()
    x.merge(b)
    return x


def test_merge_laws():
    rng = random.Random(0)
    for _ in range(300):
        a, b, c = (rand_state(i, rng) for i in range(3))
        assert merged(a, b).canonical() == merged(b, a).canonical()
        assert merged(merged(a, b), c).canonical() == merged(a, merged(b, c)).canonical()
        assert merged(a, a).canonical() == a.canonical()


def test_merge_is_monotone_and_reports_change():
    a = BeliefState(0, DEPOTS, [])
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b = BeliefState(1, DEPOTS, [])
    assert b.merge(a) is True
    assert b.merge(a) is False
    assert len(b.records.records()) == 1


def test_records_union_is_idempotent():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b.add_record(RentRecord(1, 0, (0, 0), (0, 3), 0, 7))
    a.merge(b)
    a.merge(b)
    assert [r.rent for r in a.records.records()] == [5, 7]
    assert a.census.items() == frozenset({0, 1})


def test_snapshot_is_isolated():
    a = BeliefState(0, DEPOTS, [])
    s = a.snapshot()
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    assert s.records.records() == []


def test_status_lattice_and_latest_observation_wins():
    b = BeliefState(0, DEPOTS, [])
    b.add_obstruction(obs(), 5)
    assert b.status((1, 1)) == "reported" and (1, 1) in b.believed_blocked()
    b.observe_cell((1, 1), False, 8)
    assert b.status((1, 1)) == "refuted" and (1, 1) not in b.believed_blocked()
    b.observe_cell((1, 1), True, 20)
    assert b.status((1, 1)) == "confirmed"
    b.filled.add((1, 1))
    assert b.status((1, 1)) == "filled" and (1, 1) not in b.believed_blocked()
    x, y = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    x.observe_cell((3, 3), True, 3)
    y.observe_cell((3, 3), False, 7)
    x.merge(y)
    assert x.status((3, 3)) == "refuted"


def test_static_pits_start_confirmed_and_class_lattice():
    b = BeliefState(0, DEPOTS, [(2, 2)])
    assert b.status((2, 2)) == "confirmed" and b.editable() == ((2, 2),)
    b.add_obstruction(obs(cells=((1, 1),), cls="robot_clearable"), 1)
    assert b.editable() == ((1, 1), (2, 2)) and b.hard_blocked() == frozenset()
    b.mark_needs_human((1, 1))
    assert b.editable() == ((2, 2),) and b.hard_blocked() == frozenset({(1, 1)})


def test_obstruction_same_report_smaller_node_wins():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    a.add_obstruction(obs(node=4, cls="needs_human"), 1)
    b.add_obstruction(obs(node=2, cls="robot_clearable"), 1)
    a.merge(b)
    assert a.obstructions.get("r0").node == 2


def test_approvals_veto_dominates():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    key = (((1, 1),), (3, 0))
    a.approvals.set(key, "approve")
    b.approvals.set(key, "veto")
    a.merge(b)
    assert a.approvals.get(key) == "veto"


def test_claim_effective_smallest_live_ticket():
    cs = ClaimSet()
    cs.issue((5, 2), Claim(((1, 1),), 2), expiry=10)
    cs.issue((4, 7), Claim(((1, 1),), 7), expiry=10)
    assert cs.effective((1, 1), 3)[0] == (4, 7)
    assert cs.effective((1, 1), 10) is None
    cs.renew((5, 2), 20)
    assert cs.effective((1, 1), 15)[0] == (5, 2)


def test_stock_remaining_and_mark_empty():
    b = BeliefState(0, {(0, 0): 2}, [])
    b.stock.take((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 1
    b.stock.mark_empty((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 0
    b.stock.give_back((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 1
```

`tests/doi/test_doi_evidence.py`:

```python
from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord, ObstructionRecord
from src.doi.evidence import EvidenceEngine
from src.doi.scenarios import build_scenario


def setup(name):
    s = build_scenario(SimConfig(scenario=name, n_robots=1, tasks_per_robot=1))
    return s, BeliefState(0, s.depots, s.pits), EvidenceEngine(s.grid, 144.0)


def test_series_residual_after_fill():
    s, b, e = setup("series_pits")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 15), 0, 18))
    assert e.evidence(b, 0) == {((3, 11), (3, 13)): 18.0}
    b.filled.add((3, 13))
    assert e.evidence(b, 0) == {((3, 11),): 18.0}


def test_substitute_fill_invalidates_evidence():
    s, b, e = setup("two_pits_parallel")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    b.add_record(RentRecord(0, 1, (5, 9), (5, 11), 0, 14))
    assert e.evidence(b, 0) == {((3, 10),): 18.0, ((5, 10),): 14.0}
    b.filled.add((3, 10))
    assert e.evidence(b, 0) == {((5, 10),): 4.0}


def test_per_pit_series_is_zero():
    s, b, e = setup("series_pits")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 15), 0, 18))
    assert e.evidence(b, 0, per_pit=True) == {}


def test_window_drops_old_records():
    s, b, e = setup("single_pit")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    assert e.evidence(b, 100, window=50) == {}
    assert e.evidence(b, 40, window=50) == {((3, 10),): 18.0}


def test_extrapolated_evidence():
    s, b, e = setup("single_pit")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    b.census.add(1)
    b.census.add(2)
    assert e.evidence(b, 0, extrapolate=True) == {((3, 10),): 18.0 * 3 / 1}


def test_needs_human_cell_gets_no_evidence():
    s, b, e = setup("incidents_room")
    b.add_obstruction(ObstructionRecord("r0", 0, "north door", ((2, 10),), "spill",
                                        "robot_clearable", 1, 0.9, "x"), 0)
    b.observe_cell((2, 10), True, 1)
    b.add_record(RentRecord(0, 0, (2, 9), (2, 11), 1, 10))
    assert e.evidence(b, 1) == {((2, 10),): 10.0}
    b.mark_needs_human((2, 10))
    assert e.evidence(b, 1) == {}
```

Numbers: `two_pits_parallel` record 2 goes `(5,9) -> (5,11)`: blocked detour via the door is 7 + 2 + 7 = 16, open via `(5,10)` is 2, rent 14. After `(3,10)` is filled, the blocked path goes through it (2 up, 2 across, 2 down = 6), so the remaining value of `(5,10)` is 6 - 2 = 4, and record 1 has an empty residual.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): record ledger, cell-status registers and residual evidence"`

---

### Task 6: Network

**Files:**
- Create: `src/doi/network.py`, `tests/doi/test_doi_network.py`

**Interfaces:**
- Consumes: `u01`, `SimConfig`.
- Produces:

```python
@dataclass(frozen=True)
class Message:
    sender: int
    kind: str            # "STATE" or "INTENT"
    payload: Any
    units: int
    sent_at: int

class Network:
    def __init__(self, cfg: SimConfig)
    def send(self, msg: Message, t: int, positions: Dict[int, Pos]) -> None
    def deliver(self, t: int) -> Dict[int, List[Message]]    # receiver id -> messages due at tick t, sorted by (sent_at, sender)
    stats: NetStats    # broadcasts:int, transmissions:int, dropped:int, units:int
```

**Rules.** Two channels (spec 2.4): `INTENT` uses the traffic channel (`r_traffic`, `loss_traffic`, latency 1); `STATE` uses the ledger channel (`r_comm`, `loss`, `latency`).
- `send`: `positions` maps robot id to cell at tick `t`. Receivers are all ids `j != msg.sender` in `positions` with `manhattan(positions[sender], positions[j]) <= radius` of the message's channel (infinity allowed; radius 0 means no receivers because distinct robots never share a cell). `broadcasts += 1`.
- For each receiver, `transmissions += 1`, `units += msg.units`. The copy is dropped when `u01(cfg.seed, t, msg.sender, receiver, KIND_ID[msg.kind]) < channel_loss` with `KIND_ID = {"STATE": 1, "INTENT": 2}`; then `dropped += 1`. Otherwise it is queued for delivery at tick `t + channel_latency`.
- `NetStats` is kept per channel: `stats` is the ledger channel, `traffic_stats` the traffic channel. Experiments report ledger-channel costs (`message_units`) separately from traffic-channel costs.
- `deliver(t)` pops and returns the queue for tick `t`.

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.network import Message, Network

POS = {0: (0, 0), 1: (0, 5), 2: (0, 20)}


def msg(sender=0, kind="STATE"):
    return Message(sender, kind, {"x": 1}, 3, 0)


def test_range_and_latency():
    n = Network(SimConfig(r_comm=8.0, latency=2))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {}
    out = n.deliver(2)
    assert list(out) == [1] and out[1][0].sender == 0


def test_infinite_range_reaches_all():
    n = Network(SimConfig(r_comm=float("inf"), latency=1))
    n.send(msg(), 0, POS)
    assert sorted(n.deliver(1)) == [1, 2]


def test_zero_range_reaches_none():
    n = Network(SimConfig(r_comm=0.0))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {} and n.stats.transmissions == 0


def test_loss_rate_and_determinism():
    cfg = SimConfig(r_comm=float("inf"), loss=0.3, seed=9)
    n = Network(cfg)
    for t in range(5000):
        n.send(msg(), t, {0: (0, 0), 1: (0, 1)})
    assert abs(n.stats.dropped / n.stats.transmissions - 0.3) < 0.02
    m = Network(cfg)
    for t in range(5000):
        m.send(msg(), t, {0: (0, 0), 1: (0, 1)})
    assert m.stats.dropped == n.stats.dropped


def test_full_loss_equals_no_comm():
    n = Network(SimConfig(r_comm=float("inf"), loss=1.0))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {} and n.stats.dropped == n.stats.transmissions == 2


def test_stats_units():
    n = Network(SimConfig(r_comm=float("inf")))
    n.send(msg(), 0, POS)
    assert n.stats.broadcasts == 1 and n.stats.units == 6


def test_intent_uses_traffic_channel():
    n = Network(SimConfig(r_comm=0.0, loss=1.0, r_traffic=8.0, loss_traffic=0.0, latency=3))
    n.send(msg(kind="INTENT"), 0, POS)
    out = n.deliver(1)
    assert list(out) == [1] and out[1][0].kind == "INTENT"
    assert n.stats.transmissions == 0 and n.traffic_stats.transmissions == 1
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): lossy range-limited network"`

---

### Task 7: World physics and arbitration

**Files:**
- Create: `src/doi/world.py`, `tests/doi/test_doi_world.py`

**Interfaces:**
- Consumes: `Scenario`, `SimConfig`.
- Produces:

```python
@dataclass(frozen=True)
class Move: to: Pos
@dataclass(frozen=True)
class Wait: pass
@dataclass(frozen=True)
class Pickup: pass
@dataclass(frozen=True)
class Drop: pit: Pos
@dataclass(frozen=True)
class Return: pass
Action = Union[Move, Wait, Pickup, Drop, Return]

@dataclass(frozen=True)
class ActionResult:
    ok: bool
    reason: str            # "" when ok; else one of vertex, occupied, swap, cycle, wall, empty, not_adjacent, already_filled, not_blocked, needs_human, no_bag, not_depot, has_bag, lost_priority
    pos: Pos               # position after this tick
    blocked_ticks: int     # consecutive ticks this robot's Move was blocked, after this tick
    carrying: bool

class Observation:         # returned by World.observe
    filled_pits: FrozenSet[Pos]     # filled cells (pits or cleared obstructions) within r_sense
    robot_cells: FrozenSet[Pos]     # cells of other robots within r_sense
    blocked_cells: FrozenSet[Pos]   # currently blocked editable cells within r_sense (unfilled pits, appeared obstructions)
    scanned: FrozenSet[Pos]         # every in-bounds non-OBSTACLE cell within r_sense

class World:
    def __init__(self, scenario: Scenario, cfg: SimConfig)
    pos: Dict[int, Pos]; carrying: Dict[int, bool]; stock: Dict[Pos, int]
    filled: Set[Pos]; filled_at: Dict[Pos, int]
    blocked: Set[Pos]                   # unfilled static pits plus appeared, unfilled obstruction cells
    appeared_at: Dict[int, int]         # incident oid -> tick it appeared
    overrides: int                      # movers removed by vertex/occupied/swap/cycle (not wall)
    grid: Grid                          # mutable copy: filled pits become FREE; appeared obstructions become PIT
    trajectory: Dict[int, List[Pos]]    # index = ticks since tick 0
    counters: per-robot dict with keys moves, carried_steps, waits, pickups, drops, rejected_drops, wrong_class_attempts
    fills: int
    def prefill(self, pits: Iterable[Pos]) -> None       # before tick 0; no counters change; incident cells in the set are suppressed (never appear)
    def begin_tick(self, t: int) -> None                 # incidents appear (rule 0); the runner calls it before sensing
    def observe(self, robot_id: int, r_sense: int) -> Observation
    def apply_actions(self, t: int, actions: Dict[int, Action]) -> Dict[int, ActionResult]
    def despawn(self, robot_id: int) -> None
    def active_ids(self) -> List[int]
```

**Rules for `apply_actions`.**
0. **Incidents (in `begin_tick(t)`, not in `apply_actions`):** for each incident (oid order) with `appear_tick <= t`, not yet appeared and not suppressed: if none of its cells holds a robot, set each cell to `PIT` in `grid`, add it to `blocked`, record `appeared_at[oid] = t`; otherwise retry next tick. Running this before sensing means a robot never sees a cell as free in the same tick it becomes blocked.
1. Priority key of a robot: `(-blocked_ticks[i], i)`; smaller key wins.
2. Only `Move` actions are arbitrated. Illegal move (target not 4-adjacent, or target not passable in the world's truth grid) -> `ok=False`, reason `wall`.
3. Candidate set `C` = legal movers. Repeat until no change:
   a. **Vertex:** for a cell targeted by two or more members of `C`, keep the highest priority, remove the others (reason `vertex`).
   b. **Occupied:** a mover whose target cell holds a robot that is NOT in `C` (waiting, other action, or removed) is removed (reason `occupied`).
   c. **Swap:** if `i -> pos[j]` and `j -> pos[i]` are both in `C`, remove both (reason `swap`).
   d. **Cycle:** find cycles in the graph "i wants the cell where j stands, j in C"; remove every member of every cycle of length >= 3 (reason `cycle`). Length-2 cycles are swaps.
   Following (moving into a cell vacated this tick by a robot in `C`) is allowed.
   Every mover removed in a, b, c or d adds 1 to `overrides` (the L0 intervention count, spec 3).
4. Surviving moves execute. Counters: `moves += 1`; `carried_steps += 1` if the robot was carrying.
5. `Wait`, removed movers, and robots not in `actions` count as waiting: `waits += 1` for any robot present in `actions`. Blocked ticks: for removed movers `blocked_ticks += 1`; for every other robot reset to 0.
6. **Pickup:** ok iff robot not carrying, its cell is a depot with `stock > 0`; else reason `has_bag`, `not_depot` or `empty`. On success `stock -= 1`, `carrying = True`, `pickups += 1`.
7. **Drop(pit):** ok iff carrying, `pit in blocked`, 4-adjacent to the robot, and not a `needs_human` incident cell. Several drops on one pit in a tick: the best priority key wins; losers get `lost_priority` and keep the bag. A drop on an already filled cell gets `already_filled`; on a cell that is not blocked and never was filled, `not_blocked` (a false report); on a `needs_human` cell, `needs_human` (bag kept, `wrong_class_attempts += 1`); `not_adjacent` and `no_bag` likewise. On success: `grid.set(pit, FREE)`, `blocked.discard(pit)`, `filled.add(pit)`, `filled_at[pit] = t`, `carrying = False`, `drops += 1`, `fills += 1`. Rejected drops increment `rejected_drops`.
8. **Return:** ok iff carrying and on a depot cell; `stock += 1`, `carrying = False`.
9. Order inside a tick: arbitrate and apply moves first, then Pickup/Drop/Return evaluated against positions after moves (a robot has only one action, so these robots did not move).
10. `trajectory[i].append(pos[i])` for every active robot each tick, after the tick's actions. Tick 0's pre-action position is `trajectory[i][0]` appended at construction, so `len(trajectory[i]) == ticks_played + 1`.
11. `observe`: within Manhattan `r_sense` of the robot: filled cells, other robots' cells, cells in `blocked`, and `scanned` (all in-bounds non-OBSTACLE cells). `blocked` at construction holds the static pits.
12. `despawn` removes the robot from `pos`; its trajectory is kept.
13. When `cfg.debug_checks` is True, after every tick assert: all active positions distinct, and no pair swapped cells during the tick.

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.scenarios import scenario_from_ascii
from src.doi.world import World, Move, Wait, Pickup, Drop, Return

ROWS = ["...#...", "...P..D", "...#...", "...#..."]


def world(starts, rows=ROWS, **kw):
    s = scenario_from_ascii(rows, starts, [[(0, 0)] for _ in starts], depot_stock=2)
    return World(s, SimConfig(n_robots=len(starts), debug_checks=True, **kw))


def test_vertex_lowest_id_wins_when_equal():
    w = world([(0, 0), (0, 2)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})
    assert r[0].ok and not r[1].ok and r[1].reason == "vertex"
    assert w.pos[0] == (0, 1) and w.pos[1] == (0, 2)
    assert r[1].blocked_ticks == 1


def test_aged_priority_overrides_id():
    w = world([(0, 0), (0, 2)], ["..."])
    w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})   # robot 1 loses, blocked_ticks becomes 1
    w.pos[0] = (0, 0)                                        # test-only teleport back
    w.pos[1] = (0, 2)
    r = w.apply_actions(1, {0: Move((0, 1)), 1: Move((0, 1))})
    assert r[1].ok and not r[0].ok                           # robot 1 now outranks robot 0


def test_swap_blocked_both():
    w = world([(0, 0), (0, 1)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 0))})
    assert not r[0].ok and not r[1].ok and r[0].reason == "swap"
    assert w.pos == {0: (0, 0), 1: (0, 1)}


def test_following_allowed_train_of_three():
    w = world([(0, 0), (0, 1), (0, 2)], ["....."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 2)), 2: Move((0, 3))})
    assert all(x.ok for x in r.values())
    assert w.pos == {0: (0, 1), 1: (0, 2), 2: (0, 3)}


def test_occupied_by_waiting_robot():
    w = world([(0, 0), (0, 1)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Wait()})
    assert not r[0].ok and r[0].reason == "occupied"


def test_rotation_cycle_blocked():
    rows = ["..", ".."]
    w = world([(0, 0), (0, 1), (1, 1), (1, 0)], rows)
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((1, 1)), 2: Move((1, 0)), 3: Move((0, 0))})
    assert not any(x.ok for x in r.values()) and r[0].reason == "cycle"


def test_wall_and_pit_are_illegal():
    w = world([(1, 2)])
    assert w.apply_actions(0, {0: Move((1, 3))})[0].reason == "wall"


def test_pickup_drop_cycle():
    w = world([(1, 6)])
    assert w.apply_actions(0, {0: Pickup()})[0].ok and w.stock[(1, 6)] == 1
    for t, c in enumerate([(1, 5), (1, 4)], start=1):
        assert w.apply_actions(t, {0: Move(c)})[0].ok
    assert w.counters[0]["carried_steps"] == 2
    r = w.apply_actions(3, {0: Drop((1, 3))})[0]
    assert r.ok and (1, 3) in w.filled and w.grid.get(1, 3).name == "FREE" and w.fills == 1
    assert w.apply_actions(4, {0: Move((1, 3))})[0].ok


def test_double_drop_one_fill():
    w = world([(1, 6), (1, 5)])
    w.pos[0], w.pos[1] = (1, 4), (1, 2)
    w.carrying[0] = w.carrying[1] = True
    r = w.apply_actions(0, {0: Drop((1, 3)), 1: Drop((1, 3))})
    assert [r[0].ok, r[1].ok] == [True, False] and r[1].reason == "lost_priority"
    assert w.fills == 1 and w.carrying[1] is True
    r2 = w.apply_actions(1, {1: Drop((1, 3))})
    assert not r2[1].ok and r2[1].reason == "already_filled"


def test_return_and_pickup_rejections():
    w = world([(1, 6), (0, 0)])
    assert w.apply_actions(0, {1: Pickup()})[1].reason == "not_depot"
    w.apply_actions(1, {0: Pickup()})
    assert w.apply_actions(2, {0: Pickup()})[0].reason == "has_bag"
    assert w.apply_actions(3, {0: Return()})[0].ok and w.stock[(1, 6)] == 2


def test_observe_and_prefill():
    w = world([(1, 4)])
    w.prefill([(1, 3)])
    ob = w.observe(0, 2)
    assert (1, 3) in ob.filled_pits
    assert w.observe(0, 0).filled_pits == frozenset()


def incident_world(starts, appear=0, cls="robot_clearable"):
    from src.doi.scenarios import Incident
    inc = [Incident(0, ((1, 3),), appear, "pallet", cls)]
    s = scenario_from_ascii(["......D"] * 3, starts, [[(0, 0)] for _ in starts], depot_stock=2,
                            incidents=inc)
    return World(s, SimConfig(n_robots=len(starts), debug_checks=True))


def test_incident_appears_and_blocks():
    w = incident_world([(1, 2)], appear=2)
    w.begin_tick(0)
    assert w.apply_actions(0, {0: Move((1, 3))})[0].ok
    w.begin_tick(1)
    assert w.apply_actions(1, {0: Move((1, 2))})[0].ok
    w.begin_tick(2)
    assert w.appeared_at[0] == 2 and (1, 3) in w.observe(0, 2).blocked_cells
    assert w.apply_actions(2, {0: Move((1, 3))})[0].reason == "wall"


def test_incident_waits_for_free_cell():
    w = incident_world([(1, 3)], appear=0)
    w.begin_tick(0)
    assert 0 not in w.appeared_at
    w.apply_actions(0, {0: Move((1, 4))})
    w.begin_tick(1)
    assert w.appeared_at[0] == 1


def test_needs_human_drop_rejected():
    w = incident_world([(1, 4)], appear=0, cls="needs_human")
    w.begin_tick(0)
    w.carrying[0] = True
    r = w.apply_actions(0, {0: Drop((1, 3))})[0]
    assert r.reason == "needs_human" and w.carrying[0] and w.fills == 0
    assert w.counters[0]["wrong_class_attempts"] == 1


def test_drop_on_clear_cell_rejected():
    w = incident_world([(1, 4)], appear=50)
    w.begin_tick(0)
    w.carrying[0] = True
    assert w.apply_actions(0, {0: Drop((1, 3))})[0].reason == "not_blocked"


def test_arbiter_overrides_counted():
    w = world([(0, 0), (0, 2)], ["..."])
    w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})
    assert w.overrides == 1
    w.apply_actions(1, {0: Move((0, 9))})
    assert w.overrides == 1


def test_random_fuzz_never_collides():
    import random
    rng = random.Random(1)
    rows = ["........"] * 8
    starts = [(r, c) for r, c in [(0, 0), (0, 7), (7, 0), (7, 7), (3, 3), (4, 4), (2, 5), (5, 2)]]
    w = world(starts, rows)
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for t in range(300):
        acts = {}
        for i, p in w.pos.items():
            dr, dc = rng.choice(dirs)
            acts[i] = Move((p[0] + dr, p[1] + dc)) if rng.random() < 0.9 else Wait()
        w.apply_actions(t, acts)
        assert len(set(w.pos.values())) == len(w.pos)
```

Note: `World.blocked_ticks: Dict[int, int]` is a public attribute and persists across ticks. `test_aged_priority_overrides_id` teleports positions directly through `w.pos`, which is allowed in tests only.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): world physics, arbitration and sensing"`

---

### Task 8: Local space-time planner

**Files:**
- Create: `src/doi/spacetime.py`, `tests/doi/test_doi_spacetime.py`

**Interfaces:**
- Consumes: `PassFn`.
- Produces:

```python
def plan_spacetime(passable: PassFn, start: Pos, goal: Pos, t0: int,
                   reserved: Set[Tuple[Pos, int]], h: Dict[Pos, int],
                   max_len: int) -> Optional[List[Pos]]
```

**Rules.**
- Returns a list `path` where `path[k]` is the cell at tick `t0 + k`, `path[0] == start`, ending when `path[-1] == goal`. Waiting is a repeated cell.
- Search over `(cell, k)`; successors: the 4 neighbours (passable) and waiting. A transition `pos -> nb` at step `k -> k+1` is rejected when `(nb, t0+k+1) in reserved` or `(pos, t0+k+1) in reserved` for `nb != pos` (the repository's conservative swap/follow rule), and a wait is rejected when `(pos, t0+k+1) in reserved`.
- Cost 1 per step including waits. Heuristic `h[cell]` (true BFS distance to the goal; cells missing from `h` are unreachable and pruned). Heap key `(g + h, g, counter)`.
- Reservations only exist up to some tick; beyond it they impose nothing. Search is cut at `k > max_len`; return `None` if the goal is not reached.
- `start == goal` returns `[start]`. If `(start, t0)` itself is in `reserved` that is ignored (the robot is already there).

- [ ] **Step 1: Write failing tests**

```python
from src.doi.paths import passable_fn, bfs_dist_map
from src.doi.scenarios import scenario_from_ascii
from src.doi.spacetime import plan_spacetime


def setup(rows):
    s = scenario_from_ascii(rows, [(0, 0)], [[(0, 0)]])
    pf = passable_fn(s.grid, frozenset())
    return pf, len(rows), len(rows[0])


def hmap(pf, goal, H, W):
    return bfs_dist_map(pf, goal, H, W)


def test_free_path_matches_shortest():
    pf, H, W = setup(["....", "...."])
    p = plan_spacetime(pf, (0, 0), (1, 3), 5, set(), hmap(pf, (1, 3), H, W), 30)
    assert p[0] == (0, 0) and p[-1] == (1, 3) and len(p) == 5


def test_waits_for_reservation():
    pf, H, W = setup(["..."])
    reserved = {((0, 1), 1)}
    p = plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 30)
    assert p == [(0, 0), (0, 0), (0, 1), (0, 2)]


def test_routes_around_reserved_cell():
    pf, H, W = setup(["...", "..."])
    reserved = {((0, 1), 1), ((0, 1), 2)}
    p = plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 30)
    assert len(p) == 5 and (0, 1) not in p[1:3]


def test_swap_is_rejected():
    pf, H, W = setup(["..."])
    reserved = {((0, 0), 1), ((0, 1), 0)}
    p = plan_spacetime(pf, (0, 1), (0, 0), 0, reserved, hmap(pf, (0, 0), H, W), 6)
    assert p is None or p[1] != (0, 0)


def test_unreachable_returns_none():
    pf, H, W = setup(["..#.."])
    assert plan_spacetime(pf, (0, 0), (0, 4), 0, set(), {(0, 4): 0, (0, 3): 1}, 20) is None
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): local space-time A*"`

---

### Task 9: Agent core, runner skeleton and the NeverFill arm

**Files:**
- Create: `src/doi/agent.py`, `src/doi/metrics.py`, `src/doi/runner.py`, `src/doi/policies.py` (contains only `FillPolicy` base and `NeverFillPolicy` for now), `tests/doi/test_doi_agent_core.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 8.
- Produces:

```python
@dataclass(frozen=True)
class TaskPlanInfo:
    robot: int
    task_idx: int
    tick: int
    start: Pos
    goal: Pos
    d_block: int
    d_open: int
    bundle: Tuple[Pos, ...]          # sorted, residual after belief.filled
    rent: int
    belief_filled: FrozenSet[Pos]
    belief_blocked: FrozenSet[Pos]   # belief.believed_blocked() at planning time
    hard_blocked: FrozenSet[Pos]
    single_rents: Dict[Pos, int]     # only filled when policy.needs_single_rents else {}

class FillPolicy(ABC):
    name: str
    needs_single_rents: bool = False
    def on_task_planned(self, agent: "RobotAgent", info: TaskPlanInfo, t: int) -> None
    def propose(self, agent: "RobotAgent", t: int) -> Optional["HaulProposal"]
    def prefill_set(self, scenario, cfg) -> FrozenSet[Pos]     # default frozenset()

@dataclass(frozen=True)
class HaulProposal:
    pits: Tuple[Pos, ...]
    evidence: float
    buy: float
    buy_per_pit: Optional[Tuple[float, ...]] = None   # None: split `buy` evenly over pits

class RobotAgent:
    id: int; belief: BeliefState; pos: Pos; carrying: bool
    task_idx: int; finished: bool; completed_tick: Optional[int]
    plan_infos: List[TaskPlanInfo]
    stats: dict   # replans, rent_counted
    def __init__(self, rid, scenario, cfg, policy, shared)
    def sense(self, obs: Observation, t: int) -> None
    def receive(self, msgs: List[Message], t: int) -> None
    def decide(self, t: int) -> Action
    def outgoing(self, t: int) -> List[Message]
    def after_action(self, res: ActionResult, t: int) -> None

@dataclass
class RunResult: ...   # see metrics step below

def run_episode(cfg: SimConfig, scenario: Optional[Scenario] = None,
                policy: Optional[FillPolicy] = None) -> RunResult
```

`shared` is a plain dataclass `Shared` defined in `policies.py`, one instance per run. Fields now: `triggers: List[dict]` (default empty list). Task 11 adds the fields RoF and `central` need (`engine`, `global_belief`, `agents`). `FillPolicy` also has the class attribute `uses_gossip: bool = True` and the hook `prepare(self, scenario, cfg, shared) -> None` (default no-op); `HaulProposal` is defined in `policies.py`.

**Rules for `RobotAgent`** (this task has no hauling):
1. **Static knowledge:** the agent keeps its own static copy of `scenario.grid`, `scenario.pits`, `scenario.depots`, and `BeliefState(id, depots, pits)`. It never reads `World`, `scenario.incidents` or `scenario.reports`.
2. **`sense(obs, t)`:** add every cell in `obs.filled_pits` to `belief.filled`; `belief.observe_cell(c, True, t)` for every `c` in `obs.blocked_cells`; `belief.observe_cell(c, False, t)` for every `c` in `obs.scanned - obs.blocked_cells` whose status is `reported` or `confirmed` (only cells that carry a status are written, so the registers do not grow with every visited cell); store `obs.robot_cells` as `self.sensed_cells`. A change in `belief.believed_blocked()` or `belief.filled` sets `replan_needed`.
3. **`receive(msgs, t)`:** for `STATE` messages call `belief.merge(msg.payload)`; for `INTENT` messages store `payload` as `self.intents[sender] = IntentRecord(start_tick, blocked_ticks, cells)`, replacing older ones. Any change in `belief.filled` sets `self.replan_needed = True`.
4. **Tasks:** the agent works on `scenario.tasks[id][task_idx]`. When it becomes current (tick of arrival at the previous goal plus one, or tick 0), compute `dream_path(static_grid, belief.editable(), filled, pos, goal, U, belief.hard_blocked())`, record exactly one `TaskPlanInfo`, call `policy.on_task_planned(self, info, t)`, then plan the first path. The agent itself never writes to the ledger; only policies do (as `RentRecord(id, task_idx, pos, goal, t, rent)`). Rent is counted once per task (a task resumed after a haul or a replan is not counted again).
5. **Path planning:** `pf = passable_fn(static_grid, filled, closed=belief.believed_blocked())`; `h = bfs_dist_map(pf, goal, H, W)`. If the goal is unreachable under belief, the plan is `[pos]` (wait) and the agent replans every tick; this is how a robot behaves behind a believed-blocked door. Reservations: for each stored intent from a sender with a better priority key `(-blocked_ticks, sender)` than own `(-own_blocked_ticks, id)` and age `t - start_tick <= intent_window`, reserve `(cell, start_tick + k)` for all cells of the intent with tick `>= t`; for each cell in `sensed_cells` not belonging to a sender with a fresh intent, reserve `(cell, t + 1)`. Call `plan_spacetime(..., t0=t, max_len=3*(H+W))`. If it returns `None`, the plan is `[pos]` (wait). The agent stores `self.plan` (cells from tick `t`).
6. **Replan triggers** (check at the start of `decide`): new task; `replan_needed` flag; the last action was blocked for 2 consecutive ticks; a stored better-priority intent reserves a cell of the next `intent_window` cells of the current plan at the same tick.
7. **`decide(t)`:** returns `Move(next_cell)` if `plan[1] != plan[0]`, else `Wait()`. After the move executes (`after_action`), advance `plan`.
8. **`after_action(res, t)`:** update `pos`, `blocked_ticks`; if `pos == goal`, `task_idx += 1`, `replan_needed = True` (next task plans next tick); if `task_idx == K` set `finished = True`, `completed_tick = t`. If the move was not ok keep the plan and set a `blocked_streak`.
9. **`outgoing(t)`:** `INTENT` every tick: payload `(t, blocked_ticks, tuple(plan[:intent_window+1]))`, `units = len(cells)`; `STATE` when `t % gossip_period == 0`: payload `belief.snapshot()`, `units = belief.units()` (only if `policy.uses_gossip`; the `never` policy sends no STATE).
10. Agents are processed in id order every tick. `finished` agents are not processed.

**Rules for `run_episode`:**

```
scenario = scenario or build_scenario(cfg); policy = policy or make_policy(cfg)
world = World(scenario, cfg); world.prefill(policy.prefill_set(scenario, cfg))
network = Network(cfg); agents = [RobotAgent(i, scenario, cfg, policy, shared) for i in range(n)]
for t in range(cfg.max_ticks):
    active = [a for a in agents if not a.finished]
    if not active: break
    world.begin_tick(t)
    (Task 15 inserts report delivery, intake and supervisor decisions here)
    for a in active: a.sense(world.observe(a.id, cfg.r_sense), t)
    inbox = network.deliver(t)
    for a in active: a.receive(inbox.get(a.id, []), t)
    actions = {a.id: a.decide(t) for a in active}
    for a in active:
        for m in a.outgoing(t): network.send(m, t, dict(world.pos))
    results = world.apply_actions(t, actions)
    for a in active: a.after_action(results[a.id], t)
    for a in active: if a.finished: world.despawn(a.id)
    progress tracking: tick counts as progress if any ok Move/Pickup/Drop/Return or any task completion
    if no progress for cfg.stall_ticks consecutive ticks: stalled = True; break
```

Messages are sent using positions at the START of the tick (before `apply_actions`), which is what the call above does.

**`RunResult`** (dataclass in `metrics.py`) must contain: `cfg: dict`, `policy: str`, `J: float`, `J_censored: float`, `delay: float`, `throughput: float` (tasks completed per 1000 ticks), `moves`, `waits`, `carried_steps`, `fills`, `fee_total`, `unfinished_tasks`, `stalled: bool`, `ticks: int`, `messages: dict` (ledger-channel stats), `traffic_messages: dict` (traffic-channel stats), `overrides: int` (from `World.overrides`), `fill_ticks: List[int]`, `triggers: List[dict]`, `edits: List[dict]` (one per completed fill: `pit`, `trigger_tick`, `claim_tick`, `fill_tick`, `B_est`, `B_real`, `approval_wait`; empty until Task 10), `plan_infos: List[TaskPlanInfo]`, `filled_at: Dict[Pos,int]`, `appeared_at: Dict[int,int]`, `wasted_haul_cost: float`, `wrong_class_attempts: int`, `unconfirmed_hauls: int`, `false_report_hauls: int`, `claims: Dict[str, int]` (keys `issued`, `lost`, `aborts`; all 0 until Task 10), `approvals: Dict[str, int]` (keys `requested`, `approved`, `vetoed`, `timeout_approved`, `deferred`; all 0 until Task 15), `intake: Dict[str, int]` (keys `reports`, `records`, `rejected`; all 0 until Task 15), `final_stock: Dict[Pos, int]` (world depot stock at the end), `hindsight_buy: float` (0.0 until Task 11), `runtime_ms: float`, `trajectory: Dict[int, List[Pos]]`. Computation:
- `moves`, `waits`, `carried_steps` summed over robots from `world.counters` (`moves` counts all successful moves including carried ones).
- `J = (moves - carried_steps) + waits + kappa * carried_steps + fee * fills`.
- `unfinished_tasks = sum(K - task_idx_i)`; `J_censored = J + U * unfinished_tasks`.
- `delay = sum(completed_tick_i + 1 for finished robots) + max_ticks * (number unfinished robots)`.
- `throughput = 1000 * (total tasks completed) / max(1, ticks)`.
- `U = cfg.unreachable_cost_for(H, W)`.

`make_policy(cfg) -> FillPolicy` is defined in `src/doi/policies.py` and imported by `runner.py`; it returns `NeverFillPolicy()` for `"never"`; other names raise `NotImplementedError` until Task 11.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_agent_core.py`

```python
from src.doi.config import SimConfig
from src.doi.scenarios import scenario_from_ascii, build_scenario
from src.doi.runner import run_episode

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]


def ascii_run(tasks, starts=None, **kw):
    starts = starts or [(1, 2)]
    s = scenario_from_ascii(ROWS, starts, tasks, depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=len(starts), tasks_per_robot=len(tasks[0]),
                    debug_checks=True, **kw)
    return run_episode(cfg, scenario=s)


def test_single_robot_pays_detour():
    r = ascii_run([[(1, 4), (1, 2), (1, 4), (1, 2)]])
    assert not r.stalled and r.unfinished_tasks == 0
    assert r.J == 40.0 and r.fills == 0 and r.waits == 0


def test_two_robots_finish_without_collision():
    tasks = [[(1, 4), (1, 2), (1, 4)], [(1, 2), (1, 4), (1, 2)]]
    r = ascii_run(tasks, starts=[(1, 2), (1, 4)])
    assert not r.stalled and r.unfinished_tasks == 0
    t0, t1 = r.trajectory[0], r.trajectory[1]
    for k in range(min(len(t0), len(t1))):
        assert t0[k] != t1[k]
        if k + 1 < min(len(t0), len(t1)):
            assert not (t0[k] == t1[k + 1] and t1[k] == t0[k + 1])


def test_plan_infos_record_rent_once_per_task():
    r = ascii_run([[(1, 4), (1, 2)]])
    assert [(i.task_idx, i.rent, i.bundle) for i in r.plan_infos] == [(0, 8, ((1, 3),)), (1, 8, ((1, 3),))]


def test_metrics_formula():
    r = ascii_run([[(1, 4)]])
    assert r.moves == 10 and r.J == 10.0 and r.J_censored == 10.0


def test_random_scenario_neverfill_completes():
    cfg = SimConfig(policy="never", scenario="multi_pit_wall", n_robots=6,
                    tasks_per_robot=4, seed=3, debug_checks=True)
    r = run_episode(cfg)
    assert r.unfinished_tasks == 0 and not r.stalled


def test_run_is_deterministic():
    cfg = SimConfig(policy="never", n_robots=6, tasks_per_robot=4, seed=5)
    a, b = run_episode(cfg), run_episode(cfg)
    assert (a.J, a.ticks, a.trajectory) == (b.J, b.ticks, b.trajectory)


def test_stale_belief_robot_never_planned_through_pit():
    r = ascii_run([[(1, 4)]])
    assert all(c != (1, 3) for c in r.trajectory[0])
```

(`test_other_robot_replans_when_fill_learned` from the Review Focus list is added in Task 10, because a fill needs the hauler.)

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Add a `--debug` smoke run** (not committed): run `run_episode(SimConfig(policy="never", n_robots=12, tasks_per_robot=20, seed=0))` for seeds 0 to 9 on `single_pit` and confirm none has `stalled=True`. If any stalls, the traffic layer is deadlocking: stop and report with the seed; do not change scenario geometry or the arbitration rules to hide it.
- [ ] **Step 6: Full suite green, commit** `git commit -m "feat(doi): agent core, runner and NeverFill baseline"`

---

### Task 10: Hauler, claims and forced hauling

**Files:**
- Create: `src/doi/hauler.py`
- Modify: `src/doi/agent.py` (hook the FSM into `decide`, `after_action`, `outgoing`), `src/doi/runner.py` (collect claim and haul stats)
- Test: `tests/doi/test_doi_hauler.py`

**Interfaces:**
- Consumes: `HaulProposal`, `BeliefState`, `ClaimSet`, `World` actions.
- Produces in `hauler.py`:

```python
class HaulState(Enum):
    NONE = 0; PENDING = 1; AWAIT_APPROVAL = 5; TO_DEPOT = 2; CARRY = 3; RETURN_BAG = 4

class Hauler:
    state: HaulState
    def __init__(self, agent)
    def offer(self, proposal: HaulProposal, t: int) -> None       # called by the agent when the policy proposes and the agent is idle
    def step(self, t: int) -> Optional[Action]                    # returns an action when it controls the robot this tick, else None
    def on_result(self, action: Action, res: ActionResult, t: int) -> None
    def active(self) -> bool                                       # state in TO_DEPOT, CARRY, RETURN_BAG
```

**Rules.** (Spec 4.5.)
- **Offer:** if `cfg.claim` is False the hauler goes straight to `TO_DEPOT` (no claim, no stagger). Otherwise it enters `PENDING` with `ready_tick = t + min(cfg.stagger_cap, floor(h / cfg.kappa))` where `h = d(pos, depot*) + kappa * d(depot*, pit_adj*)` using BFS distances on the agent's belief-passable grid with all pits in the proposal opened (lower bound), `depot*` the nearest depot with `belief.stock.remaining > 0`, `pit_adj*` the best cell adjacent to the first pit to haul (below). If no depot has stock, the offer is dropped.
- **PENDING each tick:** if any pit of the proposal is filled in belief, or `belief.claims.effective(pit, t)` exists for any pit and is not mine, cancel to `NONE`. If `t >= ready_tick`: **confirmation check (spec 5.4):** if any pit's `belief.status` is not `confirmed`, increment `unconfirmed_hauls` and cancel to `NONE` (policies filter for this, so the counter must stay 0; it is the test of the invariant). Otherwise `ticket = belief.next_ticket()`; if `cfg.gate` go to `AWAIT_APPROVAL` (implemented in Task 15; until then `gate=True` raises `NotImplementedError`), else `belief.claims.issue(ticket, Claim(pits, id), expiry=t + lease_ticks)`, remember `self.ticket`, go to `TO_DEPOT`. During `PENDING` the agent keeps following its task plan. With `cfg.claim` False the confirmation check runs at offer time instead.
- **Order of pits within the proposal:** the next pit to haul is, among unfilled pits of the proposal, the one that has a passable adjacent cell reachable from the chosen depot with only believed-filled pits open, minimising carry distance, ties by sorted pit. (This makes series corridors fill outer-first.) If none is reachable, cancel.
- **TO_DEPOT:** plan with the agent's space-time planner to the depot cell; on arrival issue `Pickup`. Success: `belief.stock.take(depot, id)`, go to `CARRY`. Failure `empty`: `belief.stock.mark_empty(depot, id)`; choose another depot or cancel.
- **CARRY:** plan to the adjacent cell chosen above; when adjacent to the target pit issue `Drop(pit)`. Success: `belief.filled.add(pit)`, mark proposal pit done, append an edit log entry (below); if pits remain, re-select the next pit and keep carrying toward the depot for another bag (go to `TO_DEPOT` for the next bag); else release (`state = NONE`, claim expiry set to `t` so it is no longer effective). Rejection `already_filled`, `lost_priority` or `not_blocked`: go to `RETURN_BAG` (for `not_blocked` also `belief.observe_cell(pit, False, t)`). Rejection `needs_human`: `belief.mark_needs_human(pit)`, go to `RETURN_BAG`; the world has already counted a `wrong_class_attempt`.
- **Realised cost (spec 4.4):** for every tick the hauler controls the robot from leaving `PENDING`/`AWAIT_APPROVAL` until the pit's `Drop` succeeds, add `kappa` if the tick's action was a successful carried move, else 1 (unloaded moves, waits, blocked ticks, pickup, drop). On success add `fee`; that sum is `B_real` for the pit. The edit entry is `{"pit", "trigger_tick" (offer tick), "claim_tick", "fill_tick", "B_est" (from `buy_per_pit`, or `buy / len(pits)`), "B_real", "approval_wait"}` (`approval_wait = 0` without the gate). For a multi-pit proposal the counter restarts after each successful drop.
- **Re-validation:** at the start of every tick of `TO_DEPOT` and `CARRY`: (a) renew own claim to `t + lease_ticks` when `claim` is on; (b) if the target pit is in `belief.filled` or an effective claim on it belongs to another robot with a smaller ticket than `self.ticket`: abort (`RETURN_BAG` if carrying, else `NONE` with claim released).
- **RETURN_BAG:** walk to the depot the bag came from, issue `Return`; success: `belief.stock.give_back`, `state = NONE`. Cost of this walk is counted by the world as `carried_steps` and in the runner as `wasted_haul_cost = kappa * (carried steps in RETURN_BAG + carried steps before the abort)`; the hauler keeps `self.wasted_steps`.
- While `active()` the agent does not advance its task; the task is resumed after release (replan on release).
- **Despawn:** a finished agent is removed; any claim it holds simply expires.
- `RobotAgent.decide` order: (1) if `hauler.active()` use `hauler.step`; (2) else plan/replan task; (3) `proposal = policy.propose(self, t)` if `hauler.state == NONE`, offer it; (4) if the hauler has just entered `TO_DEPOT` use `hauler.step` this tick; else follow the task plan.
- `outgoing`: claims travel inside the STATE snapshot; additionally, while a claim is held (`ticket` set and not released) a STATE message is sent every tick regardless of `gossip_period`.

For this task a test-only policy `ForcePolicy` (defined in the test file, subclass of `FillPolicy`) proposes the given pits once per listed robot. It is configured with `schedule: Dict[int, int]` (robot id to the earliest tick at which that robot proposes) and `min_task_idx: int` (the robot proposes only when `agent.task_idx >= min_task_idx`). `ForcePolicy.propose` is called by the agent only when its hauler is idle (`HaulState.NONE`), as for every policy.

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.policies import FillPolicy, HaulProposal
from src.doi.scenarios import scenario_from_ascii
from src.doi.runner import run_episode

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


class ForcePolicy(FillPolicy):
    name = "force"
    uses_gossip = True

    def __init__(self, pits, schedule, min_task_idx=0):
        self.pits = tuple(pits)
        self.schedule = dict(schedule)
        self.min_task_idx = min_task_idx
        self.fired = set()

    def on_task_planned(self, agent, info, t):
        pass

    def propose(self, agent, t):
        if (agent.id in self.schedule and agent.id not in self.fired
                and t >= self.schedule[agent.id] and agent.task_idx >= self.min_task_idx):
            self.fired.add(agent.id)
            return HaulProposal(self.pits, 100.0, 9.0)
        return None


def forced(tasks, schedule, claim, starts=None, min_task_idx=0):
    starts = starts or [(1, 2)]
    s = scenario_from_ascii(ROWS, starts, tasks, depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=len(starts), tasks_per_robot=len(tasks[0]),
                    claim=claim, debug_checks=True)
    return run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], schedule, min_task_idx))


def test_full_haul_exact_cost_without_claims():
    r = forced([FOUR], {0: 0}, claim=False, min_task_idx=1)
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.J == 29.0
    assert r.carried_steps == 2 and r.waits == 2
    e = r.edits[0]
    assert (e["pit"], e["B_est"], e["B_real"], e["approval_wait"]) == ((1, 3), 9.0, 13.0, 0)
    assert r.unconfirmed_hauls == 0


def test_robot_uses_pit_after_fill():
    r = forced([FOUR], {0: 0}, claim=False, min_task_idx=1)
    assert (1, 3) in r.trajectory[0][-6:]


def test_other_robot_replans_when_fill_learned():
    tasks = [[(1, 4), (1, 2), (1, 4), (1, 2)], [(1, 2), (1, 4), (1, 2), (1, 4)]]
    r = forced(tasks, {0: 0}, claim=False, starts=[(1, 2), (1, 4)], min_task_idx=1)
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert (1, 3) in r.trajectory[1][-6:]


def test_claim_issued_and_fill_completes():
    r = forced([FOUR], {0: 0}, claim=True, min_task_idx=1)
    assert r.claims["issued"] == 1 and r.fills == 1 and r.unfinished_tasks == 0


def test_two_haulers_one_fill_with_claims():
    s = scenario_from_ascii(ROWS, [(1, 5), (1, 4)], [[(1, 2)], [(1, 2)]], depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=2, tasks_per_robot=1, claim=True, debug_checks=True)
    r = run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], {0: 0, 1: 0}))
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.final_stock[(1, 6)] == 1


def test_abort_returns_bag():
    s = scenario_from_ascii(ROWS, [(1, 5), (1, 4)], [[(1, 2)], [(1, 2)]], depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=2, tasks_per_robot=1, claim=False, debug_checks=True)
    r = run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], {0: 0, 1: 3}))
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.final_stock[(1, 6)] == 1          # the second bag was put back
    assert r.wasted_haul_cost >= 0.0


def test_claim_lease_expires_and_reclaims():
    from src.doi.crdt import Claim, ClaimSet
    cs = ClaimSet()
    cs.issue((1, 0), Claim(((1, 3),), 0), expiry=8)
    assert cs.effective((1, 3), 7) is not None
    assert cs.effective((1, 3), 8) is None
    cs.issue((9, 1), Claim(((1, 3),), 1), expiry=20)
    assert cs.effective((1, 3), 9)[0] == (9, 1)
```

Derivation of the exact cost (so the implementer can check the arithmetic, not to be copied into code): task 1 costs 10. `min_task_idx=1` makes robot 0 propose at the start of task 2, when it stands on `(1,4)`. Haul: 2 unloaded moves to the depot `(1,6)`, 1 pickup tick (a wait), 2 carried moves back to `(1,4)` at cost 4 each, 1 drop tick (a wait), fee 1. Tasks 2, 3, 4 then cost 2 moves each through the filled pit. Total moves = 10 + 2 + 2 + 6 = 20 of which 2 carried; `J = (20 - 2) + 2 waits + 4 x 2 + 1 fee = 29`. The haul alone is `B_real = 2 + 1 + 8 + 1 + 1 = 13` against `B_est = 9` (the estimate omits the unloaded walk and the two action ticks, spec 4.4).

`test_abort_returns_bag` is deliberately robust to timing: whether robot 1 aborts before or after picking up a bag, exactly one bag is used and the depot ends with stock 1. If it ends with 0 or 2, the abort or return logic is wrong.

- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** per the rules.
- [ ] **Step 4: Run, expect pass.** If a timing-dependent test fails, do not loosen it: print the trajectories and the claim log for that run and fix the state machine.

- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): hauling state machine with claims and leases"`

---

### Task 11: Policies (all arms)

**Files:**
- Modify: `src/doi/policies.py` (policies, `make_policy`, `Shared`), `src/doi/runner.py` (per-tick `dispatch` call for the central arm, `hindsight_buy`, trigger collection)
- Test: `tests/doi/test_doi_policies.py`

**Interfaces:**
- Consumes: Tasks 3, 5, 9, 10.
- Produces: classes `NeverFillPolicy`, `MyopicPolicy`, `RoFPolicy`, `CentralPolicy`, `HindsightPolicy`, `FreeOpenPolicy`, plus `make_policy(cfg)` mapping names:

| `cfg.policy` | Class and flags |
|---|---|
| `never` | `NeverFillPolicy` |
| `myopic` | `MyopicPolicy` |
| `eager` | `RoFPolicy(theta=0.0, keys="bundle", scope="gossip", extrapolate=False)` |
| `rof` | `RoFPolicy(theta=cfg.theta, keys="bundle", scope="gossip", extrapolate=False)` |
| `rof_pit` | `RoFPolicy(..., keys="pit")` |
| `rof_local` | `RoFPolicy(..., scope="local")` |
| `rof_w` | `RoFPolicy(..., window=cfg.window)` (stretch) |
| `rof_x` | `RoFPolicy(..., extrapolate=True)` (stretch) |
| `central` | `CentralPolicy` |
| `hindsight` | `HindsightPolicy` |
| `free` | `FreeOpenPolicy` |

`cfg.ledger_keys`, `cfg.ledger_scope`, `cfg.extrapolate` are NOT in the config table of Task 1; the policy name selects the variant. (Remove any such fields if they were added.)

Note on `test_rof_equals_neverfill_when_unprofitable`: with two crossings the evidence (16) already exceeds the buy cost (9), so RoF correctly fills at the second crossing even though no third crossing follows; that is the nature of the ski-rental rule and it is not a bug. "Unprofitable" is therefore produced by raising the fee to 100 so the buy cost (100 + 4 x 2 = 108) exceeds all evidence reachable in four tasks (4 x 8 = 32).

`Shared` gains the fields `engine: EvidenceEngine` (one per run; its cache is a pure function of its inputs, so sharing it leaks no information between agents), `global_belief: BeliefState` (central arm only) and `agents: List[RobotAgent]` in this task.

**Rules.**
- **Buy estimator** (shared helper `buy_cost(agent, pits: Tuple[Pos,...]) -> Tuple[float, Tuple[float, ...]]`, total and per pit): per pit `fee + kappa * d(p)` where `d(p)` is the minimum, over depots with `belief.stock.remaining > 0` and over passable neighbours `n` of `p`, of BFS distance from depot to `n` under `passable_fn(static_grid, filled | set(pits), closed=believed_blocked - set(pits))`. If unreachable, `inf`.
- **`RoFPolicy.on_task_planned(agent, info, t)`:** if `info.rent > 0`, `agent.belief.add_record(RentRecord(agent.id, info.task_idx, info.start, info.goal, t, info.rent))`. Both `keys` variants write the same records; `keys="pit"` only changes how evidence is evaluated (`per_pit=True`). (If the bundle rent is 0, no single cell can have positive rent, so no record is lost.) `needs_single_rents` stays `False` for every arm.
- **`scope="local"`:** the agent's STATE messages are not sent (`uses_gossip = False`); the ledger holds only its own entries. Pit-filled and claim information still propagates through sensing only. (Claims then do not propagate, which is intended: this arm isolates the value of sharing rent.)
- **`RoFPolicy.propose(agent, t)`** (only called when the hauler is idle and no task is unplanned): `ev = shared.engine.evidence(agent.belief, t, window=self.window, extrapolate=self.extrapolate, per_pit=(self.keys == "pit"))`. A residual set `T` is eligible only if every cell of `T` has `belief.status == "confirmed"` (spec 4.4, 5.4; `editable()` already excludes `needs_human`). For each eligible `T` with finite buy, `ev[T] >= theta * buy`, and no effective claim on any pit of `T` by another robot (claim mode), compute `gap = ev[T] - buy`. Return `HaulProposal(T, ev[T], buy, per_pit)` for the largest `gap` (ties: smaller `len(T)`, then sorted tuple). Record a trigger dict `{"tick": t, "robot": id, "pits": T, "known": ev[T], "buy": buy}` in `shared.triggers`. `theta = 0` fires on any positive-evidence residual (that is `Eager`); require `ev[T] > 0`.
- **`MyopicPolicy`:** `on_task_planned` stores the info of the current task; `propose` considers only the current task's residual bundle `T` (same eligibility rule) and fires when `info.rent >= theta * buy_cost(T)` (theta from cfg, default 1). No ledger, no gossip (`uses_gossip=False`). Claims behave like RoF (`cfg.claim`).
- **`CentralPolicy`:** `shared.global_belief` is a `BeliefState` that the runner refreshes from ground truth at the start of each tick (filled cells, currently blocked cells as `observe_cell(c, True, t)`, true incident classes), which is allowed because this arm is explicitly omniscient. `on_task_planned` adds the record to `global_belief`. The runner calls `CentralPolicy.dispatch(agents, t)` once per tick instead of `propose`: compute evidence on `global_belief`; for the best firing `T` with no active central assignment, pick the idle (not hauling, not finished) agent with the smallest `h_i` (same formula as in Task 10), assign immediately (state `TO_DEPOT`, no stagger, no claim, no gate). `uses_gossip = False`.
- **`HindsightPolicy`:** `prefill_set(scenario, cfg) = hindsight(scenario, cfg).s_star`; never proposes. The runner adds `buy_lb` of the prefilled pits to `J` as `hindsight_buy` and exposes `J_hindsight = J + hindsight_buy`. Family S and ascii only.
- **`FreeOpenPolicy`** (spec 2.6): `prefill_set = set(scenario.pits) | {c for inc in scenario.incidents for c in inc.cells}` (prefilling an incident cell suppresses the incident); never proposes; `hindsight_buy = 0`; `uses_gossip = False`. Its `J` is `J_free`.
- Runner: `run_episode` calls `policy.prepare(scenario, cfg, shared)` once and, for `CentralPolicy`, `dispatch` once per tick before `decide`. Add `RunResult.triggers`, `RunResult.hindsight_buy` (0.0 for other arms).

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]


def run(policy, tasks, **kw):
    s = scenario_from_ascii(ROWS, [(1, 2)], [tasks], depot_stock=2)
    cfg = SimConfig(policy=policy, n_robots=1, tasks_per_robot=len(tasks),
                    claim=False, debug_checks=True, **kw)
    return run_episode(cfg, scenario=s)


FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def test_rof_fills_exactly_when_rent_reaches_buy_cost():
    r = run("rof", FOUR)
    assert r.fills == 1 and r.J == 29.0
    assert r.triggers[0]["known"] == 16.0 and r.triggers[0]["buy"] == 9.0
    assert r.triggers[0]["tick"] > 0


def test_rof_waits_when_single_task_rent_is_below_buy():
    r = run("rof", [(1, 4)])
    assert r.fills == 0 and r.J == 10.0


def test_myopic_never_fills_when_one_task_does_not_pay():
    r = run("myopic", FOUR)
    assert r.fills == 0 and r.J == 40.0


def test_eager_fills_at_first_positive_rent():
    r = run("eager", FOUR)
    assert r.fills == 1 and r.triggers[0]["tick"] == 0


def test_hindsight_prefills_and_charges_buy():
    r = run("hindsight", FOUR)
    assert r.fills == 0 and r.hindsight_buy == 9.0
    assert r.J == 8.0 and r.J + r.hindsight_buy == 17.0


def test_central_matches_rof_with_one_robot():
    a, b = run("rof", FOUR), run("central", FOUR)
    assert abs(a.J - b.J) <= 2.0 and b.fills == 1


def test_rof_equals_neverfill_when_unprofitable():
    a, b = run("rof", FOUR, fee=100.0), run("never", FOUR, fee=100.0)
    assert a.fills == 0 and a.J == b.J == 40.0
    assert len(a.triggers) == 0


def test_rof_pit_and_bundle_agree_on_single_pit():
    a, b = run("rof", FOUR), run("rof_pit", FOUR)
    assert a.J == b.J


def test_series_per_pit_ledger_never_fires_bundle_does():
    from src.doi.scenarios import build_scenario
    base = dict(scenario="series_pits", n_robots=4, tasks_per_robot=12, seed=1, claim=False)
    bundle = run_episode(SimConfig(policy="rof", **base))
    perpit = run_episode(SimConfig(policy="rof_pit", **base))
    assert bundle.fills == 2 and perpit.fills == 0
    assert bundle.unfinished_tasks == 0 and perpit.unfinished_tasks == 0
    assert bundle.J < perpit.J


def test_full_loss_equals_no_comm():
    base = dict(scenario="single_pit", n_robots=6, tasks_per_robot=10, seed=2, claim=False)
    lossy = run_episode(SimConfig(policy="rof", r_comm=float("inf"), loss=1.0, **base))
    quiet = run_episode(SimConfig(policy="rof", r_comm=0.0, **base))
    assert lossy.J == quiet.J and lossy.fills == quiet.fills


def test_free_open_is_travel_only():
    r = run("free", FOUR)
    assert r.J == 8.0 and r.fills == 0 and r.hindsight_buy == 0.0


def test_ledger_radius_does_not_change_traffic_messages():
    base = dict(scenario="single_pit", n_robots=6, tasks_per_robot=4, seed=1, claim=False, policy="never")
    a = run_episode(SimConfig(r_comm=0.0, **base))
    b = run_episode(SimConfig(r_comm=float("inf"), **base))
    assert a.traffic_messages == b.traffic_messages and a.J == b.J
```

(`never` sends no STATE, so with the traffic channel fixed the two runs must be identical; this pins Review Focus item 9 end to end.)

All numbers in the single-robot tests follow the derivation in Task 10 (rent per crossing 8, buy 9, fire at the second crossing with evidence 16, cost 29, hindsight 8 + 9 = 17, free 8). Static pits are `confirmed` from tick 0, so the eligibility rule does not delay any trigger in these tests. The two-pit series test depends on the simulator; if `bundle.fills != 2` investigate the hauler's pit-ordering rule before changing the test.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): fill policies for all experimental arms"`

---

### Task 12: Post-hoc metrics and the arms runner

**Files:**
- Modify: `src/doi/metrics.py`, `src/doi/runner.py`
- Test: `tests/doi/test_doi_metrics.py`

**Interfaces:**
- Produces:

```python
def stale_detour_cost(result: RunResult, scenario: Scenario, cfg: SimConfig) -> float
def true_rent_at_triggers(result: RunResult) -> List[Dict[str, float]]   # adds true_rent and coverage to each trigger
def false_report_cost(result: RunResult, scenario: Scenario, cfg: SimConfig) -> float
def false_report_hauls(result: RunResult, scenario: Scenario) -> int
def hindsight_ratios(alg: RunResult, hindsight: RunResult, free: RunResult) -> Dict[str, float]  # {"hr", "hr_av"}
def run_arms(cfg: SimConfig, arms: Sequence[str]) -> Dict[str, RunResult]  # same scenario object for all arms
def summary_row(result: RunResult, ratios: Optional[Dict[str, float]] = None,
                pod: Optional[float] = None) -> dict                      # flat dict for CSV
```

**Rules.**
- `stale_detour_cost`: for each `TaskPlanInfo` `p`, `truth_filled = {pit : filled_at[pit] <= p.tick}`; if `truth_filled != p.belief_filled`, compute `d_truth = dream_path(grid, pits, truth_filled, p.start, p.goal, U).d_block` and add `max(0, p.d_block - d_truth)`. (Family S; for family D use the same formula with `pits` = `p.belief_blocked | truth_filled`.)
- `false_report_cost` (spec 7.4): truth-blocked cells at tick `t` are unfilled static pits plus incident cells with `appeared_at <= t` and not filled by `t`. For each `TaskPlanInfo` `p`, `phantom = {c in p.belief_blocked : c is not truth-blocked at p.tick and never was blocked before p.tick}`. If non-empty, compute `d_without = dream_path(grid, sorted(p.belief_blocked - phantom), p.belief_filled, p.start, p.goal, U, p.hard_blocked - phantom).d_block` and add `max(0, p.d_block - d_without)`.
- `false_report_hauls`: number of entries in `result.triggers` whose `pits` contain a cell that is neither a static pit nor an incident cell. Must be 0 (spec 5.4); a non-zero value is a bug, not a result.
- `true_rent_at_triggers`: the rent a perfectly informed ledger would hold. Let `truth_filled(t) = {pit : result.filled_at[pit] <= t}` (ground truth; metrics code may read it). For each trigger `{tick, pits=T, known}`: `T_truth = tuple(sorted(set(T) - truth_filled(tick)))`. For each `info` in `plan_infos` with `info.tick <= tick`: `d = dream_path(grid, pits, truth_filled(info.tick), info.start, info.goal, U)`; `residual = tuple(sorted(set(d.bundle) - truth_filled(tick)))`; if `residual == T_truth` and `residual` is non-empty add `d.rent`. `true_rent` is that sum. Add `true_rent` and `coverage = known / true_rent` (`float("nan")` when `true_rent == 0`) to each trigger dict and return the list. No change to `TaskPlanInfo` is needed.
- `hindsight_ratios` (spec 2.6): requires `hindsight.policy == "hindsight"` and `free.policy == "free"`. `J_hind = J_censored(hindsight) + hindsight_buy`. `hr = J_censored(alg) / J_hind`. `hr_av = (J_censored(alg) - J_free) / (J_hind - J_free)`, or `nan` when `J_hind - J_free < 1` (the run is then excluded from `HR_av` statistics and counted).
- `summary_row`: keys `policy, seed, scenario, family, n_robots, r_comm, r_traffic, loss, latency, theta, claim, gate, intake, window, J, J_censored, delay, throughput, fills, carried_steps, wasted_haul_cost, stalled, unfinished_tasks, ticks, broadcasts, transmissions, dropped, message_units, traffic_units, overrides_per_1000, stale_detour_cost, false_report_cost, mean_coverage, mean_B_est, mean_B_real, mean_approval_wait, claims_issued, claims_lost, aborts, unconfirmed_hauls, false_report_hauls, wrong_class_attempts, approvals_requested, approvals_vetoed, intake_records, intake_rejected, runtime_ms, hr, hr_av, pod`. `overrides_per_1000 = 1000 * overrides / ticks`.
- `run_arms` builds the scenario once, runs each arm with `cfg.replace(policy=name)`.

- [ ] **Step 1: Write failing tests**

```python
from src.doi.config import SimConfig
from src.doi.metrics import stale_detour_cost, true_rent_at_triggers, hindsight_ratios, summary_row
from src.doi.runner import run_arms
from src.doi.scenarios import scenario_from_ascii, build_scenario

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def arms(claim=False):
    s = scenario_from_ascii(ROWS, [(1, 2)], [FOUR], depot_stock=2)
    cfg = SimConfig(n_robots=1, tasks_per_robot=4, claim=claim, debug_checks=True)
    from src.doi.runner import run_episode
    return {n: run_episode(cfg.replace(policy=n), scenario=s)
            for n in ("never", "rof", "hindsight", "free")}, s, cfg


def test_hindsight_ratios_single_robot():
    out, s, cfg = arms()
    r = hindsight_ratios(out["rof"], out["hindsight"], out["free"])
    assert abs(r["hr"] - 29.0 / 17.0) < 1e-9
    assert abs(r["hr_av"] - (29.0 - 8.0) / (17.0 - 8.0)) < 1e-9
    n = hindsight_ratios(out["never"], out["hindsight"], out["free"])
    assert abs(n["hr_av"] - (40.0 - 8.0) / 9.0) < 1e-9


def test_true_rent_and_coverage_single_robot_is_complete():
    out, s, cfg = arms()
    t = true_rent_at_triggers(out["rof"])[0]
    assert t["true_rent"] == 16.0 and t["coverage"] == 1.0


def test_stale_cost_zero_when_belief_correct():
    out, s, cfg = arms()
    assert stale_detour_cost(out["rof"], s, cfg) == 0.0


def test_summary_row_has_all_keys():
    out, s, cfg = arms()
    row = summary_row(out["rof"], ratios={"hr": 1.7, "hr_av": 2.3})
    for k in ("policy", "J", "hr", "hr_av", "pod", "mean_coverage", "message_units", "traffic_units",
              "overrides_per_1000", "mean_B_real", "unconfirmed_hauls", "wasted_haul_cost", "stalled"):
        assert k in row


def test_run_arms_share_scenario():
    cfg = SimConfig(n_robots=3, tasks_per_robot=3, seed=4, claim=False)
    out = run_arms(cfg, ["never", "rof", "hindsight", "free"])
    assert set(out) == {"never", "rof", "hindsight", "free"}
    assert out["never"].cfg["seed"] == out["rof"].cfg["seed"] == 4
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Add one more test, `test_stale_cost_positive_without_comm`:** scenario `single_pit`, `n_robots=6`, `tasks_per_robot=15`, `seed=1`, `claim=False`, `r_comm=0.0`, `r_sense=1`, policy `rof`; assert `stale_detour_cost > 0` for at least one of seeds 0 to 4 (robots far from the filled pit keep detouring). If it never happens report it; do not weaken the assertion silently.
- [ ] **Step 6: Full suite green, commit** `git commit -m "feat(doi): post-hoc metrics, coverage, hindsight ratios, arms runner"`

---

### Task 13: Incident dataset and location grounding

**Files:**
- Create: `src/doi/incidents.py`, `experiments/doi_build_incidents.py`, `data/incidents/README.md`, `tests/doi/test_doi_incidents.py`

Spec 5.2 and 5.6. No LLM is called in this task.

**Interfaces:**

```python
@dataclass(frozen=True)
class IncidentItem:
    item_id: str                     # f"{scenario}-{seed}-{report_id}" or f"human-{n}"
    split: str                       # dev | test | human | sim
    scenario: str
    seed: int
    report_id: str
    text: str
    source: str                      # template | human
    truth_location: Optional[str]    # None when the text is ambiguous (expected answer: reject)
    truth_kind: str
    truth_cls: str
    truth_kits: int
    is_false: bool                   # the report names a cell with no obstruction (text looks normal)
    location_names: Tuple[str, ...]  # list shown to the model

def location_names(scenario: Scenario) -> Tuple[str, ...]            # sorted keys of scenario.locations
def locate(scenario: Scenario, name: str) -> Tuple[Pos, ...]         # ValueError for an unknown name
def render_report(report: Report, rng: random.Random) -> Tuple[str, Optional[str], int]
    # (text, truth_location or None if rendered ambiguous, truth_kits)
def report_rng(scenario: Scenario, report: Report) -> random.Random  # stream(seed, f"report-{report_id}")
def build_items(scenario: Scenario, split: str) -> List[IncidentItem]
def write_jsonl(items: Sequence[IncidentItem], path: str) -> None
def read_jsonl(path: str) -> List[IncidentItem]
```

**Rules.**
- `render_report` draws, in this order: a template (at least 12, written as module constants with slots `{loc}` and `{what}`), a location phrasing, a kind phrase, a class cue, and a kits variant. Location phrasing: the exact name (p = 0.5), an abbreviation (`"A3 B5"` for aisle 3 bay 5, `"N door"`; p = 0.2), spelled-out numbers (`"aisle three, bay five"`; p = 0.2), or **ambiguous** (aisle without bay, or "a door on the wall"; p = 0.1, returns `truth_location = None`). Kind phrases: at least three per kind. Class cues: `needs_human` reports always contain a cue from a list such as "rack upright looks bent", "chemical smell", "keep robots away", "someone is hurt"; `robot_clearable` reports never do. Kits: "two pallets" or "a lot of debris" gives `truth_kits = 2`, otherwise 1. Then lowercase the whole text with p = 0.3 and introduce one character swap with p = 0.2.
- `build_items`: one item per report of the scenario, using `report_rng`. `is_false = report.oid is None`.
- `experiments/doi_build_incidents.py` writes `data/incidents/dev.jsonl` (scenarios `incidents_aisles` and `incidents_room`, seeds 0..19, `p_false = 0.2`) and `data/incidents/test.jsonl` (seeds 100..149, same settings). Seeds 200..229 are reserved for the simulation experiments E7 and E8 (Task 18) and must not appear in dev or test, so prompt work on dev never sees the reports the experiments replay.
- `data/incidents/README.md` gives the protocol for `human.jsonl`: no real names, real sites or personal data (items are sent to a hosted API); at least 100 items written by people who have not seen the templates; each writer gets a map image, a location name, a kind and a class and writes one short message as they would on a radio or chat; about 10% are told to leave the bay out (ambiguous). The file uses the `IncidentItem` schema with `split = "human"`, `source = "human"`. It is collected by hand and committed when ready; no code generates it.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_incidents.py`

```python
import random
import pytest
from src.doi.config import SimConfig
from src.doi.scenarios import build_scenario
from src.doi.incidents import location_names, locate, render_report, report_rng, build_items, write_jsonl, read_jsonl


def aisles(seed=1, **params):
    return build_scenario(SimConfig(scenario="incidents_aisles", n_robots=4, tasks_per_robot=3, seed=seed,
                                    scenario_params=params))


def test_locate():
    s = aisles()
    assert locate(s, "aisle 2 bay 3") == ((3, 2),)
    assert "aisle 2 bay 3" in location_names(s)
    with pytest.raises(ValueError):
        locate(s, "aisle 99 bay 1")


def test_render_is_deterministic_per_report():
    s = aisles()
    r = s.reports[0]
    assert render_report(r, report_rng(s, r)) == render_report(r, report_rng(s, r))


def test_class_cues_only_for_needs_human():
    s = aisles(seed=4, p_human=1.0)
    for r in s.reports:
        text, _, _ = render_report(r, report_rng(s, r))
        assert text
    items = build_items(s, "dev")
    assert all(it.truth_cls == "needs_human" for it in items)


def test_ambiguous_rate_and_truth():
    texts = []
    for seed in range(40):
        s = aisles(seed=seed)
        texts += build_items(s, "dev")
    amb = [it for it in texts if it.truth_location is None]
    assert 0.03 < len(amb) / len(texts) < 0.2
    assert all(it.truth_location in it.location_names for it in texts if it.truth_location)


def test_false_reports_flagged():
    s = aisles(seed=3, p_false=1.0)
    items = build_items(s, "dev")
    assert sum(it.is_false for it in items) == 4


def test_jsonl_roundtrip(tmp_path):
    items = build_items(aisles(), "dev")
    write_jsonl(items, str(tmp_path / "x.jsonl"))
    assert read_jsonl(str(tmp_path / "x.jsonl")) == items
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5:** Run `python experiments/doi_build_incidents.py` and read 20 random dev items by eye. If a template reads unnaturally or leaks the class through wording other than the cue lists, fix the template now (dev and test are regenerated, nothing is frozen yet).
- [ ] **Step 6: Full suite green, commit** `git add src/doi/incidents.py experiments/doi_build_incidents.py data/incidents tests/doi && git commit -m "feat(doi): incident report dataset and location grounding"`

---

### Task 14: LLM client, intake, approval drafter and cache

**Files:**
- Create: `src/doi/llm/__init__.py` (empty), `src/doi/llm/client.py`, `src/doi/llm/intake.py`, `src/doi/llm/drafter.py`
- Create: `experiments/doi_intake_run.py`, `experiments/doi_intake_eval.py`
- Test: `tests/doi/test_doi_llm.py`

Spec 5.1 to 5.3. Tests never touch the network.

**Interfaces:**

```python
# client.py
@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model: str                       # the `model` field of the response (resolved snapshot), else the requested id

class LLMClient(Protocol):
    model_id: str
    def complete(self, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> LLMResponse

class OpenAICompatClient:            # POST {base_url}/chat/completions with urllib.request
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout_s: float = 60.0)
class FakeLLMClient:
    def __init__(self, reply: Callable[[str, str], str], latency_s: float = 0.1, model_id: str = "fake")
def client_from_env(model_key: str) -> LLMClient   # env vars below

# intake.py
PROMPT_VERSION: str = "intake-v1"
SYSTEM_PROMPT: str
def build_user_prompt(text: str, names: Sequence[str]) -> str
@dataclass(frozen=True)
class IntakeResult:
    key: str
    ok: bool
    location: Optional[str]
    kind: Optional[str]
    cls: Optional[str]
    est_kits: Optional[int]
    confidence: Optional[float]
    rationale: str
    reject_reason: str               # "" when ok
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    raw: str
    model: str = ""                  # resolved model id from LLMResponse.model
def cache_key(text: str, names: Sequence[str]) -> str
def parse_and_validate(raw: str, names: Sequence[str]) -> Dict[str, Any]     # fields of IntakeResult except key/latency/tokens/raw
def run_intake(text: str, names: Sequence[str], client: LLMClient) -> IntakeResult
class IntakeCache:
    def __init__(self, root: str, model_key: str)        # file root/model_key/cache.jsonl
    def get(self, key: str) -> Optional[IntakeResult]
    def put(self, res: IntakeResult) -> None              # append; last write for a key wins on load
def to_record(res: IntakeResult, report_id: str, node: int, scenario: Scenario) -> ObstructionRecord

# drafter.py
@dataclass(frozen=True)
class ApprovalRequest:
    pits: Tuple[Pos, ...]
    evidence: float
    buy: float
    contributors: int
    census: int
    report_ids: Tuple[str, ...]
    reason: str
    text: str
def draft_request(pits, evidence, buy, contributors, census, records: Sequence[ObstructionRecord],
                  client: Optional[LLMClient] = None) -> ApprovalRequest
```

**Rules.**
- `OpenAICompatClient.__init__` takes, besides `base_url`, `model`, `api_key`, `timeout_s`: `token_param: str = "max_tokens"` (`"max_completion_tokens"` for OpenAI models that reject `max_tokens`), `send_temperature: bool = True` (False for OpenAI models that only accept the default temperature), `json_mode: bool = False` (adds `"response_format": {"type": "json_object"}`; supported by both the OpenAI API and Ollama).
- `OpenAICompatClient.complete`: body `{"model", "messages": [{"role": "system", ...}, {"role": "user", ...}], <token_param>: max_tokens}` plus `"temperature"` if `send_temperature` and `response_format` if `json_mode`; header `Authorization: Bearer <key>` only if a key is set; latency measured with `time.perf_counter` around the request; text from `choices[0].message.content`; tokens from `usage` (0 if absent). Network errors propagate; the run script records them. On HTTP 400, the error body is re-raised in the message (it names a rejected parameter, which tells the operator which flag to set). The API key never appears in exceptions, logs or cache files.
- `client_from_env(K)` reads `DOI_LLM_<K>_URL`, `DOI_LLM_<K>_MODEL`, `DOI_LLM_<K>_API_KEY` (optional; for `K = HOSTED` it falls back to `OPENAI_API_KEY`), `DOI_LLM_<K>_TOKEN_PARAM` (default `max_tokens`), `DOI_LLM_<K>_TEMPERATURE` (`0` default, `none` to omit), `DOI_LLM_<K>_JSON_MODE` (`1` to enable). A missing URL or model raises `RuntimeError` naming the variable, never its value.
- `SYSTEM_PROMPT` (spec 5.2), in substance: convert one warehouse incident report to JSON; use only a location from the provided list; if the report does not name exactly one listed location, or is not about an obstruction, return `{"reject": "<short reason>"}`; else return `{"location", "kind" (pallet|spill|debris|rack_damage|other), "class" (robot_clearable|needs_human), "est_kits" (1 to 5), "confidence" (0 to 1), "rationale" (at most 20 words)}`; classify `needs_human` for structural damage, hazardous material, injury, or anything the report says robots must avoid; output JSON only. Changing the prompt text requires bumping `PROMPT_VERSION`, which invalidates the cache by construction.
- `build_user_prompt`: the list of names, one per line, then the report text.
- `cache_key = sha256(PROMPT_VERSION + "\n" + "\n".join(names) + "\n" + text)` hex.
- `parse_and_validate`: take the substring from the first `{` to the last `}`; `json.loads`; a `reject` key gives `ok=False, reject_reason="model_reject: ..."`; otherwise every field must be present and valid (location in `names`, enums, `est_kits` int in 1..5, `confidence` in [0, 1]); any failure gives `ok=False` with `reject_reason` naming the first failed check (`not_json`, `missing:<field>`, `bad_location`, `bad_kind`, `bad_class`, `bad_kits`, `bad_confidence`).
- `to_record`: `cells = locate(scenario, res.location)`; requires `res.ok`.
- `draft_request`: the numbers are formatted by code into a fixed template: `"Edit {cells}: fleet detour cost so far {evidence:.1f}, estimated edit cost {buy:.1f}, from {contributors} of {census} robots, reports {ids}. Reason: {reason}"`. If `client` is given it is asked for one sentence explaining the obstruction from the records' rationales; the answer is truncated to 200 characters and every digit is removed before insertion. Without a client the reason is the rationales joined with "; " (truncated the same way). The LLM never sees or writes the numbers.
- `experiments/doi_intake_run.py --model-key K (--split dev|test|human | --scenario NAME --seeds A..B)`: for each item (or each report of each scenario seed, rendered with `report_rng`), skip if cached, else `run_intake` with `client_from_env(K)` and `cache.put`. It is the only code that calls an LLM. Run it once per model for dev, test, human, and for the E7/E8 scenario seeds 200..229.
- `experiments/doi_intake_eval.py --model-key K --split S`: per item compares with truth. Metrics: location exact-match accuracy on items with `truth_location` (false reports included: their text is indistinguishable, so the expected output is a normal record); reject rate on ambiguous items (higher is better); class accuracy and kind accuracy on ok items; `est_kits` mean absolute error; schema-failure rate (`ok=False` and not `model_reject`); latency p50 and p95; mean prompt and completion tokens. Writes `experiments/results/doi/intake/<K>-<S>.csv` and prints a table. Results on `human` are always reported separately from `test`.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_llm.py`

```python
import json
import pytest
from src.doi.config import SimConfig
from src.doi.crdt import ObstructionRecord
from src.doi.scenarios import build_scenario
from src.doi.llm.client import FakeLLMClient, OpenAICompatClient
from src.doi.llm.intake import (parse_and_validate, run_intake, cache_key, IntakeCache, to_record,
                                PROMPT_VERSION)
from src.doi.llm.drafter import draft_request

NAMES = ("aisle 2 bay 3", "north door")
GOOD = json.dumps({"location": "aisle 2 bay 3", "kind": "pallet", "class": "robot_clearable",
                   "est_kits": 1, "confidence": 0.8, "rationale": "fallen pallet"})


def test_parse_valid_with_surrounding_text():
    out = parse_and_validate("Sure:\n" + GOOD + "\nDone.", NAMES)
    assert out["ok"] and out["location"] == "aisle 2 bay 3" and out["cls"] == "robot_clearable"


@pytest.mark.parametrize("raw,reason", [
    ("no json here", "not_json"),
    (json.dumps({"location": "aisle 9 bay 9", "kind": "pallet", "class": "robot_clearable",
                 "est_kits": 1, "confidence": 0.5, "rationale": "x"}), "bad_location"),
    (json.dumps({"location": "north door", "kind": "fire", "class": "robot_clearable",
                 "est_kits": 1, "confidence": 0.5, "rationale": "x"}), "bad_kind"),
    (json.dumps({"location": "north door", "kind": "spill", "class": "robot_clearable",
                 "est_kits": 9, "confidence": 0.5, "rationale": "x"}), "bad_kits"),
    (json.dumps({"location": "north door", "kind": "spill", "class": "robot_clearable",
                 "est_kits": 1, "rationale": "x"}), "missing:confidence"),
])
def test_parse_rejects(raw, reason):
    out = parse_and_validate(raw, NAMES)
    assert not out["ok"] and out["reject_reason"] == reason


def test_model_reject():
    out = parse_and_validate('{"reject": "no bay given"}', NAMES)
    assert not out["ok"] and out["reject_reason"].startswith("model_reject")


def test_run_intake_with_fake_client_and_cache(tmp_path):
    client = FakeLLMClient(lambda system, user: GOOD, latency_s=0.7)
    res = run_intake("pallet down at aisle 2 bay 3", NAMES, client)
    assert res.ok and res.latency_s == 0.7 and res.key == cache_key("pallet down at aisle 2 bay 3", NAMES)
    cache = IntakeCache(str(tmp_path), "fake")
    cache.put(res)
    assert IntakeCache(str(tmp_path), "fake").get(res.key) == res
    assert cache_key("a", NAMES) != cache_key("b", NAMES)
    assert PROMPT_VERSION


def test_to_record_locates_cells():
    s = build_scenario(SimConfig(scenario="incidents_aisles", n_robots=2, tasks_per_robot=2))
    client = FakeLLMClient(lambda system, user: GOOD)
    rec = to_record(run_intake("x", NAMES, client), "r0", 3, s)
    assert rec.cells == ((3, 2),) and rec.node == 3 and rec.cls == "robot_clearable"


def test_drafter_numbers_are_not_generated():
    rec = ObstructionRecord("r0", 0, "north door", ((2, 10),), "pallet", "robot_clearable", 1, 0.9, "fallen pallet")
    liar = FakeLLMClient(lambda system, user: "Approve now, evidence is 9999 and cost 0.")
    req = draft_request(((2, 10),), 16.0, 9.0, 2, 6, [rec], client=liar)
    assert "16.0" in req.text and "9.0" in req.text and "2 of 6" in req.text
    assert "9999" not in req.text and not any(ch.isdigit() for ch in req.reason)


def test_openai_compat_request_shape(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": GOOD}}],
                               "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatClient("http://localhost:11434/v1", "small-model")
    r = c.complete("sys", "user", max_tokens=50)
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["model"] == "small-model" and seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["max_tokens"] == 50 and seen["body"]["temperature"] == 0.0
    assert seen["auth"] is None and r.text == GOOD and r.prompt_tokens == 11
    h = OpenAICompatClient("https://api.openai.com/v1", "hosted-model", api_key="sk-test",
                           token_param="max_completion_tokens", send_temperature=False, json_mode=True)
    h.complete("sys", "user", max_tokens=50)
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["max_completion_tokens"] == 50 and "max_tokens" not in seen["body"]
    assert "temperature" not in seen["body"] and seen["body"]["response_format"] == {"type": "json_object"}


def test_client_from_env_never_leaks_key(monkeypatch):
    from src.doi.llm.client import client_from_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    monkeypatch.delenv("DOI_LLM_HOSTED_URL", raising=False)
    with pytest.raises(RuntimeError) as e:
        client_from_env("hosted")
    assert "DOI_LLM_HOSTED_URL" in str(e.value) and "sk-secret-value" not in str(e.value)
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Pilot the prompt on dev only** (skip and report any endpoint that is not configured; never fake results).
  - **Hosted (`hosted`): `gpt-4o-mini` through the OpenAI API** (chosen by the user 2026-10-02). It accepts `max_tokens` and `temperature = 0`, so the defaults apply. The user keeps the key in their own shell profile (`~/.zshrc`), never in chat or a committed file. Variables: `OPENAI_API_KEY`, `DOI_LLM_HOSTED_URL=https://api.openai.com/v1`, `DOI_LLM_HOSTED_MODEL=gpt-4o-mini`, `DOI_LLM_HOSTED_JSON_MODE=1`. `gpt-4o-mini` is an alias; record the resolved snapshot from `IntakeResult.model` in `results.md`. If the alias is retired before the paper is final, the cached outputs still reproduce every result; only new intake runs would need a replacement model, which is then reported as a deviation.
  - **Optional extra, only if time allows (`small`): `qwen2.5:7b` through Ollama** on the development Mac (Apple M4, 16 GB). Ollama's default tag is 4-bit (Q4_K_M, about 4.7 GB). Run `ollama pull qwen2.5:7b` once (the user approves the download), keep `ollama serve` running during intake runs, then `DOI_LLM_SMALL_URL=http://localhost:11434/v1`, `DOI_LLM_SMALL_MODEL=qwen2.5:7b`, `DOI_LLM_SMALL_JSON_MODE=1`. No key. Record the exact quantisation from `ollama show qwen2.5:7b`. Latency measured here stands in for edge hardware; Stage 2 re-measures on target hardware (spec 11).
  - **Cost check before running:** print the number of items and an estimate of tokens (about 400 prompt and 80 completion tokens per item) and the user's model price, and ask the user to confirm before the first hosted run. Expected volume for the whole project is roughly 2,000 to 3,000 hosted calls (dev, test, human, pilot, and the E7/E8 seed sets for each `p_report`/`p_false` combination); the cache prevents repeat charges.
  - **Data:** only synthetic and lab-written reports are sent to the OpenAI API. The `human.jsonl` protocol (Task 13) forbids real names, real sites or personal data.
  - Run `doi_intake_run.py` and `doi_intake_eval.py` on `dev` for `hosted` (and `small` if set up). Prompt changes are allowed only in this step and only against dev; bump `PROMPT_VERSION` each time. The model runs at temperature 0, which does not guarantee identical output across calls, so run the dev set three times for each model and report the agreement rate (fraction of items with identical parsed output); experiments use the first cached answer. Record model ids, quantisation, hardware, prices and final dev numbers in `docs/research/results.md` (create it if missing).
- [ ] **Step 6: Full suite green, commit** `git add src/doi/llm experiments/doi_intake_*.py tests/doi && git commit -m "feat(doi): llm intake, approval drafter, cache and evaluation scripts"`

---

### Task 15: L2 in the simulator: reports, intake modes, supervisor and gate

**Files:**
- Create: `src/doi/supervisor.py`, `tests/doi/test_doi_l2.py`
- Modify: `src/doi/runner.py`, `src/doi/agent.py`, `src/doi/hauler.py`, `src/doi/policies.py`

Spec 5.2 to 5.5.

**Interfaces:**

```python
# supervisor.py
@dataclass(frozen=True)
class Decision:
    robot: int
    key: ApprovalKey
    decision: str                    # approve | veto
    needs_human: Tuple[Pos, ...]     # cells the supervisor says robots must not edit (veto only)
    due_tick: int

class Supervisor:
    def __init__(self, scenario: Scenario, cfg: SimConfig)
    def request(self, robot: int, key: ApprovalKey, pits: Tuple[Pos, ...], t: int) -> None
    def due(self, t: int) -> List[Decision]          # decisions whose due_tick == t, sorted by (robot, key)

# agent.py additions
def ingest_record(self, rec: ObstructionRecord, t: int) -> None      # belief.add_obstruction; replan_needed
def ingest_decision(self, d: Decision, t: int) -> None               # belief.approvals.set; mark_needs_human
```

**Rules: report delivery and intake (runner, after `world.begin_tick(t)`, before sensing).**
- For each report with `emit_tick == t`: the receiving node is the active robot nearest (Manhattan) to the report location's first cell, ties to the lowest id; if no robot is active the report is dropped. `intake["reports"] += 1`.
- `intake = "none"`: nothing else happens.
- `intake = "oracle"`: record `ObstructionRecord(report_id, node, location, cells, kind, cls, 1, 1.0, "oracle")` from the report fields, delivered at tick `t`.
- `intake = "llm:<K>"`: `text, _, _ = render_report(report, report_rng(scenario, report))`; look up `IntakeCache(cfg.intake_cache, K).get(cache_key(text, location_names(scenario)))`; a missing entry raises `KeyError("run experiments/doi_intake_run.py --model-key K --scenario ... --seeds ...")`. If `ok`, the record (`to_record`) is delivered at tick `t + max(1, ceil(latency_s / cfg.tick_seconds))`; else `intake["rejected"] += 1` and nothing enters the ledger.
- Wrong class (spec 5.3, both oracle and llm): if the report's true class is `needs_human` and `u01(cfg.seed, report_index, 77) < cfg.p_wrong_class`, the delivered record says `robot_clearable`.
- Delivery calls `agent.ingest_record(rec, due_tick)` on the receiving node (if it is still active; else on the nearest active robot). `intake["records"] += 1`. From there the record spreads only by gossip.

**Rules: gate (hauler).**
- After the confirmation check passes (in `PENDING` with claims on, at offer time with claims off), if `cfg.gate`: `ticket = belief.next_ticket()`, `key = (pits, ticket)`, `supervisor.request(id, key, pits, t)`, state `AWAIT_APPROVAL`, remember `request_tick`. `approvals["requested"] += 1`. The agent keeps following its task plan in `AWAIT_APPROVAL` (`active()` is False).
- Each tick in `AWAIT_APPROVAL`: if `belief.approvals.get(key) == "approve"`: `approval_wait = t - request_tick`, issue the claim (if claims on) and go to `TO_DEPOT`. If `"veto"`: go to `NONE`. If `t - request_tick >= cfg.approval_timeout` and no decision: timeout policy: approve (`timeout_approved += 1`) if every pit is `robot_clearable` in belief and every obstruction record naming a pit has `confidence >= cfg.approval_conf` (static pits count as confidence 1.0); else `deferred += 1`, go to `NONE`, and the policy must not propose a set containing these pits again before `t + cfg.approval_timeout`.
- `Supervisor.request`: `latency = max(1, round(cfg.sup_latency_median * exp(cfg.sup_latency_sigma * z)))` with `z = stream(cfg.seed, f"sup-{robot}-{t}").gauss(0, 1)`; decision `veto` if any pit is a `needs_human` incident cell (ground truth; the supervisor is a declared exception) and `u01(cfg.seed, robot, t, 91) < cfg.p_catch`, else `approve`; `needs_human` lists the pits that are `needs_human` in truth when vetoing. The runner calls `due(t)` after report delivery and passes each decision to `agents[d.robot].ingest_decision(d, t)` (if active). `approvals["approved"]` / `["vetoed"]` count delivered decisions.
- The central arm never uses the gate.

**Rules: LLM boundary.** `src/doi/llm/` may be imported only by `runner.py`, tests and experiment scripts. A test reads the import lines of `policies.py`, `hauler.py`, `agent.py`, `spacetime.py`, `world.py`, `evidence.py`, `belief.py` and fails if any mentions `llm`.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_l2.py`

```python
import math
import pathlib
import pytest
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii, Incident, Report
from src.doi.metrics import false_report_cost

ROWS_D = ["...#...", "......D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]
LOC = {"gap": ((1, 3),), "corner": ((5, 0),)}


def d_scenario(cls="robot_clearable", false_report=False):
    if false_report:
        inc, reps = [], [Report("r0", None, "gap", 0, "pallet", "robot_clearable")]
    else:
        inc = [Incident(0, ((1, 3),), 0, "pallet", cls)]
        reps = [Report("r0", 0, "gap", 0, "pallet", cls)]
    return scenario_from_ascii(ROWS_D, [(1, 2)], [FOUR], depot_stock=2,
                               incidents=inc, reports=reps, locations=LOC)


def d_run(s, **kw):
    base = dict(policy="rof", n_robots=1, tasks_per_robot=4, claim=False, intake="oracle", debug_checks=True)
    base.update(kw)
    return run_episode(SimConfig(**base), scenario=s)


def test_oracle_intake_matches_static_cost():
    r = d_run(d_scenario())
    assert r.fills == 1 and r.J == 29.0 and r.intake["records"] == 1 and r.unconfirmed_hauls == 0


def test_false_report_never_hauled():
    s = d_scenario(false_report=True)
    r = d_run(s, r_sense=0)
    assert r.fills == 0 and len(r.triggers) == 0 and r.unconfirmed_hauls == 0
    assert r.J == 40.0
    assert false_report_cost(r, s, SimConfig(n_robots=1, tasks_per_robot=4)) == 32.0


def test_gate_vetoes_wrong_class():
    s = d_scenario(cls="needs_human")
    off = d_run(s, p_wrong_class=1.0)
    assert off.wrong_class_attempts == 1 and off.fills == 0 and off.final_stock[(1, 6)] == 2
    on = d_run(s, p_wrong_class=1.0, gate=True, p_catch=1.0, sup_latency_median=3, sup_latency_sigma=0.0)
    assert on.wrong_class_attempts == 0 and on.fills == 0 and on.approvals["vetoed"] == 1


def test_gate_approval_wait_is_fixed_latency():
    r = d_run(d_scenario(), gate=True, sup_latency_median=5, sup_latency_sigma=0.0)
    assert r.fills == 1 and r.approvals["approved"] == 1 and r.edits[0]["approval_wait"] == 5


def test_llm_intake_reads_cache_only(tmp_path):
    from src.doi.incidents import render_report, report_rng, location_names
    from src.doi.llm.intake import IntakeCache, IntakeResult, cache_key
    s = d_scenario()
    text, _, _ = render_report(s.reports[0], report_rng(s, s.reports[0]))
    key = cache_key(text, location_names(s))
    with pytest.raises(KeyError):
        d_run(s, intake="llm:fake", intake_cache=str(tmp_path))
    IntakeCache(str(tmp_path), "fake").put(IntakeResult(key, True, "gap", "pallet", "robot_clearable", 1, 0.9,
                                                        "x", "", 1.2, 10, 10, "{}"))
    r = d_run(s, intake="llm:fake", intake_cache=str(tmp_path))
    assert r.intake["records"] == 1 and r.fills == 1


def test_llm_boundary_imports():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "doi"
    for name in ("policies", "hauler", "agent", "spacetime", "world", "evidence", "belief"):
        lines = [l for l in (root / f"{name}.py").read_text().splitlines() if "import" in l]
        assert not any("llm" in l for l in lines), name
```

Numbers: in `ROWS_D` the incident at `(1,3)` appears at tick 0 (before sensing) and the oracle record arrives at tick 0, so the run is the static case of Task 11: rent 8 per crossing, fire at the second crossing, `J = 29`. In `test_false_report_never_hauled` the robot has `r_sense = 0`, so it never refutes the report: it detours on every task (4 x 10 = 40), the cell is never `confirmed`, so nothing triggers; the phantom detour is `4 x (10 - 2) = 32`. In `test_gate_vetoes_wrong_class` with the gate off the robot hauls (2 to the station, carry 2, apply rejected), returns the kit, and the depot ends with 2.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): report delivery, intake modes, simulated supervisor and approval gate"`

---

### Task 16: Theory document

**Files:**
- Create: `docs/research/theory.md`

This is a writing task, no code. Follow spec section 6 (v2). The propositions are support for the experiments, not the headline (spec 1.6).

- [ ] **Step 1:** Write for each of P1 to P4: the exact statement (copy from spec section 6 and tighten any ambiguous constant), a proof or a proof sketch, and the assumptions.
  - P1: deterministic ski-rental with rent counted one task ahead and with `B_est` (trigger) differing from `B_real` (paid). Show the cost bound `B_est + B_real + r_max` and the ratio in terms of `B_est / B_real`; show it reduces to `2 + r_max/B` when they are equal.
  - P2: bound the rent missing from the firing robot's view by `D` and the haul-plus-approval latency by `lambda * L`; name the logged quantities (`coverage`, `trigger_tick`, `claim_tick`, `fill_tick`, `approval_wait`) that make the bound checkable.
  - P3: the isolation instance (n robots, equal rent rate, no ledger channel). Before writing, check spec 12 item 2 (arXiv 2507.15727) for an existing version and cite it if present.
  - P4: series corridor (per-cell counterfactual rent is identically zero; the record ledger reduces to P1) and parallel substitutes (after one fill, records it serves have empty residuals, so no second purchase is driven by stale evidence). State precisely what the record ledger does *not* guarantee (for example, splitting of evidence across substitutes before any fill).
- [ ] **Step 2:** End the document with a table: each proposition, status (`proved`, `sketch`, `open`), and the simulator observable that tests it (names from `summary_row`).
- [ ] **Step 3:** Do not claim a proof that does not close. If a bound needs an extra assumption, state it.
- [ ] **Step 4:** Commit `git add docs/research/theory.md && git commit -m "docs: propositions P1 to P4 with proof sketches"`

---

### Task 17: Statistics and experiment harness

**Files:**
- Create: `src/doi/stats.py`, `experiments/doi_common.py`, `tests/doi/test_doi_stats.py`

**Interfaces:**
- Produces:

```python
def bootstrap_ci(x: Sequence[float], stat: Callable = np.median, n: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]  # (stat, lo, hi)
def signflip_pvalue(a: Sequence[float], b: Sequence[float], n: int = 20000, seed: int = 0) -> float
def spearman(x: Sequence[float], y: Sequence[float]) -> float          # average ranks for ties

def run_grid(base: SimConfig, axes: Dict[str, Sequence], arms: Sequence[str],
             seeds: Sequence[int], out_dir: str, jobs: int = 1) -> pd.DataFrame
```

**Rules.**
- `bootstrap_ci`: resample with replacement using `np.random.default_rng(seed)`; percentile interval.
- `signflip_pvalue`: paired differences `d = a - b`, test statistic `|mean(d)|`, two-sided; under the null randomly flip signs `n` times; p = `(1 + count(|stat*| >= |stat|)) / (1 + n)`.
- `spearman`: rank transform with average ranks then Pearson correlation; returns `nan` if either input is constant.
- `run_grid`: for each point of the Cartesian product of `axes` (keys are `SimConfig` fields, or `scenario_params.<key>`), and each seed, run `run_arms` on `arms` plus the baselines it needs: `free` always; `hindsight` when the scenario family is S or ascii. For family S compute `hr` and `hr_av` (Task 12) for every non-baseline arm; if `central` is among the arms, add `pod = J(arm) / J(central)` for every other arm. Build `summary_row`, append `axis_*` columns for the swept values and `git_commit` (from `git rev-parse --short HEAD`, `unknown` on failure). Write `out_dir/runs.csv` incrementally and return the DataFrame. `jobs > 1` uses `concurrent.futures.ProcessPoolExecutor`; results must be identical to `jobs=1` (determinism).
- Also write `out_dir/config.json` with `base.to_dict()`, `axes`, `arms`, `seeds`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from src.doi.stats import bootstrap_ci, signflip_pvalue, spearman


def test_bootstrap_ci_contains_median_and_is_deterministic():
    x = list(range(1, 31))
    a = bootstrap_ci(x, seed=1)
    assert a == bootstrap_ci(x, seed=1)
    assert a[1] <= a[0] <= a[2]


def test_signflip_detects_shift_and_accepts_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    assert signflip_pvalue(a + 1.5, a) < 0.01
    assert signflip_pvalue(a, a + rng.normal(0, 0.01, 30)) > 0.05


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(spearman([1, 2, 2, 4], [1, 2, 2, 4]) - 1.0) < 1e-12


def test_run_grid_jobs_equal(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    from doi_common import run_grid
    from src.doi.config import SimConfig
    base = SimConfig(n_robots=3, tasks_per_robot=3, claim=False)
    a = run_grid(base, {"r_comm": [0.0, 8.0]}, ["never", "rof", "central"], [0, 1], str(tmp_path / "a"), jobs=1)
    b = run_grid(base, {"r_comm": [0.0, 8.0]}, ["never", "rof", "central"], [0, 1], str(tmp_path / "b"), jobs=2)
    assert a.drop(columns=["runtime_ms"]).equals(b.drop(columns=["runtime_ms"]))
    assert {"hr", "hr_av", "pod", "axis_r_comm", "git_commit"} <= set(a.columns)
    assert {"free", "hindsight"} <= set(a["policy"])
```

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.**
- [ ] **Step 5: Full suite green, commit** `git commit -m "feat(doi): statistics and experiment grid runner"`

---

### Task 18: Pilot, preregistration tag, then experiments (Stage 0)

**Files:**
- Create: `experiments/doi_e1_validity.py`, `doi_e2_information.py`, `doi_e3_complements.py`, `doi_e4_commitment.py`, `doi_e7_intake.py`, `doi_e8_gate.py`, `doi_e5_shift.py` (stretch)
- Create or extend: `docs/research/results.md`
- Modify (Step 2 only, before the tag): `docs/research/research-design.md` section 8, to replace each `[pilot]` with a number

**Seeds.** Pilot: 1000..1002. Family S experiments: 0..29. Family D experiments (E7, E8): 200..229 (never used for prompt work, Task 13). Incident dev/test sets: 0..19 and 100..149.

**Rules for every script.**
- CLI: `--seeds 30 --jobs 4 --out experiments/results/doi/<id> --quick` (`--quick` uses the pilot seeds and the smallest axis values).
- Uses `run_grid`; after the grid it writes `summary.csv` (median, bootstrap CI, per arm and axis point), the figures listed below (matplotlib `Agg`), and a printed table of the hypothesis test from spec section 8 (Spearman, sign-flip p, thresholds).
- Traffic channel fixed (`r_traffic = 8`, `loss_traffic = 0`) in every Stage 0 script.
- Never edit a hypothesis threshold in a script. After the tag, if a threshold looks wrong, record that in `results.md`; do not change spec section 8.

- [ ] **Step 1: Intake caches for the experiment seeds.** For each model key used (`hosted`, plus `small` only if the optional local model was set up), run `doi_intake_run.py --model-key K --scenario incidents_aisles --seeds 200..229` and the same for `incidents_room`, once for every `p_report`/`p_false` combination E7 uses (the reports differ per combination; the cache is keyed by text, so repeats are free). Also run the pilot seeds.
- [ ] **Step 2: Pilot (spec 8).** Run every script with `--quick`. Fix crashes, not hypotheses. Then check, and write into `results.md` under "Pilot":
  - E1 reaches R/B <= 0.3 and R/B >= 5 somewhere in its sweep; if not, extend the `fee` or `kappa` axis (never the threshold).
  - `HR_av` is defined (`J_hind - J_free >= 1`) in most E1 cells; report the excluded fraction.
  - `stalled` rate per arm is below 2%; if not, stop and report (traffic-layer problem, Task 9 Step 5).
  - From the Task 14 dev results and the E7 pilot, set the `[pilot]` values of H7 (b) and (c) in spec section 8, with one sentence each on how they were chosen.
- [ ] **Step 3: Preregister.** `python -m pytest tests -q -p no:cacheprovider` all green, then `git tag prereg-v1`. From this point spec section 8, the arm definitions, `PROMPT_VERSION` and the model ids are frozen. If a bug fix changes an arm's behaviour after the tag, create `prereg-v1-amend` with a note in `results.md`; do not move the tag.
- [ ] **Step 4: Intake accuracy on held-out data.** Run `doi_intake_run.py` and `doi_intake_eval.py` on `test` and on `human` for each model. Report them as separate tables. This is the first time the test and human splits are scored.
- [ ] **Step 5: E1 (H1).** `single_pit`, claim off, `r_comm=inf`, `loss=0`, gate off, arms `never, eager, myopic, rof, central`. Axes: `scenario_params.depot_dist` in `[1, 2, 4, 8]`, `fee` in `[1, 10, 50, 200]`, `kappa` in `[1, 4, 16]`, `tasks_per_robot` in `[2, 5, 10, 20, 40]`, `n_robots` in `[4, 12]`. If the pilot projects more than 24 hours of runtime, drop `tasks_per_robot` values 5 and 20 first and record that. Per config compute R/B (hindsight rent over `buy_lb`), `r_max`, and mean `B_real / B_est` over RoF edits. Figure `e1_hrav_vs_rent_over_buy.png` (log x, one line per arm, bound `2 + r_max/B_real`).
- [ ] **Step 6: E2 (H2).** `single_pit`, `multi_pit_wall`; arms `rof, rof_local, central, never`. Axes: ledger `r_comm` in `[0, 2, 4, 8, 16, inf]`, `loss` in `[0.0, 0.2, 0.5]`, `latency` in `[1, 3]`, `n_robots` in `[6, 12, 24]`. Figures: `e2_hrav_vs_range.png` (lines per loss), `e2_pod_vs_range.png`, `e2_coverage_vs_range.png`. Report only the direction of the effect for H2; its size is a Stage 1 result (Task 19).
- [ ] **Step 7: E3 (H3).** `series_pits`, `two_pits_parallel`; arms `rof, rof_pit, myopic, eager, never`; axes `n_robots` in `[4, 12]`, `tasks_per_robot` in `[10, 20]`. Report, for `two_pits_parallel`, the number of runs in which RoF fills the second pit and whether the first fill served all recorded tasks at that time (from `true_rent_at_triggers`). Figure `e3_complements.png`.
- [ ] **Step 8: E4 (H4).** `single_pit`, arm `rof`; axes `loss` in `[0.0, 0.1, 0.3, 0.5]`, `claim` in `[False, True]`, `lease_ticks` in `[4, 8, 16]`, `r_comm` in `[4, 8]`. Report `wasted_haul_cost`, `fills` (assert never greater than the pit count), `claims_lost`, `aborts`. Figure `e4_waste_vs_loss.png`.
- [ ] **Step 9: E7 (H7).** `incidents_room`, `incidents_aisles`, seeds 200..229; arms `rof, central, never`; axes `intake` in `[none, oracle, llm:hosted]` (add `llm:small` only if the optional local model was set up), `scenario_params.p_report` in `[0.5, 0.9]`, `scenario_params.p_false` in `[0.0, 0.1, 0.2]`. Report `J` relative to `intake=oracle` (paired by seed), `false_report_cost`, `unconfirmed_hauls`, `false_report_hauls` (both must be 0 in every run; a non-zero value is a bug), `intake_rejected`, and the latency-to-ticks distribution. Figures `e7_cost_vs_intake.png`, `e7_false_report_cost.png`.
- [ ] **Step 10: E8 (H8).** Same scenarios and seeds; arm `rof`; `intake` in `[oracle, llm:hosted]`; `gate` in `[False, True]`; `sup_latency_median` in `[5, 30, 120]`; `p_catch` in `[0.5, 0.9]`; `p_wrong_class = 0.1`. For H8 (a) compute the predicted increase per run as `sum over edits of lambda_T * approval_wait`, where `lambda_T = known evidence at trigger / max(1, trigger_tick - first record tick for T)`, and compare with the measured paired difference in `J`. Figures `e8_cost_vs_latency.png`, `e8_wrong_class.png`. Results say plainly that the supervisor is simulated.
- [ ] **Step 11 (stretch): E5 (H5).** `shift`; arms `rof, rof_w, rof_x, central, never`; axes `theta` in `[0.5, 1.0, 2.0]`, `window` in `[50, 200]` (for `rof_w`), `r_comm` in `[2, 4, 8]`; `multi_pit_wall` as the stationary contrast. Figure `e5_shift_regret.png` (regret fraction `(J_alg - J_hind)/(J_never - J_hind)`). Skip entirely if the timeline in spec section 10 has slipped.
- [ ] **Step 12:** Run the full scripts one at a time in the background. Results land in `experiments/results/doi/` (git-ignored); copy only `summary.csv` files and cited figures into `docs/research/results/`.
- [ ] **Step 13: Write `docs/research/results.md`:** for each of H1 to H4, H7, H8 (and H5 if run) one paragraph with numbers, verdict (`supported`, `partially`, `refuted`, `inconclusive`) and figure. Include the `stalled` rate and `overrides_per_1000` per arm, intake tables for test and human, model ids and hardware, and every deviation from the plan. Report refuted hypotheses plainly.
- [ ] **Step 14: Commit** `git add experiments/doi_*.py docs/research && git commit -m "feat(doi): stage 0 experiments and results"`

---

### Task 19: Stage 1: warehouse benchmark maps and scaling

**Files:**
- Create: `src/doi/maps.py`, `experiments/doi_e6_scaling.py`, `tests/doi/test_doi_stage1.py`
- Modify: `src/doi/config.py` (new fields below), `src/doi/scenarios.py` (two scenarios), `src/doi/crdt.py` and `src/doi/evidence.py` (aggregated records), `src/doi/network.py` and `src/doi/agent.py` (delta gossip), `src/doi/spacetime.py` (planning window), `.gitignore` (`data/maps/`)

Spec section 9, Stage 1. The LLM is not re-evaluated here: Stage 1 runs family D with `intake = oracle` and `llm:hosted` caches only if the location naming carries over; otherwise oracle only, stated in `results.md`.

**New config fields** (defaults keep Stage 0 behaviour unchanged): `horizon: Optional[int] = None` (stop at this tick; throughput becomes the primary metric), `record_epoch: Optional[int] = None` (aggregate records per epoch), `delta_gossip: bool = False`, `full_sync_period: int = 20`, `plan_window: Optional[int] = None`, `map_path: Optional[str] = None`.

**Rules.**
- **Maps.** `load_movingai(path) -> Grid`: header lines `type`, `height H`, `width W`, `map`; then H rows; `.` and `G` are FREE, `@`, `O`, `T`, `W` are OBSTACLE. Download the MovingAI warehouse maps (for example `warehouse-10-20-10-2-1.map` and `warehouse-20-40-10-2-1.map`) and one League of Robot Runners warehouse map into `data/maps/` **after asking the user** (state file names, source and size); they are not committed.
- **Scenarios.** `warehouse_pits` (family S): static pits at `n_pits` corridor cells (a free cell whose free neighbours are exactly two and opposite) chosen with stream `pits`, stations at `n_stations` free cells adjacent to the map border. `warehouse_incidents` (family D): same candidates as incident cells; location names `f"corridor {r}-{c}"`. Starts and goals uniform over free cells per robot stream; with `horizon` set, `tasks_per_robot = horizon // 5 + 50` so no robot runs out.
- **Aggregated records.** With `record_epoch = E`, the ledger key becomes `(robot, origin, dest, tick // E)` with value `(rent_sum, count)`; only the owner writes it and both fields only grow, so max-merge per field is a CRDT. Evidence contribution becomes `min(rent_sum, count * d.rent)`; windows apply per epoch. With `E = None` behaviour is exactly Stage 0.
- **Delta gossip.** Every component keeps a per-entry version (a local counter bumped on every local write or merge-in). `BeliefState.delta_since(v)` returns a `BeliefState` holding only entries with version `> v`. With `delta_gossip`, a robot sends `delta_since(last_sent_version)` every `gossip_period`, and its full state every `full_sync_period` ticks. Receivers merge either the same way.
- **Planning window.** With `plan_window = P`, space-time A* searches at most P steps with reservations, then appends the BFS path tail from the last cell without reservations; replanning happens before the tail is reached.
- **E6 script.** Maps as above; `N` in `[100, 200, 500]`; `horizon = 2000`; arms `rof, central, never, free` (`hindsight` only if `warehouse_pits` with at most 10 relevant pits); ledger `r_comm` in `[0, 4, 8, 16, 32, inf]`; `loss` in `[0.0, 0.2]`; `record_epoch = 100`, `delta_gossip = True`, `plan_window = 32`; seeds 0..9 (state the reduced n). Report throughput, `J` over the horizon, H2 effect size (Spearman and the slope of `PoD` against `r_comm`), H6 (smallest `r_comm` with `PoD <= 1.15`, as a fraction of `max(H, W)`), `runtime_ms`, `message_units` per robot per tick, `overrides_per_1000`, `stalled`. If one run takes more than 10 minutes in the pilot, reduce `N` and record it.

- [ ] **Step 1: Write failing tests** `tests/doi/test_doi_stage1.py`

```python
from src.doi.belief import BeliefState
from src.doi.crdt import RentRecord
from src.doi.maps import load_movingai


def test_load_movingai(tmp_path):
    p = tmp_path / "m.map"
    p.write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\nG.T\n")
    g = load_movingai(str(p))
    assert (g.height, g.width) == (2, 3)
    assert g.is_passable(0, 0) and not g.is_passable(0, 1) and g.is_passable(1, 0) and not g.is_passable(1, 2)


def test_delta_merge_equals_full_merge():
    a, b = BeliefState(0, {(0, 0): 2}, []), BeliefState(1, {(0, 0): 2}, [])
    b.add_record(RentRecord(1, 0, (0, 0), (0, 3), 0, 5))
    a.merge(b)
    v = b.version
    b.add_record(RentRecord(1, 1, (0, 3), (0, 0), 4, 6))
    b.observe_cell((2, 2), True, 4)
    x, y = a.snapshot(), a.snapshot()
    x.merge(b.delta_since(v))
    y.merge(b)
    assert x.canonical() == y.canonical()
```

Add one test for aggregated records (`record_epoch`): two tasks of one robot on the same OD in the same epoch produce one entry with `count == 2` and evidence equal to the per-task version on `single_pit`.

- [ ] **Step 2: Run, expect failure. Step 3: Implement. Step 4: Run, expect pass.** Stage 0 tests must still pass unchanged (defaults keep Stage 0 behaviour).
- [ ] **Step 5:** Pilot `doi_e6_scaling.py --quick` on the smallest map, then run in full. Add an H2 (effect size) and H6 section to `results.md`.
- [ ] **Step 6: Commit** `git commit -m "feat(doi): stage 1 warehouse maps, aggregated records, delta gossip, scaling experiment"`

---

### Task 20: Prior-art verification and close-out

**Files:**
- Create: `docs/research/prior-art-verification.md`
- Modify: `README.md` (append a "Decentralised DOI-MAPF simulator" section only)

- [ ] **Step 1:** Do the seven reading and search tasks in spec section 12, in a browser-capable session. For each paper record: citation, date read, what the paper does in two sentences, the explicit difference from this project, verdict `distinct`, `overlaps on <what>`, or `scooped`. Confirm or correct every item marked (check) in spec section 1.
- [ ] **Step 2:** If any verdict is `scooped` or `overlaps` on a claimed contribution (spec 1.6), apply the pivot rules in spec section 12 and write the decision in the document before anything else.
- [ ] **Step 3:** Append to `README.md` a short section: what `src/doi/` is, the three layers, how to run `python -m pytest tests/doi -q`, how to run one experiment with `--quick`, how to configure an LLM endpoint for `doi_intake_run.py` (environment variables only), and a pointer to `docs/research/research-design.md`. Do not touch other README sections.
- [ ] **Step 4:** Write a one-page outline for the Stage 2 plan (digital twin, ROS 2 DDS ledger transport, live edge inference) as `docs/superpowers/plans/<date>-stage2-twin-outline.md`, based on Stage 1 results. Stage 2 is planned separately.
- [ ] **Step 5:** Final checks: `python -m pytest tests -q -p no:cacheprovider` all green; `git status` clean except ignored results; `git log --format=%B | grep -i -E "claude|anthropic|co-authored"` returns nothing; `grep -rn "API_KEY\|Bearer " src experiments data` shows no secret values.
- [ ] **Step 6:** Commit `git add docs README.md && git commit -m "docs: prior-art verification, README section, stage 2 outline"`

---

## Self-review (done by the plan author)

**Spec coverage (v2).** Section 2 problem and families: Tasks 2, 7, 9. 2.4 two channels: Tasks 6, 11 (end-to-end test). 2.6 benchmarks and ratios: Tasks 4, 11, 12, 17. Section 3 layers and override log: Tasks 7, 9, 15 (import-boundary test). 4.1 rent: Tasks 3, 9. 4.3 record ledger and evidence: Task 5; aggregated records Task 19. 4.4 trigger with eligibility and `B_est`/`B_real`: Tasks 10, 11. 4.5 claims and hauling: Task 10. 4.6 arms: Task 11. Section 5 L2: intake Tasks 13, 14, 15; drafter Task 14; supervisor and gate Task 15; safety invariant Tasks 10, 15; incident dataset Task 13. Section 6 propositions: Task 16. Sections 7 and 8 experiments, pilot and preregistration: Task 18. Section 9 Stage 1: Task 19; Stage 2: outline only (Task 20 Step 4), by design. Section 12 verification: Task 20. Gap check: the drafted approval text is produced and tested (Task 14) but not used inside the simulation, because the supervisor is a stochastic model; this matches spec 5.3 and H8's stated limits.

**Interface consistency.** `TaskPlanInfo` is defined once in Task 9 (with `belief_blocked`, `hard_blocked`) and not changed later. `RentRecord`, `ObstructionRecord`, `BeliefState` are defined in Task 5 and only extended (versions, `delta_since`, aggregated key) in Task 19 behind config flags whose defaults keep Stage 0 behaviour. `HaulProposal.buy_per_pit` is added in Task 9 and filled in Task 11. `RunResult` fields used by Tasks 10 to 15 all exist from Task 9 with zero or empty values. `competitive_ratio` from v1 is replaced by `hindsight_ratios` everywhere. `World.begin_tick` is called by the runner before sensing (Task 9) and used by the Task 7 tests.

**Known soft spots the executor should surface rather than fix silently:** deadlocks in the traffic layer (Task 9 Step 5); the series-pits fill order (Task 11); `HindsightFill` fairness under congestion (spec 11); template text that leaks the class (Task 13 Step 5); an LLM endpoint not being available (Task 14 Step 5: skip and report, never fake results); E1 runtime (Task 18 Step 5); Stage 1 runtime in pure Python at N = 500 (Task 19).
