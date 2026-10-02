# Algorithmic core for publication: exact optimum, coverage theorem, two-step plans, randomized and predicted thresholds

Status: approved by the owner on 2026-10-03. Written for an implementing model that must follow it **literally**.
Every design decision has already been made. If something here seems wrong, do not fix it your own way: follow
the stop rule in section 0.3.

---

## 0. Rules for the implementer (read first, obey throughout)

### 0.1 What you are building, in one paragraph

The simulator in `src/doi` decides when a robot fleet should push an obstacle out of the way instead of walking
around it. Today it has no correct benchmark: its "hindsight" baseline is neither an upper bound nor a lower
bound, and its theory (`docs/research/theory.md`) describes an older model. You will add a small
**abstract model** (`src/doi/abstract/`) in which costs are exact shortest-path distances, so the offline optimum
can be computed exactly and the theorems in Appendix A hold. You will add **two-step push plans**, because today
no arm ever pushes when two obstacles must both be moved before anything is saved. You will add **randomized**
and **prediction-augmented** threshold rules. You will add one experiment script that measures competitive ratios
against the exact optimum. The publishable claim is algorithmic (bounds and measured ratios), not a demo.

### 0.2 Hard rules

1. Do the phases in order: 0, 1, 2, 3, 4, 5, 6, 7. Do not start a phase until the previous one's acceptance
   check passes.
2. Use exactly the file names, class names, function names, signatures, constants and commit messages given.
   Do not rename anything. Do not add parameters that are not listed.
3. Write the tests of each phase **before** the code of that phase (TDD). Use the test names given. You may add
   more tests, but never remove or weaken a listed one.
4. Do not change any existing test's expected value. Do not delete existing tests. The only existing test files
   you may edit are those this spec names.
5. Do not change world physics (`world.py`), motion and collision handling (`agent.py`'s follow, dodge and
   evade code, `spacetime.py`), the network (`network.py`), the CRDTs (`crdt.py`, `belief.py`), incidents, the
   LLM intake, the animation, the wizard or the builder. The only edits allowed in existing modules are listed
   in Phases 4 and 5.
6. Do not tune any parameter, threshold, map or seed to make a test pass. A failing listed test is either a bug
   in your code or a stop condition (0.3).
7. Do not add dependencies. Allowed imports: the standard library, `numpy`, `pandas`, `matplotlib`, and
   existing `src.*` modules.
8. Do not run full experiments. Run only the `--quick` modes named here.
9. Run Python with `venv/bin/python` (there is no `python` on PATH). Run tests with
   `venv/bin/python -m pytest tests/doi -q`.
10. Commits: one commit per phase, with the exact message given. **Do not add any `Co-Authored-By` line, any
    "Generated with" line, or any mention of Claude or Anthropic in commit messages.** Do not push. Do not open a
    PR. Do not commit `toy.gif` or anything under `experiments/results/`.
11. Determinism: every random draw must use `src.doi.rng.u01` or `src.doi.rng.stream` with the keys given.
    Never use `random.random()`, `numpy.random` without a seed, or the time.
12. Code style: match the surrounding code. Type hints on every function, module docstring at the top of every
    new file, comments only where the spec gives a reason. No print statements in library code.
13. Do not write any claim of novelty, any result number or any "we show" sentence anywhere except where
    Phase 7 tells you to paste text verbatim.

### 0.3 Stop rule

Stop, and do not try to work around the problem, if any of these happens:

* a listed test fails and you cannot find a bug in your own code after checking it against this spec twice;
* an instruction here contradicts the existing code in a way the spec does not anticipate (for example a
  function this spec calls does not exist or has a different signature);
* a phase's acceptance check fails;
* the existing test suite, which passes today with 214 tests, has any failure caused by your change that you
  cannot fix without breaking rule 4 or 5.

When you stop: write `docs/research/IMPLEMENTATION-STATUS.md` with (1) the phase and step, (2) the exact command
and its full output, (3) what you checked, (4) what you think is wrong, in at most 15 lines. Commit nothing
further. End your turn.

---

## Phase 0. Baseline snapshot

1. Run `venv/bin/python -m pytest tests/doi -q`. Expected: `214 passed`. If not, stop (0.3).
2. Stage everything except `toy.gif`: `git add -A && git reset toy.gif`.
3. Commit with message: `chore(doi): snapshot push-model refactor before the algorithmic core`.

**Acceptance:** `git status --short` shows only `?? toy.gif`.

---

## Phase 1. Ski-rental core (pure numbers): `src/doi/abstract/core.py`

Create the package `src/doi/abstract/` with an empty `__init__.py`.

This module knows nothing about grids. A *savings sequence* `s[0..T-1]` is a list of non-negative floats: `s[j]`
is what request `j` would save if the action had already been taken. `c > 0` is the action's cost. A *view*
`views[i]` is the set of request indices the deciding agent at request `i` knows about. It must be a subset of
`{0..i}`.

Definitions, used verbatim in code and docstrings:

* `S_i = s[0] + ... + s[i]` (prefix sum, inclusive). `S_{-1} = 0`.
* `K_i = sum(s[j] for j in views[i])` (known evidence at request `i`).
* Rule with threshold multiplier `thr`: fire at the first `i` with `K_i > 0` and `K_i >= thr * c`. The action is
  taken **before** request `i` is served, so the avoidable cost is `S_{i-1} + c` if the rule fires at `i`, and
  `S_{T-1}` if it never fires.
* `OPT = min(c, S_{T-1})` (the best of "take it before request 0" and "never take it").
* Coverage `rho = min over i with S_i > 0 of K_i / S_i`, or `1.0` if no `S_i > 0`.
* Deficit `D = max over i of (S_i - K_i)`, or `0.0` if `T == 0`.

### 1.1 API (exact)

```python
E_RATIO = math.e / (math.e - 1.0)

@dataclass(frozen=True)
class SingleOutcome:
    fire: Optional[int]     # request index at which the action is taken, None if never
    alg: float              # avoidable cost paid by the rule
    opt: float              # min(c, total saving)

def prefix(s: Sequence[float]) -> List[float]
def known(s: Sequence[float], views: Sequence[AbstractSet[int]]) -> List[float]       # K_0..K_{T-1}
def opt_single(s: Sequence[float], c: float) -> float
def run_threshold(s, c, views, thr: float) -> SingleOutcome
def run_predicted(s, c, views, predicted_total: float, lam: float) -> SingleOutcome
def expected_randomized_full(s, c) -> float
def randomized_draw(seed: int, draw: int) -> float          # z in [0, 1)
def run_randomized(s, c, views, z: float) -> SingleOutcome  # run_threshold with thr = z
def coverage(s, views) -> float
def deficit(s, views) -> float
def bound_coverage(theta: float, rho: float) -> float       # (1 + theta / rho) / min(1, theta)
def bound_deficit(theta: float, d: float, c: float) -> float  # (theta + 1 + d / c) / min(1, theta)
def bound_predicted(lam: float, rho: float) -> float        # 1 + 1 / (lam * rho)
def views_full(T: int) -> List[FrozenSet[int]]
def views_own(agents: Sequence[int]) -> List[FrozenSet[int]]          # j <= i and agents[j] == agents[i]
def views_delay(agents: Sequence[int], delta: int) -> List[FrozenSet[int]]  # (j <= i - delta) or own
def views_sample(agents: Sequence[int], p: float, seed: int) -> List[FrozenSet[int]]
    # j <= i and (agents[j] == agents[i] or u01(seed, agents[i], j, 41) < p)
def round_robin(T: int, n: int) -> List[int]                # [i % n for i in range(T)]
```

Rules for the implementation:

* `run_threshold` and `run_randomized`, and `run_predicted`, must raise `ValueError` if any `s[j] < 0`, if
  `c <= 0`, or if some `views[i]` contains an index `> i` or `< 0`.
* `run_predicted`: `thr = lam if predicted_total >= c else 1.0 / lam`, then exactly `run_threshold`. Raise
  `ValueError` unless `0 < lam <= 1`.
* `randomized_draw(seed, draw) = math.log(1.0 + u01(seed, draw, 31) * (math.e - 1.0))`.
* `expected_randomized_full(s, c)` is the **exact** expectation of the avoidable cost when `z` has density
  `e^z / (e - 1)` on `[0, 1]` and views are full. Compute it like this (no sampling):

  ```
  P(a, b) = (exp(min(max(b,0),1)) - exp(min(max(a,0),1))) / (e - 1)     # probability z in (a, b]
  S = prefix(s); total = S[-1] if s else 0
  expected = 0
  for i in range(T):
      lo = (S[i-1] if i > 0 else 0) / c      # z above lo: the rule did not fire before i
      hi = S[i] / c                          # z at most hi: fires at i (if S[i] > 0)
      if S[i] > 0 and hi > lo:
          expected += P(lo, hi) * ((S[i-1] if i > 0 else 0) + c)
  expected += P(total / c, 1.0) * total     # z above total/c: never fires
  ```
  (When `total / c >= 1`, the last term is 0. The `z = 0` boundary has probability 0 and is ignored.)

### 1.2 Tests: `tests/doi/test_doi_abstract_core.py`

Write exactly these (plus any extras you like):

1. `test_threshold_hand_case`: `s=[3,3,3,3]`, `c=7`, full views, `thr=1`: `fire == 2`, `alg == 13`
   (paid `S_1 = 6`, plus 7), `opt == 7`.
2. `test_threshold_never_fires`: `s=[1,1]`, `c=7`, full: `fire is None`, `alg == 2`, `opt == 2`.
3. `test_isolation_hand_case`: `T=8`, `agents=round_robin(8,2)`, `s=[1]*8`, `c=4`, `views_own`, `thr=1`:
   `fire == 6` (agent 0's fourth own request, index 6), `alg == 10`, `opt == 4`, `coverage == 0.5`.
4. `test_bound_coverage_holds_on_random_sequences`: for 500 cases drawn with `stream(0, "core-cov")`: `T` in
   `1..60`, `s[j]` uniform integers `0..5`, `c` in `1..40`, `n` in `1..8`, view kind uniform from
   {full, own, delay with `delta` in `1..10`, sample with `p` in `{0.25, 0.5, 0.9}`}, `theta` in
   `{0.5, 1.0, 2.0}`. If `opt > 0`: assert `alg <= bound_coverage(theta, rho) * opt + 1e-9` whenever
   `rho > 0`, and `alg <= bound_deficit(theta, D, c) * opt + 1e-9` always.
5. `test_full_information_ratio_is_at_most_two`: same generator with full views, `theta=1`: `alg <= 2*opt + 1e-9`.
6. `test_isolation_lower_bound_approaches_n_plus_one`: `n=4`, `c=400`, `s=[1]*4000`,
   `agents=round_robin(4000,4)`, `views_own`, `thr=1`: `alg / opt >= 4.9` and `<= 5.0 + 1e-9`.
7. `test_randomized_expectation_bound`: 500 random cases (as in 4, full views): `expected_randomized_full(s,c) <=
   E_RATIO * opt + 1e-9`.
8. `test_randomized_expectation_matches_sampling`: `s=[1]*30`, `c=10`: the mean of `run_randomized(...,
   randomized_draw(7, d)).alg` over `d in range(20000)` is within `0.1` of `expected_randomized_full`.
9. `test_predicted_consistency_and_robustness`: 500 random cases, full views, `lam` in `{0.25, 0.5, 1.0}`,
   `predicted_total` either the true total `S_{T-1}` (consistent case) or adversarial
   (`0` if `S_{T-1} >= c` else `2*c + 1`). Consistent case: `alg <= (1 + lam) * opt + 1e-9`. Both cases:
   `alg <= bound_predicted(lam, 1.0) * opt + 1e-9`.
10. `test_rejects_bad_input`: negative saving, `c=0`, a view containing `i+1`, `lam=0`: each raises `ValueError`.

**Acceptance:** all tests pass. Commit: `feat(doi): ski-rental core with coverage, deficit, randomized and predicted thresholds`.

---

## Phase 2. Abstract grid model: `src/doi/abstract/instance.py`

A deterministic, motion-free version of the push problem. There are no robots moving, no congestion and no walking
cost. Requests are served one at a time, in the given order, at their exact shortest-path distance.

### 2.1 Semantics (exact)

* Map: a `Grid` (from `src.environment.grid`) with permanent walls. Build instances with
  `src.doi.scenarios.scenario_from_ascii` and take `.grid` and `.obstacles` from the result.
* Configuration: `Config = Tuple[Tuple[Pos, str], ...]`, the obstacles as `(cell, kind)` pairs sorted by cell.
  `blocked(config) = frozenset(cell for cell, _ in config)`.
* Request `j`: `(origin, dest)`. Serving cost in configuration `X`:
  `U` if `origin in blocked(X)` or `dest in blocked(X)`; else the BFS distance from `origin` to `dest` with
  `blocked(X)` closed (`bfs_dist_map(passable_fn(grid, closed=blocked), origin, H, W)`), or `U` if unreachable.
  `U = 4 * (H + W)` (same as `SimConfig.unreachable_cost_for`). **Endpoints are not treated as open.**
* Action `a = (obstacle, direction, steps)`, `direction` in `pushplan.DIRS`, `1 <= steps <= push_max`. It is
  legal in `X` iff all of these hold:
  * `obstacle in blocked(X)`;
  * approach cell `obstacle - direction` is in bounds, not a wall, and not in `blocked(X)`;
  * every cell `obstacle + m*direction` for `m = 1..steps` is in bounds, not a wall and not in `blocked(X)`;
  * the landing `obstacle + steps*direction` is not dead: `not pushplan._dead(passable_fn(grid), landing)`.

  Reachability of the approach cell is **not** required. This is a relaxation, stated in the theory.
* Cost of an action: `fee + steps * kappa * weight(kind)` (`src.doi.kinds.weight`).
* Applying an action moves that one obstacle to the landing and keeps its kind.
* A schedule may apply any number of actions before any request. Total cost = sum of action costs + sum of
  serving costs.

### 2.2 API (exact)

```python
Pos = Tuple[int, int]
Config = Tuple[Tuple[Pos, str], ...]

@dataclass(frozen=True)
class Action:
    obstacle: Pos
    kind: str
    direction: Pos
    steps: int
    landing: Pos
    cost: float
    def key(self) -> Tuple: return (self.obstacle, self.direction, self.steps)

@dataclass
class Instance:
    grid: Grid
    obstacles: Dict[Pos, str]
    requests: List[Tuple[Pos, Pos]]
    agents: List[int]                 # agents[j] serves request j
    kappa: float = 1.0
    fee: float = 1.0
    push_max: int = 6
    def unreachable(self) -> float     # 4 * (H + W)
    def initial(self) -> Config

def instance_from_ascii(rows: List[str], requests: List[Tuple[Pos, Pos]], agents: Optional[List[int]] = None,
                        kappa: float = 1.0, fee: float = 1.0, push_max: int = 6) -> Instance
    # agents defaults to [0] * len(requests)

class Model:
    """Serving costs and legal actions of one instance, with caches."""
    def __init__(self, inst: Instance) -> None
    def blocked(self, x: Config) -> FrozenSet[Pos]
    def serve(self, x: Config, j: int) -> float           # cache key (blocked(x), origin)
    def legal_actions(self, x: Config) -> List[Action]    # sorted by Action.key()
    def apply(self, x: Config, a: Action) -> Config
    def bundle_cells(self, x: Config, j: int) -> FrozenSet[Pos]
        # dream_path(grid, sorted(blocked(x)), origin, dest, U).bundle
    def random_small_instance(...)  # NOT here: see 2.4
```

### 2.3 The hand instance `G1` (use it in tests and in the experiment)

```python
G1_ROWS = ["..#....",
           "..L....",
           "..#....",
           "..#....",
           "......."]
G1_REQUEST = ((1, 0), (1, 6))
def g1(m: int, n_agents: int = 1, fee: float = 1.0) -> Instance:
    # m copies of G1_REQUEST, agents round_robin(m, n_agents), kappa=1, the given fee, push_max=6
```

Put `G1_ROWS`, `G1_REQUEST` and `g1` in `src/doi/abstract/instance.py`.

Facts about G1, all derived by hand. The tests in 2.5 assert them.
* Open distance 6. With the pallet at `(1,2)`: 12 (via row 4).
* Legal actions initially: east `steps` 1..4 (landings `(1,3)`..`(1,6)`), west `steps` 1..2 (landings `(1,1)`,
  `(1,0)`). North and south are illegal, because the approach is a wall or out of bounds.
* Serving cost after east 1: 12. After east 2: 8. After east 3: 8. After east 4: `U = 48` (the pallet sits on the
  destination).
* From the pallet at `(1,3)`: south 1 (approach `(0,3)`, landing `(2,3)`) is legal and costs 2. North 1 (landing
  `(0,3)`) is illegal because `(0,3)` is dead. After east 1 then south 1, serving costs 6.
* Symmetrically, from the pallet at `(1,1)` (after west 1), south 1 (landing `(2,1)`) is legal and costs 2, and
  north 1 is illegal (dead).
* Exact optimum: `OPT(g1(1)) = 10` (east 1 or west 1, then south 1, then serve 6). `OPT(g1(2)) = 16`.
  `OPT(g1(4)) = 28`. Never pushing: `cost = 12*m`.
* Vanish lower bound: `LB(g1(1)) = 8` (2 + 6). `LB(g1(2)) = 14`.

### 2.4 Random small instances

```python
def random_small_instance(seed: int, n_obstacles: int, T: int, n_agents: int) -> Instance:
```
* `H = 7`, `W = 9`. Walls at column 4, rows 0..5. Row 6 is fully free (the long way round).
* Gap rows are `(1, 3, 5)[:n_obstacles]`. Each gap cell `(r, 4)` holds a pallet (`L`). Raise `ValueError` unless
  `1 <= n_obstacles <= 3`.
* Requests from `stream(seed, "abstract-req")`: request `j` alternates direction (`j` even: west to east;
  odd: east to west). The origin is drawn uniformly from the free non-obstacle cells of the source side
  (columns 0..3, or 5..8), and the destination likewise from the other side. Use `rng.choice` on sorted lists.
* `agents = round_robin(T, n_agents)`, `kappa = 1`, `fee = 1`, `push_max = 4`.

### 2.5 Tests: `tests/doi/test_doi_abstract_instance.py`

1. `test_g1_serving_costs`: the serving costs listed in 2.3 (initial, after east 1/2/3/4, after east 1 + south 1).
2. `test_g1_legal_actions`: the exact legal action keys initially (6 actions), and that north 1 from `(1,3)` is
   absent while south 1 is present.
3. `test_blocked_endpoint_costs_unreachable`: a configuration with an obstacle on the destination serves at `U`.
4. `test_action_cost_uses_kind_weight`: a crate (`C`, weight 0.5) pushed 2 steps with `kappa=4`, `fee=1` costs
   `1 + 2*4*0.5 = 5`.
5. `test_random_small_instance_is_deterministic_and_valid`: same seed gives equal requests; no endpoint is on an
   obstacle; `len(requests) == T`.

**Acceptance:** tests pass. Commit: `feat(doi): abstract push model with exact serving costs and legal actions`.

---

## Phase 3. Offline optimum and lower bound: `src/doi/abstract/offline.py`

### 3.1 Exact optimum (layered Dijkstra)

```python
MAX_STATES = 20000

class StateSpaceTooLarge(Exception): ...

@dataclass(frozen=True)
class OptResult:
    cost: float
    schedule: Tuple[Tuple[int, Action], ...]   # (request index the action is applied before, action), in order
    n_states: int

def reachable_configs(model: Model, start: Config, max_states: int = MAX_STATES) -> List[Config]
    # BFS over legal actions from start; returned in BFS discovery order with start first.
    # Raise StateSpaceTooLarge as soon as more than max_states configurations are discovered.

def exact_opt(inst: Instance, max_states: int = MAX_STATES) -> OptResult
```

`exact_opt` algorithm:
1. `configs = reachable_configs(...)`, `index = {x: k}`.
2. Precompute `succ[k] = [(index[apply(x,a)], a) for a in legal_actions(x)]` for every config.
3. Dijkstra over nodes `(i, k)`, `i` in `0..T` (number of requests served so far), source `(0, index[initial])`.
   Edges: `(i,k) -> (i,k2)` with cost `a.cost` for each `(k2, a)` in `succ[k]`; and if `i < T`,
   `(i,k) -> (i+1,k)` with cost `model.serve(configs[k], i)`. Heap entries are `(dist, i, k)`, so ties break
   deterministically. The answer is the first popped node with `i == T`.
4. Rebuild the schedule from predecessor pointers and record only the action edges, as `(i, action)`.

### 3.2 Vanish lower bound

```python
MAX_LB_OBSTACLES = 12

def vanish_lower_bound(inst: Instance) -> float
```
`min over subsets M of the initial obstacles of [ sum_{p in M} (fee + kappa * weight(kind_p))
 + sum_j serve(initial config without M, j) ]`, by exhaustive enumeration. Raise `ValueError` if there are more
than `MAX_LB_OBSTACLES` obstacles.

Why it is a valid lower bound (put this in the docstring): every obstacle that a schedule ever moves costs at
least one push (`fee + kappa*w`). The obstacles it never moves stay where they started, so at every request the
blocked set contains `initial \ M`, and serving costs can only rise as the blocked set grows.

### 3.3 Tests: `tests/doi/test_doi_abstract_offline.py`

1. `test_g1_exact_optimum`: `exact_opt(g1(1)).cost == 10`, `exact_opt(g1(2)).cost == 16`,
   `exact_opt(g1(4)).cost == 28`. For `g1(1)`, the schedule has exactly two actions, both before request 0: the
   first has `steps == 1` and direction `(0, 1)` or `(0, -1)`, and the second has `steps == 1` and direction
   `(1, 0)`. Do not assert which of the two symmetric optima is returned. `exact_opt(g1(1)).n_states == 26`.
2. `test_g1_vanish_bound`: `8` and `14` for `m = 1, 2`.
3. `test_never_is_an_upper_bound_and_vanish_a_lower_bound`: for seeds `0..9`, `n_obstacles in (1, 2)`, `T = 8`:
   `vanish_lower_bound <= exact_opt.cost <= sum(serve(initial, j))`.
4. `test_state_limit_raises`: `exact_opt(..., max_states=5)` on `g1(1)` raises `StateSpaceTooLarge`.
5. `test_schedule_replays_to_its_cost`: replaying the schedule (apply the actions before their request index,
   sum the action costs and serving costs) gives exactly `cost`, for seeds `0..4`, `n_obstacles=2`, `T=8`.

**Acceptance:** tests pass. `test_never_is_an_upper_bound...` finishes in under 60 s. If it does not, stop (0.3);
do not lower `T`. Commit: `feat(doi): exact offline optimum and vanish lower bound for the abstract model`.

---

## Phase 4. Online algorithms on the abstract model: `src/doi/abstract/online.py`

### 4.1 API (exact)

```python
MAX_FIRES_PER_REQUEST = 10

@dataclass(frozen=True)
class Rule:
    mode: str                 # "never" | "threshold" | "randomized" | "predicted"
    theta: float = 1.0
    lam: float = 0.5
    bundle_max: int = 1       # 1: single actions; 2: also two-step plans
    view: str = "full"        # "full" | "own" | "delay:<int>" | "sample:<float>"
    predictor: str = "oracle" # "oracle" | "inverted" | "noisy:<float>"   (predicted mode only)
    seed: int = 0

@dataclass
class OnlineResult:
    cost: float
    action_cost: float
    serve_cost: float
    fires: List[dict]         # one per plan taken: see 4.3

def views_for(rule: Rule, agents: Sequence[int]) -> List[FrozenSet[int]]     # reuse core.views_*
def candidates(model: Model, x: Config, j: int, bundle_max: int) -> List[Tuple[Action, ...]]
def run_online(inst: Instance, rule: Rule) -> OnlineResult
```

### 4.2 Candidates (exact)

* Singles: every `a` in `legal_actions(x)` with `a.obstacle in bundle_cells(x, j)`, as 1-tuples.
* If `bundle_max == 2`: also, for every single `a1`, let `x1 = apply(x, a1)`. For every `a2` in
  `legal_actions(x1)` with `a2.obstacle in bundle_cells(x1, j)`, add `(a1, a2)`. (`a2` may move the same obstacle
  again.)
* Order: singles in `Action.key()` order, then pairs in `(a1.key(), a2.key())` order.

### 4.3 `run_online` (exact)

```
x = initial; views = views_for(rule, agents); frozen = {}   # predicted mode: (agent, plan key) -> bool
for i in range(T):
    g = agents[i]
    for _ in range(MAX_FIRES_PER_REQUEST):
        if rule.mode == "never": break
        best = None
        for plan in candidates(model, x, i, rule.bundle_max):
            y = apply all actions of plan to x in order
            c = sum(a.cost for a in plan)
            E = sum(serve(x, j) - serve(y, j) for j in views[i])          # known evidence, signed
            thr = threshold(rule, g, plan, x, y, c)                      # see below
            if E > 0 and E >= thr * c:
                score = (E - c, tuple(-v for v in ...))                  # see tie rule
                keep the plan with the largest E - c; ties: smallest tuple(a.key() for a in plan)
        if best is None: break
        record fire; x = y_best; action_cost += c_best
    serve_cost += serve(x, i)
```
Thresholds:
* `threshold`: `rule.theta`.
* `randomized`: `max(z(a.obstacle) for a in plan)`, with
  `z(p) = math.log(1 + u01(rule.seed, p[0], p[1], 31) * (math.e - 1))`. Because it is keyed on the cell, all
  agents share the draw.
* `predicted`: `key = (g, tuple(a.key() for a in plan))`. If `key` is not in `frozen`, compute the predicted
  total saving once and store `frozen[key] = predicted >= c`. Then `thr = rule.lam if frozen[key] else 1/rule.lam`.
  The predicted total saving:
  * `oracle`: `sum(serve(x, j) - serve(y, j) for j in range(T))` (the whole sequence, evaluated in the current
    `x`);
  * `inverted`: `0.0` if oracle `>= c` else `2*c + 1`;
  * `noisy:<sigma>`: `oracle * math.exp(sigma * n)`, where `n = stream(rule.seed, f"pred-{g}-{key}").gauss(0, 1)`.

Each fire record is the dict `{"request": i, "agent": g, "plan": [a.key() for a in plan], "cost": c,
"known": E, "true": E_true, "rho": E / E_true if E_true > 0 else 1.0}`, where
`E_true = sum(serve(x,j) - serve(y,j) for j in range(i + 1))`.

### 4.4 Tests: `tests/doi/test_doi_abstract_online.py`

1. `test_g1_two_step_plan_reaches_the_optimum`: `run_online(g1(m), Rule("threshold", bundle_max=2)).cost` is
   `10, 16, 28` for `m = 1, 2, 4`. For `m = 1` there is exactly one fire, before request 0, and its plan is
   `[((1,2),(0,-1),1), ((1,1),(1,0),1)]` (west 1 then south 1). It ties with east 1 then south 1 on `E - c`, and
   the smaller key wins.
2. `test_g1_single_steps_cost_eleven`: with `bundle_max=1` the cost on `g1(1)` is `11`: east 2 (cost 3) fires
   first, then north 1 of the pallet at `(1,4)` (plan `[((1,4),(-1,0),1)]`, cost 2), then serve 6. Costs for
   `m = 2, 4` are `17, 29`.
3. `test_never_rule_equals_never_cost`: `Rule("never")` gives `12*m` on `g1(m)` for `m` in `1..4`.
4. `test_two_step_plan_matches_core`: on `g1(m)` with `fee=4` (`g1` gets a `fee` argument for this; see
   below), for `m` in `1..12`, `n_agents` in `{1,2,3}`, `view="own"`, `bundle_max=2`. The plan that fires is a
   two-step plan of cost 10 that saves 6 per request. Assert that `run_online(...).cost - 6*m` equals
   `core.run_threshold([6]*m, 10, views_own(round_robin(m, n_agents)), 1.0).alg`. Values for `n_agents = 1`:
   `6` for `m = 1`, `16` for `m >= 2`; for 2 agents `6, 12`, then `22`; for 3 agents `6, 12, 18`, then `28`.
5. `test_online_never_beats_exact_opt`: for seeds `0..9`, `n_obstacles` in `(1, 2)`, `T=8`, and every mode with
   `bundle_max=2`: `run_online(...).cost >= exact_opt(...).cost - 1e-9`.
6. `test_predicted_values_on_g1`: on `g1(m, fee=4)` with `bundle_max=2`, `lam=0.5`, full view: `cost - 6*m`
   for `m = 1, 2, 3, 4` is `6, 10, 13, 13` with predictor `oracle`, and `8, 19, 26, 28` with predictor
   `inverted`.
7. `test_multi_candidate_counterexample_on_g1`: the same rule with `inverted` on `g1(6, fee=4)` gives
   `cost - 6*6 == 33`. That exceeds `bound_predicted(0.5, 1.0) * 10 = 30`. Several candidate plans (east 2, then
   north 1, and the two-step plans) each pass a raised threshold one after another, so Theorem 3 does not
   extend beyond the single-candidate setting. Keep this test: it documents a limit the paper must state.

**Acceptance:** tests pass. Commit: `feat(doi): online threshold, randomized and predicted rules on the abstract model`.

---

## Phase 5. Simulator changes (the only edits allowed in existing `src/doi` modules)

### 5.1 Config (`src/doi/config.py`)

* Add to `POLICIES`: `"rof_r"`, `"rof_p"`.
* Add fields after `theta`: `bundle_max: int = 1` and `lam: float = 0.5`.
* Add checks: `(self.bundle_max in (1, 2), "bundle_max must be 1 or 2")` and
  `(0 < self.lam <= 1, "lam must be in (0, 1]")`.
* The default `bundle_max = 1` keeps every existing number unchanged. Do not change it.

### 5.2 Two-step plans (`src/doi/pushplan.py`)

Add:
```python
@dataclass(frozen=True)
class BundlePlan:
    first: PushPlan
    second: PushPlan
    @property
    def obstacle(self) -> Pos: return self.first.obstacle
    @property
    def kind(self) -> str: return self.first.kind
    @property
    def landing(self) -> Pos: return self.first.landing
    @property
    def steps(self) -> int: return self.first.steps
    @property
    def before(self) -> FrozenSet[Pos]: return self.first.before
    @property
    def after(self) -> FrozenSet[Pos]: return self.second.after
    @property
    def total(self) -> float:
        return self.first.walk_in + self.first.push_cost + self.second.walk_in + self.second.push_cost + self.second.walk_on

def bundle_plans(grid, distance, blocked, firsts: List[PushPlan], eligible_after: Callable[[PushPlan], List[Pos]],
                 kind_of, goal, kappa, fee, max_steps, unreachable) -> List[BundlePlan]
```
Also add (decision of 2026-10-03, binding):

```python
def all_length_plans(grid, distance, blocked, candidates, kind_of, pos, goal, kappa, fee, max_steps, unreachable,
                     skip=frozenset(), dead_end_guard=True) -> List[PushPlan]

def plan_key(plan) -> Tuple[Tuple[Pos, Pos, int], ...]
    # PushPlan:   ((obstacle, direction, steps),)
    # BundlePlan: ((first.obstacle, first.direction, first.steps), (second.obstacle, second.direction, second.steps))
```

* `all_length_plans` has the same arguments and legality rules as `candidate_plans`: the approach is free, cells
  are checked in order and the loop stops at the first cell that is not free, `landing == goal` is skipped, and
  so are dead landings. The difference is that it returns **every** legal `(obstacle, direction, steps)` plan,
  not only the cheapest per direction. Sort the result by `(total, plan_key(plan))`. This matches the abstract
  model, which enumerates every push length. It is needed because the first leg of a good two-step plan is often
  not the cheapest single push for this robot. In G1, east 1 is useless alone but is the first leg of the optimum.
* `plan_key` mirrors the abstract model's `tuple(a.key() for a in plan)`. Use it everywhere a plan needs an
  identity: sorting, tie-breaking, predictions, randomized draws and trigger logs.

`bundle_plans`: `firsts` is the output of `all_length_plans`. For each `p1` in `firsts`, call
`all_length_plans(grid, distance, p1.after, eligible_after(p1), kind_of2, p1.end, goal, kappa, fee, max_steps,
unreachable)` for the second legs. Here `kind_of2` returns `p1.kind` for `p1.landing` and `kind_of(c)`
otherwise. Make a `BundlePlan(p1, p2)` for each result `p2`. Return them sorted by `(total, plan_key(plan))`.
Do not change `candidate_plans`.

### 5.3 Agent (`src/doi/agent.py`, `_consider_push` only, plus one line in `decide`)

* If `self.cfg.bundle_max == 1`, leave everything exactly as it is (`plans = candidate_plans(...)`).
* If `self.cfg.bundle_max == 2`, replace the call: `singles = all_length_plans(...)` (same arguments as the existing
  `candidate_plans` call, including `skip`), and `plans = singles + bundle_plans(..., firsts=singles, ...)`. Then
  return `None` if `plans` is empty, as before. Singles and pairs therefore both cover every push length, the
  same candidate set as `abstract.online.candidates`. `bundle_plans` uses:
  `eligible_after(p1) = [c for c in dream_path(self.grid, tuple(sorted(p1.after - b.hard_blocked())), p1.end,
  self.goal, self.U, b.hard_blocked()).bundle if c == p1.landing or (c in eligible_set)]`,
  where `eligible_set = set(eligible)`.
* Keep the existing selection loop unchanged. It already works on any object with `total`, `before` and `after`.
* In `decide`, replace `self.pusher.start(plan, t)` with
  `self.pusher.start(plan.first if isinstance(plan, BundlePlan) else plan, t)`.
  The robot carries out the first leg only. After it, the ordinary rule re-decides the second obstacle, and its
  evidence is then the full complement saving. Put this sentence in a comment.
* `trigger(...)` in `policies.py` must add `"bundle": isinstance(plan, BundlePlan)` and `"plan": plan_key(plan)`
  to its dict.
* The selection loop's tie-break stays `(score, -plan.total)`. When that also ties, the first plan in list order
  wins, and the list is sorted by `(total, plan_key)`, so the result is deterministic.

### 5.4 New arms (`src/doi/policies.py`)

* `make_policy`: `"rof_r"` returns `RandomizedPolicy(theta=cfg.theta)`. `"rof_p"` returns
  `PredictedPolicy(lam=cfg.lam)`.
* `class RandomizedPolicy(LedgerPolicy)`: `name = "rof_r"`, gossip scope. In `assess`, the threshold is
  `z = max(math.log(1 + u01(self.cfg.seed, o[0], o[1], 31) * (math.e - 1)) for o, _d, _k in plan_key(plan))`
  instead of `self.theta`: the largest draw over the obstacles the plan moves, as in the abstract model.
  Everything else is as in `LedgerPolicy.assess`.
* `class PredictedPolicy(LedgerPolicy)`: `name = "rof_p"`, gossip scope. On each agent object, keep a dict
  `agent.predictions` (create it lazily with `getattr(agent, "predictions", None)`), keyed by
  `plan_key(plan)`. Never use `plan.direction`: `BundlePlan` has no `direction`. The first time a key is seen, compute the forecast exactly as
  `LedgerPolicy.saving` does with `forecast=True` (call the parent logic; do not copy it), and store
  `forecast >= price`. Threshold: `self.lam` if the stored value is True, else `1 / self.lam`. Known saving:
  the non-forecast `saving` (ledger + mine).
  Exact recipe: `__init__(self, lam)` calls `super().__init__("rof_p", theta=1.0, forecast=False)` and stores
  `self.lam`. A helper `_forecast(self, agent, plan) -> float` sets `self.forecast = True`, returns
  `self.saving(agent, plan)`, and resets `self.forecast = False` in a `finally` block. `assess` calls
  `self.saving(agent, plan)` for the known saving (with `self.forecast` False).
* `narrate.ARM_NOTES`: add
  `"rof_r": "Rent-or-Fill with a random threshold between 0 and 1 times the price (expected ratio e/(e-1))"`
  and `"rof_p": "Rent-or-Fill that lowers its threshold when its forecast says the push will pay, and raises it when not"`.

### 5.5 Complements scenario (`src/doi/scenarios.py`, new scenario `complements`)

Do not touch `series_blocks`. It deliberately shows "no room to push".

* `DEFAULTS["complements"] = dict(H=15, W=23, wall_cols=(7, 15), gap_row=3, door_rows=(12, 13, 14),
  kinds=("pallet",), band=(0, 6))`.
* New builder `_complements(cfg, params)`:
  * a grid `W x H`; for each `wc` in `wall_cols`, set column `wc` to a wall in every row except `door_rows` and
    `gap_row`;
  * obstacles `{(gap_row, wc): kinds[k % len(kinds)] for k, wc in enumerate(wall_cols)}`;
  * rooms: west `cols 0..wall_cols[0]-1`, east `cols wall_cols[1]+1..W-1`;
  * starts: robots `0..n//2-1` west, the rest east (use `_draw_starts` with pools restricted to rows inside
    `band`);
  * tasks: every task goes to the other outer room (`q_cross = 1.0`), goal drawn with
    `stream(cfg.seed, f"tasks-{i}")` uniformly from that room's free cells with row in `band`, excluding the
    previous position;
  * `family="S"`, `meta={"params": params, "seed": cfg.seed}`.
* Dispatch `complements` in `build_scenario`.
* `narrate.py` scenario table: add `"complements": ("S", "Three rooms in a row; each wall has a doorway at the
  top blocked by a pallet and open doors at the bottom. Opening one doorway saves nothing; opening both saves a
  lot (the obstacles are complements).")`. Follow the format of the neighbouring entries exactly.
* `docs/research/demo-guide.md` scenario table: add the row
  `| \`complements\` | two doorways in series | only a two-step plan sees the saving |`.

Why the saving is zero for one doorway (put it in the builder's docstring): from row `r1 <= 6` in the west to
`r2 <= 6` in the east, the doors cost `(12-r1) + (12-r2)` rows of vertical travel. Opening only the west doorway
costs `|r1-3| + 9 + (12-r2)`, which is the same or more. The same holds for the east doorway by symmetry.

### 5.6 Reporting fixes (`run_doi.py`, `src/doi/metrics.py`)

* `print_summary` and `print_comparison`: for the `hindsight` row, print `r.J_censored + r.hindsight_buy` in the
  cost column (today the buy is left out). Directly under each table, add the line:
  `hindsight is a relaxed lower reference (obstacles vanish, no walking); it is not achievable and not an upper bound.`
* `metrics.summary_row`: add the column `"static_travel"`, computed only when `full` (else `nan`). It is
  `sum(_dist(scenario.grid, _obstacles_at(result, p.tick), p.start, p.goal, U) for p in result.plan_infos)`.
  Also add `"congestion_excess" = result.J - result.push_cost - result.fee_total - static_travel`
  (`nan` when not full). Use `U = _unreachable(result)`.

### 5.7 Tests

New file `tests/doi/test_doi_bundles.py`:
1. `test_complements_one_doorway_saves_nothing`: build `complements` (`n_robots=2`, `tasks_per_robot=4`,
   `seed=1`). For every task pair, the BFS distance with only the west pallet removed equals the distance with
   both present, and likewise for the east pallet. With both removed it is strictly smaller.
2. `test_single_step_arms_never_push_on_complements`: `run_arms(SimConfig(scenario="complements", n_robots=6,
   tasks_per_robot=8, seed=1), ["never", "rof", "central"])`: every arm has `removals == 0`.
3. `test_two_step_plans_push_on_complements`: the same with `bundle_max=2` and the arms `["never", "rof",
   "central"]`: `central.removals >= 2`, `rof.removals >= 2`, and `rof.J_censored < never.J_censored`.
4. `test_bundle_max_one_changes_nothing`: for `single_block`, `two_blocks_parallel` and `multi_block_wall`
   (`n_robots=6`, `tasks_per_robot=6`, `seed=2`), `rof` with `bundle_max=1` gives the same `J` as the default
   config.
5. `test_new_arms_run_and_finish`: `rof_r` and `rof_p` on `single_block` and `complements` (`bundle_max=2`):
   `not stalled`, `unfinished_tasks == 0`.
6a. `test_all_length_plans_includes_every_length`: on the G1 map (`scenario_from_ascii(G1_ROWS, ...)`), robot
   at `(1,0)`, goal `(1,6)`, `kappa=1`, `fee=1`, `push_max=6`, pallet eligible. The keys of `all_length_plans` are
   exactly east 1, 2, 3 and west 1, 2. East 4 lands on the goal and is skipped. West plans are approached from
   `(1,3)` by the long way round, and robots are not obstacles to the legality check. The existing
   `candidate_plans` still returns at most one plan per direction.
6b. `test_plan_key_shapes`: `plan_key` of a `PushPlan` is a 1-tuple and of a `BundlePlan` a 2-tuple of
   `(obstacle, direction, steps)`, and both are hashable.
6. `test_hindsight_row_includes_buy`: capture `print_summary` output for `single_block` with `free` and
   `hindsight`; the hindsight cost printed equals `J_censored + hindsight_buy`.

If test 3 fails, that is a **stop condition** (0.3). Do not change the map, `kappa`, `fee`, the seeds or the
counts. Record the per-arm `removals`, `J_censored`, the triggers list and the push log in the status file.

**Acceptance:** the full suite passes (214 existing + new). Commit:
`feat(doi): two-step push plans, randomized and predicted arms, complements scenario, honest benchmark rows`.

---

## Phase 6. Experiment E1: `experiments/doi_e1_ratio.py`

One script, two parts, flags `--quick`, `--part {core,grid,all}` (default `all`) and `--out DIR` (default
`experiments/results/doi/e1`). It writes `core.csv` and `grid.csv` and prints a summary table. It exits with
code 1 if any bound check fails.

### 6.1 Part `core` (pure numbers, uses `core.py` only)

Families of savings sequences (integers):
* `constant`: `s = [1] * L`, with `L` in `{c // 2, c, 2*c, 10*c}`;
* `stop`: `s = [1] * c` (the adversary stops right after full information would fire);
* `random`: `T = 5*c` uniform integers `0..3` from `stream(seed, f"e1-{c}")`.

Grid: `c` in `{5, 20, 80}`; seeds `0..9`; views `full`, `own` with `n` in `{2, 4, 8, 16}` (round robin),
`delay` with `delta` in `{2, 8, 32}` (round robin `n = 4`), `sample` with `p` in `{0.25, 0.5}` (`n = 4`).
Arms:
* `threshold` with `theta` in `{0.5, 1, 2}`;
* `randomized` (full view: `expected_randomized_full`; other views: the mean over draws `0..199` of
  `run_randomized`);
* `predicted` with `lam` in `{0.25, 0.5, 1}` and predictor `oracle` (the true total), `inverted`, or
  `noisy:1.0` (`true * exp(n)`, `n = stream(seed, "e1-noise").gauss(0,1)`).

Columns: `family, c, seed, view, n, delta, p, arm, theta, lam, predictor, alg, opt, ratio, rho, D, bound,
bound_holds`. `bound` is, for threshold, `min(bound_coverage(theta, rho) if rho > 0 else inf,
bound_deficit(theta, D, c))`; for predicted, `bound_predicted(lam, rho)` if `rho > 0`, else `nan`; for
randomized with a full view, `E_RATIO`; `nan` otherwise. `bound_holds = alg <= bound * opt + 1e-9`, and it is
`True` when `bound` is `nan`. `ratio = alg/opt`
when `opt > 0`, else `1.0` if `alg == 0`.

`--quick`: `c = 20` only, seeds `0..2`, `own` with `n` in `{2, 4}` and `full`, families `constant` and `stop`.

### 6.2 Part `grid` (abstract model)

Instances:
* `g1(m)` for `m` in `{1, 2, 4, 8, 16}`, `n_agents` in `{1, 4}`;
* `random_small_instance(seed, k, T, n_agents)` for seeds `0..19`, `k` in `{1, 2}`, `T` in `{10, 30}`,
  `n_agents` in `{1, 4}`.

Arms (all with views `full` and `own`):
* `never`;
* `threshold` with `bundle_max` 1 and 2;
* `randomized` with `bundle_max=2`, averaged over `seed` in `0..19` of the rule;
* `predicted` with `bundle_max=2`, `lam=0.5`, predictor `oracle` and `inverted`.

Columns: `instance, m_or_seed, k, T, n_agents, arm, bundle_max, view, alg, opt, lb, ratio_opt, ratio_lb,
n_states, fires`. If `exact_opt` raises `StateSpaceTooLarge`, write `opt = nan`, `n_states = -1` and keep going.

`--quick`: `g1(m)` for `m` in `{1, 4}` with `n_agents = 1`, plus seeds `0..2` with `k = 1`, `T = 10`,
`n_agents = 1`.

### 6.3 Summary printed by the script

For each `(part, arm, view)` print `median ratio, max ratio, share of runs where bound_holds`, in a fixed-width
table sorted by part, arm and view. Nothing else.

### 6.4 Test: `tests/doi/test_doi_e1_script.py`

`test_e1_quick_runs`: run the script with `--quick --out <tmp_path>` via `subprocess` (using `sys.executable`).
It must return 0, create both CSVs, and every `bound_holds` in `core.csv` must be True.

**Acceptance:** tests pass. Run `venv/bin/python experiments/doi_e1_ratio.py --quick` once and keep its printed
summary for Phase 7. Commit: `feat(doi): E1 competitive-ratio experiment against the exact optimum`.

---

## Phase 7. Documents

1. Replace the whole of `docs/research/theory.md` with the text of **Appendix A**, verbatim. (The old version
   stays in git history.)
2. In `docs/research/results.md`, append a section `## E1 quick pilot (2026-10-03)` containing exactly: the
   command you ran, the printed summary in a code block, and the sentence
   `Pilot only (quick mode); not a result. The full run has not been made.`
   Change nothing else in that file.
3. In `src/doi/README.md`, add `rof_r` and `rof_p` to the arms table, in the same format as the existing rows,
   with the `ARM_NOTES` text. Add one row to the experiments section:
   `E1 | competitive ratio against the exact optimum in the abstract model | experiments/doi_e1_ratio.py`.
   Change nothing else.

**Acceptance:** `venv/bin/python -m pytest tests/doi -q` passes. Commit:
`docs(doi): theory for the push model (coverage theorem, randomized and predicted thresholds, complements)`.

When Phase 7 is committed, end your turn with: the list of commits, the test count, and the E1 quick summary.
Do not add commentary on what the results mean.

---

## What NOT to do (summary, all binding)

* Do not alter physics, motion, gossip, CRDTs, incidents, the LLM code, the animation, the wizard or the builder.
* Do not change `candidate_plans` (add `all_length_plans` beside it), `series_blocks`, existing defaults, or any existing expected value.
* Do not make `bundle_max = 2` the default.
* Do not put an LLM anywhere on the decision path. This spec adds no LLM work at all.
* Do not "improve" the theorems, rename the arms, or add arms, metrics, flags or scenarios that are not listed.
* Do not cap, shorten or skip a test because it is slow. Stop instead (0.3).
* Do not run the full E1, and do not write results or interpretation beyond Phase 7 step 2.
* Do not commit `toy.gif` or `experiments/results/`, and do not add co-author or tool attribution to commits.

---

## Appendix A. Text of `docs/research/theory.md` (paste verbatim)

```markdown
# Theory: rent-or-push with partial information

The propositions are stated for the **abstract model** (`src/doi/abstract/`): requests are served one at a time
at their exact shortest-path distance; an action pushes one obstacle `k` cells in a straight line at cost
`fee + k * kappa * w`; approach walking and congestion are not modelled. The simulator in `src/doi` is the
empirical check of how far the executed system departs from this model (`static_travel`,
`congestion_excess`).

## Setting

Requests `j = 0..T-1` arrive in order; request `j` is served by agent `g(j)`. For one candidate action (or
two-step plan) `a` with cost `c > 0`, `s_j >= 0` is what request `j` saves if `a` has already been taken
(`s_j = serve_j(X) - serve_j(X after a)`), `S_i = s_0 + ... + s_i`. Agent `g(i)` knows the savings of the
requests in its view `V_i`, a subset of `{0..i}`, so its known evidence is `K_i = sum_{j in V_i} s_j <= S_i`.

**Rule** (threshold `theta > 0`): take `a` before serving the first request `i` with `K_i > 0` and
`K_i >= theta * c`.

**Comparator** `OPT_a = min(c, S_{T-1})`: take `a` before the first request, or never. All costs are
*avoidable* costs: the cost of serving every request as if `a` had been taken at time 0 is subtracted from both
sides.

**Single-candidate setting.** The propositions assume that `a` is the only action the rule can take and that
savings are non-negative (no collateral: the landing cell never lengthens a request) and do not depend on time.
Beyond this setting (several obstacles, collateral, executed motion) the bounds are not claimed; E1 measures the
ratio against the exact optimum of the whole abstract problem instead.

Coverage `rho = min_{i: S_i > 0} K_i / S_i`; deficit `D = max_i (S_i - K_i)`.

## Theorem 1 (coverage)

In the single-candidate setting the rule's avoidable cost satisfies

    ALG <= (1 + theta / rho) / min(1, theta) * OPT_a        and        ALG <= (theta + 1 + D / c) / min(1, theta) * OPT_a.

*Proof.* Suppose the rule fires at `i`. Every earlier agent saw `K_{i-1} < theta c`. Since `K_{i-1} >= rho S_{i-1}`
(or `S_{i-1} = 0`), the rent paid is `S_{i-1} < theta c / rho`; likewise `S_{i-1} <= K_{i-1} + D < theta c + D`.
So `ALG < theta c / rho + c` (resp. `theta c + D + c`). Because `S_{T-1} >= S_i >= K_i >= theta c`,
`OPT_a >= min(c, theta c) = min(1, theta) c`. Dividing gives both bounds. If the rule never fires,
`ALG = S_{T-1}`; when `S_{T-1} <= c` the ratio is 1, otherwise `OPT_a = c` and the last agent's
`K_{T-1} < theta c` gives `S_{T-1} < theta c / rho` (resp. `theta c + D`), which is within both bounds. QED.

With full information (`rho = 1`, `D = 0`, `theta = 1`) this is the classical ratio 2 of ski rental
(Karlin, Manasse, Rudolph, Sleator 1988). The current request is counted before it is served, so there is no
additive one-request term.

**Corollary 1 (isolation, tight).** If `n` agents serve requests round robin and each knows only its own
requests, then `rho >= 1/n` and `ALG <= (n + 1) OPT_a` for `theta = 1`. The bound is tight: with equal savings
`s` per request and `s / c -> 0`, every agent crosses the threshold at about the same time, so the fleet pays
about `n c` of rent before the first purchase, and the ratio tends to `n + 1`. (Test:
`test_isolation_lower_bound_approaches_n_plus_one`.)

**Corollary 2 (delay).** If each agent knows every request up to `i - delta` and all its own requests, then
`D <= (delta - 1) s_max` and `ALG <= (2 + (delta - 1) s_max / c) OPT_a` for `theta = 1`.

Coverage is the single quantity through which communication range, loss and latency enter the guarantee; the
simulator logs it at every trigger (`known` against the true fleet saving).

## Theorem 2 (randomized threshold, full information)

Draw `z` with density `e^z / (e - 1)` on `[0, 1]` and use `theta = z`. In the single-candidate setting with full
views, `E[ALG] <= e / (e - 1) * OPT_a`.

*Proof.* If the rule fires at `i`, the rent paid is `S_{i-1} < z c`, so for every `z` the cost is at most that of
the continuous rule that pays exactly `z c`. For `S = S_{T-1} >= c`:
`E[ALG] <= int_0^1 (z c + c) e^z / (e - 1) dz = c e / (e - 1)`. For `S < c`, with `x = S / c`:
`E[ALG] <= int_0^x (z c + c) e^z / (e - 1) dz + S (e - e^x) / (e - 1) = S e^x / (e - 1) + S (e - e^x) / (e - 1)
= S e / (e - 1)`. QED. (Classical: Karlin, Manasse, McGeoch, Owicki 1994. The discrete, count-before-serve form
is computed exactly by `expected_randomized_full`.) With partial views the expectation is measured, not bounded.

## Theorem 3 (predicted threshold)

A prediction says whether the total saving of `a` will reach `c`; it is made once per agent and action. Use
`theta = lambda` if the prediction says yes and `theta = 1 / lambda` otherwise, with `0 < lambda <= 1`. In the
single-candidate setting:

* robustness, whatever the prediction: `ALG <= (1 + 1 / (lambda rho)) OPT_a`;
* consistency, if the prediction is right and views are full: `ALG <= (1 + lambda) OPT_a`.

*Proof.* Robustness: Theorem 1 with `theta = lambda` gives `1/lambda + 1/rho`, and with `theta = 1/lambda` gives
`1 + 1/(lambda rho)`, which is the larger (their difference is `(1 - 1/lambda)(1 - 1/rho) >= 0`). Consistency:
if `S_{T-1} >= c` and the prediction says yes, the rule fires with rent below `lambda c` and `OPT_a = c`; if
`S_{T-1} < c` and the prediction says no, the threshold `c / lambda > S_{T-1}` is never reached and
`ALG = S_{T-1} = OPT_a`. QED. (This is the deterministic algorithm of Purohit, Svitkina and Kumar, NeurIPS 2018,
here with coverage; the predictions in the simulator come from the forecast of the remaining traffic.)

## Proposition 4 (complements need multi-step plans)

If two obstacles are complements, so that moving either one alone saves nothing for any recorded request, then
every rule that evaluates single actions has zero evidence for each of them, never fires, and pays `S_{T-1}`
against `OPT_a = min(c_1 + c_2, S_{T-1})`. That ratio is unbounded in `T`. Evaluating two-step plans makes the
pair a single candidate, so Theorem 1 applies with `c = c_1 + c_2`. The `complements` scenario realises this:
with both pallets in place, opening one doorway saves no vertical travel for traffic in rows 0..6. (Tests:
`test_single_step_arms_never_push_on_complements`, `test_two_step_plans_push_on_complements`,
`test_g1_two_step_plan_reaches_the_optimum`.)

## Benchmarks

* `exact_opt`: the optimum of the abstract problem over all push schedules (layered Dijkstra over reachable
  configurations). It is exact when it returns; it refuses above 20000 configurations.
* `vanish_lower_bound`: a valid lower bound on `exact_opt`. Moved obstacles are charged one push and vanish;
  unmoved obstacles stay where they are.
* The simulator's `hindsight` arm is the vanish bound executed with motion. It is a relaxed reference: it is
  neither achievable nor an upper bound.

## What is not claimed

* No bound for several interacting candidates, for collateral, or for executed motion with congestion. The
  bounds really do fail there: on `g1(6)` with fee 4, the predicted rule with an adversarial prediction pays 33
  against the robustness bound of 30, because several candidate plans each cross a raised threshold in turn
  (`test_multi_candidate_counterexample_on_g1`).
* No claim of novelty for Theorems 1 to 3 until the prior-art reads in `prior-art-verification.md` are done.
  The coverage form and its use as the measure of decentralisation are the parts to check first.
```
