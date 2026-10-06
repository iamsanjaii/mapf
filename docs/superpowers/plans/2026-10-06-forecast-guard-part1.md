# Forecast Guard, Part 1 (no model) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the guard rule, text notices, the `shift_notice` scenario and the arm `rof_a` with the four forecasters that need no model, then measure how much any forecaster can change fleet cost.

**Architecture:** A robot asks a forecaster one yes/no question per candidate action ("will the fleet's saving reach the price?") once its known saving reaches `lam * price`. The existing predicted-threshold rule turns the answer into a threshold: `lam` on yes, `1 / lam` on no, 1 while there is no answer. What the robot knows is frozen into a plain `ForecastCase`, so forecasters never touch the simulator.

**Tech Stack:** Python 3 in `venv/`, standard library, `numpy`, `pandas`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-forecast-agent-guard-design.md`. This plan implements build-order steps 1 to 5 of its section 16. Steps 6 to 9 (chat client, agent loop, model forecaster, datasets, E9) are Part 2, written after the gate in Task 8.

## Global Constraints

- Run Python as `venv/bin/python` (there is no `python` on PATH). Run tests with `venv/bin/python -m pytest tests/doi -q`.
- Every existing test must pass unchanged. Do not edit an existing test's expected value.
- No test and no script in this plan calls a language model.
- Every random draw uses `src.doi.rng.stream` or `src.doi.rng.u01`. Never `random.random()`, unseeded `numpy.random`, or the clock.
- Edit existing modules only where a task says so. Do not change `world.py`, `spacetime.py`, `network.py`, `pushplan.py`, `carryplan.py`, `evidence.py`.
- Commit messages carry no `Co-Authored-By` line, no "Generated with" line and no mention of Claude or Anthropic.
- Never commit `toy.gif` or anything under `experiments/results/`.
- Match the surrounding code: type hints on every function, a module docstring on every new file, no print statements in library code.
- `lam` is in `(0, 1]`. The threshold is `lam` on a yes, `1 / lam` on a no, and exactly `1.0` with no visible answer.

## Departures from the spec

Each is small and deliberate. The owner approves them by approving this plan.

1. `NoticeFeed` has one method, `emit`. Delivery has no delay, so a separate `deliver` would do nothing.
2. `RunResult` gains `notice_reach` (how many robots knew each notice at the end) and `forecast_cases`.
3. No forecast is requested when the price is 0 or when `lam == 1`. The answer cannot change a decision there.
4. `shift_notice` refuses a run in which the busy band could never move (`tasks_per_robot <= shift_after_task` in modes `true` and `missing`).
5. `ForecastResult` lives in `src/doi/forecast/result.py`, so Part 2's agent loop and the forecasters can both import it.
6. Config gains only `forecaster` and `agent_max_forecasts` here. `agent_mode`, `agent_cache` and `agent_live` come with Part 2.

## Review Focus

1. `shift_notice` run with too few tasks for the band to move: a clear error, not notices silently labelled true (Task 4).
2. A notice due when every robot has finished: ignored, no error (Task 4).
3. `rof_a` on a scenario with no zones and no notices: runs, empty trip table, the keyword forecaster falls back to the numeric one (Task 7).
4. A candidate with price 0, or `lam = 1`: no forecast is requested (Task 7).
5. The same notice id arriving with different text from two neighbours: every robot converges on one version (Task 2).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/doi/abstract/core.py` | modify | `run_guarded`, `guarded_wait`, `bound_guarded_consistency` |
| `src/doi/crdt.py` | modify | `NoticeRecord`, `NoticeSet` |
| `src/doi/belief.py` | modify | the `notices` component |
| `src/doi/notices.py` | create | wording banks, `render_text`, `NoticeFeed` |
| `src/doi/scenarios.py` | modify | `Notice`, `Scenario.notices`, `Scenario.zones`, scenario `shift_notice` |
| `src/doi/forecast/__init__.py` | create | empty |
| `src/doi/forecast/result.py` | create | `ForecastResult` |
| `src/doi/forecast/case.py` | create | `ForecastCase`, `build_case`, the truth label |
| `src/doi/forecast/forecasters.py` | create | numeric, keyword, oracle, inverted; `make_forecaster` |
| `src/doi/policies.py` | modify | `on_tick` hook, `ForecastState`, `GuardedPolicy`, `make_policy` |
| `src/doi/agent.py` | modify | forecast fields, re-evaluation key, `ingest_notice` |
| `src/doi/runner.py` | modify | notice feed, `on_tick`, result fields |
| `src/doi/metrics.py` | modify | result fields, `forecast_columns`, summary columns |
| `src/doi/config.py` | modify | arm `rof_a`, `forecaster`, `agent_max_forecasts` |
| `src/doi/narrate.py` | modify | notes for `shift_notice` and `rof_a` |
| `experiments/doi_e9_headroom.py` | create | the headroom pilot |
| `data/forecasts/README.md` | create | how to collect human-written notices |
| `docs/research/theory.md`, `docs/research/results.md`, `src/doi/README.md` | modify | Proposition 5, pilot output, arm and scenario rows |

---

### Task 1: The guard on pure numbers, and Proposition 5

**Files:**
- Modify: `src/doi/abstract/core.py` (append after `bound_predicted`)
- Modify: `docs/research/theory.md` (insert before `## Benchmarks`)
- Test: `tests/doi/test_doi_abstract_guard.py`

**Interfaces:**
- Consumes: `prefix`, `known`, `opt_single`, `_check`, `SingleOutcome`, `run_predicted`, `run_threshold`, `bound_predicted`, `coverage` from `src/doi/abstract/core.py`.
- Produces: `run_guarded(s, c, views, lam, says_yes, delay=0) -> SingleOutcome`, `guarded_wait(s, c, views, lam, delay) -> float`, `bound_guarded_consistency(lam, w, c) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_abstract_guard.py`:

```python
import pytest
from src.doi.abstract.core import (
    bound_guarded_consistency, bound_predicted, coverage, guarded_wait, round_robin, run_guarded, run_predicted,
    run_threshold, views_delay, views_full, views_own, views_sample,
)
from src.doi.rng import stream


def _cases(name, full_only=False, n_cases=500):
    rng = stream(0, name)
    for k in range(n_cases):
        T = rng.randint(1, 60)
        s = [float(rng.randint(0, 5)) for _ in range(T)]
        c = float(rng.randint(1, 40))
        n = rng.randint(1, 8)
        agents = round_robin(T, n)
        kind = "full" if full_only else rng.choice(["full", "own", "delay", "sample"])
        if kind == "full":
            views = views_full(T)
        elif kind == "own":
            views = views_own(agents)
        elif kind == "delay":
            views = views_delay(agents, rng.randint(1, 10))
        else:
            views = views_sample(agents, rng.choice([0.25, 0.5, 0.9]), k)
        lam = rng.choice([0.25, 0.5, 1.0])
        delay = rng.randint(0, 20)
        yield s, c, views, lam, delay


def test_guarded_hand_case():
    s, views = [1.0] * 30, views_full(30)
    now = run_guarded(s, 10.0, views, 0.5, True, 0)
    assert now.fire == 4 and now.alg == 14.0 and now.opt == 10.0
    late = run_guarded(s, 10.0, views, 0.5, True, 3)
    assert late.fire == 7 and late.alg == 17.0
    assert guarded_wait(s, 10.0, views, 0.5, 3) == 3.0
    assert bound_guarded_consistency(0.5, 3.0, 10.0) == pytest.approx(1.8)


def test_guarded_delay_zero_equals_predicted():
    for s, c, views, lam, _delay in _cases("guard-eq"):
        for says_yes in (True, False):
            assert run_guarded(s, c, views, lam, says_yes, 0) == run_predicted(s, c, views, c if says_yes else 0.0, lam)


def test_guarded_robustness_bound():
    checked = 0
    for s, c, views, lam, delay in _cases("guard-rob"):
        rho = coverage(s, views)
        for says_yes in (True, False, None):
            out = run_guarded(s, c, views, lam, says_yes, delay)
            if out.opt > 0 and rho > 0:
                checked += 1
                assert out.alg <= bound_predicted(lam, rho) * out.opt + 1e-9
    assert checked > 1000


def test_guarded_consistency_bound():
    for s, c, views, lam, delay in _cases("guard-con", full_only=True):
        if sum(s) >= c:                                   # a right "yes"
            out = run_guarded(s, c, views, lam, True, delay)
            w = guarded_wait(s, c, views, lam, delay)
            assert out.alg <= bound_guarded_consistency(lam, w, c) * out.opt + 1e-9
        else:                                             # a right "no"
            out = run_guarded(s, c, views, lam, False, delay)
            assert out.alg == out.opt


def test_guarded_no_answer_is_the_classical_rule():
    for s, c, views, lam, delay in _cases("guard-none"):
        assert run_guarded(s, c, views, lam, None, delay) == run_threshold(s, c, views, 1.0)


def test_guarded_rejects_bad_input():
    with pytest.raises(ValueError):
        run_guarded([1.0], 2.0, views_full(1), 0.0, True)
    with pytest.raises(ValueError):
        run_guarded([1.0], 2.0, views_full(1), 0.5, True, -1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/doi/test_doi_abstract_guard.py -q`
Expected: collection error, `ImportError: cannot import name 'bound_guarded_consistency'`.

- [ ] **Step 3: Implement**

In `src/doi/abstract/core.py`, add after `bound_predicted`:

```python
def _request_index(K: Sequence[float], c: float, lam: float) -> Optional[int]:
    """First request at which the known evidence reaches lam * c: the forecast is asked for there."""
    for i, k in enumerate(K):
        if k > 0 and k >= lam * c:
            return i
    return None


def run_guarded(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]], lam: float,
                says_yes: Optional[bool], delay: int = 0) -> SingleOutcome:
    """Predicted-threshold rule whose forecast is asked for lazily and may arrive late or never.

    The forecast is requested at the first request r with K_r > 0 and K_r >= lam * c, and is visible from request
    r + delay. Before that, and always when says_yes is None, the threshold is 1 (the classical rule). After it the
    threshold is lam on a yes and 1 / lam on a no."""
    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    _check(s, c, views)
    S = prefix(s)
    K = known(s, views)
    opt = opt_single(s, c)
    r = _request_index(K, c, lam)
    for i in range(len(s)):
        thr = 1.0
        if says_yes is not None and r is not None and i >= r + delay:
            thr = lam if says_yes else 1.0 / lam
        if K[i] > 0 and K[i] >= thr * c:
            return SingleOutcome(i, (S[i - 1] if i > 0 else 0.0) + c, opt)
    return SingleOutcome(None, S[-1] if S else 0.0, opt)


def guarded_wait(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]], lam: float, delay: int) -> float:
    """W: the saving that passes between the request for a forecast and its arrival (0.0 if never requested)."""
    S = prefix(s)
    r = _request_index(known(s, views), c, lam)
    if r is None:
        return 0.0
    a = min(r + delay, len(s))

    def before(i: int) -> float:
        return S[i - 1] if i > 0 else 0.0

    return before(a) - before(r)


def bound_guarded_consistency(lam: float, w: float, c: float) -> float:
    return 1.0 + min(1.0, lam + w / c)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/doi/test_doi_abstract_guard.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Add Proposition 5 to the theory document**

In `docs/research/theory.md`, insert this block immediately before the line `## Benchmarks`:

```markdown
## Proposition 5 (the guard with late or missing answers)

The forecast of Theorem 3 need not be available at once. Let `r` be the first request with `K_r > 0` and
`K_r >= lambda c` (the forecast is requested there) and let the answer become visible at request `a >= r`
(`a` infinite: no answer). The threshold is `theta_i = 1` for `i < a`, and for `i >= a` it is `lambda` on a yes
and `1 / lambda` on a no. In the single-candidate setting:

* robustness, whatever the answer and whenever it arrives: `ALG <= (1 + 1 / (lambda rho)) OPT_a`;
* consistency, if the answer is right and views are full: `ALG <= (1 + min(1, lambda + W / c)) OPT_a` on a yes,
  where `W = S_{a-1} - S_{r-1}` is the saving that passed while waiting, and `ALG = OPT_a` on a no;
* no answer: the classical rule, `ALG <= (1 + 1 / rho) OPT_a`.

*Proof of robustness.* Suppose the rule fires at `i`. If `i = 0` then `ALG = c` and `OPT_a >= min(c, K_0) >=
lambda c`. Otherwise the rent is `S_{i-1}`, and because the rule did not fire at `i - 1`,
`S_{i-1} <= K_{i-1} / rho < theta_{i-1} c / rho` (or `S_{i-1} = 0`). (1) `theta_i = 1 / lambda`: `K_i >= c`, so
`OPT_a = c`, and `theta_{i-1} <= 1 / lambda` gives rent below `c / (lambda rho)`. (2) `theta_i = 1`: `OPT_a = c`
and `theta_{i-1} = 1`, rent below `c / rho`. (3) `theta_i = theta_{i-1} = lambda`: rent below `lambda c / rho`
and `OPT_a >= lambda c`, ratio below `1 / rho + 1 / lambda`. (4) `theta_i = lambda`, `theta_{i-1} = 1`: if
`S_{i-1} >= c` then `OPT_a = c` and the ratio is below `1 + 1 / rho`; if `S_{i-1} < c` then
`OPT_a >= max(S_{i-1}, lambda c)` and `ALG / OPT_a <= 1 + 1 / lambda`. Each is at most `1 + 1 / (lambda rho)`,
since `rho <= 1` and `(1 - 1 / lambda)(1 - 1 / rho) >= 0`. If the rule never fires, `ALG = S_{T-1}`; either
`S_{T-1} <= c` and the ratio is 1, or `OPT_a = c` and `S_{T-1} < c / (lambda rho)`. QED.

*Proof of consistency.* With full views `K = S`. A right yes means `S_{T-1} >= c`; the rule fires at `a` at the
latest, the rent is below `c` (the waiting threshold is 1) and at most `S_{a-1} = S_{r-1} + W < lambda c + W`. A
right no means `S_{T-1} < c`; neither threshold 1 nor `1 / lambda` is reached, so `ALG = S_{T-1} = OPT_a`. QED.
(Functions `run_guarded`, `guarded_wait`, `bound_guarded_consistency`; tests in `test_doi_abstract_guard.py`.)
```

In the same file, under `## What is not claimed`, append this bullet:

```markdown
* No claim of novelty for Proposition 5 until learning-augmented ski rental with delayed predictions has been
  checked in the literature.
```

- [ ] **Step 6: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass (6 more than before).

```bash
git add src/doi/abstract/core.py tests/doi/test_doi_abstract_guard.py docs/research/theory.md
git commit -m "feat(doi): guard rule with late or missing forecasts, with its bounds"
```

---

### Task 2: Notice records in the belief, and a pin on existing runs

**Files:**
- Modify: `src/doi/crdt.py` (append at the end)
- Modify: `src/doi/belief.py`
- Test: `tests/doi/test_doi_notice_records.py`, `tests/doi/test_doi_guard_neutral.py`

**Interfaces:**
- Consumes: `_Versioned`, `Clock` from `src/doi/crdt.py`; `BeliefState` from `src/doi/belief.py`.
- Produces: `NoticeRecord(notice_id: str, tick: int, text: str)`; `NoticeSet` with `add(rec)`, `get(notice_id) -> Optional[NoticeRecord]`, `records() -> List[NoticeRecord]`, `merge`, `units`, `copy`, `delta`, `canonical`; `BeliefState.notices` and `BeliefState.add_notice(rec)`.

- [ ] **Step 1: Pin the cost of existing runs (this test passes before any change)**

Create `tests/doi/test_doi_guard_neutral.py`:

```python
"""The forecast-guard work must not change any existing arm. These costs were recorded before it started."""
import pytest
from src.doi.config import SimConfig
from src.doi.runner import run_arms

PINNED = {
    "single_block": {"rof": 519.0, "rof_p": 519.0, "central": 519.0},
    "multi_block_wall": {"rof": 507.0, "rof_p": 503.0, "central": 498.0},
    "shift": {"rof": 501.0, "rof_p": 501.0, "central": 501.0},
}


@pytest.mark.parametrize("scenario", sorted(PINNED))
def test_existing_arms_cost_what_they_did(scenario):
    res = run_arms(SimConfig(scenario=scenario, n_robots=6, tasks_per_robot=6, seed=2), ["rof", "rof_p", "central"])
    assert {name: r.J for name, r in res.items()} == PINNED[scenario]
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_guard_neutral.py -q`
Expected: `3 passed`. If a value differs, stop: the baseline moved and the pinned numbers need the owner's review.

- [ ] **Step 2: Write the failing tests for the records**

Create `tests/doi/test_doi_notice_records.py`:

```python
from src.doi.belief import BeliefState
from src.doi.crdt import NoticeRecord, NoticeSet

A = NoticeRecord("n0", 5, "wave 2 is all in the south bays")
B = NoticeRecord("n1", 5, "the north bays are finished after this wave")
A_LATER = NoticeRecord("n0", 9, "a different wording")


def _set(*recs):
    s = NoticeSet()
    for r in recs:
        s.add(r)
    return s


def test_merge_is_commutative_associative_idempotent():
    ab = _set(A)
    ab.merge(_set(B))
    ba = _set(B)
    ba.merge(_set(A))
    assert ab.canonical() == ba.canonical() and ab.records() == [A, B]
    again = ab.copy()
    assert again.merge(ab) is False and again.canonical() == ab.canonical()
    left = _set(A)
    left.merge(_set(B))
    left.merge(_set(A_LATER))
    right = _set(B)
    right.merge(_set(A_LATER))
    right.merge(_set(A))
    assert left.canonical() == right.canonical()


def test_same_id_with_different_content_converges_on_one_version():
    one, two = _set(A), _set(A_LATER)
    one.merge(_set(A_LATER))
    two.merge(_set(A))
    assert one.get("n0") == A and two.get("n0") == A          # the smaller (tick, text) wins on both sides
    assert one.get("missing") is None


def test_belief_carries_notices_in_full_and_delta_gossip():
    sender, full, delta = BeliefState(0, {}), BeliefState(1, {}), BeliefState(2, {})
    assert sender.units() == 0 and sender.version == 0        # an empty notice set adds nothing
    before = sender.version
    sender.add_notice(A)
    assert sender.version == before + 1 and sender.units() == 1
    full.merge(sender.snapshot())
    delta.merge(sender.delta_since(before))
    assert full.notices.records() == [A] and delta.notices.records() == [A]
    assert sender.delta_since(sender.version).units() == 0
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_notice_records.py -q`
Expected: `ImportError: cannot import name 'NoticeRecord'`.

- [ ] **Step 3: Implement the records**

Append to `src/doi/crdt.py`:

```python
@dataclass(frozen=True)
class NoticeRecord:
    notice_id: str
    tick: int                 # tick the notice was issued
    text: str


class NoticeSet(_Versioned):
    """Grow-only set of notices keyed by id. On a clash the smaller (tick, text) wins, so merges commute."""

    def __init__(self) -> None:
        self._init_versions()
        self._d: Dict[str, NoticeRecord] = {}

    def add(self, rec: NoticeRecord) -> None:
        old = self._d.get(rec.notice_id)
        if old is None or (rec.tick, rec.text) < (old.tick, old.text):
            self._d[rec.notice_id] = rec
            self._bump(rec.notice_id)

    def get(self, notice_id: str) -> Optional[NoticeRecord]:
        return self._d.get(notice_id)

    def records(self) -> List[NoticeRecord]:
        return [self._d[k] for k in sorted(self._d)]

    def merge(self, other: "NoticeSet") -> bool:
        before = dict(self._d)
        for rec in other._d.values():
            self.add(rec)
        return self._d != before

    def units(self) -> int:
        return len(self._d)

    def copy(self) -> "NoticeSet":
        out = NoticeSet()
        out._d = dict(self._d)
        self._copy_versions_to(out)
        return out

    def delta(self, v: int) -> "NoticeSet":
        out = NoticeSet()
        out._d = {k: r for k, r in self._d.items() if self._newer(k, v)}
        return out

    def canonical(self) -> Any:
        return frozenset(self._d.items())
```

In `src/doi/belief.py`:

Change the import line to add the two names:

```python
from src.doi.crdt import (AggRecordSet, Clock, GSet, MaxRegisterMap, NoticeRecord, NoticeSet, ObstructionRecord,
                          ObstructionSet, RecordSet, RentRecord)
```

In `BeliefState.__init__`, directly after the line `self.slots_full = GSet()            # slots known to hold an obstacle; a slot is never emptied`, add:

```python
        self.notices = NoticeSet()          # text notices about future traffic; read only by forecasters
```

Replace the `_COMPONENTS` tuple with:

```python
    _COMPONENTS = ("records", "agg", "census", "report_tick", "blocked_tick", "free_tick", "cls", "kind",
                   "obstructions", "slots_full", "notices")
```

Add this method after `add_obstruction`:

```python
    def add_notice(self, rec: NoticeRecord) -> None:
        self.notices.add(rec)
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_notice_records.py tests/doi/test_doi_guard_neutral.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/crdt.py src/doi/belief.py tests/doi/test_doi_notice_records.py tests/doi/test_doi_guard_neutral.py
git commit -m "feat(doi): notice records as a gossiped belief component"
```

---

### Task 3: Notice wording

**Files:**
- Create: `src/doi/notices.py`
- Create: `data/forecasts/README.md`
- Test: `tests/doi/test_doi_notice_text.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `KINDS`, `BANKS`, `PREFIXES`, `SURGE_CUES`, `DROP_CUES`, `load_human(path) -> List[dict]`, `render_text(kind, group, other, bank, rng, human_path=None) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_notice_text.py`:

```python
import json
import pytest
from src.doi.notices import BANKS, DROP_CUES, KINDS, SURGE_CUES, load_human, render_text
from src.doi.rng import stream

CUES = SURGE_CUES + DROP_CUES


def _has(text, cues):
    return any(c in text.lower() for c in cues)


def test_banks_share_no_template_and_test_bank_has_no_dev_cue():
    for kind in KINDS:
        assert len(BANKS["dev"][kind]) == 4 and len(BANKS["test"][kind]) == 4
        assert not set(BANKS["dev"][kind]) & set(BANKS["test"][kind])
        for template in BANKS["test"][kind]:
            assert not _has(template, CUES), template


def test_dev_bank_cues_match_their_kind():
    for template in BANKS["dev"]["surge"]:
        assert _has(template, SURGE_CUES) and not _has(template, DROP_CUES)
    for template in BANKS["dev"]["drop"]:
        assert _has(template, DROP_CUES) and not _has(template, SURGE_CUES)
    for template in BANKS["dev"]["distractor"]:
        assert not _has(template, CUES)


def test_render_is_deterministic_and_names_the_group():
    a = render_text("surge", "south bays", "north bays", "dev", stream(7, "notice-n0"))
    b = render_text("surge", "south bays", "north bays", "dev", stream(7, "notice-n0"))
    assert a == b and "south bays" in a and "north" not in a
    assert "north bays" in render_text("drop", "north bays", "south bays", "test", stream(7, "notice-n1"))


def test_render_rejects_unknown_kind_and_bank():
    with pytest.raises(ValueError):
        render_text("rumour", "south bays", "north bays", "dev", stream(0, "x"))
    with pytest.raises(ValueError):
        render_text("surge", "south bays", "north bays", "prod", stream(0, "x"))


def test_human_bank_reads_the_file_and_explains_a_missing_one(tmp_path):
    with pytest.raises(ValueError, match="data/forecasts/README.md"):
        load_human(str(tmp_path / "absent.jsonl"))
    path = tmp_path / "human.jsonl"
    rows = [{"kind": "surge", "group": "south bays", "text": "everything's heading south after lunch"},
            {"kind": "drop", "group": "north bays", "text": "north side is wrapped up"},
            {"kind": "distractor", "group": "", "text": "who left the pallet jack on charge"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert render_text("surge", "south bays", "north bays", "human", stream(0, "x"), str(path)) == rows[0]["text"]
    assert render_text("distractor", "north bays", "south bays", "human", stream(0, "x"), str(path)) == rows[2]["text"]
    with pytest.raises(ValueError, match="no human notice"):
        render_text("surge", "north bays", "south bays", "human", stream(0, "x"), str(path))
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_notice_text.py -q`
Expected: `ModuleNotFoundError: No module named 'src.doi.notices'`.

- [ ] **Step 2: Implement the wording**

Create `src/doi/notices.py`:

```python
"""Notices: short operational messages about future traffic. Their wording, and (see NoticeFeed) their delivery.

Two wording banks, `dev` and `test`, share no template and no cue phrase. The keyword forecaster's cue lists come
from `dev` only, so a rule that matches phrases cannot look as good as a model on `test` text by construction.
A third source, `human`, reads texts written by people (see data/forecasts/README.md).
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

KINDS = ("surge", "drop", "distractor")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANKS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "dev": {
        "surge": ("wave 2 is all in the {group}",
                  "the next batch of orders goes to the {group}",
                  "the {group} get busy after the break",
                  "picking moves to the {group} for the second half"),
        "drop": ("the {group} are finished after this wave",
                 "no more orders for the {group} after the break",
                 "the {group} go quiet for the second half",
                 "picking in the {group} ends soon"),
        "distractor": ("fire drill reminder: Thursday at 10",
                       "label printer at pack station 2 is out of ribbon",
                       "please return scanners to the charging dock",
                       "reminder: tidy the signage in the {group} this week"),
    },
    "test": {
        "surge": ("afternoon orders are concentrated on the {group}",
                  "expect most pick tasks in the {group} from now until end of shift",
                  "inbound has been re-slotted: the {group} take the remaining volume",
                  "work shifts from the {other} over to the {group}"),
        "drop": ("the {group} are done for today once current jobs clear",
                 "nothing further is scheduled in the {group}",
                 "remaining volume has been pulled out of the {group}",
                 "we are winding down the {group} and sending everything to the {other}"),
        "distractor": ("forklift training is in the yard at 3pm",
                       "the dock door sensor is being recalibrated",
                       "canteen closes early today",
                       "cycle count paperwork for the {group} is due Friday"),
    },
}
PREFIXES: Dict[str, Tuple[str, ...]] = {
    "dev": ("", "FYI: ", "supervisor: ", "radio: "),
    "test": ("", "ops update: ", "shift lead: ", "note from planning: "),
}
# Cue phrases of the keyword forecaster. Taken from the dev bank only; never tuned on test or human text.
SURGE_CUES = ("all in", "goes to", "get busy", "moves to")
DROP_CUES = ("finished", "no more", "go quiet", "ends soon")


def load_human(path: str) -> List[dict]:
    """Rows {"kind", "group", "text"} written by people. A relative path is taken from the repository root."""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        raise ValueError(f"human notice file {path} not found: see data/forecasts/README.md for how to collect it")
    with open(full) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_text(kind: str, group: str, other: str, bank: str, rng: random.Random,
                human_path: Optional[str] = None) -> str:
    """The text of one notice of `kind` about zone group `group` (`other` is the other group)."""
    if kind not in KINDS:
        raise ValueError(f"unknown notice kind {kind!r}: choose from {', '.join(KINDS)}")
    if bank == "human":
        rows = [r for r in load_human(human_path or "")
                if r["kind"] == kind and (kind == "distractor" or r["group"] == group)]
        if not rows:
            raise ValueError(f"no human notice of kind {kind!r} for group {group!r} in {human_path}")
        return rng.choice(rows)["text"]
    if bank not in BANKS:
        raise ValueError(f"unknown notice bank {bank!r}: choose dev, test or human")
    template = rng.choice(BANKS[bank][kind])
    return rng.choice(PREFIXES[bank]) + template.format(group=group, other=other)
```

- [ ] **Step 3: Write the collection protocol**

Create `data/forecasts/README.md`:

```markdown
# Forecast datasets

| file | origin | use |
|---|---|---|
| `human_notices.jsonl` | written by people | notice texts for the held-out `human` cases |

`dev.jsonl`, `test.jsonl` and `human.jsonl` (forecast cases) are produced by a script in Part 2 of the build.

## Protocol for `human_notices.jsonl`

No code generates this file. It is collected by hand and committed when ready.

* One JSON object per line: `{"kind": "surge" | "drop" | "distractor", "group": "north bays" | "south bays" | "", "text": "..."}`.
* At least 100 texts: 40 surge, 40 drop, 20 distractor.
* Writers have not seen the templates in `src/doi/notices.py`.
* Each writer is told the kind and the zone group and writes one short message as they would on a radio or chat.
  A surge says work is about to move into that group. A drop says work there is about to stop. A distractor is
  any message that does not change where robots will go; its `group` is empty.
* No real names, sites or personal data. The texts are sent to a hosted API.
```

- [ ] **Step 4: Run the tests and commit**

Run: `venv/bin/python -m pytest tests/doi/test_doi_notice_text.py -q`
Expected: `5 passed`.

```bash
git add src/doi/notices.py data/forecasts/README.md tests/doi/test_doi_notice_text.py
git commit -m "feat(doi): notice wording banks and the protocol for human-written notices"
```

---

### Task 4: The `shift_notice` scenario and notice delivery

**Files:**
- Modify: `src/doi/notices.py` (append `NoticeFeed`)
- Modify: `src/doi/scenarios.py`
- Modify: `src/doi/agent.py` (one method)
- Modify: `src/doi/runner.py`
- Modify: `src/doi/metrics.py` (one field)
- Modify: `src/doi/narrate.py` (one entry)
- Test: `tests/doi/test_doi_shift_notice.py`

**Interfaces:**
- Consumes: `render_text` (Task 3); `NoticeRecord`, `BeliefState.add_notice`, `NoticeSet.get` (Task 2).
- Produces: `Notice(notice_id, emit_tick, kind, group, is_true, text)`; `Scenario.notices: List[Notice]`; `Scenario.zones: Dict[str, Tuple[Pos, ...]]`; scenario `shift_notice` with parameters `notice_mode`, `notice_tick`, `n_distractors`, `notice_bank`, `human_notices`; `NOTICE_MODES`; `NoticeFeed(scenario)` with `emit(t, agents)` and `delivered: int`; `RobotAgent.ingest_notice(rec)`; `RunResult.notice_reach: Dict[str, int]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_shift_notice.py`:

```python
import pytest
from src.doi.config import SimConfig
from src.doi.notices import NoticeFeed
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario


def _build(mode, seed=3, tasks=12, **params):
    cfg = SimConfig(scenario="shift_notice", n_robots=4, tasks_per_robot=tasks, seed=seed,
                    scenario_params={"notice_mode": mode, **params})
    return cfg, build_scenario(cfg)


def _about_the_band(s):
    return sorted((n.kind, n.group, n.is_true) for n in s.notices if n.kind != "distractor")


def test_modes_decide_the_tasks_and_the_notices():
    (_, t), (_, m), (_, f), (_, q) = (_build(x) for x in ("true", "missing", "false", "quiet"))
    assert t.tasks == m.tasks and f.tasks == q.tasks and t.tasks != f.tasks
    assert [g[:10] for g in t.tasks] == [g[:10] for g in f.tasks] and t.starts == f.starts
    plain = build_scenario(SimConfig(scenario="shift", n_robots=4, tasks_per_robot=12, seed=3))
    assert plain.tasks == m.tasks and plain.obstacles == m.obstacles
    assert _about_the_band(t) == [("drop", "north bays", True), ("surge", "south bays", True)]
    assert _about_the_band(f) == [("drop", "north bays", False), ("surge", "south bays", False)]
    assert _about_the_band(m) == [] and _about_the_band(q) == []
    for s in (t, m, f, q):
        distractors = [n for n in s.notices if n.kind == "distractor"]
        assert len(distractors) == 1 and distractors[0].group is None and 0 <= distractors[0].emit_tick <= 10


def test_zones_are_the_four_bay_blocks():
    _, s = _build("true")
    assert sorted(s.zones) == ["east north bays", "east south bays", "west north bays", "west south bays"]
    assert all(c < 10 and 0 <= r <= 6 for r, c in s.zones["west north bays"])
    assert all(c > 10 and 8 <= r <= 14 for r, c in s.zones["east south bays"])
    assert not any(cell in s.obstacles for cells in s.zones.values() for cell in cells)


def test_notice_text_is_deterministic_and_follows_the_bank():
    _, a = _build("true")
    _, b = _build("true")
    _, dev = _build("true", notice_bank="dev")
    assert [n.text for n in a.notices] == [n.text for n in b.notices]
    assert [n.text for n in a.notices] != [n.text for n in dev.notices]


def test_bad_mode_and_missing_human_file_are_errors(tmp_path):
    with pytest.raises(ValueError, match="notice_mode"):
        _build("sometimes")
    with pytest.raises(ValueError, match="data/forecasts/README.md"):
        _build("true", notice_bank="human", human_notices=str(tmp_path / "absent.jsonl"))


def test_too_few_tasks_for_the_band_to_move_is_an_error():
    with pytest.raises(ValueError, match="shift_after_task"):
        _build("true", tasks=10)
    with pytest.raises(ValueError, match="shift_after_task"):
        _build("missing", tasks=6)
    _build("quiet", tasks=6)                                   # the band is not meant to move: allowed


class _Stub:
    def __init__(self, rid, pos, finished=False):
        self.id, self.pos, self.finished, self.got = rid, pos, finished, []

    def ingest_notice(self, rec):
        self.got.append(rec)


def test_feed_delivers_each_notice_once_to_the_robot_nearest_the_centre():
    _, s = _build("true")
    feed = NoticeFeed(s)
    far, near = _Stub(0, (0, 0)), _Stub(1, (7, 10))
    for t in range(20):
        feed.emit(t, [far, near])
    assert far.got == [] and feed.delivered == len(s.notices) == 3
    assert sorted((r.notice_id, r.tick, r.text) for r in near.got) == \
        sorted((n.notice_id, n.emit_tick, n.text) for n in s.notices)


def test_feed_with_no_live_robot_does_nothing():
    _, s = _build("true")
    feed = NoticeFeed(s)
    done = _Stub(0, (7, 10), finished=True)
    for t in range(20):
        feed.emit(t, [done])
    assert done.got == [] and feed.delivered == 0


def test_notices_spread_by_gossip_and_not_without_it():
    cfg, s = _build("true")
    shared = run_episode(cfg.replace(policy="rof"), scenario=s)
    alone = run_episode(cfg.replace(policy="rof", r_comm=0.0), scenario=s)
    assert sorted(shared.notice_reach) == sorted(n.notice_id for n in s.notices)
    assert all(v == 1 for v in alone.notice_reach.values())
    assert max(shared.notice_reach.values()) >= 2
    assert not shared.stalled and shared.unfinished_tasks == 0
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_shift_notice.py -q`
Expected: `ImportError: cannot import name 'NoticeFeed'`.

- [ ] **Step 2: Add the feed**

Append to `src/doi/notices.py`, and add `from src.doi.crdt import NoticeRecord` to its imports:

```python
class NoticeFeed:
    """Hands each notice, at its tick, to the live robot nearest the map centre. Gossip does the rest.

    There is no broadcast to the fleet: who knows a notice depends on range, loss and delay, as with the ledger."""

    def __init__(self, scenario) -> None:
        self.by_tick: Dict[int, list] = {}
        for n in scenario.notices:
            self.by_tick.setdefault(n.emit_tick, []).append(n)
        self.centre = (scenario.grid.height // 2, scenario.grid.width // 2)
        self.delivered = 0

    def emit(self, t: int, agents) -> None:
        for n in self.by_tick.get(t, []):
            live = [a for a in agents if not a.finished]
            if not live:
                continue
            target = min(live, key=lambda a: (abs(a.pos[0] - self.centre[0]) + abs(a.pos[1] - self.centre[1]), a.id))
            target.ingest_notice(NoticeRecord(n.notice_id, t, n.text))
            self.delivered += 1
```

- [ ] **Step 3: Add the scenario**

In `src/doi/scenarios.py`:

Add to the imports:

```python
from src.doi.notices import render_text
```

Directly after the closing `}` of the `DEFAULTS` dict, add:

```python
DEFAULTS["shift_notice"] = dict(DEFAULTS["shift"], notice_mode="true", notice_tick=5, n_distractors=1,
                                notice_bank="test", human_notices="data/forecasts/human_notices.jsonl")
NOTICE_MODES = ("true", "false", "missing", "quiet")
```

Replace the `_TWO_ROOM` line with:

```python
_TWO_ROOM = ("single_block", "two_blocks_parallel", "series_blocks", "multi_block_wall", "shift", "shift_notice")
```

Add this dataclass directly above `class Scenario`:

```python
@dataclass(frozen=True)
class Notice:
    notice_id: str
    emit_tick: int
    kind: str                # "surge" | "drop" | "distractor"
    group: Optional[str]     # zone group the notice is about; None for a distractor
    is_true: bool            # hidden from robots; used only for scoring
    text: str                # what robots receive
```

Add two fields at the end of `class Scenario`, after `slots`:

```python
    notices: List[Notice] = field(default_factory=list)
    zones: Dict[str, Tuple[Pos, ...]] = field(default_factory=dict)    # named regions, for forecasting only
```

In `_two_room`, replace these lines:

```python
    shift = params if cfg.scenario == "shift" else None
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], set(obstacles), shift)
    return Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts, tasks=tasks,
                    meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                          "params": params, "seed": cfg.seed})
```

with:

```python
    shift = params if cfg.scenario in ("shift", "shift_notice") else None
    if cfg.scenario == "shift_notice":
        mode = params["notice_mode"]
        if mode not in NOTICE_MODES:
            raise ValueError(f"unknown notice_mode {mode!r}: choose from {', '.join(NOTICE_MODES)}")
        if mode in ("false", "quiet"):                       # the busy band never moves
            shift = dict(params, shift_after_task=cfg.tasks_per_robot)
        elif cfg.tasks_per_robot <= params["shift_after_task"]:
            raise ValueError(f"notice_mode {mode!r} needs tasks_per_robot above shift_after_task "
                             f"({params['shift_after_task']}), or the busy band never moves")
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], set(obstacles), shift)
    scenario = Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts, tasks=tasks,
                        meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                              "params": params, "seed": cfg.seed})
    if cfg.scenario == "shift_notice":
        _add_notices(scenario, cfg, params, rooms)
    return scenario
```

Add this function directly above `_two_room`:

```python
def _add_notices(scenario: Scenario, cfg: SimConfig, params: Dict[str, Any], rooms: Dict[str, List[Pos]]) -> None:
    """Zones and notices of shift_notice. The band that is busy first is the "north bays", the later one the
    "south bays". A notice's kind, group and is_true stay in the scenario; a robot only ever receives its text."""
    bands = {"north bays": params["hot_before"], "south bays": params["hot_after"]}
    scenario.zones = {f"{side} {group}": tuple(p for p in cells if lo <= p[0] <= hi and p not in scenario.obstacles)
                      for side, cells in sorted(rooms.items()) for group, (lo, hi) in bands.items()}
    bank, path = params["notice_bank"], params["human_notices"]
    notices: List[Notice] = []

    def add(nid: str, tick: int, kind: str, group: str, other: str, is_true: bool) -> None:
        text = render_text(kind, group, other, bank, stream(cfg.seed, f"notice-{nid}"), path)
        notices.append(Notice(nid, tick, kind, None if kind == "distractor" else group, is_true, text))

    if params["notice_mode"] in ("true", "false"):
        is_true = params["notice_mode"] == "true"
        add("n0", params["notice_tick"], "surge", "south bays", "north bays", is_true)
        add("n1", params["notice_tick"], "drop", "north bays", "south bays", is_true)
    rng = stream(cfg.seed, "notices")
    for d in range(params["n_distractors"]):
        tick = rng.randint(0, 2 * params["notice_tick"])
        group, other = rng.sample(["north bays", "south bays"], 2)
        add(f"d{d}", tick, "distractor", group, other, True)
    scenario.notices = sorted(notices, key=lambda n: (n.emit_tick, n.notice_id))
```

- [ ] **Step 4: Wire delivery into the robot and the tick loop**

In `src/doi/agent.py`, add this method directly after `ingest_record`:

```python
    def ingest_notice(self, rec) -> None:
        self.belief.add_notice(rec)
```

In `src/doi/metrics.py`, add this field at the end of `RunResult`:

```python
    notice_reach: Dict[str, int] = field(default_factory=dict)          # notice id -> robots that knew it at the end
```

In `src/doi/runner.py`:

Add to the imports:

```python
from src.doi.notices import NoticeFeed
```

Directly after the line `intake = Intake(scenario, cfg)`, add:

```python
    feed = NoticeFeed(scenario)
```

Directly after the line `intake.deliver(t, agents)`, add:

```python
        feed.emit(t, agents)
```

In `build_result`, add this argument to the `RunResult(...)` call, after `pick_log=[dict(e) for e in world.pick_log]`:

```python
        notice_reach={n.notice_id: sum(1 for a in agents if a.belief.notices.get(n.notice_id) is not None)
                      for n in scenario.notices},
```

- [ ] **Step 5: Describe the scenario**

In `src/doi/narrate.py`, add this entry to `SCENARIO_NOTES` directly after the `"shift"` entry:

```python
    "shift_notice": ("S", "The shift scenario with text notices: a short message tells one robot that work is about "
                          "to move to the other bays, and it spreads by gossip. The notice can be true, false or "
                          "missing (scenario parameter notice_mode), so a forecaster that reads it can be tested."),
```

- [ ] **Step 6: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_shift_notice.py tests/doi/test_doi_guard_neutral.py -q`
Expected: `11 passed`.

- [ ] **Step 7: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/notices.py src/doi/scenarios.py src/doi/agent.py src/doi/runner.py src/doi/metrics.py src/doi/narrate.py tests/doi/test_doi_shift_notice.py
git commit -m "feat(doi): shift_notice scenario, zones and notice delivery by gossip"
```

---

### Task 5: The forecast case and the truth label

**Files:**
- Create: `src/doi/forecast/__init__.py` (empty)
- Create: `src/doi/forecast/case.py`
- Test: `tests/doi/test_doi_forecast_case.py`

**Interfaces:**
- Consumes: `BeliefState.notices` (Task 2); `Scenario.zones` (Task 4); `EvidenceEngine.distance(origin, dest, blocked)` and `EvidenceEngine.fleet_cost(belief, blocked)` from `src/doi/evidence.py`; `BundlePlan` from `src/doi/pushplan.py`; `CarryPlan` from `src/doi/carryplan.py`.
- Produces: `ForecastCase` (fields below); `plan_key(plan) -> Tuple`; `plan_mode(plan) -> str`; `true_total_saving(scenario, engine, before, after) -> float`; `numeric_saving(ledger) -> float`; `trip_table(zones, engine, before, after, seed) -> Dict[str, Dict[str, float]]`; `build_case(agent, plan, tick, price, known, scenario, engine) -> ForecastCase`; `case_to_dict(case) -> dict`; `case_from_dict(d) -> ForecastCase`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_forecast_case.py`:

```python
import dataclasses
import json
from types import SimpleNamespace

from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.crdt import NoticeRecord, RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.forecast.case import (build_case, case_from_dict, case_to_dict, numeric_saving, plan_key, plan_mode,
                                   true_total_saving, trip_table)
from src.doi.policies import PredictedPolicy, Shared
from src.doi.scenarios import scenario_from_ascii

# A wall at column 2 with a pallet in its only gap (row 1); row 4 is the long way round.
ROWS = ["..#....",
        "..L....",
        "..#....",
        "..#....",
        "......."]
WEST = ((0, 0), (0, 1), (1, 0), (1, 1))
EAST = ((0, 5), (0, 6), (1, 5), (1, 6))


def setup(price_total=16.0):
    s = scenario_from_ascii(ROWS, starts=[(1, 0), (4, 0)], tasks=[[(1, 6), (1, 0), (1, 6)], [(4, 6)]])
    s.zones = {"west": WEST, "east": EAST}
    cfg = SimConfig(n_robots=2, tasks_per_robot=3, policy="rof_p")
    shared = Shared(engine=EvidenceEngine(s.grid, cfg.unreachable_cost_for(5, 7)))
    policy = PredictedPolicy(0.5)
    policy.prepare(s, cfg, shared)
    agent = RobotAgent(0, s, cfg, policy, shared)
    agent.belief.add_record(RentRecord(0, 0, (1, 0), (1, 6), 0, 6))
    agent.belief.add_record(RentRecord(1, 0, (1, 0), (1, 6), 0, 6))
    # A stand-in plan that lifts the pallet out of the gap: only `before` and `after` matter to a case.
    plan = SimpleNamespace(obstacle=(1, 2), kind="pallet", landing=(1, 2), key=("lift", (1, 2)),
                           before=frozenset({(1, 2)}), after=frozenset(), total=price_total)
    return s, cfg, shared, policy, agent, plan


def test_truth_counts_every_task_of_every_robot():
    s, _cfg, shared, _policy, _agent, plan = setup()
    # robot 0 crosses three times and saves 12 - 6 each time; robot 1 walks along row 4 and saves nothing
    assert true_total_saving(s, shared.engine, plan.before, plan.after) == 18.0


def test_trip_table_on_the_hand_map():
    s, cfg, shared, _policy, _agent, plan = setup()
    table = trip_table(s.zones, shared.engine, plan.before, plan.after, cfg.seed)
    assert sorted(table) == ["east|east", "east|west", "west|east", "west|west"]
    assert table["west|east"] == {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 16.0}
    assert table["east|west"] == {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 16.0}
    assert table["west|west"] == {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 12.0}
    assert trip_table({}, shared.engine, plan.before, plan.after, cfg.seed) == {}


def test_case_holds_what_the_numeric_forecast_uses():
    s, _cfg, shared, policy, agent, plan = setup()
    agent.belief.add_notice(NoticeRecord("n0", 5, "wave 2 is all in the east"))
    case = build_case(agent, plan, 7, 10.0, 12.0, s, shared.engine)
    assert case.ledger == {"robots_known": 2.0, "tasks_per_robot": 3.0, "trips_recorded": 2.0, "trips_expected": 6.0,
                           "saving_on_recorded_trips": 12.0, "saving_on_my_current_trip": 6.0,
                           "my_tasks_done": 0.0, "my_tasks_left": 3.0}
    assert numeric_saving(case.ledger) == 30.0 == policy._forecast(agent, plan)
    assert case.numeric_forecast is True and case.truth is True          # 30 >= 10 and 18 >= 10
    assert case.notices == (("n0", 5, "wave 2 is all in the east"),)
    assert case.zones == {"west": {"rows": (0, 1), "cols": (0, 1)}, "east": {"rows": (0, 1), "cols": (5, 6)}}
    assert (case.robot, case.tick, case.mode, case.kind, case.obstacle) == (0, 7, "push", "pallet", (1, 2))
    assert case.plan_key == (("lift", (1, 2)),) and case.price == 10.0 and case.known_saving == 12.0
    dear = build_case(agent, plan, 7, 20.0, 12.0, s, shared.engine)
    assert dear.truth is False and dear.numeric_forecast is True          # 18 < 20 but 30 >= 20


def test_case_survives_json_and_its_id_ignores_hidden_fields():
    s, _cfg, shared, _policy, agent, plan = setup()
    case = build_case(agent, plan, 7, 10.0, 12.0, s, shared.engine)
    again = case_from_dict(json.loads(json.dumps(case_to_dict(case))))
    assert again == case and len(case.case_id) == 16
    hidden = dataclasses.replace(case, robot=5, tick=99, truth=False, numeric_forecast=False)
    assert build_case(agent, plan, 99, 10.0, 12.0, s, shared.engine).case_id == case.case_id
    assert hidden.case_id == case.case_id
    assert build_case(agent, plan, 7, 11.0, 12.0, s, shared.engine).case_id != case.case_id


def test_plan_key_and_mode_for_a_plain_plan():
    plan = SimpleNamespace(key=((1, 2), (0, 1), 2))
    assert plan_key(plan) == (((1, 2), (0, 1), 2),) and plan_mode(plan) == "push"
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_case.py -q`
Expected: `ModuleNotFoundError: No module named 'src.doi.forecast'`.

- [ ] **Step 2: Implement the case**

Create `src/doi/forecast/__init__.py` as an empty file.

Create `src/doi/forecast/case.py`:

```python
"""ForecastCase: what one robot knows about one candidate action, frozen into plain data, and the truth label.

A forecaster sees a case and nothing else, so forecasters can be tested, and models scored, without a simulation.
The fields a model may see are listed in VISIBLE; `truth`, `numeric_forecast`, `robot` and `tick` are not shown.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from src.doi.carryplan import CarryPlan
from src.doi.pushplan import BundlePlan
from src.doi.rng import stream

Pos = Tuple[int, int]
ORIGINS_PER_ZONE, DESTS_PER_ZONE = 6, 4
VISIBLE = ("mode", "kind", "obstacle", "landing", "plan_key", "price", "known_saving", "notices", "ledger", "zones",
           "trip_saving")


@dataclass(frozen=True)
class ForecastCase:
    case_id: str
    robot: int
    tick: int
    mode: str                                        # "push" | "bundle" | "carry" | "fill"
    kind: str
    obstacle: Pos
    landing: Pos
    plan_key: Tuple
    price: float
    known_saving: float
    notices: Tuple[Tuple[str, int, str], ...]        # (notice id, tick issued, text), sorted by id
    ledger: Dict[str, float]
    zones: Dict[str, Dict[str, Tuple[int, int]]]     # name -> {"rows": (lo, hi), "cols": (lo, hi)}
    trip_saving: Dict[str, Dict[str, float]]         # "from|to" -> {"mean_saving", "share_saving", "pairs"}
    numeric_forecast: bool
    truth: Optional[bool]


def plan_key(plan) -> Tuple:
    legs = (plan.first, plan.second) if isinstance(plan, BundlePlan) else (plan,)
    return tuple(p.key for p in legs)


def plan_mode(plan) -> str:
    if isinstance(plan, BundlePlan):
        return "bundle"
    return plan.mode if isinstance(plan, CarryPlan) else "push"


def true_total_saving(scenario, engine, before: FrozenSet[Pos], after: FrozenSet[Pos]) -> float:
    """Fall in total route length over every task of every robot, past and future, if `before` became `after`.

    A static measure: it ignores congestion and later changes to the map. A trip's own endpoints count as open,
    as in EvidenceEngine.fleet_cost."""
    total = 0.0
    for start, goals in zip(scenario.starts, scenario.tasks):
        prev = start
        for goal in goals:
            if goal != prev:
                ends = {prev, goal}
                total += engine.distance(prev, goal, before - ends) - engine.distance(prev, goal, after - ends)
            prev = goal
    return total


def numeric_saving(ledger: Dict[str, float]) -> float:
    """The arithmetic forecast of LedgerPolicy.saving(forecast=True), from the ledger numbers alone."""
    seen, expected = ledger["trips_recorded"], ledger["trips_expected"]
    return (ledger["saving_on_my_current_trip"]
            + ledger["saving_on_recorded_trips"] * max(0.0, expected - seen) / max(1.0, seen))


def _sample(cells, n: int, seed: int, name: str) -> list:
    pool = sorted(cells)
    stream(seed, name).shuffle(pool)
    return pool[:n]


def trip_table(zones: Dict[str, Tuple[Pos, ...]], engine, before: FrozenSet[Pos], after: FrozenSet[Pos],
               seed: int) -> Dict[str, Dict[str, float]]:
    """For every ordered pair of zones: what one trip between them would save, on a seeded sample of cell pairs."""
    blocked = before | after
    table: Dict[str, Dict[str, float]] = {}
    for a in sorted(zones):
        origins = _sample([c for c in zones[a] if c not in blocked], ORIGINS_PER_ZONE, seed, f"trip-o-{a}")
        for b in sorted(zones):
            dests = _sample([c for c in zones[b] if c not in blocked], DESTS_PER_ZONE, seed, f"trip-d-{b}")
            savings = [engine.distance(o, d, before) - engine.distance(o, d, after)
                       for o in origins for d in dests if o != d]
            n = len(savings)
            table[f"{a}|{b}"] = {"mean_saving": round(sum(savings) / n, 3) if n else 0.0,
                                 "share_saving": round(sum(1 for v in savings if v > 0) / n, 3) if n else 0.0,
                                 "pairs": float(n)}
    return table


def _freeze(x: Any) -> Any:
    return tuple(_freeze(v) for v in x) if isinstance(x, (list, tuple)) else x


def _case_id(fields: Dict[str, Any]) -> str:
    visible = {k: fields[k] for k in VISIBLE}
    return hashlib.sha256(json.dumps(visible, sort_keys=True, default=list).encode()).hexdigest()[:16]


def build_case(agent, plan, tick: int, price: float, known: float, scenario, engine) -> ForecastCase:
    """Freeze what `agent` knows about `plan` at `tick`. `scenario` supplies the zones and the hidden truth."""
    b = agent.belief
    before, after = plan.before, plan.after
    robots = max(1, len(b.census.items()))
    seen = len(b.records.records()) + sum(c for _, (_, c) in b.agg.entries())
    ledger = {
        "robots_known": float(robots),
        "tasks_per_robot": float(len(agent.tasks)),
        "trips_recorded": float(seen),
        "trips_expected": float(robots * len(agent.tasks)),
        "saving_on_recorded_trips": round(engine.fleet_cost(b, before) - engine.fleet_cost(b, after), 3),
        "saving_on_my_current_trip": round(engine.distance(agent.pos, agent.goal, before)
                                           - engine.distance(agent.pos, agent.goal, after), 3),
        "my_tasks_done": float(agent.task_idx),
        "my_tasks_left": float(len(agent.tasks) - agent.task_idx),
    }
    zones = {name: {"rows": (min(r for r, _ in cells), max(r for r, _ in cells)),
                    "cols": (min(c for _, c in cells), max(c for _, c in cells))}
             for name, cells in sorted(scenario.zones.items()) if cells}
    fields: Dict[str, Any] = {
        "mode": plan_mode(plan), "kind": plan.kind, "obstacle": tuple(plan.obstacle), "landing": tuple(plan.landing),
        "plan_key": _freeze(plan_key(plan)), "price": round(float(price), 3), "known_saving": round(float(known), 3),
        "notices": tuple((r.notice_id, r.tick, r.text) for r in b.notices.records()),
        "ledger": ledger, "zones": zones,
        "trip_saving": trip_table(scenario.zones, engine, before, after, agent.cfg.seed),
    }
    return ForecastCase(case_id=_case_id(fields), robot=agent.id, tick=tick,
                        numeric_forecast=numeric_saving(ledger) >= price,
                        truth=true_total_saving(scenario, engine, before, after) >= price, **fields)


def case_to_dict(case: ForecastCase) -> dict:
    return asdict(case)


def case_from_dict(d: dict) -> ForecastCase:
    d = dict(d)
    for name in ("obstacle", "landing", "plan_key", "notices"):
        d[name] = _freeze(d[name])
    d["zones"] = {name: {k: tuple(v) for k, v in box.items()} for name, box in d["zones"].items()}
    return ForecastCase(**d)
```

- [ ] **Step 3: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecast_case.py -q`
Expected: `5 passed`.

If `test_case_survives_json_and_its_id_ignores_hidden_fields` fails on `hidden.case_id`, note that `case_id` is a stored field: `dataclasses.replace` keeps it, which is what the test checks.

- [ ] **Step 4: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/forecast/__init__.py src/doi/forecast/case.py tests/doi/test_doi_forecast_case.py
git commit -m "feat(doi): forecast case with ledger numbers, trip table and the truth label"
```

---

### Task 6: The forecasters that need no model

**Files:**
- Create: `src/doi/forecast/result.py`
- Create: `src/doi/forecast/forecasters.py`
- Test: `tests/doi/test_doi_forecasters.py`

**Interfaces:**
- Consumes: `ForecastCase` (Task 5); `SURGE_CUES`, `DROP_CUES` (Task 3).
- Produces: `ForecastResult(answer, confidence, reason, failed, latency_s, calls, tool_calls, prompt_tokens, completion_tokens, model)`; `plain(answer) -> ForecastResult`; classes `NumericForecaster`, `KeywordForecaster`, `OracleForecaster`, `InvertedForecaster`, each with `name: str` and `forecast(case) -> ForecastResult`; `group_word(case) -> Optional[str]`; `FORECASTERS`; `make_forecaster(name: str)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_forecasters.py`:

```python
import dataclasses
import pytest
from src.doi.forecast.case import ForecastCase
from src.doi.forecast.forecasters import (InvertedForecaster, KeywordForecaster, NumericForecaster, OracleForecaster,
                                          group_word, make_forecaster)

ZONES = {"west north bays": {"rows": (0, 6), "cols": (0, 9)}, "west south bays": {"rows": (8, 14), "cols": (0, 9)},
         "east north bays": {"rows": (0, 6), "cols": (11, 20)}, "east south bays": {"rows": (8, 14), "cols": (11, 20)}}
BASE = ForecastCase(case_id="x", robot=0, tick=1, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
                    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0, notices=(), ledger={},
                    zones=ZONES, trip_saving={}, numeric_forecast=False, truth=True)


def case(**kw):
    return dataclasses.replace(BASE, **kw)


def test_plain_forecasters():
    assert NumericForecaster().forecast(case(numeric_forecast=True)).answer is True
    assert NumericForecaster().forecast(case(numeric_forecast=False)).answer is False
    assert OracleForecaster().forecast(case(truth=True)).answer is True
    assert InvertedForecaster().forecast(case(truth=True)).answer is False
    res = OracleForecaster().forecast(case())
    assert (res.failed, res.latency_s, res.calls, res.confidence) == ("", 0.0, 0, None)
    for cls in (OracleForecaster, InvertedForecaster):
        with pytest.raises(ValueError):
            cls().forecast(case(truth=None))


def test_group_word_comes_from_the_zone_that_holds_the_obstacle_row():
    assert group_word(case(obstacle=(10, 10))) == "south" and group_word(case(obstacle=(2, 10))) == "north"
    assert group_word(case(obstacle=(7, 10))) is None and group_word(case(zones={})) is None


def test_keyword_reads_only_notices_about_its_own_group():
    kw = KeywordForecaster()
    surge = ("n0", 5, "FYI: wave 2 is all in the south bays")
    drop = ("n1", 5, "the north bays are finished after this wave")
    assert kw.forecast(case(notices=(surge, drop))).answer is True              # south obstacle, surge for south
    assert kw.forecast(case(obstacle=(2, 10), notices=(surge, drop), numeric_forecast=True)).answer is False
    assert kw.forecast(case(notices=(drop,), numeric_forecast=False)).answer is False     # nothing about south
    assert kw.forecast(case(notices=(drop,), numeric_forecast=True)).answer is True       # falls back to numeric


def test_keyword_puts_a_drop_before_a_surge_and_ignores_unknown_wording():
    kw = KeywordForecaster()
    both = (("a", 1, "the south bays get busy after the break"), ("b", 2, "NO MORE orders for the south bays"))
    assert kw.forecast(case(notices=both, numeric_forecast=True)).answer is False
    reworded = (("a", 1, "afternoon orders are concentrated on the south bays"),)
    assert kw.forecast(case(notices=reworded, numeric_forecast=False)).answer is False    # no dev cue: numeric
    assert kw.forecast(case(zones={}, notices=both, numeric_forecast=True)).answer is True  # no zones: numeric


def test_make_forecaster_names():
    for name in ("numeric", "keyword", "oracle", "inverted"):
        assert make_forecaster(name).name == name
    with pytest.raises(ValueError, match="llm"):
        make_forecaster("llm:small")
    with pytest.raises(ValueError):
        make_forecaster("crystal_ball")
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecasters.py -q`
Expected: `ModuleNotFoundError: No module named 'src.doi.forecast.forecasters'`.

- [ ] **Step 2: Implement**

Create `src/doi/forecast/result.py`:

```python
"""ForecastResult: what any forecaster returns for one case."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForecastResult:
    answer: Optional[bool]          # None: the forecast failed, so the guard keeps the classical threshold
    confidence: Optional[float]
    reason: str
    failed: str                     # "" on success, else a failure code
    latency_s: float
    calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    model: str
```

Create `src/doi/forecast/forecasters.py`:

```python
"""Forecasters: one question per case ("will the fleet's saving reach the price?"), several ways to answer it.

  numeric   the arithmetic extrapolation of the ledger (what rof_p uses)
  keyword   reads notices by matching cue phrases; falls back to numeric
  oracle    the truth label (the best a forecaster can be)
  inverted  the opposite of the truth (the worst)
"""
from typing import Optional

from src.doi.forecast.case import ForecastCase
from src.doi.forecast.result import ForecastResult
from src.doi.notices import DROP_CUES, SURGE_CUES


def plain(answer: bool) -> ForecastResult:
    return ForecastResult(answer, None, "", "", 0.0, 0, 0, 0, 0, "")


def _truth(case: ForecastCase) -> bool:
    if case.truth is None:
        raise ValueError("this forecaster needs a case with a truth label")
    return case.truth


class NumericForecaster:
    name = "numeric"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(case.numeric_forecast)


class OracleForecaster:
    name = "oracle"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(_truth(case))


class InvertedForecaster:
    name = "inverted"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        return plain(not _truth(case))


def group_word(case: ForecastCase) -> Optional[str]:
    """The word a notice would use for the obstacle's region: "west north bays" holds its row -> "north".

    A zone's group is its name without the first word; a one-word zone is its own group."""
    for name in sorted(case.zones):
        lo, hi = case.zones[name]["rows"]
        if lo <= case.obstacle[0] <= hi:
            words = name.split()
            return (words[1:] or words)[0].lower()
    return None


class KeywordForecaster:
    """The answer a team gets with no model: among notices that name the obstacle's region, a drop cue says no,
    else a surge cue says yes, else the numeric forecast stands."""
    name = "keyword"

    def forecast(self, case: ForecastCase) -> ForecastResult:
        word = group_word(case)
        if word is not None:
            texts = [text.lower() for _nid, _tick, text in case.notices if word in text.lower()]
            if any(cue in t for t in texts for cue in DROP_CUES):
                return plain(False)
            if any(cue in t for t in texts for cue in SURGE_CUES):
                return plain(True)
        return plain(case.numeric_forecast)


FORECASTERS = {"numeric": NumericForecaster, "keyword": KeywordForecaster, "oracle": OracleForecaster,
               "inverted": InvertedForecaster}


def make_forecaster(name: str):
    if name.startswith("llm:"):
        raise ValueError("llm forecasters are not built yet (Part 2 of the forecast-guard plan)")
    if name not in FORECASTERS:
        raise ValueError(f"unknown forecaster {name!r}: choose from {', '.join(sorted(FORECASTERS))}")
    return FORECASTERS[name]()
```

- [ ] **Step 3: Run the tests and commit**

Run: `venv/bin/python -m pytest tests/doi/test_doi_forecasters.py -q`
Expected: `5 passed`.

```bash
git add src/doi/forecast/result.py src/doi/forecast/forecasters.py tests/doi/test_doi_forecasters.py
git commit -m "feat(doi): numeric, keyword, oracle and inverted forecasters"
```

---

### Task 7: The arm `rof_a`

**Files:**
- Modify: `src/doi/config.py`
- Modify: `src/doi/policies.py`
- Modify: `src/doi/agent.py`
- Modify: `src/doi/runner.py`
- Modify: `src/doi/metrics.py`
- Modify: `src/doi/narrate.py`
- Modify: `src/doi/README.md`
- Test: `tests/doi/test_doi_guarded_policy.py`

**Interfaces:**
- Consumes: `build_case`, `plan_key` (Task 5); `make_forecaster`, `ForecastResult` (Task 6); `PredictedPolicy`, `price_of`, `Shared`, `PushPolicy` from `src/doi/policies.py`.
- Produces: `SimConfig.forecaster: str = "numeric"`, `SimConfig.agent_max_forecasts: int = 200`, arm name `"rof_a"`; `ForecastState(asked, due, answer)`; `GuardedPolicy(lam, forecaster=None)`; `PushPolicy.on_tick(agents, t)`; `Shared.forecast_log`, `Shared.forecast_cases`; `RobotAgent.forecasts`, `RobotAgent.forecast_epoch`; `RunResult.forecast_log`, `RunResult.forecast_cases`; `metrics.forecast_columns(log) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/doi/test_doi_guarded_policy.py`:

```python
import math
from types import SimpleNamespace

import pytest
from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.forecast.result import ForecastResult
from src.doi.metrics import forecast_columns, summary_row
from src.doi.policies import GuardedPolicy, Shared, make_policy
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario, scenario_from_ascii

ROWS = ["..#....",
        "..L....",
        "..#....",
        "..#....",
        "......."]


class Scripted:
    """A forecaster that returns one fixed result and counts how often it is asked."""
    name = "scripted"

    def __init__(self, answer, latency_s=0.0, failed=""):
        self.result = ForecastResult(answer, None, "", failed, latency_s, 0, 0, 0, 0, "")
        self.asked = 0

    def forecast(self, case):
        self.asked += 1
        return self.result


def hand(forecaster, lam=0.5, **cfg_kw):
    """One robot at (1, 0) heading for (1, 6); known saving 12 for lifting the pallet out of the gap."""
    s = scenario_from_ascii(ROWS, starts=[(1, 0), (4, 0)], tasks=[[(1, 6), (1, 0), (1, 6)], [(4, 6)]])
    cfg = SimConfig(n_robots=2, tasks_per_robot=3, policy="rof_a", lam=lam, **cfg_kw)
    shared = Shared(engine=EvidenceEngine(s.grid, cfg.unreachable_cost_for(5, 7)))
    policy = GuardedPolicy(lam, forecaster)
    policy.prepare(s, cfg, shared)
    agent = RobotAgent(0, s, cfg, policy, shared)
    agent.belief.add_record(RentRecord(0, 0, (1, 0), (1, 6), 0, 6))
    agent.belief.add_record(RentRecord(1, 0, (1, 0), (1, 6), 0, 6))
    return policy, agent, shared


def plan(total):
    return SimpleNamespace(obstacle=(1, 2), kind="pallet", landing=(1, 2), key=("lift", (1, 2)),
                           before=frozenset({(1, 2)}), after=frozenset(), total=total)


def info(tick):
    return SimpleNamespace(tick=tick, d_open=6)        # price = plan.total - 6


def test_config_knows_the_arm_and_checks_the_forecaster():
    assert SimConfig(policy="rof_a").forecaster == "numeric" and SimConfig().agent_max_forecasts == 200
    assert make_policy(SimConfig(policy="rof_a")).name == "rof_a"
    SimConfig(forecaster="llm:small")
    with pytest.raises(ValueError):
        SimConfig(forecaster="crystal_ball")
    with pytest.raises(ValueError):
        SimConfig(agent_max_forecasts=-1)


def test_threshold_follows_the_answer():
    for answer, total, fires in ((True, 26.0, True), (False, 26.0, False),      # price 20: 12 >= 10, 12 < 40
                                 (False, 14.0, False), (None, 14.0, True),      # price 8: 12 < 16; no answer: 12 >= 8
                                 (None, 26.0, False)):                          # no answer: 12 < 20
        policy, agent, _ = hand(Scripted(answer))
        verdict = policy.assess(agent, plan(total), info(5))
        assert (verdict is not None) is fires, (answer, total)
        if fires:
            assert verdict == (12.0 - (total - 6), 12.0, total - 6)


def test_a_late_answer_is_ignored_until_it_is_due():
    policy, agent, shared = hand(Scripted(True, latency_s=2.0))                 # 2 s at 0.5 s per tick: 4 ticks
    p = plan(26.0)                                                              # price 20
    assert policy.assess(agent, p, info(5)) is None
    state = agent.forecasts[(("lift", (1, 2)),)]
    assert (state.asked, state.due, state.answer) == (5, 9, True)
    assert policy.assess(agent, p, info(8)) is None
    policy.on_tick([agent], 8)
    assert agent.forecast_epoch == 0
    policy.on_tick([agent], 9)
    assert agent.forecast_epoch == 1
    assert policy.assess(agent, p, info(9)) == (-8.0, 12.0, 20.0)
    assert policy.forecaster.asked == 1 and len(shared.forecast_log) == 1
    row = shared.forecast_log[0]
    assert (row["tick"], row["robot"], row["due"], row["answer"], row["skipped"]) == (5, 0, 9, True, False)
    assert row["truth"] is False and row["numeric_forecast"] is True and row["forecaster"] == "scripted"


def test_no_forecast_below_the_lazy_line_at_zero_price_or_with_lam_one():
    policy, agent, _ = hand(Scripted(True))
    assert policy.assess(agent, plan(36.0), info(5)) is None                    # price 30: 12 < 15, nothing asked
    assert agent.forecasts == {} and policy.forecaster.asked == 0
    assert policy.assess(agent, plan(6.0), info(5)) == (12.0, 12.0, 0.0)        # price 0: fires, nothing asked
    assert policy.forecaster.asked == 0
    policy, agent, _ = hand(Scripted(False), lam=1.0)
    assert policy.assess(agent, plan(14.0), info(5)) == (4.0, 12.0, 8.0)        # lam 1: the classical rule
    assert policy.forecaster.asked == 0


def test_budget_of_zero_skips_every_request():
    policy, agent, shared = hand(Scripted(True), agent_max_forecasts=0)
    assert policy.assess(agent, plan(26.0), info(5)) is None                    # no answer: threshold 1
    assert policy.forecaster.asked == 0 and shared.forecast_log[0]["skipped"] is True
    assert shared.forecast_cases == []


def _run(scenario, policy_obj=None, **kw):
    cfg = SimConfig(scenario=scenario, n_robots=6, tasks_per_robot=6, seed=2, **kw)
    s = build_scenario(cfg)
    return run_episode(cfg, scenario=s, policy=policy_obj), s


def test_a_forecaster_that_always_fails_leaves_the_classical_rule():
    rof, _ = _run("single_block", policy="rof")
    failing, _ = _run("single_block", GuardedPolicy(0.5, Scripted(None, failed="client_error")), policy="rof_a")
    assert failing.J == rof.J == 519.0 and failing.pushes == rof.pushes
    assert failing.forecast_log and all(r["failed"] == "client_error" for r in failing.forecast_log)


def test_requests_are_lazy_and_made_once_per_robot_and_plan():
    res, _ = _run("multi_block_wall", GuardedPolicy(0.5, Scripted(True)), policy="rof_a")
    rows = res.forecast_log
    assert rows and all(r["price"] > 0 and r["known"] >= 0.5 * r["price"] for r in rows)
    assert len({(r["robot"], r["plan_key"]) for r in rows}) == len(rows) == len(res.forecast_cases)
    assert not res.stalled and res.unfinished_tasks == 0


@pytest.mark.parametrize("mode", ["true", "false", "missing", "quiet"])
@pytest.mark.parametrize("forecaster", ["numeric", "keyword", "oracle", "inverted"])
def test_rof_a_finishes_shift_notice(mode, forecaster):
    cfg = SimConfig(scenario="shift_notice", n_robots=6, tasks_per_robot=12, seed=3, policy="rof_a",
                    forecaster=forecaster, scenario_params={"notice_mode": mode})
    res = run_episode(cfg)
    assert not res.stalled and res.unfinished_tasks == 0 and res.forecast_log
    answers = [(r["answer"], r["truth"]) for r in res.forecast_log]
    if forecaster == "oracle":
        assert all(a == t for a, t in answers)
    if forecaster == "inverted":
        assert all(a != t for a, t in answers)
    row = summary_row(res)
    assert row["forecaster"] == forecaster and row["forecasts"] == len(res.forecast_log)
    if forecaster in ("oracle", "inverted"):
        assert row["forecast_acc"] == (1.0 if forecaster == "oracle" else 0.0)


def test_a_scenario_with_no_zones_and_no_notices_still_runs():
    res, _ = _run("single_block", policy="rof_a", forecaster="keyword")
    assert not res.stalled and res.forecast_cases
    assert all(c.zones == {} and c.trip_saving == {} and c.notices == () for c in res.forecast_cases)
    assert all(r["answer"] == r["numeric_forecast"] for r in res.forecast_log)   # keyword falls back to numeric


def test_forecast_columns():
    log = [dict(answer=True, truth=True, failed="", skipped=False, calls=2, latency_s=1.5, prompt_tokens=10,
                completion_tokens=3),
           dict(answer=True, truth=False, failed="", skipped=False, calls=1, latency_s=0.5, prompt_tokens=5,
                completion_tokens=1),
           dict(answer=False, truth=True, failed="", skipped=False, calls=0, latency_s=0.0, prompt_tokens=0,
                completion_tokens=0),
           dict(answer=None, truth=True, failed="no_answer", skipped=False, calls=4, latency_s=2.0, prompt_tokens=8,
                completion_tokens=0),
           dict(answer=None, truth=None, failed="", skipped=True, calls=0, latency_s=0.0, prompt_tokens=0,
                completion_tokens=0)]
    cols = forecast_columns(log)
    assert cols == {"forecasts": 5, "forecast_failed": 1, "forecast_skipped": 1, "forecast_acc": pytest.approx(1 / 3),
                    "wrong_yes": 1, "wrong_no": 1, "agent_calls": 7, "agent_latency_s": 4.0,
                    "agent_prompt_tokens": 23, "agent_completion_tokens": 4}
    empty = forecast_columns([])
    assert empty["forecasts"] == 0 and math.isnan(empty["forecast_acc"])
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_guarded_policy.py -q`
Expected: `ImportError: cannot import name 'forecast_columns'`.

- [ ] **Step 2: Config**

In `src/doi/config.py`:

Replace the `POLICIES` definition with:

```python
POLICIES = frozenset({"never", "myopic", "eager", "rof", "rof_local", "rof_f", "central", "hindsight", "free",
                      "rof_r", "rof_p", "rof_a"})
FORECASTERS = ("numeric", "keyword", "oracle", "inverted")
```

Add these two fields directly after `lam: float = 0.5`:

```python
    forecaster: str = "numeric"          # rof_a: numeric | keyword | oracle | inverted | llm:<model key>
    agent_max_forecasts: int = 200       # rof_a: forecasts one run may request; later requests get no answer
```

Add these two checks to the `checks` list, directly after the `lam` check:

```python
            (self.forecaster in FORECASTERS or self.forecaster.startswith("llm:"),
             f"bad forecaster {self.forecaster!r}"),
            (self.agent_max_forecasts >= 0, "agent_max_forecasts must be >= 0"),
```

- [ ] **Step 3: Policy**

In `src/doi/policies.py`:

Add to the imports:

```python
from src.doi.forecast.case import build_case, plan_key
from src.doi.forecast.forecasters import make_forecaster
```

Add two fields at the end of `class Shared`:

```python
    forecast_log: List[dict] = field(default_factory=list)       # one row per forecast a robot requested
    forecast_cases: List[Any] = field(default_factory=list)      # the ForecastCase of each answered request
```

Add this method to `class PushPolicy`, directly after `on_task_planned`:

```python
    def on_tick(self, agents, t: int) -> None:
        """Called once per tick before the robots decide. Most arms have nothing to do."""
```

Add these two classes directly after `class PredictedPolicy` (before `class CentralPolicy`):

```python
@dataclass
class ForecastState:
    asked: int                  # tick the forecast was requested
    due: int                    # tick from which the answer is visible to the guard
    answer: Optional[bool]      # None: failed or skipped, so the classical threshold stays


class GuardedPolicy(PredictedPolicy):
    """rof_a: a forecaster says whether the action will pay; the predicted-threshold rule guards the answer.

    The threshold is lam on a yes, 1 / lam on a no, and 1 while there is no visible answer (Proposition 5).
    A forecast is requested once per robot and plan, and only when the known saving reaches lam * price: below
    that neither threshold can fire, so the answer could not matter."""

    def __init__(self, lam: float, forecaster=None) -> None:
        super().__init__(lam)
        self.name = "rof_a"
        self._given = forecaster

    def prepare(self, scenario: "Scenario", cfg: SimConfig, shared: Shared) -> None:
        super().prepare(scenario, cfg, shared)
        self.scenario = scenario
        self.forecaster = self._given or make_forecaster(cfg.forecaster)
        self.requests = 0

    def on_tick(self, agents, t: int) -> None:
        for a in agents:                 # an answer that lands now is a reason to look at the plans again
            if any(st.due == t and st.asked < t and st.answer is not None for st in a.forecasts.values()):
                a.forecast_epoch += 1

    def _request(self, agent, plan, key, t: int, price: float, known: float) -> ForecastState:
        row = {"tick": t, "robot": agent.id, "plan_key": key, "forecaster": self.forecaster.name,
               "known": float(known), "price": float(price)}
        if self.requests >= self.cfg.agent_max_forecasts:
            state = ForecastState(t, t, None)
            row.update(case_id="", answer=None, truth=None, numeric_forecast=None, confidence=None, failed="",
                       skipped=True, due=t, calls=0, tool_calls=0, latency_s=0.0, prompt_tokens=0,
                       completion_tokens=0, reason="")
        else:
            self.requests += 1
            case = build_case(agent, plan, t, price, known, self.scenario, self.shared.engine)
            res = self.forecaster.forecast(case)
            due = t if res.latency_s <= 0 else t + max(1, math.ceil(res.latency_s / self.cfg.tick_seconds))
            state = ForecastState(t, due, res.answer)
            self.shared.forecast_cases.append(case)
            row.update(case_id=case.case_id, answer=res.answer, truth=case.truth,
                       numeric_forecast=case.numeric_forecast, confidence=res.confidence, failed=res.failed,
                       skipped=False, due=due, calls=res.calls, tool_calls=res.tool_calls, latency_s=res.latency_s,
                       prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, reason=res.reason)
        self.shared.forecast_log.append(row)
        return state

    def assess(self, agent, plan, info):
        price = price_of(plan, info)
        known = self.saving(agent, plan)
        if known <= 0:
            return None
        key = plan_key(plan)
        state = agent.forecasts.get(key)
        if state is None and price > 0 and self.lam < 1 and known >= self.lam * price:
            state = agent.forecasts[key] = self._request(agent, plan, key, info.tick, price, known)
        thr = 1.0
        if state is not None and state.answer is not None and info.tick >= state.due:
            thr = self.lam if state.answer else 1.0 / self.lam
        if known < thr * price:
            return None
        return known - price, known, price
```

In `make_policy`, add directly after the `rof_p` branch:

```python
    if name == "rof_a":
        return GuardedPolicy(lam=cfg.lam)
```

- [ ] **Step 4: Robot and tick loop**

In `src/doi/agent.py`:

In `RobotAgent.__init__`, directly after `self._push_key = None`, add:

```python
        self.forecasts: Dict[tuple, object] = {}        # rof_a: plan key -> ForecastState
        self.forecast_epoch = 0                         # rof_a: bumped when a late forecast becomes visible
```

In `_consider_push`, replace:

```python
        key = (b.version, self.pos, self.goal, tuple(sorted(c for c, u in self.deferred_until.items() if u > t)))
```

with:

```python
        key = (b.version, self.pos, self.goal, tuple(sorted(c for c, u in self.deferred_until.items() if u > t)),
               self.forecast_epoch)
```

In `src/doi/runner.py`, in `run_episode`, directly after the loop

```python
        for a in active:
            a.receive(inbox.get(a.id, []), t)
```

add:

```python
        policy.on_tick(active, t)
```

In `build_result`, add these arguments to the `RunResult(...)` call, after `notice_reach=...`:

```python
        forecast_log=[dict(r) for r in shared.forecast_log], forecast_cases=list(shared.forecast_cases),
```

- [ ] **Step 5: Metrics**

In `src/doi/metrics.py`:

Add two fields at the end of `RunResult`:

```python
    forecast_log: List[dict] = field(default_factory=list)              # rof_a: one row per forecast requested
    forecast_cases: List[Any] = field(default_factory=list)             # rof_a: the case behind each answered one
```

Add this function directly above `summary_row`:

```python
def forecast_columns(log: List[dict]) -> dict:
    """Counts over a run's forecast log. A wrong yes pays the price early; a wrong no pays detours for longer."""
    scored = [r for r in log if r["answer"] is not None and r["truth"] is not None]
    return {
        "forecasts": len(log),
        "forecast_failed": sum(1 for r in log if r["failed"]),
        "forecast_skipped": sum(1 for r in log if r["skipped"]),
        "forecast_acc": (sum(1 for r in scored if r["answer"] == r["truth"]) / len(scored)) if scored
        else float("nan"),
        "wrong_yes": sum(1 for r in scored if r["answer"] and not r["truth"]),
        "wrong_no": sum(1 for r in scored if not r["answer"] and r["truth"]),
        "agent_calls": sum(r["calls"] for r in log),
        "agent_latency_s": float(sum(r["latency_s"] for r in log)),
        "agent_prompt_tokens": sum(r["prompt_tokens"] for r in log),
        "agent_completion_tokens": sum(r["completion_tokens"] for r in log),
    }
```

In `summary_row`, change the end of the returned dict from

```python
        "hr": ratios.get("hr", nan), "hr_av": ratios.get("hr_av", nan), "pod": pod if pod is not None else nan,
    }
```

to:

```python
        "hr": ratios.get("hr", nan), "hr_av": ratios.get("hr_av", nan), "pod": pod if pod is not None else nan,
        "forecaster": cfg.forecaster, **forecast_columns(result.forecast_log),
    }
```

- [ ] **Step 6: Describe the arm**

In `src/doi/narrate.py`, add to `ARM_NOTES` directly after the `"rof_p"` entry:

```python
    "rof_a": "Rent-or-Fill guarded: a forecaster says whether the move will pay; the threshold is lowered on a yes, "
             "raised on a no, and stays at the price while there is no answer",
```

In `src/doi/README.md`, add this row to the arms table directly after the `rof_p` row:

```markdown
| `rof_a` | Rent-or-Fill guarded: a forecaster says whether the move will pay; the threshold is lowered on a yes, raised on a no, and stays at the price while there is no answer |
```

- [ ] **Step 7: Run the tests**

Run: `venv/bin/python -m pytest tests/doi/test_doi_guarded_policy.py tests/doi/test_doi_guard_neutral.py -q`
Expected: `28 passed` (9 plain tests, 16 parametrised runs, 3 pinned).

If `test_a_forecaster_that_always_fails_leaves_the_classical_rule` gives a cost other than 519.0 for both arms, stop and report: the guard with no answer must be the classical rule.

- [ ] **Step 8: Run the whole suite and commit**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add src/doi/config.py src/doi/policies.py src/doi/agent.py src/doi/runner.py src/doi/metrics.py src/doi/narrate.py src/doi/README.md tests/doi/test_doi_guarded_policy.py
git commit -m "feat(doi): rof_a, the predicted rule guarding a pluggable forecaster"
```

---

### Task 8: The headroom pilot, and the gate

**Files:**
- Create: `experiments/doi_e9_headroom.py`
- Modify: `docs/research/results.md` (append one section)
- Test: `tests/doi/test_doi_e9_headroom.py`

**Interfaces:**
- Consumes: `run_episode`, `build_scenario`, `SimConfig` with `forecaster` (Task 7), scenario `shift_notice` (Task 4), `summary_row` columns (Task 7), `src.doi.stats.bootstrap_ci(x, n=..., seed=...) -> (stat, lo, hi)`.
- Produces: `experiments/doi_e9_headroom.py` with `ARMS`, `COSTS`, `MODES`, `run_point(task) -> List[dict]`, `summarise(df) -> pd.DataFrame`, `main(argv=None) -> int`; files `runs.csv` and `summary.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/doi/test_doi_e9_headroom.py`:

```python
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_headroom_quick_runs(tmp_path):
    out = str(tmp_path / "headroom")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "doi_e9_headroom.py"),
                           "--quick", "--out", out], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    runs = pd.read_csv(os.path.join(out, "runs.csv"))
    assert sorted(runs.arm.unique()) == ["inverted", "keyword", "numeric", "oracle", "rof"]
    assert sorted(runs["mode"].unique()) == ["false", "missing", "quiet", "true"]
    assert len(runs) == 3 * 4 * 5 and not runs.stalled.any()             # 3 seeds, 4 modes, 5 arms, 1 cost setting
    assert (runs.avoidable == runs.J - runs.J_free).all()
    summary = pd.read_csv(os.path.join(out, "summary.csv"))
    assert {"kappa", "fee", "mode", "compare", "mean_diff", "median_diff", "lo", "hi", "share_positive",
            "headroom_share"} <= set(summary.columns)
    assert "numeric - oracle" in set(summary["compare"])
    assert "no model was called" in proc.stdout
```

Run: `venv/bin/python -m pytest tests/doi/test_doi_e9_headroom.py -q`
Expected: FAIL, the script does not exist (`returncode` 2).

- [ ] **Step 2: Write the script**

Create `experiments/doi_e9_headroom.py`:

```python
"""E9 headroom pilot: how much can any forecaster change fleet cost on shift_notice? Calls no model.

`oracle` is the best a forecaster can be and `inverted` the worst, so the gap between them, and between `numeric`
and `oracle`, bounds what a language model could add. Costs are compared as avoidable cost: J minus the cost of
the `free` arm on the same seed, because most of J is travel no rule can avoid.

Usage: venv/bin/python experiments/doi_e9_headroom.py [--quick] [--seeds 100] [--jobs 4] [--out DIR]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.doi.config import SimConfig
from src.doi.metrics import summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario
from src.doi.stats import bootstrap_ci

ARMS = [("rof", "rof", "numeric"), ("numeric", "rof_a", "numeric"), ("keyword", "rof_a", "keyword"),
        ("oracle", "rof_a", "oracle"), ("inverted", "rof_a", "inverted")]      # (label, policy, forecaster)
COSTS = [(4.0, 1.0), (8.0, 1.0), (4.0, 20.0), (8.0, 40.0)]                     # (kappa, fee)
MODES = ["true", "false", "missing", "quiet"]
COMPARE = [("numeric", "oracle"), ("inverted", "oracle"), ("rof", "oracle"), ("numeric", "keyword"),
           ("rof", "numeric")]
FIRST_SEED = 200
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "doi", "e9_headroom")


def run_point(task) -> List[dict]:
    seed, mode, kappa, fee = task
    cfg = SimConfig(scenario="shift_notice", n_robots=8, tasks_per_robot=20, seed=seed, kappa=kappa, fee=fee,
                    lam=0.5, scenario_params={"notice_mode": mode})
    scenario = build_scenario(cfg)
    free = run_episode(cfg.replace(policy="free"), scenario=scenario).J_censored
    rows = []
    for label, policy, forecaster in ARMS:
        res = run_episode(cfg.replace(policy=policy, forecaster=forecaster), scenario=scenario)
        cols = summary_row(res, cheap=True)
        rows.append({"seed": seed, "mode": mode, "kappa": kappa, "fee": fee, "arm": label, "J": res.J_censored,
                     "J_free": free, "avoidable": res.J_censored - free, "removals": res.removals,
                     "stalled": res.stalled, "forecasts": cols["forecasts"], "forecast_acc": cols["forecast_acc"],
                     "wrong_yes": cols["wrong_yes"], "wrong_no": cols["wrong_no"]})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per cost setting and notice mode: paired differences in avoidable cost between arms, with a bootstrap
    interval on the median, and the share of the numeric arm's avoidable cost a perfect forecast would remove."""
    out = []
    for (kappa, fee, mode), g in df.groupby(["kappa", "fee", "mode"]):
        wide = g.pivot(index="seed", columns="arm", values="avoidable")
        headroom = float((wide["numeric"] - wide["oracle"]).mean() / max(1e-9, wide["numeric"].mean()))
        for a, b in COMPARE:
            d = (wide[a] - wide[b]).to_numpy(dtype=float)
            med, lo, hi = bootstrap_ci(d, n=2000, seed=0)
            out.append({"kappa": kappa, "fee": fee, "mode": mode, "compare": f"{a} - {b}", "mean_diff": float(d.mean()),
                        "median_diff": med, "lo": lo, "hi": hi, "share_positive": float((d > 0).mean()),
                        "headroom_share": headroom, "mean_avoidable_numeric": float(wide["numeric"].mean()),
                        "n": len(d)})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="3 seeds and the default costs only")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or (DEFAULT_OUT + ("_quick" if args.quick else ""))
    os.makedirs(out, exist_ok=True)
    seeds = range(FIRST_SEED, FIRST_SEED + (3 if args.quick else args.seeds))
    costs = COSTS[:1] if args.quick else COSTS
    tasks = [(seed, mode, kappa, fee) for kappa, fee in costs for mode in MODES for seed in seeds]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            batches = list(pool.map(run_point, tasks))
    else:
        batches = [run_point(t) for t in tasks]
    df = pd.DataFrame([row for batch in batches for row in batch])
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    print(f"E9 headroom pilot: shift_notice, 8 robots, 20 tasks, lam 0.5, seeds {seeds[0]}..{seeds[-1]}; "
          f"no model was called")
    means = df.groupby(["kappa", "fee", "mode", "arm"])["avoidable"].mean().unstack("arm").round(1)
    print("\nmean avoidable cost (J minus the free arm's J):")
    print(means[[label for label, _, _ in ARMS]].to_string())
    acc = df[df.arm.isin(["numeric", "keyword"])].groupby(["kappa", "fee", "mode", "arm"])["forecast_acc"].mean()
    print("\nmean forecast accuracy against the truth label:")
    print(acc.unstack("arm").round(2).to_string())
    print("\npaired differences in avoidable cost (median, 95% interval of the median, share of seeds above 0):")
    for _, r in summary.iterrows():
        print(f"  kappa {r.kappa:g} fee {r.fee:g} {r['mode']:8s} {r['compare']:20s} median {r.median_diff:7.1f} "
              f"[{r.lo:7.1f}, {r.hi:7.1f}]  above 0: {r.share_positive:.2f}")
    top = summary[summary["compare"] == "numeric - oracle"][["kappa", "fee", "mode", "headroom_share"]]
    print("\nshare of the numeric arm's avoidable cost a perfect forecast removes:")
    print(top.round(3).to_string(index=False))
    print(f"\nstalled runs: {int(df.stalled.sum())} of {len(df)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the test**

Run: `venv/bin/python -m pytest tests/doi/test_doi_e9_headroom.py -q`
Expected: `1 passed` (about a minute: 72 simulation runs).

- [ ] **Step 4: Run the whole suite and commit the script**

Run: `venv/bin/python -m pytest tests/doi -q`
Expected: all pass.

```bash
git add experiments/doi_e9_headroom.py tests/doi/test_doi_e9_headroom.py
git commit -m "feat(doi): E9 headroom pilot for the forecast guard (no model)"
```

- [ ] **Step 5: Run the pilot in full**

Run: `venv/bin/python experiments/doi_e9_headroom.py --seeds 100 --jobs 4`
Expected: exit code 0 and `stalled runs: 0 of 8000`. It is 9,600 simulation runs at a little under a second each, so allow about half an hour with four jobs. Keep the whole printed output.

If any run stalled, do not go on: report the seeds, modes and arms that stalled.

- [ ] **Step 6: Record the output**

Append to `docs/research/results.md` a section `## E9 headroom pilot (<today's date>)` that contains exactly: the command from Step 5, the full printed output in a code block, and these two sentences:

```markdown
No model was called. This measures the best (`oracle`) and worst (`inverted`) any forecaster can do on
`shift_notice`, to decide whether the model parts of the forecast guard are worth building.
```

Add no interpretation. Change nothing else in the file.

```bash
git add docs/research/results.md
git commit -m "docs(doi): E9 headroom pilot output"
```

- [ ] **Step 7: Stop for the owner's decision**

End the work here. Report to the owner: the list of commits, the test count, and the pilot's printed output. Do not start Part 2. The owner decides, from the `numeric - oracle` and `inverted - oracle` rows, whether to build the model parts, change the scenario or costs, or stop at this result.

---

## Self-Review Notes

- Spec sections covered: 3 (arm), 4 (theory), 5 (notices), 6 (scenario), 7 (case), 10 (forecasters except `llm`), 11 (edits to existing modules, less the `llm/client.py` and command-line rows), 13 (logging), 14.4 (headroom gate), 15.1, 15.2, 15.3, 15.5 (tests, less record-and-replay).
- Left for Part 2 by design: spec sections 8 (tools), 9 (agent loop), 12 (model calls), 14.1 to 14.3 (datasets, evaluation, E9), 15.4, 15.6, and the `run_doi.py` flags.
- The pinned costs in Task 2 and the hand-case numbers in Task 1 were measured on the code as it stood at commit `8ac1cdd`.
