# Forecast agent with a rule as its guard: design

Status: design approved by the owner on 2026-10-06. Part 1 (everything without a model) was built and measured on
2026-10-07; Part 2 (the model parts: sections 8, 9, 12, 14.1 to 14.3) is built to
`docs/superpowers/plans/2026-10-07-forecast-guard-part2.md`. No model call has been made.

Target venue: IEEE T-ASE special issue "Agentic Intelligence for Materials Handling, Warehousing, and
Logistics 5.0", submission deadline 2027-02-15.

---

## 1. What this adds, and why

Today a language model appears in one place only: it turns incident report text into a record
(`src/doi/llm/intake.py`). Every decision to move an obstacle is made by a numeric rule. The call for papers asks
for agents built on foundation models that act as the decision-and-control layer, so a model at the edge of the
system is on topic but not what the call centrally asks for.

This design puts a model on the decision path with a proven cap on the harm it can do:

* A **forecast agent**, one per robot, answers one question about one candidate action: *will the fleet's total
  saving from this action reach its price?* It answers yes or no.
* The **guard** is the existing predicted-threshold rule (`rof_p`, Theorem 3 in `docs/research/theory.md`). A yes
  lowers the threshold to `lam`, a no raises it to `1 / lam`, and while there is no answer the threshold is 1,
  the classical ski-rental rule.
* The agent can only read. It cannot move a robot, move an obstacle or change a threshold directly.

The agent's reason to do better than arithmetic is **text notices about future traffic** ("second wave is all in
the south bays"). The numeric forecast extrapolates past traffic and cannot read them.

### 1.1 Decisions already made (binding)

| Decision | Choice |
|---|---|
| What the agent may propose | A yes/no forecast only. No ranking, no free-form actions. |
| Whose information it uses | One agent per robot, using only that robot's belief. No fleet-level agent. |
| What it knows that the numbers do not | Text notices about future traffic. |
| How it works | Tool-using, a bounded number of rounds. No memory between forecasts. |
| Provider | OpenAI, through the existing OpenAI-compatible client. |
| Model calls | Made live during a run and stored, so a re-run replays them without calling. |

### 1.2 What this supersedes

`docs/superpowers/specs/2026-10-03-algorithmic-core.md` says "Do not put an LLM anywhere on the decision path".
That rule is lifted for the new arm `rof_a` only. Every existing arm keeps its behaviour, and every existing test
must pass unchanged.

### 1.3 Fit with the call

| Phrase in the call | Part of this design |
|---|---|
| LLM agents as the decision-and-control layer | The forecast sets the guard's threshold for each candidate action |
| Tool-using agents | Three read-only tools and a final answer call (section 8) |
| sense, simulate, decide, act | read notices and ledger; `trip_saving` what-if on the robot's believed map; forecast; guard acts |
| Safety assurance and runtime monitoring of LLM-driven control | The guard and Proposition 5 (section 4) |
| Reliability and graceful degradation | No answer, a late answer or a bad answer leaves the classical rule in force |
| Latency-, energy- and cost-aware deployment | Latency is charged in ticks; calls, tokens and latency are reported |
| Decentralised, communication-efficient coordination | Per-robot agents; notices spread by gossip; coverage `rho` unchanged |

Not covered here: the human approval gate, validation on warehouse benchmark maps and a digital twin. Each is its
own spec (section 17).

---

## 2. Architecture

```
scenario notices ──► NoticeFeed ──► nearest robot's belief ──► gossip to neighbours
                                             │
robot assesses a candidate action; known saving reaches lam * price for the first time
                                             │
                                             ▼
                          build_case(robot, plan)  ──►  ForecastCase (plain data, no live objects)
                                             │
                                             ▼
                          Forecaster.forecast(case)  ──►  ForecastResult
                          numeric | keyword | oracle | inverted | llm:<model key>
                                             │   answer becomes visible `latency` ticks later
                                             ▼
                          GuardedPolicy.assess: threshold lam (yes), 1/lam (no), 1 (no answer yet or failed)
                                             │
                                             ▼
                                   existing push / carry / fill machinery
```

Units and their one job:

| Unit | File | Job | Depends on |
|---|---|---|---|
| Guard rule on pure numbers | `src/doi/abstract/core.py` | `run_guarded` and its bounds | nothing |
| Notices | `src/doi/notices.py` | text templates, rendering, delivery | scenarios, rng |
| Forecast case | `src/doi/forecast/case.py` | freeze what a robot knows into plain data; the truth label | belief, evidence engine |
| Tools | `src/doi/forecast/tools.py` | pure functions of a case, and their schemas | case |
| Agent loop | `src/doi/forecast/agent.py` | run the model on a case for a bounded number of turns | tools, chat client |
| Forecasters | `src/doi/forecast/forecasters.py` | one interface, five backends | case, agent loop |
| Chat client and store | `src/doi/llm/client.py`, `src/doi/llm/chatcache.py` | tool-calling requests; record and replay | nothing in `src/doi` |
| Policy | `src/doi/policies.py` | `GuardedPolicy`, the arm `rof_a` | forecasters |

The agent loop never touches a belief, a world or a robot. It sees a `ForecastCase` only. That makes it testable
without a simulation, and it makes the same cases usable for scoring models offline (section 14).

---

## 3. The arm `rof_a`

`GuardedPolicy(LedgerPolicy)`, `name = "rof_a"`, gossip scope, `theta = 1.0`, `forecast = False`.

State on each robot: `agent.forecasts: Dict[key, ForecastState]` and `agent.forecast_epoch: int`, both created in
`RobotAgent.__init__`. `key` is the plan key already used by `PredictedPolicy`
(`tuple(p.key for p in legs)`), so pushes, two-step plans, carries and fills are all covered.

```python
@dataclass
class ForecastState:
    asked: int                  # tick the forecast was requested
    due: int                    # tick from which the answer is visible to the guard
    answer: Optional[bool]      # None if the forecast failed or was skipped
```

`assess(agent, plan, info)`:

1. `price = price_of(plan, info)`, `known = self.saving(agent, plan)`. If `known <= 0`, return `None`.
2. If the key has no state and `known >= lam * price`: request a forecast (below).
3. `thr = 1.0`. If the key has a state with `answer is not None` and `info.tick >= due`:
   `thr = lam if answer else 1 / lam`.
4. If `known < thr * price`, return `None`. Otherwise return `(known - price, known, price)`.

Requesting a forecast at tick `t`:

1. If the run has already made `cfg.agent_max_forecasts` requests, store `ForecastState(t, t, None)`, count it as
   skipped and stop.
2. `case = build_case(...)` (section 7), `res = forecaster.forecast(case)`.
3. `due = t` if `res.latency_s == 0`, else `t + max(1, ceil(res.latency_s / cfg.tick_seconds))`.
4. Store `ForecastState(t, due, res.answer)` and append one row to `shared.forecast_log` (section 13).

**Why the request is lazy.** Below `lam * price` neither threshold can fire, since `lam <= 1 <= 1 / lam`. The
answer cannot change a decision there, so asking earlier only costs calls. Section 4 proves the guarantee with
the lazy request.

**Hook for late answers.** `PushPolicy.on_tick(self, agents, t)` is a new no-op method, called by the runner once
per tick before `decide`. `GuardedPolicy.on_tick` adds 1 to `agent.forecast_epoch` for every robot that has a
forecast with `due == t` and a non-`None` answer. `RobotAgent._consider_push` adds `self.forecast_epoch` to its
re-evaluation key, so a robot standing still reconsiders when an answer lands.

---

## 4. Theory: the guard with late or missing answers

This extends Theorem 3. It goes into `docs/research/theory.md` as Proposition 5, with the proof below.

**Setting.** As in `theory.md`: one candidate action of cost `c`, non-negative savings `s_j`, prefix sums `S_i`,
known evidence `K_i`, coverage `rho`. Let `r` be the first request with `K_r > 0` and `K_r >= lam c` (the forecast
is requested there), and let the answer become visible at request `a >= r` (`a` may be infinite: no answer).
The threshold is `theta_i = 1` for `i < a`, and for `i >= a` it is `lam` if the answer is yes and `1 / lam` if no.
The rule takes the action before the first request `i` with `K_i > 0` and `K_i >= theta_i c`.

**Proposition 5.**

* Robustness, whatever the answer and whenever it arrives: `ALG <= (1 + 1 / (lam rho)) OPT_a`. This is the same
  bound as Theorem 3.
* Consistency, if the answer is right and views are full: `ALG <= (1 + min(1, lam + W / c)) OPT_a` when the answer
  is yes, where `W = S_{a-1} - S_{r-1}` is the saving that passed while the robot waited; and `ALG = OPT_a` when
  the answer is no.
* No answer at all: the rule is the classical one, `ALG <= (1 + 1 / rho) OPT_a` (Theorem 1 with `theta = 1`).

*Proof of robustness.* Suppose the rule fires at `i`. If `i = 0` then `ALG = c` and `OPT_a >= min(c, K_0) >=
lam c`, so the ratio is at most `1 / lam`. Otherwise the rent is `S_{i-1}`, and since the rule did not fire at
`i - 1`, `S_{i-1} <= K_{i-1} / rho < theta_{i-1} c / rho` (or `S_{i-1} = 0`). Four cases:

1. `theta_i = 1 / lam`. Then `K_i >= c`, so `OPT_a = c`, and `theta_{i-1} <= 1 / lam` gives rent below
   `c / (lam rho)`.
2. `theta_i = 1`. Then `OPT_a = c` and `theta_{i-1} = 1`, so the rent is below `c / rho`.
3. `theta_i = theta_{i-1} = lam`. Rent below `lam c / rho`, `OPT_a >= lam c`: the ratio is below
   `1 / rho + 1 / lam`.
4. `theta_i = lam`, `theta_{i-1} = 1` (the answer arrived at `i`). If `S_{i-1} >= c` then `OPT_a = c` and the ratio
   is below `1 + 1 / rho`. If `S_{i-1} < c` then `OPT_a >= max(S_{i-1}, lam c)`, so
   `ALG / OPT_a <= S_{i-1} / OPT_a + c / OPT_a <= 1 + 1 / lam`.

Each is at most `1 + 1 / (lam rho)`, because `rho <= 1` and `(1 - 1 / lam)(1 - 1 / rho) >= 0`. If the rule never
fires, `ALG = S_{T-1}`; either `S_{T-1} <= c` and the ratio is 1, or `OPT_a = c` and
`S_{T-1} < c / (lam rho)`. QED.

*Proof of consistency.* Views are full, so `K = S`. A right yes means `S_{T-1} >= c`. The rule fires at `a` at the
latest, because `S_a >= S_r >= lam c`. The rent is below `c` (the waiting threshold is 1) and at most
`S_{a-1} = S_{r-1} + W < lam c + W`. A right no means `S_{T-1} < c`; neither threshold 1 nor `1 / lam` is ever
reached, so `ALG = S_{T-1} = OPT_a`. QED.

**New functions in `src/doi/abstract/core.py`:**

```python
def run_guarded(s, c, views, lam: float, says_yes: Optional[bool], delay: int = 0) -> SingleOutcome
def guarded_wait(s, c, views, lam: float, delay: int) -> float      # W; 0.0 if the forecast is never requested
def bound_guarded_consistency(lam: float, w: float, c: float) -> float    # 1 + min(1, lam + w / c)
```

`delay` is in requests: the answer is visible from request `r + delay`. `says_yes=None` means no answer.

**What the proposition does not cover.** It is a statement about one candidate action in the abstract model. In
the simulator a robot's price changes with its position, several candidates compete, and motion adds congestion.
The repo already has a test where the predicted rule exceeds its bound with several candidates
(`test_multi_candidate_counterexample_on_g1`). For the simulator the result is measured, not claimed.

---

## 5. Notices

### 5.1 Data

In `src/doi/scenarios.py`:

```python
@dataclass(frozen=True)
class Notice:
    notice_id: str
    emit_tick: int
    kind: str               # "surge" | "drop" | "distractor"
    group: Optional[str]    # the zone group it is about, e.g. "south bays"; None for a distractor
    is_true: bool           # hidden from robots; used only for scoring
    text: str               # what robots receive
```

`Scenario` gains `notices: List[Notice]` and `zones: Dict[str, Tuple[Pos, ...]]`, both empty by default.

In `src/doi/crdt.py`: `NoticeRecord(notice_id, tick, text)` and `NoticeSet`, a keyed grow-only set built like
`ObstructionSet` (one record per id; on a clash the smaller `(tick, text)` wins, so merges commute).

In `src/doi/belief.py`: a new component `notices`, added to `_COMPONENTS`, and `add_notice(rec)`. Because it is a
component it gossips with the rest of the belief, in full and delta mode, with no change to the network.

A robot receives only `NoticeRecord`. `kind`, `group` and `is_true` never leave the scenario.

### 5.2 Delivery

`NoticeFeed` in `src/doi/notices.py`, built like `Intake` in `runner.py`. At `emit_tick` the notice goes to the
live robot nearest the map centre (ties by id), then spreads by gossip. This matches the project's premise that
there is no reliable central link: who knows a notice depends on communication range, loss and delay, as with the
ledger.

A notice changes nothing by itself. Only the `keyword` and `llm` forecasters read notices.

### 5.3 Text

`src/doi/notices.py` holds two disjoint template banks, `dev` and `test`, with no template and no key phrase in
common, for each of the three kinds. This is deliberate: the incident templates leak labels through shared
phrases (see `data/incidents/README.md`), and the same mistake here would make the keyword baseline look as good
as a model.

* `dev` bank: each surge or drop notice names exactly one zone group.
* `test` bank: wording differs, and some notices name both groups ("work moves from the north bays to the south
  bays").
* Distractors are about things that do not change where robots go (a fire drill reminder, a label printer fault).
* A third source, `human`, reads texts written by people from `data/forecasts/human_notices.jsonl` (section 14.2).

Rendering is deterministic: `stream(seed, f"notice-{notice_id}")`.

---

## 6. Scenario `shift_notice`

Built on the existing `shift` scenario and its defaults: a 15 by 21 map, a wall at column 10 with doors at rows 6
to 8, pallets in the wall at rows 2, 4, 10 and 12, and goals drawn mostly from a busy band of rows that moves
from rows 0 to 6 to rows 8 to 14 after each robot's task 10.

Added parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `notice_mode` | `"true"` | `"true"`, `"false"`, `"missing"` or `"quiet"` (table below) |
| `notice_tick` | `5` | tick at which the surge and drop notices are emitted |
| `n_distractors` | `1` | distractor notices, at ticks drawn from `0 .. 2 * notice_tick` |
| `notice_bank` | `"test"` | `"dev"`, `"test"` or `"human"` |

| `notice_mode` | Does the busy band move? | Notices about it |
|---|---|---|
| `true` | yes | surge for the south bays, drop for the north bays, both true |
| `false` | no | the same two notices, both false |
| `missing` | yes | none |
| `quiet` | no | none |

"The band does not move" means the first band stays busy for every task. Distractors are emitted in all four modes.

Zones, for the `trip_saving` tool: `west north bays`, `west south bays`, `east north bays`, `east south bays`
(the free cells of each room inside each band). Zone groups, for notices: `north bays` and `south bays`.

Task lists depend on the seed and on whether the band moves, and on nothing else, so every arm in a comparison
runs the same tasks.

---

## 7. The forecast case

`src/doi/forecast/case.py`. A `ForecastCase` is plain data that can be written to JSON and read back.

```python
@dataclass(frozen=True)
class ForecastCase:
    case_id: str                       # first 16 hex digits of the sha256 of the model-visible fields
    robot: int
    tick: int
    mode: str                          # "push" | "bundle" | "carry" | "fill"
    kind: str                          # obstacle kind
    obstacle: Pos
    landing: Pos                       # where the obstacle ends up
    plan_key: Tuple
    price: float
    known_saving: float
    notices: Tuple[Tuple[str, int, str], ...]      # (notice_id, tick received, text), sorted by id
    ledger: Dict[str, float]
    zones: Dict[str, Dict[str, Tuple[int, int]]]   # name -> {"rows": (lo, hi), "cols": (lo, hi)}
    trip_saving: Dict[str, Dict[str, float]]       # "from|to" -> {"mean_saving", "share_saving", "pairs"}
    numeric_forecast: bool             # hidden from the model
    truth: Optional[bool]              # hidden from the model
```

`ledger` holds exactly what the numeric forecast uses, so an agent with no notices can reproduce it:
`robots_known`, `tasks_per_robot`, `trips_recorded`, `trips_expected`, `saving_on_recorded_trips`,
`saving_on_my_current_trip`, `my_tasks_done`, `my_tasks_left`.

`trip_saving` is computed for every ordered pair of zones on the robot's **believed** map: 6 origin cells from
the first zone and 4 destination cells from the second, chosen by a seeded shuffle, and for each pair the route
length with `plan.before` blocked minus the length with `plan.after` blocked. If the scenario has no zones the
dict is empty.

`numeric_forecast` is `LedgerPolicy.saving` with `forecast=True`, compared with `price`: what `rof_p` would say at
this moment.

**Truth label.** `truth = true_total_saving >= price`, where `true_total_saving` is the sum, over every task of
every robot in the scenario (past and future), of the route length with `plan.before` blocked minus the length
with `plan.after` blocked, using the evidence engine's rule that a trip's own endpoints count as open. This is the
simulator's version of "the total saving reaches the cost" in Theorem 3. It is a static measure: it ignores
congestion and later changes to the map. `truth` is `None` only when a case is built without a scenario.

---

## 8. Tools

`src/doi/forecast/tools.py`. Each tool is a pure function of a `ForecastCase`.

| Tool | Arguments | Returns |
|---|---|---|
| `read_notices` | none | the notices this robot has received |
| `ledger_summary` | none | the `ledger` dict, plus `price` and `known_saving` |
| `trip_saving` | `from_zone`, `to_zone` (each one of the case's zone names) | mean saving per trip, the share of trips that save anything, and the number of pairs sampled |
| `answer` | `will_pay: bool`, `confidence: number in [0, 1]`, `reason: string` | ends the forecast |

A call to an unknown tool, or with a bad argument, returns `{"error": "<reason>"}` to the model and counts against
the budget. No tool writes anything.

---

## 9. The agent loop

`src/doi/forecast/agent.py`.

```python
PROMPT_VERSION = "forecast-v1"
MAX_TURNS = 4            # model responses per forecast
MAX_TOOL_CALLS = 8       # tool calls per forecast, not counting `answer`

@dataclass(frozen=True)
class ForecastResult:
    answer: Optional[bool]
    confidence: Optional[float]
    reason: str
    failed: str              # "" on success, else the failure code
    latency_s: float         # sum over the model calls
    calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    model: str

def run_agent(case: ForecastCase, chat: Chat, mode: str = "tools") -> ForecastResult
```

`Chat` is any object with the `chat` method of section 12.1: the real client, the fake, or the record-and-replay
wrapper.

**Mode `tools`.** The conversation starts with the system prompt and a user message that describes the candidate
action (mode, obstacle kind and cell, landing cell, price, known saving so far) and lists the zone names with
their row and column ranges. On each turn the model must call at least one tool. On the last turn, or once
`MAX_TOOL_CALLS` is used up, the request forces a call to `answer`. Tool results are appended as tool messages and
the loop continues until `answer` is called.

**Mode `single`.** One request. The user message also contains the notices, the ledger summary and the whole
`trip_saving` table, and the request forces `answer`. This is the ablation that shows what the tools add.

**System prompt (version `forecast-v1`):**

```
You advise one warehouse robot. The robot can move an obstacle out of the way now, which costs a one-off
price, or keep walking round it. Give one forecast: over the rest of this shift, will the whole fleet's
travel saving from moving this obstacle reach the price?

You know only what this robot knows. Use the tools.
- read_notices: messages the robot has received about future work. They can be wrong or irrelevant.
- ledger_summary: the traffic recorded so far and how many trips are still to come.
- trip_saving: how much one trip between two zones would save if the obstacle were moved.

Past traffic may not continue if a notice says the work is moving. A notice that does not change where
robots travel should not change your forecast. When you are ready, call answer. You have 4 turns.
```

**Failure.** Each of these gives `answer=None` and a failure code, and the guard then keeps threshold 1:

| Code | Cause |
|---|---|
| `no_tool_call` | a response with no tool call |
| `no_answer` | the turn budget ran out without `answer` |
| `bad_answer` | `will_pay` is not a boolean, or `confidence` is not a number in `[0, 1]` |
| `client_error` | the request raised (timeout, HTTP error) |

Temperature is 0. The `reason` is cut to 200 characters and is logged, never acted on.

---

## 10. Forecasters

`src/doi/forecast/forecasters.py`. One interface: `forecast(case) -> ForecastResult`.

| `cfg.forecaster` | Answer | Latency |
|---|---|---|
| `numeric` | `case.numeric_forecast` | 0 |
| `keyword` | section 10.1 | 0 |
| `oracle` | `case.truth` | 0 |
| `inverted` | `not case.truth` | 0 |
| `llm:<model key>` | `run_agent(case, chat, cfg.agent_mode)` | as measured |

`oracle` and `inverted` are references: the best and the worst a forecaster can be. They need `truth`, so they
raise `ValueError` on a case without one.

### 10.1 The keyword baseline

The answer a team would get with no model. Find the zone group whose row band contains the obstacle's row. Among
the robot's notices whose text contains that group's first word ("north" or "south", case-insensitive): if any
contains a drop cue, answer no; otherwise if any contains a surge cue, answer yes; otherwise answer
`case.numeric_forecast`. The cue lists are fixed from the `dev` bank and written down in `notices.py`. They are
not tuned on `test` or `human`.

This baseline should do well on `dev`-style text. Whether a model beats it on `test` and `human` text is one of
the questions the evaluation answers.

---

## 11. Changes to existing modules

Only these edits are allowed. Everything else in `src/doi` stays as it is.

| File | Edit |
|---|---|
| `config.py` | `"rof_a"` in `POLICIES`; fields `forecaster: str = "numeric"`, `agent_mode: str = "tools"`, `agent_cache: str = "experiments/results/doi/agent_cache"`, `agent_live: bool = False`, `agent_max_forecasts: int = 200`; validation of each |
| `crdt.py` | `NoticeRecord`, `NoticeSet` |
| `belief.py` | the `notices` component and `add_notice` |
| `scenarios.py` | `Notice`; `Scenario.notices` and `Scenario.zones`; the `shift_notice` scenario |
| `runner.py` | build a `NoticeFeed` and call its `emit` and `deliver` each tick; call `policy.on_tick`; copy the forecast log and counters into the result |
| `agent.py` | `self.forecasts = {}` and `self.forecast_epoch = 0` in `__init__`; `forecast_epoch` in the `_consider_push` key; `ingest_notice` |
| `policies.py` | `PushPolicy.on_tick` (no-op); `Shared.forecast_log`; `GuardedPolicy`; `make_policy` |
| `metrics.py` | the fields and summary columns of section 13 |
| `narrate.py`, `run_doi.py` | the arm note, the scenario note, and flags `--forecaster`, `--agent-mode`, `--agent-live` |
| `llm/client.py` | `chat` (section 12); the key rule in section 12.3 |
| `abstract/core.py` | the three functions of section 4 |

**These edits must not change any existing run.** An empty notice set adds no gossip units and never bumps the
belief version; `on_tick` does nothing for other arms; `forecast_epoch` stays 0. A test asserts that `rof`, `rof_p`
and `central` give the same `J` as before on three existing scenarios.

---

## 12. Model calls

### 12.1 Chat with tools

`OpenAICompatClient` gains one method and keeps `complete` untouched:

```python
@dataclass(frozen=True)
class ChatResponse:
    message: dict            # the assistant message as returned: role, content, tool_calls
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model: str

def chat(self, messages: list, tools: list, tool_choice, max_tokens: int) -> ChatResponse
```

It posts to `/chat/completions` with `tools` and `tool_choice`. `tool_choice` is `"required"` on ordinary turns and
`{"type": "function", "function": {"name": "answer"}}` on the forced turn. `FakeChatClient` takes a function from
`(messages, tools)` to an assistant message, for tests.

### 12.2 Record and replay

`src/doi/llm/chatcache.py`: `CachedChat(client, root, model_key, live)`.

* The key of a call is the sha256 of the canonical JSON of `(PROMPT_VERSION, model_key, messages, tools,
  tool_choice)`. Every call in a multi-turn forecast has its own key.
* A hit returns the stored response, including its stored latency. A re-run of the same seed therefore makes the
  same decisions at the same ticks and makes no calls.
* A miss with `live=False` raises `KeyError` with the command that fills the store. Runs never call a model unless
  `cfg.agent_live` is set.
* A miss with `live=True` calls the client and appends the response to `<root>/<model_key>/chat.jsonl`.

Changing the prompt, the tools or the case contents changes the keys, so a stale store cannot be replayed by
mistake. `PROMPT_VERSION` must be raised whenever the system prompt or a tool schema changes.

### 12.3 OpenAI configuration

Model keys are set through the existing environment variables: `DOI_LLM_<KEY>_URL`, `DOI_LLM_<KEY>_MODEL`, and the
optional `_TOKEN_PARAM`, `_TEMPERATURE`, `_JSON_MODE`. The experiments use two keys, `small` and `large`.

The API key is read from `DOI_LLM_<KEY>_API_KEY`. Today `OPENAI_API_KEY` is a fallback for the key named `hosted`
only. That rule stays, and the fallback is extended to any model key whose URL starts with
`https://api.openai.com/`, so `small` and `large` work with one exported key and the key is not sent to any other
host under those names.

### 12.4 Keeping cost bounded

* Lazy requests (section 3) and one forecast per robot and plan key.
* `agent_max_forecasts` per run; requests past it get no answer and are counted.
* At most `MAX_TURNS` calls per forecast.
* Every script that can call a model first runs the same grid with the `numeric` forecaster, counts the forecast
  requests, prints `requests x MAX_TURNS` as the upper bound on calls with a token estimate, and asks for
  confirmation unless `--yes` is given.

---

## 13. What is logged

`RunResult.forecast_log`, one dict per request: `tick`, `robot`, `plan_key`, `case_id`, `forecaster`, `answer`,
`truth`, `numeric_forecast`, `confidence`, `failed`, `skipped`, `due`, `known`, `price`, `calls`, `tool_calls`,
`latency_s`, `prompt_tokens`, `completion_tokens`, `reason`.

`metrics.summary_row` columns: `forecasts`, `forecast_failed`, `forecast_skipped`, `forecast_acc`, `wrong_yes`,
`wrong_no`, `agent_calls`, `agent_latency_s`, `agent_prompt_tokens`, `agent_completion_tokens`. They are 0 or
`nan` for arms that make no forecasts.

`wrong_yes` is a yes when the truth is no; `wrong_no` is the reverse. They are kept apart because they cost
different things: a wrong yes pays the price early, a wrong no pays detours up to `price / lam`.

---

## 14. Showing the model is fit for the purpose

Fitness is a measured claim at four levels. A single model's score shows nothing about fitness, so every level
is reported for the baselines as well.

### 14.1 The four levels

| Level | Question | Measurement | Script |
|---|---|---|---|
| 1. Accuracy | Is the forecast right? | accuracy, `wrong_yes` rate, `wrong_no` rate on `test` cases | `doi_agent_eval.py` |
| 2. Robustness | Does that hold on text it was not developed on? | the same on `human` cases; the drop from `dev` to `test` to `human` | `doi_agent_eval.py` |
| 3. Risk and cost | Are its mistakes cheap, and what does it cost to run? | calibration of `confidence` (expected calibration error, 10 bins); failure rate by code; calls, tool calls, latency (p50, p95, and in ticks), tokens per forecast | `doi_agent_eval.py` |
| 4. End to end | Does it lower fleet cost, and how bad is it when the notices are wrong? | `J_censored` per arm and notice mode, paired by seed | `doi_e9_agent.py` |

The cap on the harm of a wrong forecast is not a measurement: it is Proposition 5, checked exactly by the property
tests of section 15.1.

### 14.2 Datasets

`experiments/doi_agent_cases.py` runs `rof_a` with the `oracle` forecaster on `shift_notice` in all four notice
modes and writes every requested case.

| File | Seeds | Notice bank | Use |
|---|---|---|---|
| `data/forecasts/dev.jsonl` | 0..19 | `dev` | prompt development only |
| `data/forecasts/test.jsonl` | 100..149 | `test` | held-out scoring |
| `data/forecasts/human.jsonl` | 300..319 | `human` | held-out scoring, always reported separately |

Seeds 200..229 are reserved for E9 and appear in no dataset.

`human_notices.jsonl` is collected by hand, as the incident `human.jsonl` protocol describes: at least 100 texts
(40 surge, 40 drop, 20 distractor), written by people who have not seen the templates. Each writer is given the
kind and the zone group and writes one message as they would on a radio or chat. No real names, sites or personal
data, because the texts are sent to a hosted API. Until this file exists, level 2 is reported as not done.

### 14.3 Experiment E9

`experiments/doi_e9_agent.py`, on the shared grid runner in `doi_common.py`.

* Scenario `shift_notice`, 8 robots, 20 tasks each, `lam = 0.5`, `bundle_max = 1`.
* Seeds 200..299 for arms that need no model; model arms run on 200..229 unless the budget allows more.
* Measure: **avoidable cost**, `J_censored` minus the `J_censored` of the `free` arm on the same seed. Total cost
  is mostly travel that no rule can avoid, so differences between forecasters are a small share of it
  (section 14.4).
* Axis: `notice_mode` in `true`, `false`, `missing`, `quiet`.
* Arms: `never`, `rof`, `rof_p`, and `rof_a` with each of `numeric`, `keyword`, `oracle`, `inverted`,
  `llm:small` (tools), `llm:large` (tools), `llm:small` (single).
* Secondary sweeps, on the `true` and `false` modes only: `lam` in `0.25, 1.0`; `notice_tick` in `5, 150`.
* `--quick`: seeds 200..202, no secondary sweeps, and the `llm` arms only if their store already holds the calls.

Hypotheses, stated before any full run:

| | Claim |
|---|---|
| H9a | With true notices, `rof_a` with a model costs less than `rof_a` with `numeric`. |
| H9b | With false notices, `rof_a` with a model costs no more than `rof_a` with `inverted`; its cost relative to `rof` is reported. |
| H9c | With no notices (`missing`, `quiet`), `rof_a` with a model is within a fixed margin of `rof_a` with `numeric`. |
| H9d | On `human` cases, a model's forecast accuracy is above the keyword baseline's. No claim is made on `dev` or `test`. |

The margin in H9c and the tests used are fixed after the `--quick` pilot and before the full run, written into
`docs/research/results.md`, and marked with the git tag `prereg-agent-v1`. They are not changed afterwards.

### 14.4 Headroom gate (added 2026-10-06, after a scratch check)

A model can only help as much as a perfect forecast helps. A scratch check on the existing `shift` scenario (8
robots, 20 tasks, seeds 200..229, `lam = 0.5`, default costs, lazy guard) gave these mean avoidable costs:
no forecast 142, numeric forecast 142, perfect forecast 112, inverted forecast 172. A perfect forecast therefore
removes about a fifth of the avoidable cost, which is about 1.3% of total cost, and its paired advantage over the
numeric forecast was not clearly separated from zero at 30 seeds (median 3.5, 95% interval 0 to 30.5). This was
not run on `shift_notice`, which did not exist yet.

So the build has a gate. After the guard, the notices, the scenario and the forecasters that need no model are
built, `experiments/doi_e9_headroom.py` measures `oracle`, `numeric`, `inverted`, `keyword` and no forecast on
`shift_notice` over 100 seeds and a small set of cost settings. The model parts are built only if the owner
decides the headroom is worth it. If it is not, the options are a different scenario or cost setting, or stopping
at the non-model result. No model call is made before this gate.

---

## 15. Testing

No test calls a model. All use `FakeChatClient` or the non-model forecasters.

### 15.1 Guard on pure numbers (`tests/doi/test_doi_abstract_guard.py`)

1. Hand case: `s = [1] * 30`, `c = 10`, `lam = 0.5`, yes. With `delay = 0` the rule fires at 4 and `alg = 14`.
   With `delay = 3` it fires at 7, `alg = 17`, `W = 3`, and `bound_guarded_consistency = 1.8`.
2. `delay = 0` equals `run_predicted` exactly, for both answers, on 500 random cases over every view kind.
3. Robustness: 500 random cases, `delay` in `0..20`, answer yes, no or none, `lam` in `0.25, 0.5, 1`:
   `alg <= bound_predicted(lam, rho) * opt + 1e-9` whenever `rho > 0` and `opt > 0`.
4. Consistency: full views and a right answer. Yes: `alg <= bound_guarded_consistency(lam, W, c) * opt + 1e-9`.
   No: `alg == opt`.
5. No answer equals `run_threshold` with `thr = 1`.

### 15.2 Notices and the scenario

Merge laws for `NoticeSet` (commutative, associative, idempotent); a notice reaches a second robot by gossip and
not without it; the four notice modes produce the stated task lists and notices; the two template banks share no
template; the same seed gives the same texts.

### 15.3 Case and tools

On a small hand map: `trip_saving` values checked by hand; the ledger numbers reproduce `rof_p`'s forecast;
`truth` checked by hand for a case that pays and one that does not; a case survives a JSON round trip;
`case_id` ignores the hidden fields.

### 15.4 Agent loop

A scripted model that reads notices then answers; one that never calls `answer` (`no_answer`); one that returns
text only (`no_tool_call`); a malformed `answer` (`bad_answer`); a client that raises (`client_error`); more than
`MAX_TOOL_CALLS` calls; an unknown tool. Mode `single` makes exactly one call.

### 15.5 Policy and runs

1. Existing arms are unchanged: `rof`, `rof_p`, `central` give the same `J` as before on `single_block`,
   `multi_block_wall` and `shift`.
2. `rof_a` with a forecaster that always fails makes the same decisions as `rof` on `single_block`.
3. A forecast is requested once per robot and plan key, and never below `lam * price`.
4. An answer with latency is ignored before `due` and used from `due`.
5. `agent_max_forecasts = 0` gives the same result as test 2 and counts every request as skipped.
6. Record and replay: a run with a live fake, then the same run with `live=False`, gives an identical `RunResult`
   and zero client calls; a miss with `live=False` raises.
7. `rof_a` with `oracle`, `inverted`, `numeric` and `keyword` finishes `shift_notice` in all four modes with no
   stall and no unfinished task.

### 15.6 Scripts

`doi_agent_cases.py`, `doi_agent_eval.py` and `doi_e9_agent.py` each run in `--quick` mode with non-model
forecasters, return 0 and write their files.

---

## 16. Build order

Each phase ends with the whole suite passing and one commit. Commit messages carry no tool or co-author
attribution.

1. Guard on pure numbers and Proposition 5 in `theory.md`.
2. Notices: records, belief component, feed, template banks; scenario `shift_notice`.
3. Forecast case, truth label and tools.
4. The forecasters that need no model, `GuardedPolicy`, runner and robot hooks, metrics.
5. The headroom pilot (section 14.4). **The build stops here for the owner's decision.**
6. Chat client, record and replay, the agent loop, the model forecaster, command-line flags.
7. Dataset builder and `doi_agent_eval.py`.
8. `doi_e9_agent.py`, quick mode only.
9. Documents: README arms table, `narrate.py`, demo guide, results status.

No full experiment and no paid model call is part of the build. Those are run by the owner afterwards, after the
pilot and the preregistration tag.

---

## 17. Out of scope

* Agent memory or learning between forecasts.
* A fleet-level agent, or agents that talk to each other.
* Ranking candidates or proposing actions.
* Local or fine-tuned models. The client already reaches any OpenAI-compatible server, so they need no new code,
  but they are not part of this build or its experiments.
* The human approval gate (next spec).
* Warehouse benchmark maps and a digital twin (the spec after that).
* Image input.

---

## 18. Risks and what is not claimed

* **Small headroom.** On the existing `shift` scenario a perfect forecast removes about a fifth of the avoidable
  cost and about 1.3% of total cost (section 14.4). A real model will capture only part of that, so the end-to-end
  effect may be too small to show. The headroom gate exists to find this out before any model work.
* **No bound in the simulator.** Proposition 5 is for one candidate action in the abstract model. With several
  candidates, a price that moves with the robot, and congestion, cost is measured.
* **The scenario is built for the agent.** `shift_notice` exists so that text carries information the ledger
  lacks. The paper must say so, and the `missing` and `quiet` modes are there to show what happens when it does
  not.
* **The keyword baseline may match the model on template text.** That is a finding, not a failure. The claim
  about models rests on the `human` set.
* **Truth is static.** The label ignores congestion and later map changes, so "accuracy" is accuracy against that
  definition.
* **Forecast count.** One forecast per plan key can mean several per obstacle for one robot. If the dry-run count
  is too high for the budget, the fallback is to key forecasts by `(mode, obstacle, landing)`; that is a change
  to this spec and needs the owner's approval.
* **Latency.** A four-call forecast may take several seconds, which is many ticks at `tick_seconds = 0.5`. The
  consistency bound degrades with the saving that passes while waiting (`W`). This is reported, not hidden.
* **No novelty claim** for Proposition 5 until the prior-art reads in `prior-art-verification.md` are done.
  Learning-augmented ski rental with delayed predictions is the area to check.

## 19. What needs the owner

* The human-written notice texts (section 14.2). Nobody else can produce them.
* OpenAI model names for the `small` and `large` keys, and a spending limit for the pilot and the full run.
* The H9c margin, after the pilot.
