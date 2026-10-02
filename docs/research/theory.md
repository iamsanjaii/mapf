# Theory: propositions P1 to P4

Support for the experiments, not the headline (spec section 1.6). Each proposition states its
assumptions, gives a proof or a sketch, and names the simulator quantity that tests it. Where a bound
needs an assumption the simulator does not enforce, the gap is stated.

> **Scope note (2026-10-02).** These propositions were written for the earlier pit-and-kit model, where one
> "edit" has a purchase cost `B`. In the push model `B` is the price of a push run (walk, push steps at
> `kappa` times the obstacle's weight, fee, walk on, over the ideal route). The ski-rental argument only uses
> "rent paid so far" against "price", so P1 to P4 carry over in form. Two things are new and are **not**
> covered by the proofs: a pushed obstacle can land on a route and create new rent (collateral), and the
> price of a push depends on where the robot is. Treat the bounds below as the intended shape, not a result.

## Setting and notation

One permanent edit (a "pit"), tasks indexed by `t = 1, 2, ..., T`.

* `r_t` in `[0, r_max]`: rent of task `t`, `rent = max(0, d_block - d_open)`, counted when the task is
  planned (spec 4.1). `S_t = r_1 + ... + r_t` is the counted rent after task `t`; `R = S_T`.
* `B` is the realised purchase cost `B_real` (fee plus carried steps times `kappa` plus unloaded steps and
  the pickup and apply ticks, spec 4.4). `B_est` is the estimate the trigger uses (fee plus `kappa` times
  the carry only). `B_est <= B` in the simulator because `B_est` leaves out the unloaded walk.
* `m = min(B, B_est)`, `M = max(B, B_est)`.
* Hindsight comparator `OPT = min(B, R)`: buy at time 0, or never buy. This is the benchmark of spec 2.6
  restricted to one pit; `HR_av` in the simulator removes the travel no arm can avoid.
* Trigger (theta = 1): fire at the first task `tau` whose planning brings the counted rent to
  `S_tau >= B_est`.

Assumptions used throughout:

* **A1** The counted rent of a task equals the extra cost of not having the edit for that task.
* **A2** After the edit, every later task pays its open-path cost (rent 0).
* **A3** The haul is atomic for the firing robot: it replaces the detour of task `tau` and ends before the
  next task begins.

## P1. Full information

**Statement.** Under A1 to A3 with one robot (equivalently, a fleet whose ledger is instantaneous and
complete),

```
ALG <= (1 + M/m + r_max/m) * OPT   if the algorithm buys,
ALG <  max(1, B_est/B) * OPT       if it never buys (never larger than the first bound),
```

where `M/m = max(B/B_est, B_est/B)`. With `B_est = B` this is the classical `2 + r_max/B`.

**Proof.** Let `tau` be the first task with `S_tau >= B_est`.

*The algorithm buys.* Before the edit is usable the fleet has walked the detours of tasks `1..tau-1`, which
cost `S_{tau-1} < B_est` by minimality of `tau`. Task `tau` is counted at planning but replaced by the haul;
in the worst case (the trigger is evaluated after the robot already started the task) its rent is also
paid, adding at most `r_max`. So paid rent is below `B_est + r_max`, and `ALG <= B_est + r_max + B`.
Since `R >= S_tau >= B_est`, `OPT = min(B, R) >= min(B, B_est) = m`. Then

```
ALG / OPT <= (B_est + B + r_max) / m = 1 + M/m + r_max/m.
```

*The algorithm never buys.* Then `S_T < B_est`, `R = S_T`, and `ALG = R`. If `R <= B`, `OPT = R` and the ratio
is 1. Otherwise `B < R < B_est`, so `OPT = B` and the ratio is `R/B < B_est/B`, which is at most
`1 + M/m`. Both cases are bounded by the first line. QED.

**Tightness.** For `B_est = B` the adversary that stops the rent right after the trigger forces ratio
`2 - epsilon` against any deterministic threshold rule (classical ski rental). The `r_max/m` term is the
cost of counting rent one task ahead.

**Where the simulator departs.** Congestion makes realised rent differ from counted rent (E1 reports both),
and a single-cell gap can add waiting on every arm alike. `B_real/B_est` is logged per edit
(`mean_B_real`, `mean_B_est`); for the toy map of Task 10 it is `13/9`, so the bound is stated with
`B_real` in the denominator, never `B_est`.

**Observables.** `hr_av` against `2 + r_max/B_real`; `mean_B_est`, `mean_B_real`; counted against realised
rent. **Status: proved** under A1 to A3. The multi-robot case with a perfect instantaneous ledger reduces to
this proof; everything that breaks that reduction is P2.

## P2. Information delay

**Statement.** Let `K` be the rent the firing robot knows at the trigger and `S*` the true counted fleet
rent at that moment, `D = S* - K >= 0` (the rent missing from the firing robot's view). Let `L` be the
number of ticks from the trigger to the fill (stagger plus approval wait plus haul), and `lambda` the
maximum fleet rent accrued per tick. Then, in addition to the P1 terms,

```
ALG <= B_est + r_max + D + lambda * L + B   and   ALG / OPT <= (1 + M/m + r_max/m) + (D + lambda * L) / m.
```

**Proof sketch.** The robot fires when its known rent first reaches `B_est`; between consecutive evaluations
the known rent grows by at most `max(r_max, J)` where `J` is the largest chunk a merge can add. Absorbing
`J` into `D`, the known rent at the trigger is below `B_est + r_max`, hence the true counted rent at the
trigger is below `B_est + r_max + D`. During the `L` ticks until the edit is usable the fleet keeps paying
rent at rate at most `lambda`. Adding the purchase `B` gives the first inequality; dividing by
`OPT >= m` (as in P1) gives the second.

**Checkable quantities.** Per edit: `trigger_tick`, `claim_tick`, `fill_tick`, `approval_wait`. Per trigger:
`coverage = known / true_rent` (`true_rent_at_triggers`), so `D = (1 - coverage) * true_rent`. `L` is
`fill_tick - trigger_tick`. The gate (H8) enters through `approval_wait` only: the robot keeps working
meanwhile, so the wait is part of `L`, not of `B_real`.

**Not covered.** With `claim = off`, `k` robots may haul concurrently inside the window `L`; each pays its
own `B` and all but one apply is rejected by the world, so the additive term becomes `(k - 1) * B_real`
(this is what E4 measures as `wasted_haul_cost`). The bound also assumes `D` and `L` are small relative to
the horizon; it says nothing about how `D` depends on `r_comm`, which is the empirical content of H2.

**Status: sketch** (the algebra closes; the bound on the per-evaluation jump `J` is absorbed into `D`
rather than derived from the gossip model).

## P3. Isolation

**Instance.** `n` robots, identical task streams so each pays the same per-task rent `r` on the same pit,
no ledger channel (`r_comm = 0`), and none of them learns of the fill before it would itself fire.

**Statement.** Every robot fires after `ceil(B_est / r)` tasks of its own rent. When the first robot has
bought, each robot has paid at least `B_est - r` in rent, so

```
ALG >= n * (B_est - r) + B,     OPT <= B   (buy at time 0, when R = n*r*T >= B),
ALG / OPT >= n * (B_est - r)/B + 1  ->  n + 1  as B_est = B and r/B -> 0.
```

With a ledger the same fleet pools its rent, fires after paying about `B_est` in total, and the ratio is about
`2`. The price of isolation is therefore about `n - 1` purchase-equivalents of rent.

**Proof.** Immediate from the construction: rent is paid by each robot independently until the first
purchase, the first purchase happens at the first robot's threshold crossing, and all robots cross
together because their streams are identical. The upper bound `ALG <= n * (B_est + r_max) + B` follows from
P1 applied per robot, which gives `Theta(n)` overall.

**Prior art check (spec section 12, item 2).** The abstract of arXiv 2507.15727 (Wang, Sun, Beyhaghi, Lui,
Hajiesmaili, Wierman, "Competitive Algorithms for Multi-Agent Ski-Rental Problems", read on 2026-10-02)
describes agents that choose between daily rental, individual purchase, or a discounted group pass, with
agents exiting over time, and analyses overall, state-dependent and individual-rational ratios. It does not
mention a no-communication lower bound or an `n + 1` ratio. That is an abstract-only check; the full paper
is read in Task 20 and the verdict is recorded in `docs/research/prior-art-verification.md`. Until then P3
is not claimed as new.

**Observables.** `hr_av` at `r_comm = 0` against `n_beneficiaries + 1` in the equal-rent case (H2);
`mean_coverage`, predicted near `1/n` when each robot sees only its own rent. **Status: sketch** (the lower bound is proved for the stated instance; its
sensitivity to sensing the filled cell on approach is not analysed).

## P4. Complements and substitutes

**Series corridor (complements).** Let `k` pits lie in series on the only open route, so every open path
uses all `k` and the blocked route is the same detour.

* *Per-cell ledger.* Opening any single pit leaves the others closed, so `d_block - d_p = 0` for every `p`:
  `single_pit_rents` is empty and the per-cell evidence is identically zero. A per-cell rule never fires.
  Its cost equals NeverFill, `ALG = R`, while `OPT = min(B_total, R)`; the ratio `R / B_total` is unbounded
  in the horizon. (Tested: `test_per_pit_series_is_zero`, `test_series_per_pit_ledger_never_fires_bundle_does`.)
* *Record ledger.* The canonical open path of every record uses the same bundle `T` of all `k` pits, so
  `evidence(T) = sum of rents` and `buy_est(T) = sum of per-pit buy_est`. This is P1 with
  `B_est = buy_est(T)` and `B = B_real(T)`, hence the P1 bound. After `j < k` pits are filled (the hauler
  fills the outer pit first, which keeps the next pit reachable), the residual bundle shrinks while each
  record's contribution `min(rent_counted, d.rent)` is unchanged, so evidence stays above the smaller
  `buy_est` and the remaining pits fire at once.

**Parallel substitutes.** `k` pits each of which alone opens a path for a record.

* *Post-fill soundness (proved).* Once a robot's belief contains the fill of `p`, every record `r` that `p`
  serves has `d_block(F) = d_open`, hence `d.rent = 0` and an empty bundle; the evidence engine drops it.
  Records not served by `p` contribute `min(rent_counted, d_block(F) - d_open)`, the marginal rent beyond
  what `p` already provides. A second purchase therefore needs marginal rent that pays for itself, never
  stale evidence of records `p` already serves. (Tested: `test_substitute_fill_invalidates_evidence`.)
* *Evidence is a function of CRDT state.* It depends only on the record set and the filled set, so robots
  with equal states compute equal evidence and merge cannot make two robots disagree about a trigger.

**What the record ledger does not guarantee.**

1. *Splitting before any fill.* Records choose their canonical bundle by tie-break
   `(steps, pits used, insertion order)`, so evidence can split across substitutes by origin-destination
   geometry and delay the first trigger relative to a ledger that pooled it.
2. *Concurrent purchases inside the latency window.* A claim names the pits of one proposal, so a claim on
   `p` does not stop a robot from proposing the substitute `q` before it learns of `p`'s fill. At most one
   extra purchase per substitute group per window `L` is possible; the world's rejection of a drop on a
   filled cell bounds the damage to one wasted haul.
3. *Optimality.* Nothing here says the chosen subset matches the hindsight optimum for substitutes; it
   shows only that stale evidence does not drive a purchase after the fill is known.

**Observables.** Number of fills on `series_pits` (`rof` fills all pits, `rof_pit` none); on
`two_pits_parallel`, the number of runs in which the second pit is filled and whether the first fill served
every recorded task at that time (`true_rent_at_triggers`). **Status: proved** for the series corridor and
for post-fill substitutes; **open** for the pre-fill splitting effect.

## Summary

| Proposition | Status | Simulator observable (names from `summary_row`) |
|---|---|---|
| P1 full information, `1 + M/m + r_max/m` | proved (A1 to A3) | `hr_av`, `mean_B_est`, `mean_B_real` (E1) |
| P2 information delay, `+ (D + lambda L)/m` | sketch | `mean_coverage`, `mean_approval_wait`, `claims_issued`, `wasted_haul_cost`, `J` difference (E2, E4, E8) |
| P3 isolation, ratio about `n + 1` | sketch (instance lower bound proved); prior art not yet fully checked | `hr_av` and `mean_coverage` at `r_comm = 0` (E2) |
| P4a series: per-cell never fires, records restore P1 | proved | `fills` for `rof_pit` against `rof` on `series_pits` (E3) |
| P4b substitutes: no purchase from served evidence | proved after the fill is known; open before | `fills` on `two_pits_parallel`, `mean_coverage` (E3) |
