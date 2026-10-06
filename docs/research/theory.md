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
* No claim of novelty for Proposition 5 until learning-augmented ski rental with delayed predictions has been
  checked in the literature.
