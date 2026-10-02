# Results (Stage 0)

Status as of 2026-10-02: simulator, scripts and the quick pilot are done. **The preregistration tag
`prereg-v1` has not been created and no full experiment has been run.** Reasons are listed under
"What blocks the tag". Nothing below is a hypothesis verdict; the pilot uses 3 seeds (1000..1002) and the
smallest axis values.

## Pilot

Command for each: `python experiments/doi_eN_*.py --quick --jobs 4` (outputs in
`experiments/results/doi/eN_quick/`, git-ignored).

### Checks the plan asks for

| Check | Result |
|---|---|
| E1 reaches R/B <= 0.3 and R/B >= 5 somewhere | The quick grid does not (R/B 1.5 to 18). The full sweep does: corner cells (N = 4, `fee` 1/200, `kappa` 1/16, `depot_dist` 1/8, K 2/40) give R/B from 0.037 to 302; 28% of those cells are <= 0.3 and 28% are >= 5. No axis change needed. |
| `HR_av` defined (`J_hind - J_free >= 1`) in most E1 cells | Defined in 100% of RoF rows in the quick grid and in the corner check. Excluded fraction: 0%. |
| `stalled` below 2% per arm | 0% for every arm in E1, E2, E3, E4, E5, E7, E8 pilots (about 600 runs). |
| Set `[pilot]` values of H7 (b), (c) | **Not done.** They need the LLM dev results (below). |
| Intake prompt pilot on dev (Task 14 Step 5) | **Skipped, not faked.** No endpoint is configured (`DOI_LLM_HOSTED_URL`, `DOI_LLM_HOSTED_MODEL` and `OPENAI_API_KEY` are unset) and the plan requires the user to confirm the cost first. |

### What the pilot shows per experiment (3 seeds, indicative only)

* **E1 (H1).** H1a as specified fails in the pilot: worst margin 1.79 with claims off (the plan's setting)
  and 2.73 with claims on. Diagnosis, in the order the spec asks for (section 12):
  * `B_real / B_est` has median 1.84 (for example 23 or 27 against 9): the estimate leaves out the
    unloaded walk and the two action ticks, as intended.
  * The absolute bound of P1 holds: in the worked cell RoF's avoidable cost is 49 against a bound of
    `B_est + B_real + r_max = 50`.
  * `HR_av` divides by hindsight's charge, which is the lower bound `buy_lb` (9 there), while the H1
    threshold uses `B_real` (23). The ratio is inflated by about `B_real / buy_lb`.
  * As a labelled diagnostic (not H1), charging hindsight the realised buy cost makes the same check pass
    (worst margin -0.42).
  * With claims off and a global ledger, all four robots trigger within one tick, three abort, and the
    cell shows 38 waits and a total cost above NeverFill. That is the P2 concurrency term, not a bug.

  The threshold was not changed. This needs a decision before the tag (see below).
* **E2 (H2, direction only).** Spearman(`r_comm`, HR_av(RoF)) = -0.50 on `single_pit`; PoD at
  `r_comm = inf`, loss 0 is 0.983.
* **E3 (H3).** Series: RoF-Pit made no edit in 3/3 runs, RoF made two in 3/3, median HR_av(RoF) 2.53
  (threshold 2.5, so this pilot cell narrowly misses). Parallel: RoF filled the second pit in 2/3 runs and
  in none of them had the first fill already served every recorded task.
* **E4 (H4).** Fails in the pilot: with claims on, wasted haul cost is 10.7 even at loss 0 (6 claims lost
  in 3 seeds) against 16.0 with claims off; at loss 0.5 claims off wastes 8.0, less than claims on. Fills
  never exceeded 1. A possible cause (not verified) is two robots reaching the same stagger tick, so the loser
  learns of the smaller ticket one tick late; to be investigated with 30 seeds before concluding anything.
* **E5 (stretch).** Regret fraction under `shift`: RoF 2.94, RoF-W 0.54, RoF-X 0.23; on `multi_pit_wall`
  0.119, 0.119, 0.124.
* **E7 (H7).** H7a holds in all 96 pilot runs (`unconfirmed_hauls = 0`, `false_report_hauls = 0`).
  J(none) / J(oracle) has median 1.094 over 12 pairs. `false_report_cost` is 0 at `p_false = 0.2` in this
  pilot because robots refute false reports by sensing early.
* **E8 (H8).** The measured increase in J from the gate is about 2.1 times the predicted
  `sum(lambda_T * approval_wait)` (19.0 against 8.8 at latency 5; 79.0 against 38.1 at latency 30), so
  H8a fails in the pilot; the predictor ignores that waiting robots are also out of the fleet's task flow
  while the claim-holder walks. H8b has no wrong-class attempts to compare in 3 seeds.

### Projected full runtimes (4 jobs)

From timing the heaviest cells: E1 about 3 h, E2 about 10 h (N = 24 dominates), E3 and E4 under 1 h each,
E7 and E8 a few hours each (family D runs are slower). In total roughly one day of compute, below the
24 h threshold at which the plan allows dropping E1 axis values.

## What blocks the tag

1. H7 (b) and (c) thresholds are `[pilot]` values from the dev split and need the LLM run.
2. H1a as written compares against `B_real` but `HR_av` is normalised by `buy_lb`; this cannot pass when
   `B_real` is much larger than `buy_lb`. Options for the owner of the spec: keep the hypothesis and report
   the failure; or redefine the H1 comparator before the tag (a change to spec section 8, which the plan
   forbids after the tag). It is a decision, not a bug fix.
3. Preregistration freezes spec section 8, the arms, `PROMPT_VERSION` and the model ids. Several arms
   changed behaviour during the build (see "Rulings"), so the freeze should be made deliberately.

## Rulings and deviations made while implementing (for review)

| # | Where | What | Why | Cost if wrong |
|---|---|---|---|---|
| 1 | Task 2 | With one robot it starts in the west room (`max(1, n // 2)` west starters). | The plan's rule gave an east start but said west. | Different starts for n = 1 only. |
| 2 | Task 8 | Planner rejects only true swaps, not any move out of a cell reserved next tick. | The literal rule left lower-priority robots no way to step aside and deadlocked head-on traffic. | None found; tests cover both cases. |
| 3 | Tasks 9, 10, 15 | Traffic layer additions: reservations for stationary and waiting neighbours, a deterministic random side-step, an evade mode for dead-end standoffs, a verify-fallback through merely reported cells, a failed-plan cache. | Without them 12 robots stalled in 50% of seeds on `single_pit` and in several incident runs. The plan forbids hiding stalls by changing geometry or arbitration, and neither was changed. | The traffic layer is now more elaborate than the spec's "local intents plus priority"; every arm shares it. |
| 4 | Task 10 | `ClaimSet.release` tombstones. | A max-merge expiry cannot be shortened; a stale copy would resurrect a released claim. | Slightly larger STATE messages. |
| 5 | Task 11 | Triggers require class `robot_clearable`, not just `confirmed` (spec 4.4). | The plan's shorthand allowed unknown-class cells, which produced wrong-class hauls with `intake = none`. | With `intake = none` no family D edit ever fires; E7 reports `none` as pure detours. |
| 6 | Task 12 | `RunResult` carries its scenario. | Post-hoc metrics need the grid. | Memory only. |
| 7 | Task 14 Step 5 | Prompt pilot skipped. | No endpoint, no cost approval. | `PROMPT_VERSION` not yet tuned on dev. |
| 8 | Task 18 | No tag, no full runs. | See above. | Time. |
