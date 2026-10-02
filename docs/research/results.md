# Results status

Status as of 2026-10-02: **no full experiment has been run on the push model, and the preregistration tag
`prereg-v1` was never created.**

The earlier pilot numbers in this file described the pit-and-kit model (fetch a kit from a depot, fill a pit).
That model was replaced by removable obstacles that robots push, so those pilots no longer describe the
simulator and have been removed. They are in git history (commit `29fdf45`) for reference.

## What exists now

* `experiments/doi_e2_information.py`: price of information (ledger range, loss, delay, fleet size).
* `experiments/doi_e7_intake.py`: cost of none/oracle/llm intake and false reports in the incident scenarios.
* `experiments/doi_common.py`: the shared grid runner.

Both scripts have a `--quick` pilot mode (3 seeds, smallest axes). Neither has been run in full.

## Removed with the pit model (to be re-expressed for pushing)

* E1 validity of the ski-rental bound on a single resource
* E3 complements and substitutes
* E4 claims, leases and wasted hauls
* E5 shifting demand and the windowed and extrapolating arms
* E6 scaling on benchmark warehouse maps
* E8 the human approval gate

## What a hand-check of the new model gives

Toy (`run_doi.py --demo toy`): never 40, myopic 40, eager 23, rof 31, central 31, free 8, hindsight 17.
Robustness check: 1,120 runs (7 scenarios, 20 fresh seeds, 8 arms) completed with no stalled or unfinished
run before the last evidence change; a smaller sweep after it is recorded in the commit that follows.
These are checks that the simulator behaves, not results about the method.
