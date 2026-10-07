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

## E1 quick pilot (2026-10-03)

Command: `venv/bin/python experiments/doi_e1_ratio.py --quick`

```
part  arm                    view     median ratio    max ratio  bound holds
core  predicted              full            1.450        4.950         1.00
core  predicted              own             1.000       10.000         1.00
core  randomized             full            1.557        1.563         1.00
core  randomized             own             1.175        3.375         1.00
core  threshold              full            1.450        2.950         1.00
core  threshold              own             1.000        8.800         1.00
grid  never                  full            1.299        1.714            -
grid  never                  own             1.299        1.714            -
grid  predicted:inverted     full            1.046        1.400            -
grid  predicted:inverted     own             1.046        1.400            -
grid  predicted:oracle       full            1.000        1.011            -
grid  predicted:oracle       own             1.000        1.011            -
grid  randomized             full            1.000        1.022            -
grid  randomized             own             1.000        1.022            -
grid  threshold              full            1.011        1.100            -
grid  threshold              own             1.011        1.100            -
```

Pilot only (quick mode); not a result. The full run has not been made.


## E9 headroom pilot (2026-10-07)

Commands:

```bash
venv/bin/python experiments/doi_e9_headroom.py --seeds 100 --jobs 4
venv/bin/python experiments/doi_e9_headroom.py --resummarise experiments/results/doi/e9_headroom
```

The first command made the 8,000 runs (16 of them stalled). The second recomputed the summaries with stalled
points left out, as the owner decided after the stall diagnosis; its output follows. `runs.csv` keeps every run.

```
E9 headroom pilot: shift_notice, 8 robots, 20 tasks, lam 0.5, seeds 200..299; no model was called
stalled runs are left out of every summary below, with the other arms of the same seed, mode and cost setting: dropped 16 of 1600 points

mean avoidable cost (J minus the free arm's J):
arm                   rof  numeric  keyword  oracle  inverted
kappa fee  mode                                              
4.0   1.0  false    104.6     75.2     75.2    72.1     142.2
           missing  146.6    129.9    129.9   111.6     178.1
           quiet    104.6     75.2     75.2    72.1     142.2
           true     146.6    129.9    129.9   111.6     178.1
      20.0 false    252.3    216.4    216.4   203.7     282.2
           missing  279.0    283.2    283.2   253.2     338.5
           quiet    252.3    216.4    216.4   203.7     282.2
           true     279.0    283.2    283.2   253.2     338.5
8.0   1.0  false    243.2    227.1    227.1   228.5     256.0
           missing  234.2    233.5    233.5   226.6     255.0
           quiet    243.2    227.1    227.1   228.5     256.0
           true     234.2    233.5    233.5   226.6     255.0
      40.0 false    339.3    324.9    324.9   325.5     376.7
           missing  400.0    403.7    403.7   377.2     431.2
           quiet    339.3    324.9    324.9   325.5     376.7
           true     400.0    403.7    403.7   377.2     431.2

mean forecast accuracy against the truth label:
arm                 keyword  numeric
kappa fee  mode                     
4.0   1.0  false       0.61     0.61
           missing     0.54     0.54
           quiet       0.61     0.61
           true        0.54     0.54
      20.0 false       0.61     0.61
           missing     0.46     0.46
           quiet       0.61     0.61
           true        0.46     0.46
8.0   1.0  false       0.67     0.67
           missing     0.62     0.62
           quiet       0.67     0.67
           true        0.62     0.62
      40.0 false       0.67     0.67
           missing     0.25     0.25
           quiet       0.67     0.67
           true        0.25     0.25

paired differences in avoidable cost (median, 95% interval of the median, share of seeds above 0):
  kappa 4 fee 1 false    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.25  n 100
  kappa 4 fee 1 false    inverted - oracle    median    63.5 [   44.5,    82.0]  above 0: 0.78  n 100
  kappa 4 fee 1 false    rof - oracle         median    23.0 [    9.0,    36.0]  above 0: 0.71  n 100
  kappa 4 fee 1 false    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 4 fee 1 false    rof - numeric        median    17.0 [    8.0,    30.0]  above 0: 0.65  n 100
  kappa 4 fee 1 missing  numeric - oracle     median     9.5 [    1.0,    18.0]  above 0: 0.61  n 100
  kappa 4 fee 1 missing  inverted - oracle    median    56.5 [   43.0,    75.0]  above 0: 0.79  n 100
  kappa 4 fee 1 missing  rof - oracle         median    31.5 [   22.0,    40.0]  above 0: 0.77  n 100
  kappa 4 fee 1 missing  numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 4 fee 1 missing  rof - numeric        median    12.0 [    4.0,    20.5]  above 0: 0.64  n 100
  kappa 4 fee 1 quiet    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.25  n 100
  kappa 4 fee 1 quiet    inverted - oracle    median    63.5 [   44.5,    82.0]  above 0: 0.78  n 100
  kappa 4 fee 1 quiet    rof - oracle         median    23.0 [    9.0,    36.0]  above 0: 0.71  n 100
  kappa 4 fee 1 quiet    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 4 fee 1 quiet    rof - numeric        median    17.0 [    8.0,    30.0]  above 0: 0.65  n 100
  kappa 4 fee 1 true     numeric - oracle     median     9.5 [    1.0,    18.0]  above 0: 0.61  n 100
  kappa 4 fee 1 true     inverted - oracle    median    56.5 [   43.0,    75.0]  above 0: 0.79  n 100
  kappa 4 fee 1 true     rof - oracle         median    31.5 [   22.0,    40.0]  above 0: 0.77  n 100
  kappa 4 fee 1 true     numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 4 fee 1 true     rof - numeric        median    12.0 [    4.0,    20.5]  above 0: 0.64  n 100
  kappa 4 fee 20 false    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.23  n 99
  kappa 4 fee 20 false    inverted - oracle    median    82.0 [   63.0,   102.0]  above 0: 0.79  n 99
  kappa 4 fee 20 false    rof - oracle         median    33.0 [   10.0,    49.0]  above 0: 0.68  n 99
  kappa 4 fee 20 false    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 99
  kappa 4 fee 20 false    rof - numeric        median    21.0 [    8.0,    40.0]  above 0: 0.62  n 99
  kappa 4 fee 20 missing  numeric - oracle     median    30.0 [   20.0,    46.0]  above 0: 0.67  n 99
  kappa 4 fee 20 missing  inverted - oracle    median    88.0 [   73.0,   107.0]  above 0: 0.85  n 99
  kappa 4 fee 20 missing  rof - oracle         median    23.0 [    5.0,    39.0]  above 0: 0.64  n 99
  kappa 4 fee 20 missing  numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 99
  kappa 4 fee 20 missing  rof - numeric        median   -13.0 [  -36.0,     5.0]  above 0: 0.42  n 99
  kappa 4 fee 20 quiet    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.23  n 99
  kappa 4 fee 20 quiet    inverted - oracle    median    82.0 [   63.0,   102.0]  above 0: 0.79  n 99
  kappa 4 fee 20 quiet    rof - oracle         median    33.0 [   10.0,    49.0]  above 0: 0.68  n 99
  kappa 4 fee 20 quiet    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 99
  kappa 4 fee 20 quiet    rof - numeric        median    21.0 [    8.0,    40.0]  above 0: 0.62  n 99
  kappa 4 fee 20 true     numeric - oracle     median    30.0 [   20.0,    46.0]  above 0: 0.67  n 99
  kappa 4 fee 20 true     inverted - oracle    median    88.0 [   73.0,   107.0]  above 0: 0.85  n 99
  kappa 4 fee 20 true     rof - oracle         median    23.0 [    5.0,    39.0]  above 0: 0.64  n 99
  kappa 4 fee 20 true     numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 99
  kappa 4 fee 20 true     rof - numeric        median   -13.0 [  -36.0,     5.0]  above 0: 0.42  n 99
  kappa 8 fee 1 false    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.15  n 100
  kappa 8 fee 1 false    inverted - oracle    median    30.0 [    0.0,    51.0]  above 0: 0.60  n 100
  kappa 8 fee 1 false    rof - oracle         median     8.5 [   -7.5,    22.5]  above 0: 0.54  n 100
  kappa 8 fee 1 false    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 8 fee 1 false    rof - numeric        median     7.5 [   -8.0,    18.0]  above 0: 0.55  n 100
  kappa 8 fee 1 missing  numeric - oracle     median     0.0 [    0.0,     1.0]  above 0: 0.40  n 100
  kappa 8 fee 1 missing  inverted - oracle    median    23.5 [   13.5,    51.5]  above 0: 0.64  n 100
  kappa 8 fee 1 missing  rof - oracle         median     5.5 [  -14.5,    30.5]  above 0: 0.52  n 100
  kappa 8 fee 1 missing  numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 8 fee 1 missing  rof - numeric        median     0.5 [  -16.0,    13.0]  above 0: 0.50  n 100
  kappa 8 fee 1 quiet    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.15  n 100
  kappa 8 fee 1 quiet    inverted - oracle    median    30.0 [    0.0,    51.0]  above 0: 0.60  n 100
  kappa 8 fee 1 quiet    rof - oracle         median     8.5 [   -7.5,    22.5]  above 0: 0.54  n 100
  kappa 8 fee 1 quiet    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 8 fee 1 quiet    rof - numeric        median     7.5 [   -8.0,    18.0]  above 0: 0.55  n 100
  kappa 8 fee 1 true     numeric - oracle     median     0.0 [    0.0,     1.0]  above 0: 0.40  n 100
  kappa 8 fee 1 true     inverted - oracle    median    23.5 [   13.5,    51.5]  above 0: 0.64  n 100
  kappa 8 fee 1 true     rof - oracle         median     5.5 [  -14.5,    30.5]  above 0: 0.52  n 100
  kappa 8 fee 1 true     numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 100
  kappa 8 fee 1 true     rof - numeric        median     0.5 [  -16.0,    13.0]  above 0: 0.50  n 100
  kappa 8 fee 40 false    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 false    inverted - oracle    median    39.0 [   20.9,    67.0]  above 0: 0.66  n 97
  kappa 8 fee 40 false    rof - oracle         median    36.0 [  -13.0,    58.0]  above 0: 0.57  n 97
  kappa 8 fee 40 false    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 false    rof - numeric        median    36.0 [  -13.0,    58.0]  above 0: 0.57  n 97
  kappa 8 fee 40 missing  numeric - oracle     median    32.0 [   18.0,    53.0]  above 0: 0.70  n 97
  kappa 8 fee 40 missing  inverted - oracle    median    61.0 [   40.0,    78.0]  above 0: 0.78  n 97
  kappa 8 fee 40 missing  rof - oracle         median    32.0 [   13.0,    53.0]  above 0: 0.63  n 97
  kappa 8 fee 40 missing  numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 missing  rof - numeric        median    -4.0 [  -23.0,    12.0]  above 0: 0.49  n 97
  kappa 8 fee 40 quiet    numeric - oracle     median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 quiet    inverted - oracle    median    39.0 [   20.9,    67.0]  above 0: 0.66  n 97
  kappa 8 fee 40 quiet    rof - oracle         median    36.0 [  -13.0,    58.0]  above 0: 0.57  n 97
  kappa 8 fee 40 quiet    numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 quiet    rof - numeric        median    36.0 [  -13.0,    58.0]  above 0: 0.57  n 97
  kappa 8 fee 40 true     numeric - oracle     median    32.0 [   18.0,    53.0]  above 0: 0.70  n 97
  kappa 8 fee 40 true     inverted - oracle    median    61.0 [   40.0,    78.0]  above 0: 0.78  n 97
  kappa 8 fee 40 true     rof - oracle         median    32.0 [   13.0,    53.0]  above 0: 0.63  n 97
  kappa 8 fee 40 true     numeric - keyword    median     0.0 [    0.0,     0.0]  above 0: 0.00  n 97
  kappa 8 fee 40 true     rof - numeric        median    -4.0 [  -23.0,    12.0]  above 0: 0.49  n 97

share of the numeric arm's avoidable cost a perfect forecast removes:
 kappa  fee    mode  headroom_share
   4.0  1.0   false           0.041
   4.0  1.0 missing           0.141
   4.0  1.0   quiet           0.041
   4.0  1.0    true           0.141
   4.0 20.0   false           0.058
   4.0 20.0 missing           0.106
   4.0 20.0   quiet           0.058
   4.0 20.0    true           0.106
   8.0  1.0   false          -0.006
   8.0  1.0 missing           0.029
   8.0  1.0   quiet          -0.006
   8.0  1.0    true           0.029
   8.0 40.0   false          -0.002
   8.0 40.0 missing           0.065
   8.0 40.0   quiet          -0.002
   8.0 40.0    true           0.065

stalled runs (all of them, kept in runs.csv): 16 of 8000
wrote experiments/results/doi/e9_headroom
```

No model was called. This measures the best (`oracle`) and worst (`inverted`) any forecaster can do on
`shift_notice`, to decide whether the model parts of the forecast guard are worth building.
