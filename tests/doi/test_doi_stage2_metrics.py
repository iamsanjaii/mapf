"""Stage 2 metrics: carries, fills, slot conflicts, slot use and waiting near the slots."""
import dataclasses

from src.doi.config import SimConfig
from src.doi.metrics import near_slot_waits, summary_row
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

DOOR_CRATE = ["...T#....", "....#....", "....C....", "....#....", "....#...."]


def door_run():
    s = scenario_from_ascii(DOOR_CRATE, [(2, 1)], [[(2, 7)]])
    return run_episode(SimConfig(n_robots=1, tasks_per_robot=1, policy="myopic", kappa=50.0), scenario=s)


def test_summary_row_reports_the_carry():
    row = summary_row(door_run())
    assert (row["carries"], row["fills"], row["picks"], row["drops"]) == (1, 0, 1, 1)
    assert (row["carry_steps"], row["carry_cost"], row["slot_conflicts"]) == (1, 1.0, 0)
    assert row["slot_utilisation"] == 1.0                      # the one rack holds a crate
    assert row["near_slot_waits"] == 0                         # lifting and dropping are not waiting


def test_summary_row_for_a_run_with_no_slots_has_zero_utilisation():
    s = scenario_from_ascii(["...L...", "......."], [(1, 1)], [[(1, 5)]])
    r = run_episode(SimConfig(n_robots=1, tasks_per_robot=1, policy="never"), scenario=s)
    row = summary_row(r)
    assert row["slot_utilisation"] == 0.0 and row["carries"] == 0


def test_near_slot_waits_counts_standing_still_close_to_a_slot_but_not_lifts():
    r = door_run()
    still = [(1, 2), (1, 2), (1, 2), (2, 2)]                   # (1,2) is two cells from the rack at (0,3)
    fake = dataclasses.replace(r, trajectory={0: still}, lift_ticks=[])
    assert near_slot_waits(fake) == 2
    far = dataclasses.replace(r, trajectory={0: [(4, 8), (4, 8), (4, 8)]}, lift_ticks=[])
    assert near_slot_waits(far) == 0
    lifted = dataclasses.replace(r, trajectory={0: still}, lift_ticks=[(0, 0)])
    assert near_slot_waits(lifted) == 1
