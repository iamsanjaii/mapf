"""Stage 2 end to end: a robot carries a crate to a rack, or fills a pit with debris, and the books balance."""
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

# Left room (cols 0-3), a wall in col 4 with one door at row 2, right room (cols 5-8).
DOOR_CRATE = ["...T#....",
              "....#....",
              "....C....",                 # (2,4) is the door, holding a crate
              "....#....",
              "....#...."]
PIT_ROOM = ["....#....",
            "....#....",
            "....P....",
            "..R.#....",
            "....#...."]


def cfg(**kw):
    base = dict(n_robots=1, tasks_per_robot=1, policy="myopic", kappa=50.0, debug_checks=True)
    return SimConfig(**{**base, **kw})


def run(rows, **kw):
    s = scenario_from_ascii(rows, [(2, 1)], [[(2, 7)]])
    return run_episode(cfg(**kw), scenario=s)


def test_robot_carries_a_crate_from_the_door_to_a_rack():
    r = run(DOOR_CRATE)
    # 2 steps to the crate, pick 1, one loaded step (2.0 * 0.5), drop 1, 5 steps on to the goal
    assert r.J == 10.0 and not r.stalled and r.unfinished_tasks == 0
    assert (r.picks, r.drops, r.carries, r.fills, r.removals) == (1, 1, 1, 0, 0)
    assert r.carry_steps == 1 and r.carry_cost == 1.0
    assert abs(sum(r.tick_cost) - r.J) < 1e-9
    assert r.carry_log[0]["mode"] == "carry" and r.carry_log[0]["target"] == (0, 3)


def test_robot_fills_a_pit_with_debris_and_walks_through():
    r = run(PIT_ROOM)
    # 1 step to the debris, pick 1, one loaded step (2.0 * 0.5), drop 1, 4 steps on to the goal
    assert r.J == 8.0 and not r.stalled and r.unfinished_tasks == 0
    assert (r.picks, r.drops, r.carries, r.fills) == (1, 1, 0, 1)
    assert abs(sum(r.tick_cost) - r.J) < 1e-9
    assert r.carry_log[0]["mode"] == "fill" and r.carry_log[0]["target"] == (2, 4)


def test_the_never_arm_does_not_carry():
    r = run(DOOR_CRATE, policy="never")
    assert r.picks == 0 and r.drops == 0 and r.stalled
