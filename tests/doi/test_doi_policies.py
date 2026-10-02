from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

# One robot, a one-cell gap with a shelf unit in it (push step = kappa 4 * weight 2.0 = 8), a long way round.
# Each crossing pays 8 rent against the detour; the cheapest plan pushes the shelf two cells east.
ROWS = ["...#...", "...S...", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def run(policy, tasks, rows=ROWS, **kw):
    s = scenario_from_ascii(rows, [(1, 2)], [tasks])
    cfg = SimConfig(policy=policy, n_robots=1, tasks_per_robot=len(tasks), debug_checks=True, **kw)
    return run_episode(cfg, scenario=s)


def test_rof_pushes_exactly_when_the_ledger_saving_reaches_the_price():
    r = run("rof", FOUR)
    assert r.removals == 1 and r.push_steps == 2 and r.J == 31.0
    t = r.triggers[0]
    assert t["known"] == 16.0 and t["buy"] == 15.0 and t["tick"] == 10 and t["steps"] == 2
    assert t["cells"] == ((1, 3),) and t["kind"] == "shelf_unit"


def test_rof_waits_when_a_single_task_does_not_pay():
    r = run("rof", [(1, 4)])
    assert r.removals == 0 and r.J == 10.0


def test_myopic_never_pushes_when_the_push_costs_more_than_the_detour():
    r = run("myopic", FOUR)
    assert r.removals == 0 and r.J == 40.0


def test_myopic_pushes_when_it_is_cheaper_for_me():
    r = run("myopic", FOUR, kappa=1.0)          # a push run now costs 2 * 1 * 2.0 + 1 = 5 against a detour of 10
    assert r.removals == 1 and r.triggers[0]["tick"] == 0


def test_eager_pushes_at_the_first_positive_saving():
    r = run("eager", FOUR)
    assert r.removals == 1 and r.triggers[0]["tick"] == 0 and r.J == 23.0


def test_forecast_arm_pushes_early_when_the_run_has_many_tasks_left():
    r = run("rof_f", FOUR)
    assert r.removals == 1 and r.triggers[0]["tick"] == 0 and r.J == 23.0
    assert r.triggers[0]["known"] == 32.0               # 8 on the task in hand, 8 on each of three more like it


def test_hindsight_removes_at_tick_zero_and_charges_the_lowest_price():
    r = run("hindsight", FOUR)
    assert r.removals == 0 and r.hindsight_buy == 9.0
    assert r.J == 8.0 and r.J + r.hindsight_buy == 17.0


def test_central_and_local_match_rof_with_one_robot():
    a = run("rof", FOUR)
    assert run("central", FOUR).J == a.J == run("rof_local", FOUR).J == 31.0


def test_rof_equals_never_when_pushing_is_not_worth_it():
    a, b = run("rof", FOUR, fee=100.0), run("never", FOUR, fee=100.0)
    assert a.removals == 0 and a.J == b.J == 40.0 and len(a.triggers) == 0


def test_an_obstacle_with_no_room_to_push_is_never_pushed():
    rows = ["..S#..", "......"]                    # wall behind the shelf, no side room, no approach cell
    r = run("eager", [(0, 5)], rows)
    assert r.removals == 0 and r.push_rejected == 0 and r.unfinished_tasks == 0


def test_full_loss_equals_no_comm():
    base = dict(scenario="single_block", n_robots=6, tasks_per_robot=10, seed=2)
    lossy = run_episode(SimConfig(policy="rof", r_comm=float("inf"), loss=1.0, **base))
    quiet = run_episode(SimConfig(policy="rof", r_comm=0.0, **base))
    assert lossy.J == quiet.J and lossy.removals == quiet.removals


def test_free_open_is_travel_only():
    r = run("free", FOUR)
    assert r.J == 8.0 and r.removals == 0 and r.hindsight_buy == 0.0


def test_ledger_radius_does_not_change_traffic_messages():
    base = dict(scenario="single_block", n_robots=6, tasks_per_robot=4, seed=1, policy="never")
    a = run_episode(SimConfig(r_comm=0.0, **base))
    b = run_episode(SimConfig(r_comm=float("inf"), **base))
    assert a.traffic_messages == b.traffic_messages and a.J == b.J


def test_job_cost_equals_the_per_tick_costs():
    for policy in ("never", "myopic", "eager", "rof", "central"):
        r = run_episode(SimConfig(policy=policy, scenario="multi_block_wall", n_robots=5, tasks_per_robot=6,
                                  seed=3, debug_checks=True))
        assert abs(r.J - sum(r.tick_cost)) < 1e-6
