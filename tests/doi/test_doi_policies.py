from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]


def run(policy, tasks, **kw):
    s = scenario_from_ascii(ROWS, [(1, 2)], [tasks], depot_stock=2)
    cfg = SimConfig(policy=policy, n_robots=1, tasks_per_robot=len(tasks),
                    claim=False, debug_checks=True, **kw)
    return run_episode(cfg, scenario=s)


FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


def test_rof_fills_exactly_when_rent_reaches_buy_cost():
    r = run("rof", FOUR)
    assert r.fills == 1 and r.J == 29.0
    assert r.triggers[0]["known"] == 16.0 and r.triggers[0]["buy"] == 9.0
    assert r.triggers[0]["tick"] > 0


def test_rof_waits_when_single_task_rent_is_below_buy():
    r = run("rof", [(1, 4)])
    assert r.fills == 0 and r.J == 10.0


def test_myopic_never_fills_when_one_task_does_not_pay():
    r = run("myopic", FOUR)
    assert r.fills == 0 and r.J == 40.0


def test_eager_fills_at_first_positive_rent():
    r = run("eager", FOUR)
    assert r.fills == 1 and r.triggers[0]["tick"] == 0


def test_hindsight_prefills_and_charges_buy():
    r = run("hindsight", FOUR)
    assert r.fills == 0 and r.hindsight_buy == 9.0
    assert r.J == 8.0 and r.J + r.hindsight_buy == 17.0


def test_central_matches_rof_with_one_robot():
    a, b = run("rof", FOUR), run("central", FOUR)
    assert abs(a.J - b.J) <= 2.0 and b.fills == 1


def test_rof_equals_neverfill_when_unprofitable():
    a, b = run("rof", FOUR, fee=100.0), run("never", FOUR, fee=100.0)
    assert a.fills == 0 and a.J == b.J == 40.0
    assert len(a.triggers) == 0


def test_rof_pit_and_bundle_agree_on_single_pit():
    a, b = run("rof", FOUR), run("rof_pit", FOUR)
    assert a.J == b.J


def test_series_per_pit_ledger_never_fires_bundle_does():
    base = dict(scenario="series_pits", n_robots=4, tasks_per_robot=12, seed=1, claim=False)
    bundle = run_episode(SimConfig(policy="rof", **base))
    perpit = run_episode(SimConfig(policy="rof_pit", **base))
    assert bundle.fills == 2 and perpit.fills == 0
    assert bundle.unfinished_tasks == 0 and perpit.unfinished_tasks == 0
    assert bundle.J < perpit.J


def test_full_loss_equals_no_comm():
    base = dict(scenario="single_pit", n_robots=6, tasks_per_robot=10, seed=2, claim=False)
    lossy = run_episode(SimConfig(policy="rof", r_comm=float("inf"), loss=1.0, **base))
    quiet = run_episode(SimConfig(policy="rof", r_comm=0.0, **base))
    assert lossy.J == quiet.J and lossy.fills == quiet.fills


def test_free_open_is_travel_only():
    r = run("free", FOUR)
    assert r.J == 8.0 and r.fills == 0 and r.hindsight_buy == 0.0


def test_ledger_radius_does_not_change_traffic_messages():
    base = dict(scenario="single_pit", n_robots=6, tasks_per_robot=4, seed=1, claim=False, policy="never")
    a = run_episode(SimConfig(r_comm=0.0, **base))
    b = run_episode(SimConfig(r_comm=float("inf"), **base))
    assert a.traffic_messages == b.traffic_messages and a.J == b.J
