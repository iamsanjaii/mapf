"""Stage 2 scenarios: warehouse_racks, dump_central, site_pits and mixed."""
import pytest

from src.doi.config import SimConfig
from src.doi.runner import run_arms, run_episode
from src.doi.scenarios import build_scenario

STAGE2 = ("warehouse_racks", "dump_central", "site_pits", "mixed")


def cfg(name, **kw):
    base = dict(scenario=name, n_robots=4, tasks_per_robot=3, seed=1, debug_checks=True)
    return SimConfig(**{**base, **kw})


@pytest.mark.parametrize("name", STAGE2)
def test_nobody_starts_or_aims_at_an_obstacle_or_a_rack(name):
    s = build_scenario(cfg(name))
    fixed = set(s.obstacles) | {c for c, k in s.slots.items() if k == "rack"}
    assert not (set(s.starts) & fixed)
    assert not ({g for goals in s.tasks for g in goals} & fixed)


def test_warehouse_racks_has_crates_and_racks_only():
    s = build_scenario(cfg("warehouse_racks"))
    assert set(s.obstacles.values()) == {"crate"} and set(s.slots.values()) == {"rack"}
    assert all(not s.grid.is_passable(*c) for c in s.slots)


def test_dump_central_is_one_region_of_many_walkable_slots():
    s = build_scenario(cfg("dump_central"))
    assert set(s.slots.values()) == {"dump"} and len(s.slots) >= 6
    assert all(s.grid.is_passable(*c) for c in s.slots)
    rows = [c[0] for c in s.slots]
    cols = [c[1] for c in s.slots]
    assert len(s.slots) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)       # a solid block
    assert "shelf_unit" in s.obstacles.values()                                          # only the dump takes it


def test_site_pits_has_pits_and_enough_debris():
    s = build_scenario(cfg("site_pits"))
    kinds = list(s.obstacles.values())
    assert kinds.count("pit") >= 2 and kinds.count("debris") >= kinds.count("pit")
    assert s.slots == {}


def test_mixed_has_every_kind_of_target():
    s = build_scenario(cfg("mixed"))
    assert set(s.slots.values()) == {"rack", "dump"} and "pit" in s.obstacles.values()


@pytest.mark.parametrize("name", STAGE2)
def test_unknown_parameters_are_rejected(name):
    with pytest.raises(ValueError):
        build_scenario(cfg(name, scenario_params={"nonsense": 1}))


@pytest.mark.parametrize("name", STAGE2)
def test_scenarios_are_deterministic_for_a_seed(name):
    a, b = build_scenario(cfg(name)), build_scenario(cfg(name))
    assert (a.starts, a.tasks, a.obstacles, a.slots) == (b.starts, b.tasks, b.obstacles, b.slots)


@pytest.mark.parametrize("name", STAGE2)
def test_every_arm_finishes_and_the_books_balance(name):
    results = run_arms(cfg(name, r_comm=float("inf")), ["never", "myopic", "rof", "central"])
    for arm, r in results.items():
        assert not r.stalled and r.unfinished_tasks == 0, (name, arm)
        assert abs(sum(r.tick_cost) - r.J) < 1e-6, (name, arm)
    assert results["never"].picks == 0


def test_the_ledger_arms_carry_where_pushing_cannot_help():
    r = run_arms(cfg("site_pits", r_comm=float("inf"), tasks_per_robot=8), ["never", "central"])
    assert r["central"].fills >= 1
    assert r["central"].J_censored < r["never"].J_censored
