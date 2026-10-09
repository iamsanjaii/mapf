"""Stage 2 agent behaviour: where a loaded robot goes, and what happens when slots fill up."""
from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.evidence import EvidenceEngine
from src.doi.policies import Shared, make_policy
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii
from src.doi.world import Drop, Pick, World


def agent_for(rows, start, goal, load, **kw):
    s = scenario_from_ascii(rows, [start], [[goal]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1, policy="myopic", **kw)
    engine = EvidenceEngine(s.grid, cfg.unreachable_cost_for(len(rows), len(rows[0])))
    a = RobotAgent(0, s, cfg, make_policy(cfg), Shared(engine=engine))
    a.load = load
    return a


def test_loaded_robot_picks_the_cheapest_free_slot():
    a = agent_for(["T.....T", "......."], start=(1, 1), goal=(1, 2), load="crate")
    # rack (0,0) from (0,1): one loaded step (1.0) + drop 1 + two steps on = 4; the far rack costs 10
    assert a.unload_target(0) == ("carry", (0, 0), (0, 1))


def test_a_full_slot_is_skipped_and_none_is_left_when_all_are_full():
    a = agent_for(["T.....T", "......."], start=(1, 1), goal=(1, 2), load="crate")
    a.belief.slots_full.add((0, 0))
    assert a.unload_target(0)[1] == (0, 6)
    a.belief.slots_full.add((0, 6))
    assert a.unload_target(0) is None


def test_only_the_dump_takes_a_shelf_unit():
    a = agent_for(["T.....D", "......."], start=(1, 1), goal=(1, 5), load="shelf_unit")
    assert a.unload_target(0)[:2] == ("carry", (0, 6))


def test_debris_goes_to_a_believed_pit_when_no_dump_is_free():
    a = agent_for(["T...P..", "......."], start=(1, 1), goal=(1, 5), load="debris")
    assert a.unload_target(0) == ("fill", (0, 4), (1, 4))                     # a rack does not take debris


def test_an_unloaded_robot_has_no_target():
    a = agent_for(["T.....T", "......."], start=(1, 1), goal=(1, 5), load=None)
    assert a.unload_target(0) is None


def test_a_dump_region_of_two_slots_takes_exactly_two():
    s = scenario_from_ascii(["DD.", "CCC"], [(0, 0)], [[(0, 2)]])
    w = World(s, SimConfig(n_robots=1, debug_checks=True))
    w.apply_actions(0, {0: Pick((1, 0))})
    assert w.apply_actions(1, {0: Drop((0, 1))})[0].ok            # first slot filled
    w.pos[0] = (0, 1)                                              # test-only: step onto the filled dump cell
    w.apply_actions(2, {0: Pick((1, 1))})
    assert w.apply_actions(3, {0: Drop((0, 0))})[0].ok            # second slot filled
    w.pos[0] = (0, 2)
    w.apply_actions(4, {0: Pick((1, 2))})
    w.pos[0] = (0, 0)
    r = w.apply_actions(5, {0: Drop((0, 1))})[0]
    assert not r.ok and r.reason == "full" and w.load[0] == "crate"
    assert len(w.slot_items) == 2


def test_two_robots_one_rack_only_one_crate_is_parked():
    rows = ["T.#..",
            "..C..",
            "..#..",
            "..C..",
            "..#.."]
    s = scenario_from_ascii(rows, [(1, 0), (3, 0)], [[(1, 4)], [(3, 4)]])
    r = run_episode(SimConfig(n_robots=2, tasks_per_robot=1, policy="myopic", kappa=50.0, debug_checks=True),
                    scenario=s)
    assert r.unfinished_tasks == 0 and not r.stalled
    assert (r.picks, r.drops, r.carries) == (2, 1, 1)             # the second robot cannot unload and carries on
    assert abs(sum(r.tick_cost) - r.J) < 1e-6
