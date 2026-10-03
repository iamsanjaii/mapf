"""Stage 2 physics: pick up, carry, drop into a rack or dump slot, fill a pit.

Tests teleport robots by writing `w.pos`, as the existing world tests do, to set up a drop without a walk."""
from src.doi.config import SimConfig
from src.doi.scenarios import scenario_from_ascii
from src.doi.world import Drop, Move, Pick, Push, World


def world(rows, starts, **kw):
    s = scenario_from_ascii(rows, starts, [[(0, 0)] for _ in starts])
    return World(s, SimConfig(n_robots=len(starts), debug_checks=True, **kw))


def test_pick_removes_the_obstacle_and_loads_the_robot():
    w = world(["..C.."], [(0, 1)])
    r = w.apply_actions(0, {0: Pick((0, 2))})[0]
    assert r.ok and w.load[0] == "crate" and w.obstacles == {}
    assert w.grid.get(0, 2).name == "FREE" and w.pos[0] == (0, 1)
    assert w.counters[0]["picks"] == 1 and w.tick_cost == [1.0]          # the pick fee, nothing else


def test_pick_rejections():
    w = world(["..C.W", ".P..."], [(0, 1), (1, 3)])
    assert w.apply_actions(0, {0: Pick((0, 0))})[0].reason == "no_obstacle"
    assert w.apply_actions(1, {1: Pick((0, 4))})[1].reason == "not_adjacent"
    w2 = world(["..W", "..."], [(0, 1)])
    assert w2.apply_actions(0, {0: Pick((0, 2))})[0].reason == "not_carriable"      # a spill cannot be lifted
    w3 = world(["P.", ".."], [(1, 0)])
    assert w3.apply_actions(0, {0: Pick((0, 0))})[0].reason == "not_carriable"      # nor can a pit
    w4 = world(["CC."], [(0, 2)])
    assert w4.apply_actions(0, {0: Pick((0, 1))})[0].ok
    assert w4.apply_actions(1, {0: Pick((0, 1))})[0].reason == "loaded"
    assert w4.counters[0]["picks"] == 1


def test_a_loaded_robot_cannot_pick_or_push():
    w = world(["CC.."], [(0, 2)])
    w.apply_actions(0, {0: Pick((0, 1))})
    assert w.apply_actions(1, {0: Pick((0, 0))})[0].reason == "loaded"
    w.pos[0] = (0, 1)
    assert w.apply_actions(2, {0: Push((0, 0))})[0].reason == "loaded"


def test_loaded_step_costs_kappa_c_times_weight():
    w = world([".S.....", "......."], [(0, 0)], kappa_c=2.0)           # a shelf unit weighs 2.0
    w.apply_actions(0, {0: Pick((0, 1))})
    r = w.apply_actions(1, {0: Move((1, 0))})[0]
    assert r.ok and w.tick_cost == [1.0, 4.0]                          # the pick fee, then 2.0 * 2.0
    assert w.counters[0]["carry_steps"] == 1 and w.carry_cost == 4.0


def test_an_empty_robot_still_pays_one_per_step():
    w = world(["...."], [(0, 0)])
    w.apply_actions(0, {0: Move((0, 1))})
    assert w.tick_cost == [1.0] and w.carry_cost == 0.0


def test_drop_into_a_rack_slot():
    w = world(["T.", "C."], [(0, 1)])
    w.pos[0] = (1, 1)
    w.apply_actions(0, {0: Pick((1, 0))})
    w.apply_actions(1, {0: Move((0, 1))})
    r = w.apply_actions(2, {0: Drop((0, 0))})[0]
    assert r.ok and w.load[0] is None and w.slot_items == {(0, 0): "crate"}
    assert w.counters[0]["drops"] == 1 and w.tick_cost[-1] == 1.0       # the drop fee
    assert w.obstacles == {}                                            # a stored crate is not an obstacle


def test_a_full_slot_rejects_and_counts_a_conflict():
    w = world(["TCC", "..."], [(1, 1), (1, 2)])
    w.apply_actions(0, {0: Pick((0, 1)), 1: Pick((0, 2))})
    w.pos[0] = (0, 1)
    assert w.apply_actions(1, {0: Drop((0, 0))})[0].ok
    w.pos[1] = (1, 0)
    r = w.apply_actions(2, {1: Drop((0, 0))})[1]
    assert not r.ok and r.reason == "full"
    assert w.counters[1]["slot_conflicts"] == 1 and w.load[1] == "crate"          # still carrying it


def test_two_drops_on_one_slot_in_one_tick_fill_it_once():
    w = world(["TCC", "..."], [(1, 1), (1, 2)])
    w.apply_actions(0, {0: Pick((0, 1)), 1: Pick((0, 2))})
    w.pos[0], w.pos[1] = (0, 1), (1, 0)
    r = w.apply_actions(1, {0: Drop((0, 0)), 1: Drop((0, 0))})
    assert [r[0].ok, r[1].ok] == [True, False] and r[1].reason == "full"
    assert w.slot_items == {(0, 0): "crate"}
    assert w.counters[0]["slot_conflicts"] + w.counters[1]["slot_conflicts"] == 1


def test_a_kind_goes_only_to_slots_that_accept_it():
    w = world([".S.", "T.."], [(0, 0)])
    w.apply_actions(0, {0: Pick((0, 1))})
    w.pos[0] = (1, 1)
    r = w.apply_actions(1, {0: Drop((1, 0))})[0]
    assert not r.ok and r.reason == "wrong_slot" and w.slot_items == {}


def test_a_dump_cell_is_walkable_and_its_item_does_not_block():
    w = world(["DC", ".."], [(1, 1)])
    w.apply_actions(0, {0: Pick((0, 1))})
    w.pos[0] = (1, 0)
    assert w.apply_actions(1, {0: Drop((0, 0))})[0].ok
    assert w.apply_actions(2, {0: Move((0, 0))})[0].ok              # walk onto the filled dump cell


def test_debris_fills_a_pit_and_the_pit_stays_filled():
    w = world(["P.R."], [(0, 1)])
    w.apply_actions(0, {0: Pick((0, 2))})
    r = w.apply_actions(1, {0: Drop((0, 0))})[0]
    assert r.ok and (0, 0) not in w.obstacles and w.grid.get(0, 0).name == "FREE"
    assert w.fills == 1 and w.load[0] is None
    assert w.apply_actions(2, {0: Move((0, 0))})[0].ok            # the pit is ordinary floor now
    w.apply_actions(3, {0: Move((0, 1))})
    assert (0, 0) not in w.obstacles                                # and does not come back


def test_only_debris_goes_into_a_pit():
    w = world(["PC.."], [(0, 2)])
    w.apply_actions(0, {0: Pick((0, 1))})
    w.pos[0] = (0, 1)
    r = w.apply_actions(1, {0: Drop((0, 0))})[0]
    assert not r.ok and r.reason == "wrong_slot" and (0, 0) in w.obstacles


def test_observation_reports_full_slots_in_range():
    w = world(["TC.", "..."], [(1, 1)])
    w.apply_actions(0, {0: Pick((0, 1))})
    w.pos[0] = (0, 1)
    w.apply_actions(1, {0: Drop((0, 0))})
    assert w.observe(0, 2).slots_full == frozenset({(0, 0)})
    w.pos[0] = (1, 2)
    assert w.observe(0, 1).slots_full == frozenset()
