from src.doi.config import SimConfig
from src.doi.scenarios import scenario_from_ascii
from src.doi.world import World, Move, Wait, Pickup, Drop, Return

ROWS = ["...#...", "...P..D", "...#...", "...#..."]


def world(starts, rows=ROWS, **kw):
    s = scenario_from_ascii(rows, starts, [[(0, 0)] for _ in starts], depot_stock=2)
    return World(s, SimConfig(n_robots=len(starts), debug_checks=True, **kw))


def test_vertex_lowest_id_wins_when_equal():
    w = world([(0, 0), (0, 2)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})
    assert r[0].ok and not r[1].ok and r[1].reason == "vertex"
    assert w.pos[0] == (0, 1) and w.pos[1] == (0, 2)
    assert r[1].blocked_ticks == 1


def test_aged_priority_overrides_id():
    w = world([(0, 0), (0, 2)], ["..."])
    w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})   # robot 1 loses, blocked_ticks becomes 1
    w.pos[0] = (0, 0)                                        # test-only teleport back
    w.pos[1] = (0, 2)
    r = w.apply_actions(1, {0: Move((0, 1)), 1: Move((0, 1))})
    assert r[1].ok and not r[0].ok                           # robot 1 now outranks robot 0


def test_swap_blocked_both():
    w = world([(0, 0), (0, 1)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 0))})
    assert not r[0].ok and not r[1].ok and r[0].reason == "swap"
    assert w.pos == {0: (0, 0), 1: (0, 1)}


def test_following_allowed_train_of_three():
    w = world([(0, 0), (0, 1), (0, 2)], ["....."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 2)), 2: Move((0, 3))})
    assert all(x.ok for x in r.values())
    assert w.pos == {0: (0, 1), 1: (0, 2), 2: (0, 3)}


def test_occupied_by_waiting_robot():
    w = world([(0, 0), (0, 1)], ["..."])
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Wait()})
    assert not r[0].ok and r[0].reason == "occupied"


def test_rotation_cycle_blocked():
    rows = ["..", ".."]
    w = world([(0, 0), (0, 1), (1, 1), (1, 0)], rows)
    r = w.apply_actions(0, {0: Move((0, 1)), 1: Move((1, 1)), 2: Move((1, 0)), 3: Move((0, 0))})
    assert not any(x.ok for x in r.values()) and r[0].reason == "cycle"


def test_wall_and_pit_are_illegal():
    w = world([(1, 2)])
    assert w.apply_actions(0, {0: Move((1, 3))})[0].reason == "wall"


def test_pickup_drop_cycle():
    w = world([(1, 6)])
    assert w.apply_actions(0, {0: Pickup()})[0].ok and w.stock[(1, 6)] == 1
    for t, c in enumerate([(1, 5), (1, 4)], start=1):
        assert w.apply_actions(t, {0: Move(c)})[0].ok
    assert w.counters[0]["carried_steps"] == 2
    r = w.apply_actions(3, {0: Drop((1, 3))})[0]
    assert r.ok and (1, 3) in w.filled and w.grid.get(1, 3).name == "FREE" and w.fills == 1
    assert w.apply_actions(4, {0: Move((1, 3))})[0].ok


def test_double_drop_one_fill():
    w = world([(1, 6), (1, 5)])
    w.pos[0], w.pos[1] = (1, 4), (1, 2)
    w.carrying[0] = w.carrying[1] = True
    r = w.apply_actions(0, {0: Drop((1, 3)), 1: Drop((1, 3))})
    assert [r[0].ok, r[1].ok] == [True, False] and r[1].reason == "lost_priority"
    assert w.fills == 1 and w.carrying[1] is True
    r2 = w.apply_actions(1, {1: Drop((1, 3))})
    assert not r2[1].ok and r2[1].reason == "already_filled"


def test_return_and_pickup_rejections():
    w = world([(1, 6), (0, 0)])
    assert w.apply_actions(0, {1: Pickup()})[1].reason == "not_depot"
    w.apply_actions(1, {0: Pickup()})
    assert w.apply_actions(2, {0: Pickup()})[0].reason == "has_bag"
    assert w.apply_actions(3, {0: Return()})[0].ok and w.stock[(1, 6)] == 2


def test_observe_and_prefill():
    w = world([(1, 4)])
    w.prefill([(1, 3)])
    ob = w.observe(0, 2)
    assert (1, 3) in ob.filled_pits
    assert w.observe(0, 0).filled_pits == frozenset()


def incident_world(starts, appear=0, cls="robot_clearable"):
    from src.doi.scenarios import Incident
    inc = [Incident(0, ((1, 3),), appear, "pallet", cls)]
    s = scenario_from_ascii(["......D"] * 3, starts, [[(0, 0)] for _ in starts], depot_stock=2,
                            incidents=inc)
    return World(s, SimConfig(n_robots=len(starts), debug_checks=True))


def test_incident_appears_and_blocks():
    w = incident_world([(1, 2)], appear=2)
    w.begin_tick(0)
    assert w.apply_actions(0, {0: Move((1, 3))})[0].ok
    w.begin_tick(1)
    assert w.apply_actions(1, {0: Move((1, 2))})[0].ok
    w.begin_tick(2)
    assert w.appeared_at[0] == 2 and (1, 3) in w.observe(0, 2).blocked_cells
    assert w.apply_actions(2, {0: Move((1, 3))})[0].reason == "wall"


def test_incident_waits_for_free_cell():
    w = incident_world([(1, 3)], appear=0)
    w.begin_tick(0)
    assert 0 not in w.appeared_at
    w.apply_actions(0, {0: Move((1, 4))})
    w.begin_tick(1)
    assert w.appeared_at[0] == 1


def test_needs_human_drop_rejected():
    w = incident_world([(1, 4)], appear=0, cls="needs_human")
    w.begin_tick(0)
    w.carrying[0] = True
    r = w.apply_actions(0, {0: Drop((1, 3))})[0]
    assert r.reason == "needs_human" and w.carrying[0] and w.fills == 0
    assert w.counters[0]["wrong_class_attempts"] == 1


def test_drop_on_clear_cell_rejected():
    w = incident_world([(1, 4)], appear=50)
    w.begin_tick(0)
    w.carrying[0] = True
    assert w.apply_actions(0, {0: Drop((1, 3))})[0].reason == "not_blocked"


def test_arbiter_overrides_counted():
    w = world([(0, 0), (0, 2)], ["..."])
    w.apply_actions(0, {0: Move((0, 1)), 1: Move((0, 1))})
    assert w.overrides == 1
    w.apply_actions(1, {0: Move((0, 9))})
    assert w.overrides == 1


def test_random_fuzz_never_collides():
    import random
    rng = random.Random(1)
    rows = ["........"] * 8
    starts = [(r, c) for r, c in [(0, 0), (0, 7), (7, 0), (7, 7), (3, 3), (4, 4), (2, 5), (5, 2)]]
    w = world(starts, rows)
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for t in range(300):
        acts = {}
        for i, p in w.pos.items():
            dr, dc = rng.choice(dirs)
            acts[i] = Move((p[0] + dr, p[1] + dc)) if rng.random() < 0.9 else Wait()
        w.apply_actions(t, acts)
        assert len(set(w.pos.values())) == len(w.pos)


def test_dense_fuzz_with_cycles_never_collides():
    import random
    rng = random.Random(7)
    rows = ["...."] * 4
    starts = [(r, c) for r in range(4) for c in range(4)][:14]
    w = world(starts, rows)
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for t in range(1000):
        acts = {}
        for i, p in w.pos.items():
            dr, dc = rng.choice(dirs)
            acts[i] = Move((p[0] + dr, p[1] + dc))
        w.apply_actions(t, acts)
        assert len(set(w.pos.values())) == len(w.pos)
    assert w.overrides > 0
