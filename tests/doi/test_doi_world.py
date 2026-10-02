from src.doi.config import SimConfig
from src.doi.scenarios import Incident, scenario_from_ascii
from src.doi.world import World, Move, Push, Wait

ROWS = ["...#...", "...L...", "...#...", "...#..."]


def world(starts, rows=ROWS, **kw):
    s = scenario_from_ascii(rows, starts, [[(0, 0)] for _ in starts])
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


def test_wall_and_obstacle_are_illegal_to_walk_into():
    w = world([(1, 2)])
    assert w.apply_actions(0, {0: Move((1, 3))})[0].reason == "wall"
    assert w.apply_actions(1, {0: Move((0, 2)), })[0].ok
    w2 = world([(0, 2)])
    assert w2.apply_actions(0, {0: Move((0, 3))})[0].reason == "wall"


def test_push_slides_the_obstacle_and_moves_the_robot():
    w = world([(1, 2)], kappa=4.0, fee=1.0)
    r = w.apply_actions(0, {0: Push((1, 3))})[0]
    assert r.ok and w.pos[0] == (1, 3) and w.obstacles == {(1, 4): "pallet"}
    assert w.grid.get(1, 3).name == "FREE" and w.grid.get(1, 4).name == "OBSTACLE"
    assert w.counters[0]["push_steps"] == 1 and w.counters[0]["moves"] == 1 and w.removals == 1
    assert w.tick_cost == [4.0 + 1.0]                         # kappa * weight + one fee for the run


def test_a_straight_run_costs_one_fee_and_a_pause_starts_a_new_run():
    w = world([(1, 2)], kappa=4.0, fee=1.0)
    w.apply_actions(0, {0: Push((1, 3))})
    w.apply_actions(1, {0: Push((1, 4))})
    assert w.removals == 1 and w.tick_cost == [5.0, 4.0] and w.push_log[0]["steps"] == 2
    assert w.push_log[0]["origin"] == (1, 3) and w.push_log[0]["landing"] == (1, 5)
    w.apply_actions(2, {0: Wait()})
    w.apply_actions(3, {0: Push((1, 5))})
    assert w.removals == 2 and w.tick_cost[-1] == 5.0 and len(w.push_log) == 2


def test_kind_weight_scales_the_push_cost():
    w = World(scenario_from_ascii(["..C..", "..S.."], [(0, 1), (1, 1)], [[(0, 4)], [(1, 4)]]),
              SimConfig(n_robots=2, kappa=4.0, fee=0.0))
    w.apply_actions(0, {0: Push((0, 2))})
    assert w.push_cost == 2.0 and w.tick_cost == [2.0]        # crate: 4 * 0.5
    w.apply_actions(1, {1: Push((1, 2))})
    assert w.push_cost == 10.0 and w.tick_cost[-1] == 8.0     # shelf unit: 4 * 2.0


def test_push_rejections_cost_a_wait():
    w = world([(1, 2), (1, 6)], ["...L#..", ".......", "......."])
    w.pos[0] = (0, 2)
    r = w.apply_actions(0, {0: Push((0, 3))})[0]            # the cell beyond is a wall
    assert not r.ok and r.reason == "no_room" and w.pos[0] == (0, 2) and w.obstacles == {(0, 3): "pallet"}
    assert w.counters[0]["waits"] == 1 and w.counters[0]["push_rejected"] == 1 and w.tick_cost == [1.0]
    assert w.apply_actions(1, {0: Push((1, 2))})[0].reason == "no_obstacle"
    w.pos[0] = (1, 0)
    assert w.apply_actions(2, {0: Push((0, 3))})[0].reason == "not_adjacent"
    assert w.removals == 0


def test_push_needs_the_cell_beyond_free_of_robots():
    w = world([(1, 2), (1, 4)])
    r = w.apply_actions(0, {0: Push((1, 3)), 1: Wait()})[0]
    assert r.reason == "no_room" and w.obstacles == {(1, 3): "pallet"}


def test_two_pushers_one_obstacle_only_one_succeeds():
    rows = [".......", ".......", "...L...", ".......", "......."]
    w = world([(2, 2), (3, 3)], rows)
    r = w.apply_actions(0, {0: Push((2, 3)), 1: Push((2, 3))})
    assert [r[0].ok, r[1].ok] == [True, False] and r[1].reason == "no_obstacle"
    assert w.removals == 1 and w.obstacles == {(2, 4): "pallet"}


def test_a_robot_walking_into_the_landing_cell_is_refused():
    w = world([(1, 2), (0, 4)], [".......", "...L...", "......."])
    r = w.apply_actions(0, {0: Push((1, 3)), 1: Move((1, 4))})
    assert r[0].ok and not r[1].ok and r[1].reason == "wall"


def test_cost_per_tick_adds_up_to_the_run_total():
    w = world([(1, 2), (0, 0)], kappa=3.0, fee=2.0)
    for t, acts in enumerate([{0: Push((1, 3)), 1: Move((0, 1))}, {0: Push((1, 4)), 1: Wait()},
                              {0: Move((0, 3)), 1: Move((0, 2))}]):
        w.apply_actions(t, acts)
    moves = sum(c["moves"] for c in w.counters.values())
    pushes = sum(c["push_steps"] for c in w.counters.values())
    waits = sum(c["waits"] for c in w.counters.values())
    j = (moves - pushes) + waits + w.push_cost + 2.0 * w.removals
    assert abs(sum(w.tick_cost) - j) < 1e-9


def test_prefill_removes_obstacles_and_observe_sees_nearby_ones():
    w = world([(1, 4)])
    ob = w.observe(0, 2)
    assert ob.blocked_cells == frozenset({(1, 3)}) and ob.kinds == (((1, 3), "pallet"),)
    assert (1, 3) not in ob.scanned and (1, 5) in ob.scanned
    assert w.observe(0, 0).blocked_cells == frozenset()
    w.prefill([(1, 3)])
    assert w.obstacles == {} and w.observe(0, 2).blocked_cells == frozenset()
    assert w.obstacle_trace[0] == {} and w.apply_actions(0, {0: Move((1, 3))})[0].ok


def test_obstacle_trace_follows_the_pushes():
    w = world([(1, 2)])
    w.apply_actions(0, {0: Push((1, 3))})
    assert w.obstacle_trace == [{(1, 3): "pallet"}, {(1, 4): "pallet"}]


def incident_world(starts, appear=0, cls="robot_clearable"):
    inc = [Incident(0, ((1, 3),), appear, "pallet", cls)]
    s = scenario_from_ascii(["......."] * 3, starts, [[(0, 0)] for _ in starts], incidents=inc)
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


def test_incident_obstacle_can_be_pushed_unless_it_needs_a_human():
    w = incident_world([(1, 2)], appear=0)
    w.begin_tick(0)
    assert w.apply_actions(0, {0: Push((1, 3))})[0].ok and (1, 4) in w.obstacles
    w = incident_world([(1, 2)], appear=0, cls="needs_human")
    w.begin_tick(0)
    r = w.apply_actions(0, {0: Push((1, 3))})[0]
    assert r.reason == "needs_human" and w.removals == 0 and w.counters[0]["wrong_class_attempts"] == 1


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
