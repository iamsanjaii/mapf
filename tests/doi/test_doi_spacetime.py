from src.doi.paths import passable_fn, bfs_dist_map
from src.doi.scenarios import scenario_from_ascii
from src.doi.spacetime import plan_spacetime


def setup(rows):
    s = scenario_from_ascii(rows, [(0, 0)], [[(0, 0)]])
    pf = passable_fn(s.grid, frozenset())
    return pf, len(rows), len(rows[0])


def hmap(pf, goal, H, W):
    return bfs_dist_map(pf, goal, H, W)


def test_free_path_matches_shortest():
    pf, H, W = setup(["....", "...."])
    p = plan_spacetime(pf, (0, 0), (1, 3), 5, set(), hmap(pf, (1, 3), H, W), 30)
    assert p[0] == (0, 0) and p[-1] == (1, 3) and len(p) == 5


def test_waits_for_reservation():
    pf, H, W = setup(["..."])
    reserved = {((0, 1), 1)}
    p = plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 30)
    assert p == [(0, 0), (0, 0), (0, 1), (0, 2)]


def test_routes_around_reserved_cell():
    pf, H, W = setup(["...", "..."])
    reserved = {((0, 1), 1), ((0, 1), 2)}
    p = plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 30)
    assert len(p) == 5 and (0, 1) not in p[1:3]


def test_swap_is_rejected():
    pf, H, W = setup(["..."])
    reserved = {((0, 0), 1), ((0, 1), 0)}
    p = plan_spacetime(pf, (0, 1), (0, 0), 0, reserved, hmap(pf, (0, 0), H, W), 6)
    assert p is None or p[1] != (0, 0)


def test_unreachable_returns_none():
    pf, H, W = setup(["..#.."])
    assert plan_spacetime(pf, (0, 0), (0, 4), 0, set(), {(0, 4): 0, (0, 3): 1}, 20) is None


def test_start_equals_goal_and_ignores_own_reservation():
    pf, H, W = setup(["..."])
    assert plan_spacetime(pf, (0, 1), (0, 1), 3, {((0, 1), 3)}, {(0, 1): 0}, 10) == [(0, 1)]


def test_cannot_wait_in_a_cell_reserved_next_tick_but_can_step_aside():
    pf, H, W = setup(["...", "..."])
    reserved = {((0, 0), 1)}
    p = plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 10)
    assert p is not None and p[1] != (0, 0) and p[-1] == (0, 2)


def test_swap_with_reserved_cell_is_still_rejected_when_no_room():
    pf, H, W = setup(["..."])
    reserved = {((0, 0), 1), ((0, 1), 0)}
    assert plan_spacetime(pf, (0, 0), (0, 2), 0, reserved, hmap(pf, (0, 2), H, W), 10) is None
