import random
from src.doi.belief import BeliefState
from src.doi.crdt import RentRecord, ObstructionRecord

OBST = {(2, 2): "pallet"}


def obs(rid="r0", node=0, cells=((1, 1),), cls="robot_clearable"):
    return ObstructionRecord(rid, node, "x", cells, "pallet", cls, 1, 0.9, "why")


def rand_state(rid: int, rng: random.Random) -> BeliefState:
    b = BeliefState(rid, OBST)
    for k in range(rng.randint(0, 4)):
        b.add_record(RentRecord(rid, k, (0, 0), (0, rng.randint(1, 5)), rng.randint(0, 9), rng.randint(1, 9)))
    if rng.random() < 0.5:
        b.observe_cell((1, 1), rng.random() < 0.5, rng.randint(0, 9), kind=rng.choice(["pallet", "crate", None]))
    if rng.random() < 0.5:
        b.add_obstruction(obs(node=rid), rng.randint(0, 9))
    return b


def merged(a, b):
    x = a.snapshot()
    x.merge(b)
    return x


def test_merge_laws():
    rng = random.Random(0)
    for _ in range(300):
        a, b, c = (rand_state(i, rng) for i in range(3))
        assert merged(a, b).canonical() == merged(b, a).canonical()
        assert merged(merged(a, b), c).canonical() == merged(a, merged(b, c)).canonical()
        assert merged(a, a).canonical() == a.canonical()


def test_merge_is_monotone_and_reports_change():
    a = BeliefState(0, {})
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b = BeliefState(1, {})
    assert b.merge(a) is True
    assert b.merge(a) is False
    assert len(b.records.records()) == 1


def test_records_union_is_idempotent():
    a, b = BeliefState(0, {}), BeliefState(1, {})
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b.add_record(RentRecord(1, 0, (0, 0), (0, 3), 0, 7))
    a.merge(b)
    a.merge(b)
    assert [r.rent for r in a.records.records()] == [5, 7]
    assert a.census.items() == frozenset({0, 1})


def test_snapshot_is_isolated():
    a = BeliefState(0, {})
    s = a.snapshot()
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    assert s.records.records() == []


def test_status_lattice_and_latest_observation_wins():
    b = BeliefState(0, {})
    b.add_obstruction(obs(), 5)
    assert b.status((1, 1)) == "reported" and (1, 1) in b.believed_blocked()
    b.observe_cell((1, 1), False, 8)
    assert b.status((1, 1)) == "refuted" and (1, 1) not in b.believed_blocked()
    b.observe_cell((1, 1), True, 20)
    assert b.status((1, 1)) == "confirmed"
    x, y = BeliefState(0, {}), BeliefState(1, {})
    x.observe_cell((3, 3), True, 3)
    y.observe_cell((3, 3), False, 7)
    x.merge(y)
    assert x.status((3, 3)) == "refuted"


def test_initial_obstacles_start_confirmed_and_class_lattice():
    b = BeliefState(0, OBST)
    assert b.status((2, 2)) == "confirmed" and b.editable() == ((2, 2),) and b.kind_of((2, 2)) == "pallet"
    b.add_obstruction(obs(cells=((1, 1),), cls="robot_clearable"), 1)
    assert b.editable() == ((1, 1), (2, 2)) and b.hard_blocked() == frozenset()
    b.mark_needs_human((1, 1))
    assert b.editable() == ((2, 2),) and b.hard_blocked() == frozenset({(1, 1)})


def test_a_moved_obstacle_is_learned_from_observations_in_any_order():
    a, b = BeliefState(0, OBST), BeliefState(1, OBST)
    a.observe_cell((2, 2), False, 5)                       # robot 0 saw the pallet leave (2, 2) ...
    a.observe_cell((2, 3), True, 5, kind="pallet")         # ... and arrive at (2, 3)
    assert a.believed_blocked() == frozenset({(2, 3)}) and a.kind_of((2, 3)) == "pallet"
    assert b.believed_blocked() == frozenset({(2, 2)})     # robot 1 is out of date until they talk
    assert merged(a, b).canonical() == merged(b, a).canonical()
    b.merge(a)
    assert b.believed_blocked() == frozenset({(2, 3)})


def test_the_kind_register_keeps_the_latest_sighting():
    b = BeliefState(0, {})
    b.observe_cell((1, 1), True, 3, kind="crate")
    b.observe_cell((1, 1), True, 9, kind="shelf_unit")
    b.observe_cell((1, 1), True, 5, kind="pallet")          # an older sighting arriving late does not win
    assert b.kind_of((1, 1)) == "shelf_unit"


def test_obstruction_same_report_smaller_node_wins():
    a, b = BeliefState(0, {}), BeliefState(1, {})
    a.add_obstruction(obs(node=4, cls="needs_human"), 1)
    b.add_obstruction(obs(node=2, cls="robot_clearable"), 1)
    a.merge(b)
    assert a.obstructions.get("r0").node == 2
