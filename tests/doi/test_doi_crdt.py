import random
from src.doi.belief import BeliefState
from src.doi.crdt import Claim, ClaimSet, RentRecord, ObstructionRecord

DEPOTS = {(0, 0): 3}


def obs(rid="r0", node=0, cells=((1, 1),), cls="robot_clearable"):
    return ObstructionRecord(rid, node, "x", cells, "pallet", cls, 1, 0.9, "why")


def rand_state(rid: int, rng: random.Random) -> BeliefState:
    b = BeliefState(rid, DEPOTS, [(2, 2)])
    for k in range(rng.randint(0, 4)):
        b.add_record(RentRecord(rid, k, (0, 0), (0, rng.randint(1, 5)), rng.randint(0, 9), rng.randint(1, 9)))
    if rng.random() < 0.5:
        b.filled.add((rng.randint(0, 2), 1))
    if rng.random() < 0.5:
        b.stock.take((0, 0), rid)
    if rng.random() < 0.5:
        b.claims.issue(b.next_ticket(), Claim(((1, 1),), rid), rng.randint(1, 20))
    if rng.random() < 0.5:
        b.observe_cell((1, 1), rng.random() < 0.5, rng.randint(0, 9))
    if rng.random() < 0.5:
        b.add_obstruction(obs(node=rid), rng.randint(0, 9))
    if rng.random() < 0.3:
        b.approvals.set(((((1, 1),), (1, rid))), rng.choice(["approve", "veto"]))
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
    a = BeliefState(0, DEPOTS, [])
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b = BeliefState(1, DEPOTS, [])
    assert b.merge(a) is True
    assert b.merge(a) is False
    assert len(b.records.records()) == 1


def test_records_union_is_idempotent():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    b.add_record(RentRecord(1, 0, (0, 0), (0, 3), 0, 7))
    a.merge(b)
    a.merge(b)
    assert [r.rent for r in a.records.records()] == [5, 7]
    assert a.census.items() == frozenset({0, 1})


def test_snapshot_is_isolated():
    a = BeliefState(0, DEPOTS, [])
    s = a.snapshot()
    a.add_record(RentRecord(0, 0, (0, 0), (0, 3), 0, 5))
    assert s.records.records() == []


def test_status_lattice_and_latest_observation_wins():
    b = BeliefState(0, DEPOTS, [])
    b.add_obstruction(obs(), 5)
    assert b.status((1, 1)) == "reported" and (1, 1) in b.believed_blocked()
    b.observe_cell((1, 1), False, 8)
    assert b.status((1, 1)) == "refuted" and (1, 1) not in b.believed_blocked()
    b.observe_cell((1, 1), True, 20)
    assert b.status((1, 1)) == "confirmed"
    b.filled.add((1, 1))
    assert b.status((1, 1)) == "filled" and (1, 1) not in b.believed_blocked()
    x, y = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    x.observe_cell((3, 3), True, 3)
    y.observe_cell((3, 3), False, 7)
    x.merge(y)
    assert x.status((3, 3)) == "refuted"


def test_static_pits_start_confirmed_and_class_lattice():
    b = BeliefState(0, DEPOTS, [(2, 2)])
    assert b.status((2, 2)) == "confirmed" and b.editable() == ((2, 2),)
    b.add_obstruction(obs(cells=((1, 1),), cls="robot_clearable"), 1)
    assert b.editable() == ((1, 1), (2, 2)) and b.hard_blocked() == frozenset()
    b.mark_needs_human((1, 1))
    assert b.editable() == ((2, 2),) and b.hard_blocked() == frozenset({(1, 1)})


def test_obstruction_same_report_smaller_node_wins():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    a.add_obstruction(obs(node=4, cls="needs_human"), 1)
    b.add_obstruction(obs(node=2, cls="robot_clearable"), 1)
    a.merge(b)
    assert a.obstructions.get("r0").node == 2


def test_approvals_veto_dominates():
    a, b = BeliefState(0, DEPOTS, []), BeliefState(1, DEPOTS, [])
    key = (((1, 1),), (3, 0))
    a.approvals.set(key, "approve")
    b.approvals.set(key, "veto")
    a.merge(b)
    assert a.approvals.get(key) == "veto"


def test_claim_effective_smallest_live_ticket():
    cs = ClaimSet()
    cs.issue((5, 2), Claim(((1, 1),), 2), expiry=10)
    cs.issue((4, 7), Claim(((1, 1),), 7), expiry=10)
    assert cs.effective((1, 1), 3)[0] == (4, 7)
    assert cs.effective((1, 1), 10) is None
    cs.renew((5, 2), 20)
    assert cs.effective((1, 1), 15)[0] == (5, 2)


def test_stock_remaining_and_mark_empty():
    b = BeliefState(0, {(0, 0): 2}, [])
    b.stock.take((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 1
    b.stock.mark_empty((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 0
    b.stock.give_back((0, 0), 0)
    assert b.stock.remaining((0, 0)) == 1
