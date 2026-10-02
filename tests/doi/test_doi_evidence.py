from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.scenarios import build_scenario


def setup(epoch=None):
    s = build_scenario(SimConfig(scenario="single_block", n_robots=1, tasks_per_robot=1))
    return s, BeliefState(0, s.obstacles), EvidenceEngine(s.grid, 144.0, epoch)


def rec(k, origin, dest):
    return RentRecord(0, k, origin, dest, 0, 0.0)


def test_distance_round_the_barrier_and_through_the_gap():
    s, b, e = setup()
    assert e.distance((3, 9), (3, 13), frozenset({(3, 10)})) == 22
    assert e.distance((3, 9), (3, 13), frozenset()) == 4
    assert e.distance((3, 9), (3, 11), frozenset({(3, 11)})) == 144.0      # the goal cell itself is blocked


def test_evidence_is_the_travel_the_push_would_have_saved():
    s, b, e = setup()
    b.add_record(rec(0, (3, 9), (3, 13)))
    b.add_record(rec(1, (3, 13), (3, 9)))
    before, after = frozenset({(3, 10)}), frozenset({(3, 12)})            # pushed east two cells
    assert e.evidence(b, before, after) == 2 * (22 - 6)


def test_collateral_on_a_recorded_route_lowers_the_evidence():
    s, b, e = setup()
    b.add_record(rec(0, (3, 9), (3, 13)))
    b.add_record(rec(1, (3, 11), (3, 13)))                                 # a task the landing cell will block
    before, after = frozenset({(3, 10)}), frozenset({(3, 12)})
    saved = 22 - 6
    hurt = e.distance((3, 11), (3, 13), after) - e.distance((3, 11), (3, 13), before)
    assert hurt == 2 and e.evidence(b, before, after) == saved - hurt


def test_traffic_with_no_rent_still_counts_against_a_bad_landing():
    s, b, e = setup()
    b.add_record(rec(0, (3, 11), (3, 13)))                                 # this task never paid any rent
    assert e.evidence(b, frozenset({(3, 10)}), frozenset({(3, 12)})) == -2.0


def test_aggregated_records_count_each_traversal():
    s, b, e = setup(epoch=10)
    for k in range(3):
        b.add_rent(RentRecord(0, k, (3, 9), (3, 13), k, 18), 10)
    assert e.evidence(b, frozenset({(3, 10)}), frozenset({(3, 12)})) == 3 * (22 - 6)


def test_an_empty_ledger_gives_no_evidence():
    s, b, e = setup()
    assert e.evidence(b, frozenset({(3, 10)}), frozenset()) == 0.0
