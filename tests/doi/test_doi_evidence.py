from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord, ObstructionRecord
from src.doi.evidence import EvidenceEngine
from src.doi.scenarios import build_scenario


def setup(name):
    s = build_scenario(SimConfig(scenario=name, n_robots=1, tasks_per_robot=1))
    return s, BeliefState(0, s.depots, s.pits), EvidenceEngine(s.grid, 144.0)


def test_series_residual_after_fill():
    s, b, e = setup("series_pits")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 15), 0, 18))
    assert e.evidence(b, 0) == {((3, 11), (3, 13)): 18.0}
    b.filled.add((3, 13))
    assert e.evidence(b, 0) == {((3, 11),): 18.0}


def test_substitute_fill_invalidates_evidence():
    s, b, e = setup("two_pits_parallel")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    b.add_record(RentRecord(0, 1, (5, 9), (5, 11), 0, 14))
    assert e.evidence(b, 0) == {((3, 10),): 18.0, ((5, 10),): 14.0}
    b.filled.add((3, 10))
    assert e.evidence(b, 0) == {((5, 10),): 4.0}


def test_per_pit_series_is_zero():
    s, b, e = setup("series_pits")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 15), 0, 18))
    assert e.evidence(b, 0, per_pit=True) == {}


def test_window_drops_old_records():
    s, b, e = setup("single_pit")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    assert e.evidence(b, 100, window=50) == {}
    assert e.evidence(b, 40, window=50) == {((3, 10),): 18.0}


def test_extrapolated_evidence():
    s, b, e = setup("single_pit")
    b.add_record(RentRecord(0, 0, (3, 9), (3, 11), 0, 18))
    b.census.add(1)
    b.census.add(2)
    assert e.evidence(b, 0, extrapolate=True) == {((3, 10),): 18.0 * 3 / 1}


def test_needs_human_cell_gets_no_evidence():
    s, b, e = setup("incidents_room")
    b.add_obstruction(ObstructionRecord("r0", 0, "north door", ((2, 10),), "spill",
                                        "robot_clearable", 1, 0.9, "x"), 0)
    b.observe_cell((2, 10), True, 1)
    b.add_record(RentRecord(0, 0, (2, 9), (2, 11), 1, 10))
    assert e.evidence(b, 1) == {((2, 10),): 10.0}
    b.mark_needs_human((2, 10))
    assert e.evidence(b, 1) == {}
