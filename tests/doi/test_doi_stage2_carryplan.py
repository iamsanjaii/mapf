"""Carry and fill plans: pick up an obstacle, haul it to a slot or a pit, and walk on to the goal."""
from src.doi.carryplan import carry_plans
from src.doi.evidence import EvidenceEngine
from src.doi.kinds import KINDS
from src.doi.scenarios import scenario_from_ascii

UNREACHABLE = 99.0
COSTS = dict(kappa_c=2.0, pick_fee=1.0, drop_fee=1.0, unreachable=UNREACHABLE)


def plans(rows, pos, goal, carry=None, fill=None, full=frozenset(), skip=frozenset()):
    s = scenario_from_ascii(rows, [pos], [[goal]])
    engine = EvidenceEngine(s.grid, UNREACHABLE)
    blocked = frozenset(s.obstacles)
    pits = frozenset(c for c, k in s.obstacles.items() if k == "pit")
    carry = sorted(c for c, k in s.obstacles.items() if k != "pit") if carry is None else carry
    fill = carry if fill is None else fill
    return carry_plans(s.grid, engine.distance, blocked, carry, fill, pits, s.obstacles.get, pos, goal,
                       slots=s.slots, full=full, skip=skip, **COSTS)


def test_crate_one_step_from_a_rack_costs_what_a_hand_count_says():
    # Robot under the crate at (0,1); the rack at (0,0) is reached by standing on the crate's old cell.
    ps = plans(["TC...", "....."], pos=(1, 1), goal=(0, 4))
    assert len(ps) == 1
    p = ps[0]
    assert p.mode == "carry" and p.obstacle == (0, 1) and p.target == (0, 0)
    assert (p.approach, p.access) == ((1, 1), (0, 1))
    # walk_in 0 + pick 1 + one loaded step (2.0 * crate weight 0.5 = 1.0) + drop 1 + walk on 3
    assert (p.walk_in, p.haul, p.walk_on) == (0.0, 1.0, 3.0)
    assert p.total == 6.0
    assert p.before == frozenset({(0, 1)}) and p.after == frozenset()


def test_a_full_slot_is_not_a_target():
    assert plans(["TC...", "....."], pos=(1, 1), goal=(0, 4), full=frozenset({(0, 0)})) == []


def test_a_kind_only_goes_to_slots_that_accept_it():
    rows = ["TS...", "....."]                      # a shelf unit may not go on a rack
    assert plans(rows, pos=(1, 1), goal=(0, 4)) == []
    ps = plans(["DS...", "....."], pos=(1, 1), goal=(0, 4))      # but it may go to the dump region
    assert len(ps) == 1 and ps[0].target == (0, 0)
    assert ps[0].haul_cost == 2.0 * KINDS["shelf_unit"].weight * ps[0].haul == 4.0      # one loaded step


def test_the_nearest_free_slot_wins():
    rows = ["TC..T", "....."]
    ps = plans(rows, pos=(1, 1), goal=(0, 2))
    assert [p.target for p in ps] == [(0, 0)]       # one cheapest plan per obstacle, not one per slot


def test_a_fill_plan_also_opens_the_pit():
    ps = plans([".P.", "R.."], pos=(1, 1), goal=(1, 2), carry=[], fill=[(1, 0)])
    assert len(ps) == 1
    p = ps[0]
    assert p.mode == "fill" and p.obstacle == (1, 0) and p.target == (0, 1)
    assert (p.approach, p.access) == ((1, 1), (1, 1))                  # lift and drop from the same cell
    assert p.haul == 0.0 and p.walk_on == 1.0 and p.total == 3.0       # pick 1, no haul, drop 1, one step on
    assert p.before == frozenset({(0, 1), (1, 0)}) and p.after == frozenset()


def test_only_debris_can_fill_a_pit():
    assert plans([".P.", "C.."], pos=(1, 1), goal=(0, 2), carry=[], fill=[(1, 0)]) == []


def test_skipped_obstacles_are_not_planned():
    assert plans(["TC...", "....."], pos=(1, 1), goal=(0, 4), skip=frozenset({(0, 1)})) == []


def test_unreachable_pick_up_cell_gives_no_plan():
    rows = ["TC#..", "..#.."]                     # the crate is reachable only from the west side
    assert plans(rows, pos=(1, 4), goal=(0, 4)) == []
