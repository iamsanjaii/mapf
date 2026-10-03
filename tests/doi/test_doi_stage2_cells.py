"""Stage 2 cells: pits, rack slots and dump-region slots, and the kinds that may go where."""
import pytest

from src.doi.config import SimConfig
from src.doi.kinds import KINDS, GLYPHS, can_carry, can_fill, pushable
from src.doi.scenarios import scenario_from_ascii


def test_pit_is_an_unpushable_kind_with_glyph_p():
    assert GLYPHS["P"] == "pit"
    assert not pushable("pit") and pushable("pallet")
    assert not can_carry("pit")


def test_slot_rules_per_kind():
    assert can_carry("crate") and "rack" in KINDS["crate"].slots
    assert KINDS["shelf_unit"].slots == ("dump",)          # a shelf unit only goes to the dump zone
    assert can_fill("debris") and not can_fill("crate")
    assert not can_carry("spill")


def test_ascii_parses_slots_and_pits():
    rows = ["D.T.P",
            "DD..."]
    s = scenario_from_ascii(rows, [(1, 4)], [[(0, 1)]])
    assert s.slots == {(0, 0): "dump", (1, 0): "dump", (1, 1): "dump", (0, 2): "rack"}
    assert s.obstacles == {(0, 4): "pit"}
    assert s.grid.is_passable(0, 0)                         # a dump cell is walkable yard floor
    assert not s.grid.is_passable(0, 2)                     # a rack cell is a fixture, not floor


def test_scenarios_without_slots_have_none():
    s = scenario_from_ascii(["..."], [(0, 0)], [[(0, 2)]])
    assert s.slots == {}


def test_carry_costs_validated():
    with pytest.raises(ValueError):
        SimConfig(kappa_c=1.0)                              # a loaded step must cost at least an empty one
    cfg = SimConfig()
    assert cfg.kappa_c == 2.0 and cfg.pick_fee == 1.0 and cfg.drop_fee == 1.0
