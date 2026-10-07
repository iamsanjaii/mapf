import json

from src.doi.forecast.case import ForecastCase
from src.doi.forecast.tools import ANSWER, LEDGER_SUMMARY, READ_NOTICES, TOOLS, TRIP_SAVING, call_tool

LEDGER = {"robots_known": 8.0, "tasks_per_robot": 20.0, "trips_recorded": 10.0, "trips_expected": 160.0,
          "saving_on_recorded_trips": 6.0, "saving_on_my_current_trip": 0.0, "my_tasks_done": 1.0,
          "my_tasks_left": 19.0}
CASE = ForecastCase(
    case_id="x", robot=0, tick=9, mode="push", kind="pallet", obstacle=(10, 10), landing=(10, 11),
    plan_key=(((10, 10), (0, 1), 1),), price=10.0, known_saving=6.0,
    notices=(("n0", 5, "wave 2 is all in the south bays"), ("d0", 3, "the label printer is out of ribbon")),
    ledger=LEDGER,
    zones={"north": {"rows": (0, 6), "cols": (0, 9)}, "south": {"rows": (8, 14), "cols": (0, 9)}},
    trip_saving={"north|south": {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0},
                 "south|north": {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0},
                 "north|north": {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 20.0},
                 "south|south": {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 20.0}},
    numeric_forecast=False, truth=True)


def test_schemas_are_json_and_name_the_four_tools():
    assert [t["function"]["name"] for t in TOOLS] == ["read_notices", "ledger_summary", "trip_saving", "answer"]
    assert json.loads(json.dumps(TOOLS)) == TOOLS
    assert all(t["type"] == "function" for t in TOOLS)
    assert TRIP_SAVING["function"]["parameters"]["required"] == ["from_zone", "to_zone"]
    assert ANSWER["function"]["parameters"]["required"] == ["will_pay", "confidence", "reason"]
    assert ANSWER["function"]["parameters"]["properties"]["will_pay"]["type"] == "boolean"
    assert READ_NOTICES["function"]["parameters"]["properties"] == {} == LEDGER_SUMMARY["function"]["parameters"]["properties"]


def test_read_notices_returns_what_the_robot_received():
    out = call_tool(CASE, "read_notices", {})
    assert out == {"notices": [{"id": "n0", "tick": 5, "text": "wave 2 is all in the south bays"},
                               {"id": "d0", "tick": 3, "text": "the label printer is out of ribbon"}]}
    assert call_tool(CASE, "read_notices", None) == out                      # no arguments are needed


def test_ledger_summary_adds_price_and_known_saving():
    assert call_tool(CASE, "ledger_summary", {}) == {**LEDGER, "price": 10.0, "known_saving": 6.0}


def test_trip_saving_by_zone_pair_and_its_errors():
    assert call_tool(CASE, "trip_saving", {"from_zone": "north", "to_zone": "south"}) == \
        {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 24.0}
    unknown = call_tool(CASE, "trip_saving", {"from_zone": "attic", "to_zone": "south"})
    assert "attic" in unknown["error"] and "north" in unknown["error"] and "south" in unknown["error"]
    assert "error" in call_tool(CASE, "trip_saving", {"from_zone": "north"})            # a missing argument
    assert "error" in call_tool(CASE, "trip_saving", {"from_zone": 3, "to_zone": ["south"]})
    assert "error" in call_tool(CASE, "trip_saving", None)
    assert "error" in call_tool(CASE, "trip_saving", ["north", "south"])
    nozones = ForecastCase(**{**CASE.__dict__, "zones": {}, "trip_saving": {}})
    assert "no zones" in call_tool(nozones, "trip_saving", {"from_zone": "north", "to_zone": "south"})["error"]


def test_unknown_tools_and_answer_are_errors_here():
    for name in ("teleport", "answer", None, 7):
        out = call_tool(CASE, name, {})
        assert list(out) == ["error"] and "read_notices" in out["error"]
