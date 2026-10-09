import dataclasses
import json
from types import SimpleNamespace

from src.doi.agent import RobotAgent
from src.doi.config import SimConfig
from src.doi.crdt import NoticeRecord, RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.forecast.case import (build_case, case_from_dict, case_to_dict, numeric_saving, plan_key, plan_mode,
                                   true_total_saving, trip_table)
from src.doi.policies import PredictedPolicy, Shared
from src.doi.scenarios import scenario_from_ascii

# A wall at column 2 with a pallet in its only gap (row 1); row 4 is the long way round.
ROWS = ["..#....",
        "..L....",
        "..#....",
        "..#....",
        "......."]
WEST = ((0, 0), (0, 1), (1, 0), (1, 1))
EAST = ((0, 5), (0, 6), (1, 5), (1, 6))


def setup(price_total=16.0):
    s = scenario_from_ascii(ROWS, starts=[(1, 0), (4, 0)], tasks=[[(1, 6), (1, 0), (1, 6)], [(4, 6)]])
    s.zones = {"west": WEST, "east": EAST}
    cfg = SimConfig(n_robots=2, tasks_per_robot=3, policy="rof_p")
    shared = Shared(engine=EvidenceEngine(s.grid, cfg.unreachable_cost_for(5, 7)))
    policy = PredictedPolicy(0.5)
    policy.prepare(s, cfg, shared)
    agent = RobotAgent(0, s, cfg, policy, shared)
    agent.belief.add_record(RentRecord(0, 0, (1, 0), (1, 6), 0, 6))
    agent.belief.add_record(RentRecord(1, 0, (1, 0), (1, 6), 0, 6))
    # A stand-in plan that lifts the pallet out of the gap: only `before` and `after` matter to a case.
    plan = SimpleNamespace(obstacle=(1, 2), kind="pallet", landing=(1, 2), key=("lift", (1, 2)),
                           before=frozenset({(1, 2)}), after=frozenset(), total=price_total)
    return s, cfg, shared, policy, agent, plan


def test_truth_counts_every_task_of_every_robot():
    s, _cfg, shared, _policy, _agent, plan = setup()
    # robot 0 crosses three times and saves 12 - 6 each time; robot 1 walks along row 4 and saves nothing
    assert true_total_saving(s, shared.engine, plan.before, plan.after) == 18.0


def test_trip_table_on_the_hand_map():
    s, cfg, shared, _policy, _agent, plan = setup()
    table = trip_table(s.zones, shared.engine, plan.before, plan.after, cfg.seed)
    assert sorted(table) == ["east|east", "east|west", "west|east", "west|west"]
    assert table["west|east"] == {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 16.0}
    assert table["east|west"] == {"mean_saving": 6.0, "share_saving": 1.0, "pairs": 16.0}
    assert table["west|west"] == {"mean_saving": 0.0, "share_saving": 0.0, "pairs": 12.0}
    assert trip_table({}, shared.engine, plan.before, plan.after, cfg.seed) == {}


def test_case_holds_what_the_numeric_forecast_uses():
    s, _cfg, shared, policy, agent, plan = setup()
    agent.belief.add_notice(NoticeRecord("n0", 5, "wave 2 is all in the east"))
    case = build_case(agent, plan, 7, 10.0, 12.0, s, shared.engine)
    assert case.ledger == {"robots_known": 2.0, "tasks_per_robot": 3.0, "trips_recorded": 2.0, "trips_expected": 6.0,
                           "saving_on_recorded_trips": 12.0, "saving_on_my_current_trip": 6.0,
                           "my_tasks_done": 0.0, "my_tasks_left": 3.0}
    assert numeric_saving(case.ledger) == 30.0 == policy._forecast(agent, plan)
    assert case.numeric_forecast is True and case.truth is True          # 30 >= 10 and 18 >= 10
    assert case.notices == (("n0", 5, "wave 2 is all in the east"),)
    assert case.zones == {"west": {"rows": (0, 1), "cols": (0, 1)}, "east": {"rows": (0, 1), "cols": (5, 6)}}
    assert (case.robot, case.tick, case.mode, case.kind, case.obstacle) == (0, 7, "push", "pallet", (1, 2))
    assert case.plan_key == (("lift", (1, 2)),) and case.price == 10.0 and case.known_saving == 12.0
    dear = build_case(agent, plan, 7, 20.0, 12.0, s, shared.engine)
    assert dear.truth is False and dear.numeric_forecast is True          # 18 < 20 but 30 >= 20


def test_case_survives_json_and_its_id_ignores_hidden_fields():
    s, _cfg, shared, _policy, agent, plan = setup()
    case = build_case(agent, plan, 7, 10.0, 12.0, s, shared.engine)
    again = case_from_dict(json.loads(json.dumps(case_to_dict(case))))
    assert again == case and len(case.case_id) == 16
    hidden = dataclasses.replace(case, robot=5, tick=99, truth=False, numeric_forecast=False)
    assert build_case(agent, plan, 99, 10.0, 12.0, s, shared.engine).case_id == case.case_id
    assert hidden.case_id == case.case_id
    assert build_case(agent, plan, 7, 11.0, 12.0, s, shared.engine).case_id != case.case_id


def test_plan_key_and_mode_for_a_plain_plan():
    plan = SimpleNamespace(key=((1, 2), (0, 1), 2))
    assert plan_key(plan) == (((1, 2), (0, 1), 2),) and plan_mode(plan) == "push"
