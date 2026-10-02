from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.crdt import RentRecord
from src.doi.evidence import EvidenceEngine
from src.doi.policies import Shared, make_policy
from src.doi.pushplan import best_push_plan, candidate_plans
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario, scenario_from_ascii

TOY = ["...#...", "...L...", "...#...", "...#...", "...#...", "......."]


def plans(rows, pos, goal, kappa=4.0, fee=1.0, **kw):
    s = scenario_from_ascii(rows, [pos], [[goal]])
    eng = EvidenceEngine(s.grid, 144.0)
    blocked = frozenset(s.obstacles)
    return candidate_plans(s.grid, eng.distance, blocked, sorted(s.obstacles), lambda c: s.obstacles[c], pos, goal,
                           kappa, fee, 6, 144.0, **kw)


def test_the_cheapest_plan_pushes_the_pallet_through_the_gap_and_clear_of_it():
    p = plans(TOY, (1, 2), (1, 4))[0]
    assert (p.obstacle, p.direction, p.approach, p.steps, p.landing, p.end) == \
        ((1, 3), (0, 1), (1, 2), 2, (1, 5), (1, 4))
    assert p.walk_in == 0 and p.push_cost == 9.0 and p.walk_on == 0 and p.total == 9.0
    assert p.before == frozenset({(1, 3)}) and p.after == frozenset({(1, 5)})
    assert p.landing_after(1) == (1, 4) and p.landing_after(2) == (1, 5)


def test_one_step_is_useless_in_a_one_cell_gap_so_the_run_goes_on_to_clear_the_gap():
    p = plans(TOY, (1, 2), (1, 6))[0]
    assert p.steps == 2 and p.landing == (1, 5)           # k=1 would leave the pallet blocking the robot's way out


def test_a_heavier_kind_costs_more_per_step():
    light = plans(["..C.."], (0, 1), (0, 4), fee=0.0)[0]
    heavy = plans(["..S.."], (0, 1), (0, 4), fee=0.0)[0]
    assert (light.push_cost, heavy.push_cost) == (light.steps * 2.0, heavy.steps * 8.0)


def test_no_plan_when_the_cell_beyond_is_a_wall_or_there_is_no_approach():
    assert plans(["..L#.."], (0, 1), (0, 5)) == []         # wall behind it, edges above and below
    assert plans(["#L#"], (0, 1), (0, 1)) == []


def test_a_plan_never_lands_on_the_goal_and_the_dead_end_guard_refuses_corners():
    assert all(p.landing != (1, 4) for p in plans(TOY, (1, 2), (1, 4)))
    rows = [".....", ".L...", "....."]
    corner = plans(rows, (1, 0), (2, 4))
    assert all(p.landing != (0, 0) for p in corner)
    unguarded = plans([".L..", "...."], (0, 0), (1, 3), dead_end_guard=False)
    guarded = plans([".L..", "...."], (0, 0), (1, 3))
    assert len(unguarded) >= len(guarded)


def test_cooldown_cells_are_skipped():
    assert plans(TOY, (1, 2), (1, 4), skip=frozenset({(1, 3)})) == []


class FakeAgent:
    id, task_idx, plan_infos = 0, 0, []

    def __init__(self, belief, pos, goal):
        self.belief, self.pos, self.goal, self.tasks = belief, pos, goal, [goal] * 10


def test_the_policy_prefers_the_push_that_does_not_block_the_gap():
    """The cheapest push for the robot would park the pallet in the only cell that leads to the gap."""
    rows = [".........#........", ".........#........", ".........#........", ".......L..........",
            ".........#........", ".........#........", ".........#........", "........."+"#........"]
    s = scenario_from_ascii(rows, [(3, 6)], [[(3, 7)]])
    cfg = SimConfig(policy="rof", n_robots=1, tasks_per_robot=10)
    pol = make_policy(cfg)
    pol.prepare(s, cfg, Shared(engine=EvidenceEngine(s.grid, cfg.unreachable_cost_for(8, 18))))
    b = BeliefState(0, s.obstacles)
    for k in range(5):                                    # traffic that goes through the gap at (3, 9), via (3, 8)
        b.add_record(RentRecord(1, k, (3, 4), (3, 14), 0, 0.0))
    agent = FakeAgent(b, (3, 6), (3, 7))
    info = type("I", (), {"d_open": 1, "d_block": 144.0, "rent": 143})()
    blocked = b.believed_blocked()
    eng = pol.shared.engine
    found = candidate_plans(s.grid, eng.distance, blocked, [(3, 7)], b.kind_of, agent.pos, agent.goal,
                            cfg.kappa, cfg.fee, cfg.push_max, 144.0)
    by_dir = {p.direction: p for p in found}
    east, south = by_dir[(0, 1)], by_dir[(1, 0)]
    assert east.landing == (3, 8) and east.total < south.total   # east is the cheaper push for this robot
    assert pol.assess(agent, east, info) is None          # but it would cut the fleet off from the gap
    assert pol.assess(agent, south, info) is not None


def test_a_run_with_an_obstacle_parked_on_a_goal_cell_still_finishes():
    """Seed 203 used to end with three tasks unfinished: a push parked a pallet on another robot's goal."""
    cfg = SimConfig(scenario="single_block", seed=203, n_robots=8, tasks_per_robot=10, policy="rof")
    r = run_episode(cfg, scenario=build_scenario(cfg))
    assert not r.stalled and r.unfinished_tasks == 0
