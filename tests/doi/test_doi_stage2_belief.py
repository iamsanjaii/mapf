"""Stage 2 belief: which slots are known to be full, and a filled pit that stale gossip cannot bring back."""
from src.doi.agent import RobotAgent
from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.evidence import EvidenceEngine
from src.doi.policies import Shared, make_policy
from src.doi.scenarios import scenario_from_ascii
from src.doi.world import Observation


def test_slot_news_merges_by_union_and_is_idempotent():
    a, b = BeliefState(0, {}), BeliefState(1, {})
    b.slots_full.add((0, 0))
    a.slots_full.add((2, 2))
    assert a.merge(b) and a.slots_full.items() == frozenset({(0, 0), (2, 2)})
    assert not a.merge(b)                                         # nothing new the second time
    b.merge(a)
    assert a.canonical() == b.canonical()


def test_slot_news_travels_in_a_delta():
    a, b = BeliefState(0, {}), BeliefState(1, {})
    v = b.version
    b.slots_full.add((3, 1))
    a.merge(b.delta_since(v))
    assert (3, 1) in a.slots_full


def test_a_filled_pit_stays_filled_when_a_stale_belief_gossips_it():
    pit = {(0, 1): "pit"}
    stale, fresh = BeliefState(0, pit), BeliefState(1, pit)
    fresh.observe_cell((0, 1), False, 50)                         # seen filled at tick 50
    stale.merge(fresh)
    assert stale.status((0, 1)) == "refuted" and (0, 1) not in stale.believed_blocked()
    fresh.merge(BeliefState(2, pit))                              # a robot that never saw the fill gossips back
    assert fresh.status((0, 1)) == "refuted"


def test_sensing_a_full_slot_is_learned():
    s = scenario_from_ascii(["TC.", "..."], [(1, 1)], [[(1, 2)]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1)
    agent = RobotAgent(0, s, cfg, make_policy(cfg), Shared(engine=EvidenceEngine(s.grid, 99.0)))
    obs = Observation(frozenset(), frozenset(), (), frozenset(), frozenset(), slots_full=frozenset({(0, 0)}))
    agent.sense(obs, 5)
    assert (0, 0) in agent.belief.slots_full
