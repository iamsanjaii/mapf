from src.doi.config import SimConfig
from src.doi.policies import FillPolicy, HaulProposal
from src.doi.scenarios import scenario_from_ascii
from src.doi.runner import run_episode

ROWS = ["...#...", "...P..D", "...#...", "...#...", "...#...", "......."]
FOUR = [(1, 4), (1, 2), (1, 4), (1, 2)]


class ForcePolicy(FillPolicy):
    name = "force"
    uses_gossip = True

    def __init__(self, pits, schedule, min_task_idx=0):
        self.pits = tuple(pits)
        self.schedule = dict(schedule)
        self.min_task_idx = min_task_idx
        self.fired = set()

    def on_task_planned(self, agent, info, t):
        pass

    def propose(self, agent, t):
        if (agent.id in self.schedule and agent.id not in self.fired
                and t >= self.schedule[agent.id] and agent.task_idx >= self.min_task_idx):
            self.fired.add(agent.id)
            return HaulProposal(self.pits, 100.0, 9.0)
        return None


def forced(tasks, schedule, claim, starts=None, min_task_idx=0):
    starts = starts or [(1, 2)]
    s = scenario_from_ascii(ROWS, starts, tasks, depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=len(starts), tasks_per_robot=len(tasks[0]),
                    claim=claim, debug_checks=True)
    return run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], schedule, min_task_idx))


def test_full_haul_exact_cost_without_claims():
    r = forced([FOUR], {0: 0}, claim=False, min_task_idx=1)
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.J == 29.0
    assert r.carried_steps == 2 and r.waits == 2
    e = r.edits[0]
    assert (e["pit"], e["B_est"], e["B_real"], e["approval_wait"]) == ((1, 3), 9.0, 13.0, 0)
    assert r.unconfirmed_hauls == 0


def test_robot_uses_pit_after_fill():
    r = forced([FOUR], {0: 0}, claim=False, min_task_idx=1)
    assert (1, 3) in r.trajectory[0][-6:]


def test_other_robot_replans_when_fill_learned():
    tasks = [[(1, 4), (1, 2), (1, 4), (1, 2)], [(1, 2), (1, 4), (1, 2), (1, 4)]]
    r = forced(tasks, {0: 0}, claim=False, starts=[(1, 2), (1, 4)], min_task_idx=1)
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert (1, 3) in r.trajectory[1][-6:]


def test_claim_issued_and_fill_completes():
    r = forced([FOUR], {0: 0}, claim=True, min_task_idx=1)
    assert r.claims["issued"] == 1 and r.fills == 1 and r.unfinished_tasks == 0


def test_two_haulers_one_fill_with_claims():
    s = scenario_from_ascii(ROWS, [(1, 5), (1, 4)], [[(1, 2)], [(1, 2)]], depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=2, tasks_per_robot=1, claim=True, debug_checks=True)
    r = run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], {0: 0, 1: 0}))
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.final_stock[(1, 6)] == 1


def test_abort_returns_bag():
    s = scenario_from_ascii(ROWS, [(1, 5), (1, 4)], [[(1, 2)], [(1, 2)]], depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=2, tasks_per_robot=1, claim=False, debug_checks=True)
    r = run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], {0: 0, 1: 3}))
    assert r.fills == 1 and r.unfinished_tasks == 0
    assert r.final_stock[(1, 6)] == 1          # the second bag was put back
    assert r.wasted_haul_cost >= 0.0


def test_claim_lease_expires_and_reclaims():
    from src.doi.crdt import Claim, ClaimSet
    cs = ClaimSet()
    cs.issue((1, 0), Claim(((1, 3),), 0), expiry=8)
    assert cs.effective((1, 3), 7) is not None
    assert cs.effective((1, 3), 8) is None
    cs.issue((9, 1), Claim(((1, 3),), 1), expiry=20)
    assert cs.effective((1, 3), 9)[0] == (9, 1)


def test_released_claim_is_no_longer_effective_and_stays_released_after_merge():
    from src.doi.crdt import Claim, ClaimSet
    a = ClaimSet()
    a.issue((1, 0), Claim(((1, 3),), 0), expiry=50)
    stale = a.copy()
    a.release((1, 0))
    assert a.effective((1, 3), 5) is None
    a.merge(stale)
    assert a.effective((1, 3), 5) is None


def test_simultaneous_haulers_without_claims_do_not_livelock():
    s = scenario_from_ascii(ROWS, [(1, 5), (1, 4)], [[(1, 2)], [(1, 2)]], depot_stock=2)
    cfg = SimConfig(policy="never", n_robots=2, tasks_per_robot=1, claim=False, debug_checks=True)
    r = run_episode(cfg, scenario=s, policy=ForcePolicy([(1, 3)], {0: 0, 1: 0}))
    assert not r.stalled and r.unfinished_tasks == 0 and r.fills == 1
    assert r.final_stock[(1, 6)] == 1
