from src.doi.abstract.core import bound_predicted, round_robin, run_threshold, views_own
from src.doi.abstract.instance import g1, random_small_instance
from src.doi.abstract.offline import exact_opt
from src.doi.abstract.online import Rule, run_online


def test_g1_two_step_plan_reaches_the_optimum():
    for m, cost in ((1, 10), (2, 16), (4, 28)):
        assert run_online(g1(m), Rule("threshold", bundle_max=2)).cost == cost
    res = run_online(g1(1), Rule("threshold", bundle_max=2))
    assert len(res.fires) == 1
    assert res.fires[0]["request"] == 0
    assert res.fires[0]["plan"] == [((1, 2), (0, -1), 1), ((1, 1), (1, 0), 1)]


def test_g1_single_steps_cost_eleven():
    res = run_online(g1(1), Rule("threshold", bundle_max=1))
    assert res.cost == 11
    assert [f["plan"] for f in res.fires] == [[((1, 2), (0, 1), 2)], [((1, 4), (-1, 0), 1)]]
    assert [f["cost"] for f in res.fires] == [3, 2]
    for m, cost in ((2, 17), (4, 29)):
        assert run_online(g1(m), Rule("threshold", bundle_max=1)).cost == cost


def test_never_rule_equals_never_cost():
    for m in range(1, 5):
        assert run_online(g1(m), Rule("never")).cost == 12 * m


def test_two_step_plan_matches_core():
    expected = {1: [6] + [16] * 11, 2: [6, 12] + [22] * 10, 3: [6, 12, 18] + [28] * 9}
    for n_agents in (1, 2, 3):
        for m in range(1, 13):
            res = run_online(g1(m, n_agents, fee=4), Rule("threshold", bundle_max=2, view="own"))
            core = run_threshold([6] * m, 10, views_own(round_robin(m, n_agents)), 1.0).alg
            assert res.cost - 6 * m == core == expected[n_agents][m - 1]
            for f in res.fires:
                assert len(f["plan"]) == 2 and f["cost"] == 10


def test_online_never_beats_exact_opt():
    for seed in range(10):
        for k in (1, 2):
            inst = random_small_instance(seed, k, 8, 1)
            opt = exact_opt(inst).cost
            for mode in ("never", "threshold", "randomized", "predicted"):
                assert run_online(inst, Rule(mode, bundle_max=2)).cost >= opt - 1e-9


def test_predicted_values_on_g1():
    for predictor, want in (("oracle", [6, 10, 13, 13]), ("inverted", [8, 19, 26, 28])):
        got = [run_online(g1(m, fee=4), Rule("predicted", lam=0.5, bundle_max=2, predictor=predictor)).cost - 6 * m
               for m in range(1, 5)]
        assert got == want


def test_multi_candidate_counterexample_on_g1():
    res = run_online(g1(6, fee=4), Rule("predicted", lam=0.5, bundle_max=2, predictor="inverted"))
    assert res.cost - 6 * 6 == 33
    assert res.cost - 6 * 6 > bound_predicted(0.5, 1.0) * 10
