import pytest
from src.doi.abstract.core import (
    E_RATIO, bound_coverage, bound_deficit, bound_predicted, coverage, deficit, expected_randomized_full,
    opt_single, prefix, randomized_draw, round_robin, run_predicted, run_randomized, run_threshold, views_delay,
    views_full, views_own, views_sample,
)
from src.doi.rng import stream


def _cases(name, full_only=False, n_cases=500):
    rng = stream(0, name)
    for k in range(n_cases):
        T = rng.randint(1, 60)
        s = [float(rng.randint(0, 5)) for _ in range(T)]
        c = float(rng.randint(1, 40))
        n = rng.randint(1, 8)
        agents = round_robin(T, n)
        kind = "full" if full_only else rng.choice(["full", "own", "delay", "sample"])
        if kind == "full":
            views = views_full(T)
        elif kind == "own":
            views = views_own(agents)
        elif kind == "delay":
            views = views_delay(agents, rng.randint(1, 10))
        else:
            views = views_sample(agents, rng.choice([0.25, 0.5, 0.9]), k)
        theta = rng.choice([0.5, 1.0, 2.0])
        yield s, c, views, theta


def test_threshold_hand_case():
    out = run_threshold([3, 3, 3, 3], 7, views_full(4), 1.0)
    assert out.fire == 2 and out.alg == 13 and out.opt == 7


def test_threshold_never_fires():
    out = run_threshold([1, 1], 7, views_full(2), 1.0)
    assert out.fire is None and out.alg == 2 and out.opt == 2


def test_isolation_hand_case():
    agents = round_robin(8, 2)
    s = [1.0] * 8
    views = views_own(agents)
    out = run_threshold(s, 4, views, 1.0)
    assert out.fire == 6 and out.alg == 10 and out.opt == 4
    assert coverage(s, views) == 0.5


def test_bound_coverage_holds_on_random_sequences():
    for s, c, views, theta in _cases("core-cov"):
        out = run_threshold(s, c, views, theta)
        if out.opt <= 0:
            continue
        rho = coverage(s, views)
        if rho > 0:
            assert out.alg <= bound_coverage(theta, rho) * out.opt + 1e-9
        assert out.alg <= bound_deficit(theta, deficit(s, views), c) * out.opt + 1e-9


def test_full_information_ratio_is_at_most_two():
    for s, c, views, _ in _cases("core-cov", full_only=True):
        out = run_threshold(s, c, views, 1.0)
        assert out.alg <= 2 * out.opt + 1e-9


def test_isolation_lower_bound_approaches_n_plus_one():
    agents = round_robin(4000, 4)
    out = run_threshold([1.0] * 4000, 400, views_own(agents), 1.0)
    assert 4.9 <= out.alg / out.opt <= 5.0 + 1e-9


def test_randomized_expectation_bound():
    for s, c, _, _ in _cases("core-cov", full_only=True):
        assert expected_randomized_full(s, c) <= E_RATIO * opt_single(s, c) + 1e-9


def test_randomized_expectation_matches_sampling():
    s, c = [1.0] * 30, 10.0
    views = views_full(30)
    mean = sum(run_randomized(s, c, views, randomized_draw(7, d)).alg for d in range(20000)) / 20000
    assert abs(mean - expected_randomized_full(s, c)) < 0.1


def test_predicted_consistency_and_robustness():
    rng = stream(0, "core-pred")
    for s, c, views, _ in _cases("core-cov", full_only=True):
        lam = rng.choice([0.25, 0.5, 1.0])
        total = prefix(s)[-1]
        good = run_predicted(s, c, views, total, lam)
        bad = run_predicted(s, c, views, 0.0 if total >= c else 2 * c + 1, lam)
        assert good.alg <= (1 + lam) * good.opt + 1e-9
        for out in (good, bad):
            assert out.alg <= bound_predicted(lam, 1.0) * out.opt + 1e-9


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        run_threshold([1, -1], 3, views_full(2), 1.0)
    with pytest.raises(ValueError):
        run_threshold([1, 1], 0, views_full(2), 1.0)
    with pytest.raises(ValueError):
        run_threshold([1, 1], 3, [frozenset({0, 1}), frozenset({0, 1})], 1.0)
    with pytest.raises(ValueError):
        run_predicted([1, 1], 3, views_full(2), 2.0, 0.0)
