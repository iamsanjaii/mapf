import pytest
from src.doi.abstract.core import (
    bound_guarded_consistency, bound_predicted, coverage, guarded_wait, round_robin, run_guarded, run_predicted,
    run_threshold, views_delay, views_full, views_own, views_sample,
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
        lam = rng.choice([0.25, 0.5, 1.0])
        delay = rng.randint(0, 20)
        yield s, c, views, lam, delay


def test_guarded_hand_case():
    s, views = [1.0] * 30, views_full(30)
    now = run_guarded(s, 10.0, views, 0.5, True, 0)
    assert now.fire == 4 and now.alg == 14.0 and now.opt == 10.0
    late = run_guarded(s, 10.0, views, 0.5, True, 3)
    assert late.fire == 7 and late.alg == 17.0
    assert guarded_wait(s, 10.0, views, 0.5, 3) == 3.0
    assert bound_guarded_consistency(0.5, 3.0, 10.0) == pytest.approx(1.8)


def test_guarded_delay_zero_equals_predicted():
    for s, c, views, lam, _delay in _cases("guard-eq"):
        for says_yes in (True, False):
            assert run_guarded(s, c, views, lam, says_yes, 0) == run_predicted(s, c, views, c if says_yes else 0.0, lam)


def test_guarded_robustness_bound():
    checked = 0
    for s, c, views, lam, delay in _cases("guard-rob"):
        rho = coverage(s, views)
        for says_yes in (True, False, None):
            out = run_guarded(s, c, views, lam, says_yes, delay)
            if out.opt > 0 and rho > 0:
                checked += 1
                assert out.alg <= bound_predicted(lam, rho) * out.opt + 1e-9
    assert checked > 1000


def test_guarded_consistency_bound():
    for s, c, views, lam, delay in _cases("guard-con", full_only=True):
        if sum(s) >= c:                                   # a right "yes"
            out = run_guarded(s, c, views, lam, True, delay)
            w = guarded_wait(s, c, views, lam, delay)
            assert out.alg <= bound_guarded_consistency(lam, w, c) * out.opt + 1e-9
        else:                                             # a right "no"
            out = run_guarded(s, c, views, lam, False, delay)
            assert out.alg == out.opt


def test_guarded_no_answer_is_the_classical_rule():
    for s, c, views, lam, delay in _cases("guard-none"):
        assert run_guarded(s, c, views, lam, None, delay) == run_threshold(s, c, views, 1.0)


def test_guarded_rejects_bad_input():
    with pytest.raises(ValueError):
        run_guarded([1.0], 2.0, views_full(1), 0.0, True)
    with pytest.raises(ValueError):
        run_guarded([1.0], 2.0, views_full(1), 0.5, True, -1)
