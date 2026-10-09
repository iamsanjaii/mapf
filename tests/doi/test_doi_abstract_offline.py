import pytest
from src.doi.abstract.instance import Model, g1, random_small_instance
from src.doi.abstract.offline import StateSpaceTooLarge, exact_opt, vanish_lower_bound


def test_g1_exact_optimum():
    assert exact_opt(g1(1)).cost == 10
    assert exact_opt(g1(2)).cost == 16
    assert exact_opt(g1(4)).cost == 28
    res = exact_opt(g1(1))
    assert len(res.schedule) == 2
    (i1, a1), (i2, a2) = res.schedule
    assert i1 == 0 and i2 == 0
    assert a1.steps == 1 and a1.direction in ((0, 1), (0, -1))
    assert a2.steps == 1 and a2.direction == (1, 0)
    assert res.n_states == 26


def test_g1_vanish_bound():
    assert vanish_lower_bound(g1(1)) == 8
    assert vanish_lower_bound(g1(2)) == 14


def test_never_is_an_upper_bound_and_vanish_a_lower_bound():
    for seed in range(10):
        for k in (1, 2):
            inst = random_small_instance(seed, k, 8, 1)
            model = Model(inst)
            never = sum(model.serve(inst.initial(), j) for j in range(len(inst.requests)))
            assert vanish_lower_bound(inst) <= exact_opt(inst).cost + 1e-9
            assert exact_opt(inst).cost <= never + 1e-9


def test_state_limit_raises():
    with pytest.raises(StateSpaceTooLarge):
        exact_opt(g1(1), max_states=5)


def test_schedule_replays_to_its_cost():
    for seed in range(5):
        inst = random_small_instance(seed, 2, 8, 1)
        model = Model(inst)
        res = exact_opt(inst)
        x = inst.initial()
        total = 0.0
        sched = list(res.schedule)
        for j in range(len(inst.requests)):
            while sched and sched[0][0] == j:
                _, a = sched.pop(0)
                assert a.key() in {b.key() for b in model.legal_actions(x)}
                x = model.apply(x, a)
                total += a.cost
            total += model.serve(x, j)
        assert not sched
        assert total == res.cost
