from src.doi.abstract.instance import Model, g1, instance_from_ascii, random_small_instance

EAST, WEST, NORTH, SOUTH = (0, 1), (0, -1), (-1, 0), (1, 0)


def _act(model, x, key):
    return next(a for a in model.legal_actions(x) if a.key() == key)


def test_g1_serving_costs():
    inst = g1(1)
    model = Model(inst)
    x = inst.initial()
    assert model.serve(x, 0) == 12
    expected = {1: 12, 2: 8, 3: 8, 4: 48}
    for k, cost in expected.items():
        assert model.serve(model.apply(x, _act(model, x, ((1, 2), EAST, k))), 0) == cost
    x1 = model.apply(x, _act(model, x, ((1, 2), EAST, 1)))
    x2 = model.apply(x1, _act(model, x1, ((1, 3), SOUTH, 1)))
    assert model.serve(x2, 0) == 6


def test_g1_legal_actions():
    inst = g1(1)
    model = Model(inst)
    x = inst.initial()
    keys = [a.key() for a in model.legal_actions(x)]
    assert keys == sorted([((1, 2), EAST, k) for k in range(1, 5)] + [((1, 2), WEST, k) for k in range(1, 3)])
    x1 = model.apply(x, _act(model, x, ((1, 2), EAST, 1)))
    keys1 = {a.key() for a in model.legal_actions(x1)}
    assert ((1, 3), NORTH, 1) not in keys1
    assert ((1, 3), SOUTH, 1) in keys1
    assert _act(model, x1, ((1, 3), SOUTH, 1)).cost == 2
    xw = model.apply(x, _act(model, x, ((1, 2), WEST, 1)))
    keysw = {a.key() for a in model.legal_actions(xw)}
    assert ((1, 1), NORTH, 1) not in keysw and ((1, 1), SOUTH, 1) in keysw


def test_blocked_endpoint_costs_unreachable():
    inst = g1(1)
    model = Model(inst)
    x = (((1, 6), "pallet"),)
    assert model.serve(x, 0) == inst.unreachable() == 48
    assert model.serve((((1, 0), "pallet"),), 0) == 48


def test_action_cost_uses_kind_weight():
    inst = instance_from_ascii(["......", "..C...", "......"], [((0, 0), (2, 5))], kappa=4, fee=1)
    model = Model(inst)
    a = _act(model, inst.initial(), ((1, 2), EAST, 2))
    assert a.kind == "crate" and a.cost == 5 and a.landing == (1, 4)


def test_random_small_instance_is_deterministic_and_valid():
    for seed in range(5):
        for k in (1, 2, 3):
            a = random_small_instance(seed, k, 12, 3)
            b = random_small_instance(seed, k, 12, 3)
            assert a.requests == b.requests and a.agents == b.agents
            assert len(a.requests) == 12
            for o, d in a.requests:
                assert o not in a.obstacles and d not in a.obstacles
                assert not a.grid.get(*o) and not a.grid.get(*d)
