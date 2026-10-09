import pytest
from src.doi.rng import splitmix64, mix, u01, stream
from src.doi.config import SimConfig


def test_splitmix64_reference_vector():
    assert splitmix64(0) == 0xE220A8397B1DCDAF


def test_u01_deterministic_and_in_range():
    a = u01(7, 1, 2, 3)
    assert a == u01(7, 1, 2, 3)
    assert 0.0 <= a < 1.0
    assert u01(7, 1, 2, 3) != u01(7, 1, 2, 4)
    assert u01(7, 1, 2, 3) != u01(8, 1, 2, 3)


def test_u01_roughly_uniform():
    xs = [u01(1, i) for i in range(100000)]
    assert abs(sum(xs) / len(xs) - 0.5) < 0.01
    assert abs(sum(1 for x in xs if x < 0.3) / len(xs) - 0.3) < 0.01


def test_stream_reproducible_and_named():
    a = [stream(5, "tasks-0").random() for _ in range(3)]
    b = [stream(5, "tasks-0").random() for _ in range(3)]
    c = [stream(5, "tasks-1").random() for _ in range(3)]
    assert a == b and a != c


def test_config_validation():
    SimConfig()
    with pytest.raises(ValueError):
        SimConfig(latency=0)
    with pytest.raises(ValueError):
        SimConfig(loss=1.5)
    with pytest.raises(ValueError):
        SimConfig(policy="nope")
    with pytest.raises(ValueError):
        SimConfig(intake="gpt")
    with pytest.raises(ValueError):
        SimConfig(push_max=0)
    SimConfig(intake="oracle", policy="free", push_max=3)
    assert SimConfig().replace(seed=3).seed == 3
    assert SimConfig().unreachable_cost_for(15, 21) == 144.0
    assert SimConfig(unreachable_cost=50.0).unreachable_cost_for(15, 21) == 50.0
