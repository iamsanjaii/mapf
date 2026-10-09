from src.doi.config import SimConfig
from src.doi.network import Message, Network

POS = {0: (0, 0), 1: (0, 5), 2: (0, 20)}


def msg(sender=0, kind="STATE"):
    return Message(sender, kind, {"x": 1}, 3, 0)


def test_range_and_latency():
    n = Network(SimConfig(r_comm=8.0, latency=2))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {}
    out = n.deliver(2)
    assert list(out) == [1] and out[1][0].sender == 0


def test_infinite_range_reaches_all():
    n = Network(SimConfig(r_comm=float("inf"), latency=1))
    n.send(msg(), 0, POS)
    assert sorted(n.deliver(1)) == [1, 2]


def test_zero_range_reaches_none():
    n = Network(SimConfig(r_comm=0.0))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {} and n.stats.transmissions == 0


def test_loss_rate_and_determinism():
    cfg = SimConfig(r_comm=float("inf"), loss=0.3, seed=9)
    n = Network(cfg)
    for t in range(5000):
        n.send(msg(), t, {0: (0, 0), 1: (0, 1)})
    assert abs(n.stats.dropped / n.stats.transmissions - 0.3) < 0.02
    m = Network(cfg)
    for t in range(5000):
        m.send(msg(), t, {0: (0, 0), 1: (0, 1)})
    assert m.stats.dropped == n.stats.dropped


def test_full_loss_equals_no_comm():
    n = Network(SimConfig(r_comm=float("inf"), loss=1.0))
    n.send(msg(), 0, POS)
    assert n.deliver(1) == {} and n.stats.dropped == n.stats.transmissions == 2


def test_stats_units():
    n = Network(SimConfig(r_comm=float("inf")))
    n.send(msg(), 0, POS)
    assert n.stats.broadcasts == 1 and n.stats.units == 6


def test_intent_uses_traffic_channel():
    n = Network(SimConfig(r_comm=0.0, loss=1.0, r_traffic=8.0, loss_traffic=0.0, latency=3))
    n.send(msg(kind="INTENT"), 0, POS)
    out = n.deliver(1)
    assert list(out) == [1] and out[1][0].kind == "INTENT"
    assert n.stats.transmissions == 0 and n.traffic_stats.transmissions == 1
