"""Two-channel lossy, range-limited broadcast network (traffic channel and ledger channel)."""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.doi.config import SimConfig
from src.doi.rng import u01

Pos = Tuple[int, int]
KIND_ID = {"STATE": 1, "INTENT": 2}


@dataclass(frozen=True)
class Message:
    sender: int
    kind: str
    payload: Any
    units: int
    sent_at: int


@dataclass
class NetStats:
    broadcasts: int = 0
    transmissions: int = 0
    dropped: int = 0
    units: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"broadcasts": self.broadcasts, "transmissions": self.transmissions,
                "dropped": self.dropped, "units": self.units}


class Network:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.stats = NetStats()
        self.traffic_stats = NetStats()
        self._queue: Dict[int, List[Tuple[int, Message]]] = {}

    def _channel(self, kind: str):
        if kind == "INTENT":
            return self.cfg.r_traffic, self.cfg.loss_traffic, 1, self.traffic_stats
        return self.cfg.r_comm, self.cfg.loss, self.cfg.latency, self.stats

    def send(self, msg: Message, t: int, positions: Dict[int, Pos]) -> None:
        radius, loss, latency, stats = self._channel(msg.kind)
        stats.broadcasts += 1
        if radius <= 0 or msg.sender not in positions:
            return
        sp = positions[msg.sender]
        for j in sorted(positions):
            if j == msg.sender:
                continue
            if abs(sp[0] - positions[j][0]) + abs(sp[1] - positions[j][1]) > radius:
                continue
            stats.transmissions += 1
            stats.units += msg.units
            if u01(self.cfg.seed, t, msg.sender, j, KIND_ID[msg.kind]) < loss:
                stats.dropped += 1
                continue
            self._queue.setdefault(t + latency, []).append((j, msg))

    def deliver(self, t: int) -> Dict[int, List[Message]]:
        out: Dict[int, List[Message]] = {}
        for receiver, msg in self._queue.pop(t, []):
            out.setdefault(receiver, []).append(msg)
        for msgs in out.values():
            msgs.sort(key=lambda m: (m.sent_at, m.sender))
        return out
