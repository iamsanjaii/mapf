"""Counter-based deterministic randomness: every draw is a pure function of (seed, keys)."""
import hashlib
import random

MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def mix(seed: int, *keys: int) -> int:
    h = splitmix64(seed & MASK)
    for key in keys:
        h = splitmix64((h ^ (key & MASK)) & MASK)
    return h


def u01(seed: int, *keys: int) -> float:
    return (mix(seed, *keys) >> 11) / float(1 << 53)


def stream(seed: int, name: str) -> random.Random:
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return random.Random(mix(seed, key))
