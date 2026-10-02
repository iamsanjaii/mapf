"""Small statistics helpers (numpy only): bootstrap CI, paired sign-flip test, Spearman correlation."""
from typing import Callable, Sequence, Tuple

import numpy as np


def bootstrap_ci(x: Sequence[float], stat: Callable = np.median, n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    stats = stat(arr[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(arr)), float(lo), float(hi)


def signflip_pvalue(a: Sequence[float], b: Sequence[float], n: int = 20000, seed: int = 0) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + n))


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])
