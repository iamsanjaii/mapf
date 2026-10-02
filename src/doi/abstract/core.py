"""Ski-rental core on pure numbers: threshold, randomized and predicted rules with partial views.

A savings sequence s[0..T-1] holds what request j would save if the action had already been taken; c > 0 is the
action's cost; views[i] is the set of request indices the deciding agent at request i knows about (a subset of
{0..i}).

* S_i = s[0] + ... + s[i] (prefix sum, inclusive), S_{-1} = 0.
* K_i = sum(s[j] for j in views[i]) (known evidence at request i).
* Rule with threshold multiplier thr: fire at the first i with K_i > 0 and K_i >= thr * c. The action is taken
  before request i is served, so the avoidable cost is S_{i-1} + c if the rule fires at i, and S_{T-1} if it never
  fires.
* OPT = min(c, S_{T-1}).
* Coverage rho = min over i with S_i > 0 of K_i / S_i, or 1.0 if no S_i > 0.
* Deficit D = max over i of (S_i - K_i), or 0.0 if T == 0.
"""
import math
from dataclasses import dataclass
from typing import AbstractSet, FrozenSet, List, Optional, Sequence

from src.doi.rng import u01

E_RATIO = math.e / (math.e - 1.0)


@dataclass(frozen=True)
class SingleOutcome:
    fire: Optional[int]     # request index at which the action is taken, None if never
    alg: float              # avoidable cost paid by the rule
    opt: float              # min(c, total saving)


def prefix(s: Sequence[float]) -> List[float]:
    out: List[float] = []
    total = 0.0
    for v in s:
        total += v
        out.append(total)
    return out


def known(s: Sequence[float], views: Sequence[AbstractSet[int]]) -> List[float]:
    return [float(sum(s[j] for j in v)) for v in views]


def opt_single(s: Sequence[float], c: float) -> float:
    return min(float(c), float(sum(s)))


def _check(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]]) -> None:
    if any(v < 0 for v in s):
        raise ValueError("savings must be non-negative")
    if c <= 0:
        raise ValueError("cost must be positive")
    for i, v in enumerate(views):
        if any(j > i or j < 0 for j in v):
            raise ValueError(f"view {i} contains an index outside 0..{i}")


def run_threshold(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]], thr: float) -> SingleOutcome:
    _check(s, c, views)
    S = prefix(s)
    K = known(s, views)
    opt = opt_single(s, c)
    for i in range(len(s)):
        if K[i] > 0 and K[i] >= thr * c:
            return SingleOutcome(i, (S[i - 1] if i > 0 else 0.0) + c, opt)
    return SingleOutcome(None, S[-1] if S else 0.0, opt)


def run_predicted(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]], predicted_total: float,
                  lam: float) -> SingleOutcome:
    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    thr = lam if predicted_total >= c else 1.0 / lam
    return run_threshold(s, c, views, thr)


def expected_randomized_full(s: Sequence[float], c: float) -> float:
    _check(s, c, [])

    def P(a: float, b: float) -> float:
        return (math.exp(min(max(b, 0.0), 1.0)) - math.exp(min(max(a, 0.0), 1.0))) / (math.e - 1.0)

    S = prefix(s)
    total = S[-1] if s else 0.0
    expected = 0.0
    for i in range(len(s)):
        before = S[i - 1] if i > 0 else 0.0
        lo = before / c
        hi = S[i] / c
        if S[i] > 0 and hi > lo:
            expected += P(lo, hi) * (before + c)
    expected += P(total / c, 1.0) * total
    return expected


def randomized_draw(seed: int, draw: int) -> float:
    return math.log(1.0 + u01(seed, draw, 31) * (math.e - 1.0))


def run_randomized(s: Sequence[float], c: float, views: Sequence[AbstractSet[int]], z: float) -> SingleOutcome:
    return run_threshold(s, c, views, z)


def coverage(s: Sequence[float], views: Sequence[AbstractSet[int]]) -> float:
    S = prefix(s)
    K = known(s, views)
    ratios = [K[i] / S[i] for i in range(len(s)) if S[i] > 0]
    return min(ratios) if ratios else 1.0


def deficit(s: Sequence[float], views: Sequence[AbstractSet[int]]) -> float:
    if not s:
        return 0.0
    S = prefix(s)
    K = known(s, views)
    return max(S[i] - K[i] for i in range(len(s)))


def bound_coverage(theta: float, rho: float) -> float:
    return (1.0 + theta / rho) / min(1.0, theta)


def bound_deficit(theta: float, d: float, c: float) -> float:
    return (theta + 1.0 + d / c) / min(1.0, theta)


def bound_predicted(lam: float, rho: float) -> float:
    return 1.0 + 1.0 / (lam * rho)


def views_full(T: int) -> List[FrozenSet[int]]:
    return [frozenset(range(i + 1)) for i in range(T)]


def views_own(agents: Sequence[int]) -> List[FrozenSet[int]]:
    seen: dict = {}
    out: List[FrozenSet[int]] = []
    for i, g in enumerate(agents):
        seen.setdefault(g, []).append(i)
        out.append(frozenset(seen[g]))
    return out


def views_delay(agents: Sequence[int], delta: int) -> List[FrozenSet[int]]:
    return [frozenset(j for j in range(i + 1) if j <= i - delta or agents[j] == agents[i])
            for i in range(len(agents))]


def views_sample(agents: Sequence[int], p: float, seed: int) -> List[FrozenSet[int]]:
    return [frozenset(j for j in range(i + 1) if agents[j] == agents[i] or u01(seed, agents[i], j, 41) < p)
            for i in range(len(agents))]


def round_robin(T: int, n: int) -> List[int]:
    return [i % n for i in range(T)]
