"""Online rules on the abstract model: never, threshold, randomized and predicted, over single or two-step plans.

Before each request the serving agent may take a plan (one action, or two in sequence) whose known evidence,
the saving it would have made on the requests in the agent's view, is positive and reaches threshold * cost.
"""
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.doi.abstract.core import views_delay, views_full, views_own, views_sample
from src.doi.abstract.instance import Action, Config, Instance, Model
from src.doi.rng import stream, u01

MAX_FIRES_PER_REQUEST = 10


@dataclass(frozen=True)
class Rule:
    mode: str                 # "never" | "threshold" | "randomized" | "predicted"
    theta: float = 1.0
    lam: float = 0.5
    bundle_max: int = 1       # 1: single actions; 2: also two-step plans
    view: str = "full"        # "full" | "own" | "delay:<int>" | "sample:<float>"
    predictor: str = "oracle" # "oracle" | "inverted" | "noisy:<float>"   (predicted mode only)
    seed: int = 0


@dataclass
class OnlineResult:
    cost: float
    action_cost: float
    serve_cost: float
    fires: List[dict]         # one per plan taken


def views_for(rule: Rule, agents: Sequence[int]) -> List[FrozenSet[int]]:
    if rule.view == "full":
        return views_full(len(agents))
    if rule.view == "own":
        return views_own(agents)
    if rule.view.startswith("delay:"):
        return views_delay(agents, int(rule.view.split(":", 1)[1]))
    if rule.view.startswith("sample:"):
        return views_sample(agents, float(rule.view.split(":", 1)[1]), rule.seed)
    raise ValueError(f"unknown view {rule.view!r}")


def candidates(model: Model, x: Config, j: int, bundle_max: int) -> List[Tuple[Action, ...]]:
    bundle = model.bundle_cells(x, j)
    singles = [a for a in model.legal_actions(x) if a.obstacle in bundle]
    plans: List[Tuple[Action, ...]] = [(a,) for a in singles]
    if bundle_max == 2:
        pairs: List[Tuple[Action, ...]] = []
        for a1 in singles:
            x1 = model.apply(x, a1)
            bundle1 = model.bundle_cells(x1, j)
            pairs.extend((a1, a2) for a2 in model.legal_actions(x1) if a2.obstacle in bundle1)
        pairs.sort(key=lambda p: (p[0].key(), p[1].key()))
        plans.extend(pairs)
    return plans


def _z(rule: Rule, p: Tuple[int, int]) -> float:
    return math.log(1 + u01(rule.seed, p[0], p[1], 31) * (math.e - 1))


def run_online(inst: Instance, rule: Rule) -> OnlineResult:
    model = Model(inst)
    T = len(inst.requests)
    x = inst.initial()
    views = views_for(rule, inst.agents)
    frozen: Dict[Tuple, bool] = {}
    action_cost = serve_cost = 0.0
    fires: List[dict] = []

    def saving(x0: Config, y0: Config, js) -> float:
        return sum(model.serve(x0, j) - model.serve(y0, j) for j in js)

    for i in range(T):
        g = inst.agents[i]
        for _ in range(MAX_FIRES_PER_REQUEST):
            if rule.mode == "never":
                break
            best: Optional[Tuple[float, Tuple, Tuple[Action, ...], Config, float, float]] = None
            for plan in candidates(model, x, i, rule.bundle_max):
                y = x
                for a in plan:
                    y = model.apply(y, a)
                c = sum(a.cost for a in plan)
                E = saving(x, y, views[i])
                if rule.mode == "threshold":
                    thr = rule.theta
                elif rule.mode == "randomized":
                    thr = max(_z(rule, a.obstacle) for a in plan)
                elif rule.mode == "predicted":
                    key = (g, tuple(a.key() for a in plan))
                    if key not in frozen:
                        oracle = saving(x, y, range(T))
                        if rule.predictor == "oracle":
                            predicted = oracle
                        elif rule.predictor == "inverted":
                            predicted = 0.0 if oracle >= c else 2 * c + 1
                        elif rule.predictor.startswith("noisy:"):
                            sigma = float(rule.predictor.split(":", 1)[1])
                            predicted = oracle * math.exp(sigma * stream(rule.seed, f"pred-{g}-{key}").gauss(0, 1))
                        else:
                            raise ValueError(f"unknown predictor {rule.predictor!r}")
                        frozen[key] = predicted >= c
                    thr = rule.lam if frozen[key] else 1 / rule.lam
                else:
                    raise ValueError(f"unknown mode {rule.mode!r}")
                if E > 0 and E >= thr * c:
                    tie = tuple(a.key() for a in plan)
                    if best is None or E - c > best[0] or (E - c == best[0] and tie < best[1]):
                        best = (E - c, tie, plan, y, c, E)
            if best is None:
                break
            _, _, plan, y, c, E = best
            E_true = saving(x, y, range(i + 1))
            fires.append({"request": i, "agent": g, "plan": [a.key() for a in plan], "cost": c, "known": E,
                          "true": E_true, "rho": E / E_true if E_true > 0 else 1.0})
            x = y
            action_cost += c
        serve_cost += model.serve(x, i)
    return OnlineResult(action_cost + serve_cost, action_cost, serve_cost, fires)
