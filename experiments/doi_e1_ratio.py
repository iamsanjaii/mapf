"""E1: competitive ratio against the exact optimum. Part `core` (pure numbers) and part `grid` (abstract model)."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.doi.abstract import core
from src.doi.abstract.instance import g1, random_small_instance
from src.doi.abstract.offline import StateSpaceTooLarge, exact_opt, vanish_lower_bound
from src.doi.abstract.online import Rule, run_online
from src.doi.rng import stream

NAN = float("nan")
CORE_COLS = ["family", "c", "seed", "view", "n", "delta", "p", "arm", "theta", "lam", "predictor", "alg", "opt",
             "ratio", "rho", "D", "bound", "bound_holds"]
GRID_COLS = ["instance", "m_or_seed", "k", "T", "n_agents", "arm", "bundle_max", "view", "alg", "opt", "lb",
             "ratio_opt", "ratio_lb", "n_states", "fires"]


def _ratio(alg: float, opt: float) -> float:
    if opt > 0:
        return alg / opt
    return 1.0 if alg == 0 else NAN


def _sequence(family: str, c: int, length: int, seed: int):
    if family == "constant":
        return [1.0] * length
    if family == "stop":
        return [1.0] * c
    rng = stream(seed, f"e1-{c}")
    return [float(rng.randint(0, 3)) for _ in range(5 * c)]


def _core_views(spec, T: int, seed: int):
    kind, par = spec
    if kind == "full":
        return core.views_full(T), 1, 0, NAN
    if kind == "own":
        return core.views_own(core.round_robin(T, par)), par, 0, NAN
    agents = core.round_robin(T, 4)
    if kind == "delay":
        return core.views_delay(agents, par), 4, par, NAN
    return core.views_sample(agents, par, seed), 4, 0, par


def core_rows(quick: bool):
    cs = [20] if quick else [5, 20, 80]
    seeds = range(3) if quick else range(10)
    view_specs = ([("full", 0)] + [("own", n) for n in (2, 4)]) if quick else (
        [("full", 0)] + [("own", n) for n in (2, 4, 8, 16)] + [("delay", d) for d in (2, 8, 32)]
        + [("sample", p) for p in (0.25, 0.5)])
    rows = []
    for c in cs:
        families = [("constant", L) for L in (c // 2, c, 2 * c, 10 * c)] + [("stop", c), ("random", 5 * c)]
        if quick:
            families = [f for f in families if f[0] != "random"]
        for family, length in families:
            for seed in seeds:
                s = _sequence(family, c, length, seed)
                T = len(s)
                total = core.prefix(s)[-1] if s else 0.0
                for spec in view_specs:
                    views, n, delta, p = _core_views(spec, T, seed)
                    rho, D = core.coverage(s, views), core.deficit(s, views)
                    opt = core.opt_single(s, c)
                    base = dict(family=family, c=c, seed=seed, view=spec[0], n=n, delta=delta, p=p, opt=opt,
                                rho=rho, D=D, theta=NAN, lam=NAN, predictor="")

                    def emit(arm, alg, bound, **kw):
                        holds = True if math.isnan(bound) else alg <= bound * opt + 1e-9
                        rows.append({**base, **kw, "arm": arm, "alg": alg, "ratio": _ratio(alg, opt),
                                     "bound": bound, "bound_holds": holds})

                    for theta in (0.5, 1.0, 2.0):
                        out = core.run_threshold(s, c, views, theta)
                        bound = min(core.bound_coverage(theta, rho) if rho > 0 else math.inf,
                                    core.bound_deficit(theta, D, c))
                        emit("threshold", out.alg, bound, theta=theta)
                    if spec[0] == "full":
                        emit("randomized", core.expected_randomized_full(s, c), core.E_RATIO)
                    else:
                        mean = sum(core.run_randomized(s, c, views, core.randomized_draw(seed, d)).alg
                                   for d in range(200)) / 200
                        emit("randomized", mean, NAN)
                    noise = stream(seed, "e1-noise").gauss(0, 1)
                    forecasts = {"oracle": total, "inverted": 0.0 if total >= c else 2 * c + 1,
                                 "noisy:1.0": total * math.exp(noise)}
                    for lam in (0.25, 0.5, 1.0):
                        for name, predicted in forecasts.items():
                            out = core.run_predicted(s, c, views, predicted, lam)
                            bound = core.bound_predicted(lam, rho) if rho > 0 else NAN
                            emit("predicted", out.alg, bound, lam=lam, predictor=name)
    return rows


def grid_instances(quick: bool):
    if quick:
        for m in (1, 4):
            yield "g1", m, 0, m, 1, g1(m, 1)
        for seed in range(3):
            yield "random", seed, 1, 10, 1, random_small_instance(seed, 1, 10, 1)
        return
    for m in (1, 2, 4, 8, 16):
        for n in (1, 4):
            yield "g1", m, 0, m, n, g1(m, n)
    for seed in range(20):
        for k in (1, 2):
            for T in (10, 30):
                for n in (1, 4):
                    yield "random", seed, k, T, n, random_small_instance(seed, k, T, n)


def grid_rows(quick: bool):
    rows = []
    for name, m_or_seed, k, T, n_agents, inst in grid_instances(quick):
        try:
            res = exact_opt(inst)
            opt, n_states = res.cost, res.n_states
        except StateSpaceTooLarge:
            opt, n_states = NAN, -1
        lb = vanish_lower_bound(inst)
        arms = [("never", 1, Rule("never"))]
        arms += [("threshold", b, Rule("threshold", bundle_max=b)) for b in (1, 2)]
        arms += [("predicted", 2, Rule("predicted", lam=0.5, bundle_max=2, predictor=p))
                 for p in ("oracle", "inverted")]
        for view in ("full", "own"):
            for arm, bundle, rule in arms:
                r = run_online(inst, Rule(**{**rule.__dict__, "view": view}))
                label = arm if arm != "predicted" else f"predicted:{rule.predictor}"
                rows.append(_grid_row(name, m_or_seed, k, T, n_agents, label, bundle, view, r.cost, opt, lb,
                                      n_states, len(r.fires)))
            runs = [run_online(inst, Rule("randomized", bundle_max=2, view=view, seed=s)) for s in range(20)]
            rows.append(_grid_row(name, m_or_seed, k, T, n_agents, "randomized", 2, view,
                                  sum(r.cost for r in runs) / 20, opt, lb, n_states,
                                  sum(len(r.fires) for r in runs) / 20))
    return rows


def _grid_row(name, m_or_seed, k, T, n_agents, arm, bundle, view, alg, opt, lb, n_states, fires) -> dict:
    return {"instance": name, "m_or_seed": m_or_seed, "k": k, "T": T, "n_agents": n_agents, "arm": arm,
            "bundle_max": bundle, "view": view, "alg": alg, "opt": opt, "lb": lb,
            "ratio_opt": _ratio(alg, opt) if not math.isnan(opt) else NAN, "ratio_lb": _ratio(alg, lb),
            "n_states": n_states, "fires": fires}


def print_summary(core_df, grid_df) -> None:
    lines = []
    for part, df in (("core", core_df), ("grid", grid_df)):
        if df is None or df.empty:
            continue
        col = "ratio" if part == "core" else "ratio_opt"
        for (arm, view), g in sorted(df.groupby(["arm", "view"])):
            share = f"{g['bound_holds'].astype(bool).mean():.2f}" if part == "core" else "-"
            lines.append(f"{part:5s} {arm:22s} {view:8s} {g[col].median():12.3f} {g[col].max():12.3f} {share:>12s}")
    print(f"{'part':5s} {'arm':22s} {'view':8s} {'median ratio':>12s} {'max ratio':>12s} {'bound holds':>12s}")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", choices=["core", "grid", "all"], default="all")
    ap.add_argument("--out", default="experiments/results/doi/e1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    core_df = grid_df = None
    if args.part in ("core", "all"):
        core_df = pd.DataFrame(core_rows(args.quick), columns=CORE_COLS)
        core_df.to_csv(os.path.join(args.out, "core.csv"), index=False)
    if args.part in ("grid", "all"):
        grid_df = pd.DataFrame(grid_rows(args.quick), columns=GRID_COLS)
        grid_df.to_csv(os.path.join(args.out, "grid.csv"), index=False)
    print_summary(core_df, grid_df)
    if core_df is not None and not core_df["bound_holds"].astype(bool).all():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
