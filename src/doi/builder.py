"""Build a scenario from user-chosen sizes, counts and densities, and check it before it is run.

Three layouts: `barrier` (two zones split by a barrier with removable obstacles in its openings), `strips`
(random continuous wall strips plus removable obstacles anywhere) and `map` (an ASCII file). Every problem is
reported as a BuildError with a plain-language message.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.doi.config import SimConfig
from src.doi.kinds import GLYPHS, KINDS
from src.doi.paths import dream_path, passable_fn
from src.doi.rng import stream
from src.doi.scenarios import Scenario, _component_labels, build_scenario, scenario_from_ascii
from src.environment.grid import CellType

Pos = Tuple[int, int]
LAYOUTS = ("barrier", "strips", "map")


class BuildError(ValueError):
    """The requested map cannot be built; the message says why and what to change."""


def barrier_block_rows(height: int, doors: int, n: int) -> List[int]:
    """Rows for the obstacles, spread evenly down the barrier above the doors (which sit at the bottom)."""
    usable = list(range(1, height - doors - 1))
    if n < 1:
        raise BuildError("at least 1 removable obstacle is needed: the point is deciding whether to remove it")
    if n > len(usable):
        raise BuildError(f"{n} obstacles do not fit in a barrier {height} rows tall with {doors} door row(s) "
                         f"(room for {len(usable)}): use fewer obstacles, more rows or narrower doors")
    if n == 1:
        return [usable[0]]
    return sorted({usable[round(i * (len(usable) - 1) / (n - 1))] for i in range(n)})


def barrier_params(rows: int, cols: int, wall_col: Optional[int], blocks: int, doors: int, kind: str,
                   crossing: float) -> Dict[str, Any]:
    if rows < 8 or cols < 9:
        raise BuildError("the barrier layout needs at least 8 rows and 9 columns")
    wc = cols // 2 if wall_col is None else wall_col
    if not 2 <= wc <= cols - 4:
        raise BuildError(f"barrier column {wc} leaves no room on one side: choose a column between 2 and {cols - 4}")
    if not 1 <= doors <= rows - 4:
        raise BuildError(f"doors must be between 1 and {rows - 4} rows for a map {rows} rows tall")
    if not 0.0 <= crossing <= 1.0:
        raise BuildError("the share of trips that cross the barrier must be between 0 and 1")
    if kind != "mixed" and kind not in KINDS:
        raise BuildError(f"unknown obstacle kind {kind!r}: choose from mixed, {', '.join(KINDS)}")
    kinds = ("pallet", "crate", "shelf_unit") if kind == "mixed" else (kind,)
    block_rows = barrier_block_rows(rows, doors, blocks)
    return dict(H=rows, W=cols, wall_col=wc, wall_thick=1, block_cells=tuple((r, wc) for r in block_rows),
                door_rows=tuple(range(rows - doors, rows)), kinds=kinds, q_cross=crossing)


def strips_params(rows: int, cols: int, strips: int, strip_len: Tuple[int, int], pallets: int, crates: int,
                  shelves: int) -> Dict[str, Any]:
    if rows < 5 or cols < 5:
        raise BuildError("the strips layout needs at least 5 rows and 5 columns")
    if strips < 0 or min(pallets, crates, shelves) < 0:
        raise BuildError("counts cannot be negative")
    if not 1 <= strip_len[0] <= strip_len[1]:
        raise BuildError("strip length must satisfy 1 <= min <= max")
    if pallets + crates + shelves < 1:
        raise BuildError("at least 1 removable obstacle is needed")
    return dict(H=rows, W=cols, strips=strips, strip_len_min=strip_len[0], strip_len_max=strip_len[1],
                pallets=pallets, crates=crates, shelves=shelves)


def scenario_from_map_file(path: str, cfg: SimConfig) -> Scenario:
    """An ASCII map (# wall, . free, and kind glyphs such as L C S); robot starts and tasks are drawn from the seed."""
    try:
        with open(path) as f:
            rows = [line.rstrip("\n") for line in f if line.strip()]
    except OSError as e:
        raise BuildError(f"cannot read map file {path!r}: {e.strerror}")
    if not rows:
        raise BuildError(f"map file {path!r} is empty")
    if len({len(r) for r in rows}) != 1:
        raise BuildError("every row of the map file must have the same length")
    bad = sorted({ch for r in rows for ch in r} - set("#.") - set(GLYPHS))
    if bad:
        raise BuildError(f"map file has unknown characters {bad}: use # wall, . free, "
                         + ", ".join(f"{g} {GLYPHS[g]}" for g in GLYPHS))
    probe = scenario_from_ascii(rows, [], [], name="map")
    if not probe.obstacles:
        raise BuildError("the map needs at least one removable obstacle (for example L for a pallet)")
    label = _component_labels(probe.grid)
    sizes: Dict[int, int] = {}
    for comp in label.values():
        sizes[comp] = sizes.get(comp, 0) + 1
    main = max(sizes, key=lambda k: (sizes[k], -k))
    pool = sorted(c for c, k in label.items() if k == main and c not in probe.obstacles)
    if len(pool) < cfg.n_robots:
        raise BuildError(f"{cfg.n_robots} robots need {cfg.n_robots} free cells but the map has {len(pool)}")
    starts: List[Pos] = []
    for i in range(cfg.n_robots):
        starts.append(stream(cfg.seed, f"start-{i}").choice([c for c in pool if c not in starts]))
    tasks: List[List[Pos]] = []
    for i, s in enumerate(starts):
        rng = stream(cfg.seed, f"tasks-{i}")
        prev, goals = s, []
        for _ in range(cfg.tasks_per_robot):
            goal = rng.choice([p for p in pool if p != prev] or pool)
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return scenario_from_ascii(rows, starts, tasks, name="map")


def build(layout: str, cfg: SimConfig, a: Dict[str, Any]) -> Tuple[SimConfig, Scenario]:
    """Validated (cfg, scenario) for a layout; `a` holds the layout's answers (see the wizard)."""
    if layout == "barrier":
        params = barrier_params(a["rows"], a["cols"], a.get("wall_col"), a["blocks"], a["doors"], a["kind"],
                                a["crossing"])
        cfg = cfg.replace(scenario="multi_block_wall", scenario_params=params)
    elif layout == "strips":
        params = strips_params(a["rows"], a["cols"], a["strips"], (a["strip_min"], a["strip_max"]),
                               a["pallets"], a["crates"], a["shelves"])
        cfg = cfg.replace(scenario="random_blocks", scenario_params=params)
    elif layout == "map":
        return cfg, scenario_from_map_file(a["map"], cfg)
    else:
        raise BuildError(f"unknown layout {layout!r}: choose one of {', '.join(LAYOUTS)}")
    try:
        scenario = build_scenario(cfg)
    except ValueError as e:
        raise BuildError(str(e))
    scenario.name = layout
    return cfg, scenario


def potential_rent(scenario: Scenario, cfg: SimConfig) -> int:
    """Detour steps the fleet's tasks would save in total if every obstacle were gone."""
    U = cfg.unreachable_cost_for(scenario.grid.height, scenario.grid.width)
    cells = sorted(scenario.obstacles)
    total = 0
    for start, goals in zip(scenario.starts, scenario.tasks):
        prev = start
        for goal in goals:
            total += dream_path(scenario.grid, cells, prev, goal, U).rent
            prev = goal
    return total


def has_push_room(scenario: Scenario, cell: Pos) -> bool:
    """Could a robot ever push this obstacle (some straight line with free floor on both sides)?"""
    free = passable_fn(scenario.grid)
    return any(free((cell[0] - dr, cell[1] - dc)) and free((cell[0] + dr, cell[1] + dc))
               for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)))


def check(scenario: Scenario, cfg: SimConfig) -> List[str]:
    """Plain-language notes about a built map; lines starting 'info:' are not warnings."""
    out: List[str] = []
    for cell, kind in sorted(scenario.obstacles.items()):
        if not has_push_room(scenario, cell):
            out.append(f"the {kind.replace('_', ' ')} at {cell} has no room to be pushed (the map edge or walls box it "
                       f"in): it can never be removed")
    rent = potential_rent(scenario, cfg)
    if rent == 0:
        out.append("no obstacle lies on a shortest route for any task, so there is nothing worth removing on this "
                   "map (try more obstacles, a longer way round, or a different seed)")
    else:
        out.append(f"info: removing every obstacle would save {rent} steps over all tasks "
                   f"(the most any removal strategy can recover)")
    return out
