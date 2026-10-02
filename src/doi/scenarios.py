"""Scenario generators: static maps, starts, task streams and (family D) incidents with reports."""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.doi.config import SimConfig
from src.doi.rng import stream
from src.environment.generator import EnvironmentGenerator
from src.environment.grid import CellType, Grid

Pos = Tuple[int, int]

_INCIDENT_COMMON = dict(appear_max=150, p_human=0.25, p_report=0.9, delta_report=5, p_false=0.0, stock=3)

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "single_pit": dict(H=15, W=21, wall_col=10, wall_thick=1, pit_cells=((3, 10),),
                       door_rows=(12, 13, 14), depot_dist=2, depot_row=3, stock=2, q_cross=0.8),
    "two_pits_parallel": dict(H=15, W=21, wall_col=10, wall_thick=1, pit_cells=((3, 10), (5, 10)),
                              door_rows=(12, 13, 14), depot_dist=2, depot_row=4, stock=2, q_cross=0.8),
    "series_pits": dict(H=15, W=21, wall_col=10, wall_thick=5, pit_cells=((3, 11), (3, 13)),
                        door_rows=(12, 13, 14), depot_dist=2, depot_row=3, stock=2, q_cross=0.8),
    "multi_pit_wall": dict(H=15, W=21, wall_col=10, wall_thick=1,
                           pit_cells=((2, 10), (4, 10), (6, 10), (8, 10)),
                           door_rows=(12, 13, 14), depot_dist=2, depot_row=4, stock=4, q_cross=0.8),
    "shift": dict(H=15, W=21, wall_col=10, wall_thick=1,
                  pit_cells=((2, 10), (4, 10), (10, 10), (12, 10)),
                  door_rows=(6, 7, 8), depot_dist=2, depot_row=7, stock=4, q_cross=0.8,
                  shift_after_task=10, hot_before=(0, 6), hot_after=(8, 14), q_hot=0.8),
    "random_pits": dict(H=20, W=20, obstacle_density=0.10, pit_density=0.04, sandbag_count=2),
    "incidents_room": dict(n_incidents=2, q_cross=0.8, **_INCIDENT_COMMON),
    "incidents_aisles": dict(n_incidents=4, **_INCIDENT_COMMON),
}

_TWO_ROOM = ("single_pit", "two_pits_parallel", "series_pits", "multi_pit_wall", "shift")
_DOORS = {"north door": (2, 10), "middle door": (7, 10), "south door": (12, 10)}


@dataclass(frozen=True)
class Incident:
    oid: int
    cells: Tuple[Pos, ...]
    appear_tick: int
    kind: str
    cls: str


@dataclass(frozen=True)
class Report:
    report_id: str
    oid: Optional[int]
    location: str
    emit_tick: int
    kind: str
    cls: str


@dataclass
class Scenario:
    name: str
    family: str
    grid: Grid
    pits: List[Pos]
    depots: Dict[Pos, int]
    starts: List[Pos]
    tasks: List[List[Pos]]
    incidents: List[Incident] = field(default_factory=list)
    reports: List[Report] = field(default_factory=list)
    locations: Dict[str, Tuple[Pos, ...]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _resolve_params(cfg: SimConfig) -> Dict[str, Any]:
    if cfg.scenario not in DEFAULTS:
        raise ValueError(f"unknown scenario {cfg.scenario!r}")
    params = dict(DEFAULTS[cfg.scenario])
    for key in cfg.scenario_params:
        if key not in params:
            raise ValueError(f"unknown scenario parameter {key!r} for {cfg.scenario}")
    params.update(cfg.scenario_params)
    if cfg.scenario == "multi_pit_wall" and "stock" not in cfg.scenario_params:
        params["stock"] = len(params["pit_cells"])
    return params


def _pit_list(grid: Grid) -> List[Pos]:
    return sorted(grid.pit_positions())


def scenario_from_ascii(rows: List[str], starts: List[Pos], tasks: List[List[Pos]],
                        depot_stock: int = 2, name: str = "ascii",
                        incidents: Sequence[Incident] = (), reports: Sequence[Report] = (),
                        locations: Optional[Dict[str, Tuple[Pos, ...]]] = None) -> Scenario:
    if len({len(r) for r in rows}) != 1:
        raise ValueError("all ascii rows must have equal length")
    grid = Grid(len(rows[0]), len(rows))
    codes = {".": CellType.FREE, "#": CellType.OBSTACLE, "P": CellType.PIT, "D": CellType.SANDBAG}
    depots: Dict[Pos, int] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch not in codes:
                raise ValueError(f"bad ascii character {ch!r}")
            grid.set(r, c, codes[ch])
            if ch == "D":
                depots[(r, c)] = depot_stock
    return Scenario(name=name, family="ascii", grid=grid, pits=_pit_list(grid), depots=depots,
                    starts=list(starts), tasks=[list(t) for t in tasks],
                    incidents=list(incidents), reports=list(reports),
                    locations=dict(locations or {}), meta={"params": {}, "seed": 0})


def _draw_starts(cfg: SimConfig, pools: List[List[Pos]]) -> List[Pos]:
    """pools[i] is the sorted candidate list for robot i; per-robot streams keep starts independent of n."""
    starts: List[Pos] = []
    for i, pool in enumerate(pools):
        rng = stream(cfg.seed, f"start-{i}")
        for _ in range(1000):
            cell = rng.choice(pool)
            if cell not in starts:
                starts.append(cell)
                break
        else:
            raise ValueError(f"could not place start for robot {i}")
    return starts


def _room_cells(grid: Grid, cols: range) -> List[Pos]:
    return [(r, c) for r in range(grid.height) for c in cols if grid.get(r, c) == CellType.FREE]


def _two_room_agents(cfg: SimConfig, grid: Grid, rooms: Dict[str, List[Pos]], q_cross: float,
                     exclude: Set[Pos], shift: Optional[Dict[str, Any]] = None
                     ) -> Tuple[List[Pos], List[List[Pos]]]:
    n_west = max(1, cfg.n_robots // 2)
    start_rooms = ["west" if i < n_west else "east" for i in range(cfg.n_robots)]
    starts = _draw_starts(cfg, [rooms[r] for r in start_rooms])
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        rng = stream(cfg.seed, f"tasks-{i}")
        room, prev = start_rooms[i], starts[i]
        goals: List[Pos] = []
        for k in range(cfg.tasks_per_robot):
            other = "east" if room == "west" else "west"
            chosen = other if rng.random() < q_cross else room
            allowed = [p for p in rooms[chosen] if p != prev and p not in exclude]
            pool = allowed
            if shift is not None:
                lo, hi = shift["hot_before"] if k < shift["shift_after_task"] else shift["hot_after"]
                hot = rng.random() < shift["q_hot"]
                band = [p for p in allowed if lo <= p[0] <= hi]
                pool = band if hot and band else allowed
            goal = rng.choice(pool)
            goals.append(goal)
            room, prev = chosen, goal
        tasks.append(goals)
    return starts, tasks


def _two_room(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    H, W = params["H"], params["W"]
    wc, wt = params["wall_col"], params["wall_thick"]
    e = wc + wt - 1
    pits = sorted(tuple(p) for p in params["pit_cells"])
    depot = (params["depot_row"], e + 1 + params["depot_dist"])
    if not (0 <= depot[0] < H and 0 <= depot[1] < W) or wc <= depot[1] <= e:
        raise ValueError(f"depot {depot} outside grid or inside the wall")
    grid = Grid(W, H)
    for r in range(H):
        for c in range(wc, e + 1):
            grid.set(r, c, CellType.OBSTACLE)
    for r in params["door_rows"]:
        for c in range(wc, e + 1):
            grid.set(r, c, CellType.FREE)
    if wt > 1:
        for r in sorted({p[0] for p in pits if wc <= p[1] <= e}):
            for c in range(wc, e + 1):
                grid.set(r, c, CellType.FREE)
    for p in pits:
        grid.set(p[0], p[1], CellType.PIT)
    grid.set(depot[0], depot[1], CellType.SANDBAG)
    west_cols, east_cols = range(0, wc), range(e + 1, W)
    rooms = {"west": _room_cells(grid, west_cols), "east": _room_cells(grid, east_cols)}
    shift = params if cfg.scenario == "shift" else None
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], set(), shift)
    return Scenario(name=cfg.scenario, family="S", grid=grid, pits=pits,
                    depots={depot: params["stock"]}, starts=starts, tasks=tasks,
                    meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                          "params": params, "seed": cfg.seed})


def _components(grid: Grid, source: Pos) -> Set[Pos]:
    """Cells reachable from source when every pit is open."""
    seen = {source}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            n = (r + dr, c + dc)
            if n in seen or not grid.in_bounds(*n) or grid.get(*n) == CellType.OBSTACLE:
                continue
            seen.add(n)
            queue.append(n)
    return seen


def _random_pits(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    gen = EnvironmentGenerator(width=params["W"], height=params["H"],
                               obstacle_density=params["obstacle_density"],
                               pit_density=params["pit_density"],
                               sandbag_count=params["sandbag_count"], seed=cfg.seed)
    grid, starts, _goals, _registry = gen.generate(num_robots=cfg.n_robots)
    depots = {p: 1 for p in sorted(grid.sandbag_positions())}
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        rng = stream(cfg.seed, f"tasks-{i}")
        comp = _components(grid, starts[i])
        prev, goals = starts[i], []
        for _ in range(cfg.tasks_per_robot):
            cand = sorted(p for p in comp if p != prev
                          and grid.get(*p) in (CellType.FREE, CellType.SANDBAG))
            if not cand:
                cand = sorted(grid.neighbours(*prev))
            if not cand:
                raise ValueError(f"robot {i} has no reachable goal")
            goal = rng.choice(cand)
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return Scenario(name=cfg.scenario, family="S", grid=grid, pits=_pit_list(grid), depots=depots,
                    starts=starts, tasks=tasks, meta={"params": params, "seed": cfg.seed})


def _draw_kind(rng, needs_human: bool, human_kind: str) -> str:
    return human_kind if needs_human else rng.choice(["pallet", "spill", "debris"])


def _incidents_and_reports(cfg: SimConfig, params: Dict[str, Any], candidates: Set[Pos],
                           locations: Dict[str, Tuple[Pos, ...]], human_kind: str
                           ) -> Tuple[List[Incident], List[Report]]:
    n, appear_max, delta = params["n_incidents"], params["appear_max"], params["delta_report"]
    rng = stream(cfg.seed, "incidents")
    cells = rng.sample(sorted(candidates), n)
    incidents: List[Incident] = []
    for oid, cell in enumerate(cells):
        appear = rng.randint(0, appear_max)
        human = rng.random() < params["p_human"]
        incidents.append(Incident(oid, (cell,), appear, _draw_kind(rng, human, human_kind),
                                  "needs_human" if human else "robot_clearable"))
    name_of = {cells_[0]: name for name, cells_ in locations.items() if cells_[0] in candidates}
    rng = stream(cfg.seed, "reports")
    raw: List[Tuple[int, str, Optional[int], str, str]] = []
    for inc in incidents:
        if rng.random() < params["p_report"]:
            raw.append((inc.appear_tick + delta, name_of[inc.cells[0]], inc.oid, inc.kind, inc.cls))
    n_false = sum(rng.random() < params["p_false"] for _ in range(n))
    incident_cells = {c for inc in incidents for c in inc.cells}
    clear = sorted(name for c, name in name_of.items() if c not in incident_cells)
    for _ in range(n_false if clear else 0):
        loc = rng.choice(clear)
        emit = rng.randint(0, appear_max + delta)
        human = rng.random() < params["p_human"]
        raw.append((emit, loc, None, _draw_kind(rng, human, human_kind),
                    "needs_human" if human else "robot_clearable"))
    raw.sort(key=lambda t: (t[0], t[1]))
    reports = [Report(f"r{i}", oid, loc, emit, kind, cls)
               for i, (emit, loc, oid, kind, cls) in enumerate(raw)]
    return incidents, reports


def _incidents_room(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    if params["n_incidents"] > 2:
        raise ValueError("incidents_room supports at most 2 incidents (one door must stay open)")
    H, W, wc = 15, 21, 10
    grid = Grid(W, H)
    for r in range(H):
        grid.set(r, wc, CellType.OBSTACLE)
    for cell in _DOORS.values():
        grid.set(cell[0], cell[1], CellType.FREE)
    depots = {(7, 8): params["stock"], (7, 12): params["stock"]}
    for p in depots:
        grid.set(p[0], p[1], CellType.SANDBAG)
    locations = {name: (cell,) for name, cell in _DOORS.items()}
    incidents, reports = _incidents_and_reports(cfg, params, set(_DOORS.values()), locations, "spill")
    exclude = {c for inc in incidents for c in inc.cells}
    west_cols, east_cols = range(0, wc), range(wc + 1, W)
    rooms = {"west": _room_cells(grid, west_cols), "east": _room_cells(grid, east_cols)}
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], exclude)
    return Scenario(name=cfg.scenario, family="D", grid=grid, pits=[], depots=depots, starts=starts,
                    tasks=tasks, incidents=incidents, reports=reports, locations=locations,
                    meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                          "params": params, "seed": cfg.seed})


def _incidents_aisles(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    H, W = 13, 21
    grid = Grid(W, H)
    for r in list(range(1, 6)) + list(range(7, 12)):
        for c in range(1, W, 2):
            grid.set(r, c, CellType.OBSTACLE)
    depots = {(6, 0): params["stock"], (6, 20): params["stock"]}
    for p in depots:
        grid.set(p[0], p[1], CellType.SANDBAG)
    locations: Dict[str, Tuple[Pos, ...]] = {}
    bay_cells: List[Pos] = []
    candidates: Set[Pos] = set()
    for k in range(1, 12):
        col = 2 * (k - 1)
        for b in range(1, 11):
            row = b if b <= 5 else b + 1
            locations[f"aisle {k} bay {b}"] = ((row, col),)
            bay_cells.append((row, col))
            if 2 <= k <= 10:
                candidates.add((row, col))
    locations["west station"] = ((6, 0),)
    locations["east station"] = ((6, 20),)
    incidents, reports = _incidents_and_reports(cfg, params, candidates, locations, "rack_damage")
    blocked = {c for inc in incidents for c in inc.cells}
    goal_cells = sorted(p for p in bay_cells if p not in blocked)
    start_pool = sorted([(0, c) for c in range(W)] + [(12, c) for c in range(W)])
    starts = _draw_starts(cfg, [start_pool] * cfg.n_robots)
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        rng = stream(cfg.seed, f"tasks-{i}")
        prev, goals = starts[i], []
        for _ in range(cfg.tasks_per_robot):
            goal = rng.choice([p for p in goal_cells if p != prev])
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return Scenario(name=cfg.scenario, family="D", grid=grid, pits=[], depots=depots, starts=starts,
                    tasks=tasks, incidents=incidents, reports=reports, locations=locations,
                    meta={"params": params, "seed": cfg.seed})


def build_scenario(cfg: SimConfig) -> Scenario:
    params = _resolve_params(cfg)
    if cfg.scenario in _TWO_ROOM:
        return _two_room(cfg, params)
    if cfg.scenario == "random_pits":
        return _random_pits(cfg, params)
    if cfg.scenario == "incidents_room":
        return _incidents_room(cfg, params)
    return _incidents_aisles(cfg, params)
