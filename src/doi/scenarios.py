"""Scenario generators: static maps, starts, task streams and (family D) incidents with reports."""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.doi.config import SimConfig
from src.doi.kinds import GLYPHS, KINDS, SLOT_GLYPHS
from src.doi.maps import load_movingai
from src.doi.rng import stream
from src.environment.grid import CellType, Grid

Pos = Tuple[int, int]

_INCIDENT_COMMON = dict(appear_max=150, p_human=0.25, p_report=0.9, delta_report=5, p_false=0.0)

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "single_block": dict(H=15, W=21, wall_col=10, wall_thick=1, block_cells=((3, 10),),
                         door_rows=(12, 13, 14), kinds=("pallet",), q_cross=0.8),
    "two_blocks_parallel": dict(H=15, W=21, wall_col=10, wall_thick=1, block_cells=((3, 10), (5, 10)),
                                door_rows=(12, 13, 14), kinds=("pallet",), q_cross=0.8),
    "series_blocks": dict(H=15, W=21, wall_col=10, wall_thick=5, block_cells=((3, 11), (3, 13)),
                          door_rows=(12, 13, 14), kinds=("pallet",), q_cross=0.8),
    "multi_block_wall": dict(H=15, W=21, wall_col=10, wall_thick=1,
                             block_cells=((2, 10), (4, 10), (6, 10), (8, 10)),
                             door_rows=(12, 13, 14), kinds=("pallet", "crate", "shelf_unit", "pallet"),
                             q_cross=0.8),
    "shift": dict(H=15, W=21, wall_col=10, wall_thick=1,
                  block_cells=((2, 10), (4, 10), (10, 10), (12, 10)),
                  door_rows=(6, 7, 8), kinds=("pallet",), q_cross=0.8,
                  shift_after_task=10, hot_before=(0, 6), hot_after=(8, 14), q_hot=0.8),
    "complements": dict(H=15, W=23, wall_cols=(7, 15), gap_row=3, door_rows=(12, 13, 14),
                        kinds=("pallet",), band=(0, 6)),
    "random_blocks": dict(H=20, W=20, strips=3, strip_len_min=4, strip_len_max=8, pallets=3, crates=2, shelves=1),
    "warehouse_blocks": dict(map_path=None, n_blocks=6, kinds=("pallet",)),
    "warehouse_incidents": dict(map_path=None, n_incidents=4, **{**_INCIDENT_COMMON, "appear_max": 1000}),
    "incidents_room": dict(n_incidents=2, q_cross=0.8, **_INCIDENT_COMMON),
    "incidents_aisles": dict(n_incidents=4, **_INCIDENT_COMMON),
}

_TWO_ROOM = ("single_block", "two_blocks_parallel", "series_blocks", "multi_block_wall", "shift")
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
    obstacles: Dict[Pos, str]          # removable obstacles at tick 0: cell -> kind (their cells are FREE in grid)
    starts: List[Pos]
    tasks: List[List[Pos]]
    incidents: List[Incident] = field(default_factory=list)
    reports: List[Report] = field(default_factory=list)
    locations: Dict[str, Tuple[Pos, ...]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[Pos, str] = field(default_factory=dict)    # cells that hold a carried obstacle: cell -> rack or dump


def _resolve_params(cfg: SimConfig) -> Dict[str, Any]:
    if cfg.scenario not in DEFAULTS:
        raise ValueError(f"unknown scenario {cfg.scenario!r}")
    params = dict(DEFAULTS[cfg.scenario])
    for key in cfg.scenario_params:
        if key not in params:
            raise ValueError(f"unknown scenario parameter {key!r} for {cfg.scenario}")
    params.update(cfg.scenario_params)
    return params


def scenario_from_ascii(rows: List[str], starts: List[Pos], tasks: List[List[Pos]], name: str = "ascii",
                        incidents: Sequence[Incident] = (), reports: Sequence[Report] = (),
                        locations: Optional[Dict[str, Tuple[Pos, ...]]] = None) -> Scenario:
    """# wall, . free, a kind glyph (L pallet, C crate, S shelf unit, P pit, ...) for a removable obstacle,
    D a dump-region slot (walkable floor) and T a rack slot (a fixture: not walkable)."""
    if len({len(r) for r in rows}) != 1:
        raise ValueError("all ascii rows must have equal length")
    grid = Grid(len(rows[0]), len(rows))
    obstacles: Dict[Pos, str] = {}
    slots: Dict[Pos, str] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                grid.set(r, c, CellType.OBSTACLE)
            elif ch in SLOT_GLYPHS:
                slots[(r, c)] = SLOT_GLYPHS[ch]
                if SLOT_GLYPHS[ch] == "rack":
                    grid.set(r, c, CellType.OBSTACLE)
            elif ch in GLYPHS:
                obstacles[(r, c)] = GLYPHS[ch]
            elif ch != ".":
                raise ValueError(f"bad ascii character {ch!r}")
    return Scenario(name=name, family="ascii", grid=grid, obstacles=obstacles, starts=list(starts),
                    tasks=[list(t) for t in tasks], incidents=list(incidents), reports=list(reports),
                    locations=dict(locations or {}), meta={"params": {}, "seed": 0}, slots=slots)


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
    cells = sorted(tuple(p) for p in params["block_cells"])
    kinds = tuple(params["kinds"])
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"unknown obstacle kind {k!r}: choose from {', '.join(KINDS)}")
    for p in cells:
        if not (0 <= p[0] < H and wc <= p[1] <= e):
            raise ValueError(f"obstacle {p} is outside the barrier (columns {wc} to {e})")
    obstacles = {p: kinds[i % len(kinds)] for i, p in enumerate(cells)}
    grid = Grid(W, H)
    for r in range(H):
        for c in range(wc, e + 1):
            grid.set(r, c, CellType.OBSTACLE)
    for r in params["door_rows"]:
        for c in range(wc, e + 1):
            grid.set(r, c, CellType.FREE)
    for r in sorted({p[0] for p in cells}):                  # a removable obstacle sits in an opening of the barrier
        for c in range(wc, e + 1):
            grid.set(r, c, CellType.FREE)
    west_cols, east_cols = range(0, wc), range(e + 1, W)
    rooms = {"west": _room_cells(grid, west_cols), "east": _room_cells(grid, east_cols)}
    shift = params if cfg.scenario == "shift" else None
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], set(obstacles), shift)
    return Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts, tasks=tasks,
                    meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                          "params": params, "seed": cfg.seed})


def _complements(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    """Three rooms in a row; each wall has a doorway at the top blocked by a pallet and open doors at the bottom.

    Why the saving is zero for one doorway: from row r1 <= 6 in the west to r2 <= 6 in the east, the doors cost
    (12-r1) + (12-r2) rows of vertical travel. Opening only the west doorway costs |r1-3| + 9 + (12-r2), which is
    the same or more. The same holds for the east doorway by symmetry."""
    H, W = params["H"], params["W"]
    wall_cols, gap_row, band = tuple(params["wall_cols"]), params["gap_row"], tuple(params["band"])
    kinds = tuple(params["kinds"])
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"unknown obstacle kind {k!r}: choose from {', '.join(KINDS)}")
    grid = Grid(W, H)
    for wc in wall_cols:
        for r in range(H):
            if r not in params["door_rows"] and r != gap_row:
                grid.set(r, wc, CellType.OBSTACLE)
    obstacles = {(gap_row, wc): kinds[k % len(kinds)] for k, wc in enumerate(wall_cols)}
    in_band = lambda cells: [p for p in cells if band[0] <= p[0] <= band[1] and p not in obstacles]
    rooms = {"west": _room_cells(grid, range(0, wall_cols[0])),
             "east": _room_cells(grid, range(wall_cols[1] + 1, W))}
    n_west = cfg.n_robots // 2
    start_rooms = ["west" if i < n_west else "east" for i in range(cfg.n_robots)]
    starts = _draw_starts(cfg, [in_band(rooms[r]) for r in start_rooms])
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        rng = stream(cfg.seed, f"tasks-{i}")
        room, prev = start_rooms[i], starts[i]
        goals: List[Pos] = []
        for _ in range(cfg.tasks_per_robot):
            room = "east" if room == "west" else "west"
            goal = rng.choice([p for p in in_band(rooms[room]) if p != prev])
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts, tasks=tasks,
                    meta={"params": params, "seed": cfg.seed})


def _random_blocks(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    H, W = params["H"], params["W"]
    lo, hi = params["strip_len_min"], params["strip_len_max"]
    if lo < 1 or hi < lo:
        raise ValueError("strip length must satisfy 1 <= min <= max")
    grid = Grid(W, H)
    rng = stream(cfg.seed, "strips")
    for _ in range(params["strips"]):
        horizontal = rng.random() < 0.5
        length, r, c = rng.randint(lo, hi), rng.randrange(H), rng.randrange(W)
        for k in range(length):
            rr, cc = (r, c + k) if horizontal else (r + k, c)
            if 0 <= rr < H and 0 <= cc < W:
                grid.set(rr, cc, CellType.OBSTACLE)
    label = _component_labels(grid)
    sizes: Dict[int, int] = {}
    for comp in label.values():
        sizes[comp] = sizes.get(comp, 0) + 1
    main = max(sizes, key=lambda k: (sizes[k], -k))
    comp_cells = sorted(c for c, k in label.items() if k == main)
    wanted = ["pallet"] * params["pallets"] + ["crate"] * params["crates"] + ["shelf_unit"] * params["shelves"]
    if len(wanted) > len(comp_cells) // 4:
        raise ValueError(f"{len(wanted)} removable obstacles are too many for {len(comp_cells)} free cells")
    chosen = stream(cfg.seed, "blocks").sample(comp_cells, len(wanted))
    obstacles = dict(zip(sorted(chosen), wanted))
    pool = [c for c in comp_cells if c not in obstacles]
    if len(pool) < cfg.n_robots:
        raise ValueError(f"{cfg.n_robots} robots need {cfg.n_robots} free cells but only {len(pool)} are left")
    starts = _draw_starts(cfg, [pool] * cfg.n_robots)
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        trng = stream(cfg.seed, f"tasks-{i}")
        prev, goals = starts[i], []
        for _ in range(cfg.tasks_per_robot):
            goal = trng.choice([p for p in pool if p != prev] or pool)
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts, tasks=tasks,
                    meta={"params": params, "seed": cfg.seed})


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
    locations = {name: (cell,) for name, cell in _DOORS.items()}
    incidents, reports = _incidents_and_reports(cfg, params, set(_DOORS.values()), locations, "spill")
    exclude = {c for inc in incidents for c in inc.cells}
    west_cols, east_cols = range(0, wc), range(wc + 1, W)
    rooms = {"west": _room_cells(grid, west_cols), "east": _room_cells(grid, east_cols)}
    starts, tasks = _two_room_agents(cfg, grid, rooms, params["q_cross"], exclude)
    return Scenario(name=cfg.scenario, family="D", grid=grid, obstacles={}, starts=starts,
                    tasks=tasks, incidents=incidents, reports=reports, locations=locations,
                    meta={"rooms": {"west_cols": west_cols, "east_cols": east_cols},
                          "params": params, "seed": cfg.seed})


def _incidents_aisles(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    H, W = 13, 21
    grid = Grid(W, H)
    for r in list(range(1, 6)) + list(range(7, 12)):
        for c in range(1, W, 2):
            grid.set(r, c, CellType.OBSTACLE)
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
    return Scenario(name=cfg.scenario, family="D", grid=grid, obstacles={}, starts=starts,
                    tasks=tasks, incidents=incidents, reports=reports, locations=locations,
                    meta={"params": params, "seed": cfg.seed})


def _component_labels(grid: Grid) -> Dict[Pos, int]:
    """Component id for every non-obstacle cell (removable obstacles ignored)."""
    label: Dict[Pos, int] = {}
    n = 0
    for r in range(grid.height):
        for c in range(grid.width):
            if (r, c) in label or grid.get(r, c) == CellType.OBSTACLE:
                continue
            label[(r, c)] = n
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                    nb = (cr + dr, cc + dc)
                    if nb not in label and grid.in_bounds(*nb) and grid.get(*nb) != CellType.OBSTACLE:
                        label[nb] = n
                        queue.append(nb)
            n += 1
    return label


def _corridor_cells(grid: Grid, cells: Sequence[Pos]) -> List[Pos]:
    """Free cells whose open neighbours are exactly two and opposite (a one-cell-wide passage)."""
    out = []
    for r, c in cells:
        open_n = [(dr, dc) for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1))
                  if grid.in_bounds(r + dr, c + dc) and grid.get(r + dr, c + dc) != CellType.OBSTACLE]
        if len(open_n) == 2 and open_n[0][0] == -open_n[1][0] and open_n[0][1] == -open_n[1][1]:
            out.append((r, c))
    return out


def _warehouse_common(cfg: SimConfig, params: Dict[str, Any]):
    path = params.get("map_path") or cfg.map_path
    if not path:
        raise ValueError("warehouse scenarios need map_path (cfg.map_path or scenario_params)")
    grid = load_movingai(path)
    label = _component_labels(grid)
    sizes: Dict[int, int] = {}
    for comp in label.values():
        sizes[comp] = sizes.get(comp, 0) + 1
    main = max(sizes, key=lambda k: (sizes[k], -k))
    cells = sorted(c for c, comp in label.items() if comp == main)
    return grid, cells, _corridor_cells(grid, cells)


def _warehouse_agents(cfg: SimConfig, cells: List[Pos], exclude: Set[Pos]):
    pool = [c for c in cells if c not in exclude]
    starts = _draw_starts(cfg, [pool] * cfg.n_robots)
    k = cfg.horizon // 5 + 50 if cfg.horizon else cfg.tasks_per_robot
    tasks: List[List[Pos]] = []
    for i in range(cfg.n_robots):
        rng = stream(cfg.seed, f"tasks-{i}")
        prev, goals = starts[i], []
        for _ in range(k):
            goal = rng.choice(pool)
            while goal == prev:
                goal = rng.choice(pool)
            goals.append(goal)
            prev = goal
        tasks.append(goals)
    return starts, tasks


def _warehouse_blocks(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    grid, cells, corridors = _warehouse_common(cfg, params)
    if len(corridors) < params["n_blocks"]:
        raise ValueError("not enough one-cell corridors for the requested obstacles")
    kinds = tuple(params["kinds"])
    chosen = sorted(stream(cfg.seed, "blocks").sample(sorted(corridors), params["n_blocks"]))
    obstacles = {p: kinds[i % len(kinds)] for i, p in enumerate(chosen)}
    starts, tasks = _warehouse_agents(cfg, cells, set(obstacles))
    return Scenario(name=cfg.scenario, family="S", grid=grid, obstacles=obstacles, starts=starts,
                    tasks=tasks, meta={"params": params, "seed": cfg.seed})


def _warehouse_incidents(cfg: SimConfig, params: Dict[str, Any]) -> Scenario:
    grid, cells, corridors = _warehouse_common(cfg, params)
    if len(corridors) < params["n_incidents"]:
        raise ValueError("not enough one-cell corridors for the requested incidents")
    locations = {f"corridor {r}-{c}": ((r, c),) for r, c in corridors}
    incidents, reports = _incidents_and_reports(cfg, params, set(corridors), locations, "rack_damage")
    blocked = {c for inc in incidents for c in inc.cells}
    starts, tasks = _warehouse_agents(cfg, cells, blocked)
    return Scenario(name=cfg.scenario, family="D", grid=grid, obstacles={}, starts=starts,
                    tasks=tasks, incidents=incidents, reports=reports, locations=locations,
                    meta={"params": params, "seed": cfg.seed})


def build_scenario(cfg: SimConfig) -> Scenario:
    params = _resolve_params(cfg)
    if cfg.scenario in _TWO_ROOM:
        return _two_room(cfg, params)
    if cfg.scenario == "complements":
        return _complements(cfg, params)
    if cfg.scenario == "random_blocks":
        return _random_blocks(cfg, params)
    if cfg.scenario == "warehouse_blocks":
        return _warehouse_blocks(cfg, params)
    if cfg.scenario == "warehouse_incidents":
        return _warehouse_incidents(cfg, params)
    if cfg.scenario == "incidents_room":
        return _incidents_room(cfg, params)
    return _incidents_aisles(cfg, params)
