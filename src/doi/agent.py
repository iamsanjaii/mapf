"""RobotAgent: sense, receive, decide, act and broadcast using only the robot's own belief."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.network import Message
from src.doi.paths import bfs_dist_map, dream_path, passable_fn, single_pit_rents
from src.doi.policies import FillPolicy, Shared
from src.doi.scenarios import Scenario
from src.doi.spacetime import plan_spacetime
from src.doi.world import Action, ActionResult, Move, Observation, Wait

Pos = Tuple[int, int]


@dataclass(frozen=True)
class TaskPlanInfo:
    robot: int
    task_idx: int
    tick: int
    start: Pos
    goal: Pos
    d_block: int
    d_open: int
    bundle: Tuple[Pos, ...]
    rent: int
    belief_filled: FrozenSet[Pos]
    belief_blocked: FrozenSet[Pos]
    hard_blocked: FrozenSet[Pos]
    single_rents: Dict[Pos, int]


@dataclass(frozen=True)
class IntentRecord:
    start_tick: int
    blocked_ticks: int
    cells: Tuple[Pos, ...]


class RobotAgent:
    def __init__(self, rid: int, scenario: Scenario, cfg: SimConfig, policy: FillPolicy,
                 shared: Shared) -> None:
        self.id = rid
        self.cfg = cfg
        self.policy = policy
        self.shared = shared
        self.grid = scenario.grid
        self.H, self.W = self.grid.height, self.grid.width
        self.U = cfg.unreachable_cost_for(self.H, self.W)
        self.tasks: List[Pos] = list(scenario.tasks[rid])
        self.belief = BeliefState(rid, dict(scenario.depots), list(scenario.pits))
        self.pos: Pos = scenario.starts[rid]
        self.carrying = False
        self.task_idx = 0
        self.finished = False
        self.completed_tick: Optional[int] = None
        self.plan_infos: List[TaskPlanInfo] = []
        self.stats = {"replans": 0, "rent_counted": 0}
        self.plan: List[Pos] = [self.pos]
        self.replan_needed = True
        self.blocked_ticks = 0
        self.blocked_streak = 0
        self.intents: Dict[int, IntentRecord] = {}
        self.sensed_cells: FrozenSet[Pos] = frozenset()
        self._planned_task = -1
        self._last_action: Optional[Action] = None
        self._reserved_cache: Tuple[int, Set[Tuple[Pos, int]]] = (-1, set())

    @property
    def goal(self) -> Pos:
        return self.tasks[self.task_idx]

    def _snapshot_view(self) -> Tuple[FrozenSet[Pos], FrozenSet[Pos]]:
        return self.belief.believed_blocked(), frozenset(self.belief.filled.items())

    def sense(self, obs: Observation, t: int) -> None:
        before = self._snapshot_view()
        b = self.belief
        for c in obs.filled_pits:
            b.filled.add(c)
        for c in sorted(obs.blocked_cells):
            if b.status(c) not in ("confirmed", "filled"):
                b.observe_cell(c, True, t)
        for c in sorted(obs.scanned - obs.blocked_cells):
            if b.status(c) in ("reported", "confirmed"):
                b.observe_cell(c, False, t)
        self.sensed_cells = obs.robot_cells
        if self._snapshot_view() != before:
            self.replan_needed = True

    def receive(self, msgs: List[Message], t: int) -> None:
        before = self._snapshot_view()
        for m in msgs:
            if m.kind == "STATE":
                self.belief.merge(m.payload)
            elif m.kind == "INTENT":
                start_tick, blocked_ticks, cells = m.payload
                old = self.intents.get(m.sender)
                if old is None or start_tick >= old.start_tick:
                    self.intents[m.sender] = IntentRecord(start_tick, blocked_ticks, cells)
        if self._snapshot_view() != before:
            self.replan_needed = True

    def _reservations(self, t: int) -> Set[Tuple[Pos, int]]:
        if self._reserved_cache[0] == t:
            return self._reserved_cache[1]
        own = (-self.blocked_ticks, self.id)
        reserved: Set[Tuple[Pos, int]] = set()
        explained: Set[Pos] = set()
        for sender, rec in sorted(self.intents.items()):
            age = t - rec.start_tick
            if age < 0 or age > self.cfg.intent_window:
                continue
            if age < len(rec.cells):
                explained.add(rec.cells[age])
            if (-rec.blocked_ticks, sender) < own:
                for k, cell in enumerate(rec.cells):
                    if rec.start_tick + k >= t:
                        reserved.add((cell, rec.start_tick + k))
            else:
                nxt = rec.cells[min(age + 1, len(rec.cells) - 1)]
                reserved.add((nxt, t + 1))
        for cell in self.sensed_cells:
            if cell not in explained:
                reserved.add((cell, t + 1))
        self._reserved_cache = (t, reserved)
        return reserved

    def _plan_conflict(self, t: int) -> bool:
        reserved = self._reservations(t)
        for k in range(1, min(self.cfg.intent_window, len(self.plan) - 1) + 1):
            if (self.plan[k], t + k) in reserved:
                return True
            if (self.plan[k] != self.plan[k - 1] and (self.plan[k - 1], t + k) in reserved
                    and (self.plan[k], t + k - 1) in reserved):
                return True
        return False

    def _passable(self):
        return passable_fn(self.grid, frozenset(self.belief.filled.items()), closed=self.belief.believed_blocked())

    def _begin_task(self, t: int) -> None:
        b = self.belief
        filled = frozenset(b.filled.items())
        d = dream_path(self.grid, b.editable(), filled, self.pos, self.goal, self.U, b.hard_blocked())
        singles: Dict[Pos, int] = {}
        if self.policy.needs_single_rents:
            singles = single_pit_rents(self.grid, b.editable(), filled, self.pos, self.goal, self.U,
                                       b.hard_blocked())
        info = TaskPlanInfo(self.id, self.task_idx, t, self.pos, self.goal, d.d_block, d.d_open,
                            tuple(sorted(d.bundle)), d.rent, filled, b.believed_blocked(),
                            b.hard_blocked(), singles)
        self.plan_infos.append(info)
        self.stats["rent_counted"] += info.rent
        self._planned_task = self.task_idx
        self.policy.on_task_planned(self, info, t)

    def _replan(self, t: int) -> None:
        pf = self._passable()
        h = bfs_dist_map(pf, self.goal, self.H, self.W)
        path = plan_spacetime(pf, self.pos, self.goal, t, self._reservations(t), h, 3 * (self.H + self.W))
        self.plan = path if path else [self.pos]
        self.replan_needed = False
        self.blocked_streak = 0
        self.stats["replans"] += 1

    def decide(self, t: int) -> Action:
        new_task = self._planned_task != self.task_idx
        if new_task:
            self._begin_task(t)
        if (new_task or self.replan_needed or self.blocked_streak >= 2 or self._plan_conflict(t)
                or (len(self.plan) <= 1 and self.pos != self.goal)):
            self._replan(t)
        action: Action = Move(self.plan[1]) if len(self.plan) > 1 and self.plan[1] != self.plan[0] else Wait()
        self._last_action = action
        return action

    def outgoing(self, t: int) -> List[Message]:
        cells = tuple(self.plan[: self.cfg.intent_window + 1])
        out = [Message(self.id, "INTENT", (t, self.blocked_ticks, cells), len(cells), t)]
        if self.policy.uses_gossip and t % self.cfg.gossip_period == 0:
            out.append(Message(self.id, "STATE", self.belief.snapshot(), self.belief.units(), t))
        return out

    def after_action(self, res: ActionResult, t: int) -> None:
        action = self._last_action
        self.pos = res.pos
        self.blocked_ticks = res.blocked_ticks
        if isinstance(action, Move):
            if res.ok:
                self.plan = self.plan[1:] or [self.pos]
                self.blocked_streak = 0
            else:
                self.blocked_streak += 1
                if res.reason == "wall":
                    self.belief.observe_cell(action.to, True, t)
                    self.replan_needed = True
        elif isinstance(action, Wait) and len(self.plan) > 1:
            self.plan = self.plan[1:]
        if self.pos == self.goal:
            self.task_idx += 1
            self.replan_needed = True
            if self.task_idx >= len(self.tasks):
                self.finished = True
                self.completed_tick = t
