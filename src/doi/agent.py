"""RobotAgent: sense, receive, decide, act and broadcast using only the robot's own belief."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.doi.belief import BeliefState
from src.doi.config import SimConfig
from src.doi.hauler import Hauler, HaulState
from src.doi.network import Message
from src.doi.paths import bfs_dist_map, dream_path, passable_fn, single_pit_rents
from src.doi.policies import FillPolicy, Shared
from src.doi.rng import u01
from src.doi.scenarios import Scenario
from src.doi.spacetime import plan_spacetime
from src.doi.world import Action, ActionResult, Move, Observation, Wait

Pos = Tuple[int, int]
DODGE_PATIENCE = 6
EVADE_WINDOW = 16
EVADE_TIMEOUT = 40


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
        self._prev_sensed: FrozenSet[Pos] = frozenset()
        self.wait_streak = 0
        self.deferred_until: Dict[Pos, int] = {}
        self._fail_sig = None
        self._last_sent = 0
        self._replan_at = 1 << 60
        self.evade: Optional[Tuple[Pos, Pos, int]] = None
        self._recent: List[Pos] = []
        self._planned_task = -1
        self._plan_goal: Optional[Pos] = None
        self._hauled = False
        self.hauler = Hauler(self)
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
        self._prev_sensed = self.sensed_cells
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

    def ingest_record(self, rec, t: int) -> None:
        self.belief.add_obstruction(rec, t)
        self.replan_needed = True

    def ingest_decision(self, d, t: int) -> None:
        self.belief.approvals.set(d.key, d.decision)
        for cell in d.needs_human:
            self.belief.mark_needs_human(cell)
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
                idx = min(age, len(rec.cells) - 1)
                if idx + 1 >= len(rec.cells):
                    reserved.add((rec.cells[-1], t + 1))
                for k in range(idx + 1, len(rec.cells)):
                    if rec.cells[k] != rec.cells[idx]:
                        break
                    reserved.add((rec.cells[k], rec.start_tick + k))
        horizon = 3 * (self.H + self.W) if self.wait_streak >= 2 else 1
        for cell in self.sensed_cells:
            if cell not in explained or cell in self._prev_sensed:
                for k in range(1, horizon + 1):
                    reserved.add((cell, t + k))
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
        self._plan_goal = None
        self.policy.on_task_planned(self, info, t)

    def _replan(self, t: int, goal: Pos) -> None:
        pf = self._passable()
        reserved = self._reservations(t)
        sig = (goal, self.pos, self.belief.believed_blocked(), len(self.belief.filled.items()),
               frozenset((c, k - t) for c, k in reserved))
        self.replan_needed = False
        self.blocked_streak = 0
        self._plan_goal = goal
        if sig == self._fail_sig:
            self.plan = [self.pos]
            return
        h = bfs_dist_map(pf, goal, self.H, self.W)
        max_len = 3 * (self.H + self.W)
        if self.pos in h:
            max_len = min(max_len, 2 * h[self.pos] + 2 * self.cfg.intent_window + 10)
        window = self.cfg.plan_window
        path = plan_spacetime(pf, self.pos, goal, t, reserved, h, max_len, window)
        self._replan_at = t + max(1, window // 2) if window else 1 << 60
        if not path:
            reported = frozenset(c for c in self.belief.believed_blocked() if self.belief.status(c) == "reported")
            if reported:
                pf2 = passable_fn(self.grid, frozenset(self.belief.filled.items()),
                                  closed=self.belief.believed_blocked() - reported)
                path = plan_spacetime(pf2, self.pos, goal, t, reserved, bfs_dist_map(pf2, goal, self.H, self.W),
                                      max_len, window)
        self.plan = path if path else [self.pos]
        self._fail_sig = None if path else sig
        self.stats["replans"] += 1

    def _dodge(self, t: int) -> Optional[Pos]:
        """Deterministic random side-step used to dissolve long mutual waits (packed queues at doorways)."""
        pf = self._passable()
        reserved = self._reservations(t)
        cands = []
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            n = (self.pos[0] + dr, self.pos[1] + dc)
            if pf(n) and n not in self.sensed_cells and (n, t + 1) not in reserved:
                cands.append(n)
        if not cands:
            return None
        return cands[int(u01(self.cfg.seed, t, self.id, 5) * len(cands)) % len(cands)]

    def invalidate_plan(self) -> None:
        self._plan_goal = None
        self.replan_needed = True

    def _oscillating(self) -> bool:
        return len(self._recent) >= EVADE_WINDOW and len(set(self._recent)) <= 3

    def _blocker(self, goal: Pos) -> Optional[Pos]:
        """A stationary neighbouring robot sitting on a shortest route to the goal."""
        stationary = self.sensed_cells & self._prev_sensed
        near = [c for c in stationary if abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]) == 1]
        if not near:
            return None
        pf = passable_fn(self.grid, frozenset(self.belief.filled.items()), closed=self.belief.believed_blocked())
        h = bfs_dist_map(lambda c: pf(c) or c in near, goal, self.H, self.W)
        mine = h.get(self.pos)
        on_route = [c for c in sorted(near) if mine is not None and h.get(c, 1 << 30) < mine]
        return on_route[0] if on_route else None

    def _evade_target(self, blocker: Pos, t: int) -> Optional[Pos]:
        pf = self._passable()
        avoid = {blocker}
        for sender, rec in self.intents.items():
            age = t - rec.start_tick
            if 0 <= age < len(rec.cells) and rec.cells[age] == blocker:
                avoid |= set(rec.cells[age:])
        occupied = set(self.sensed_cells)
        dist = {self.pos: 0}
        frontier = [self.pos]
        while frontier:
            nxt = []
            for cur in frontier:
                for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                    n = (cur[0] + dr, cur[1] + dc)
                    if n in dist or n in occupied or not pf(n):
                        continue
                    dist[n] = dist[cur] + 1
                    nxt.append(n)
            frontier = nxt
        cands = [c for c in dist if c != self.pos and c not in avoid]
        return min(cands, key=lambda c: (dist[c], c)) if cands else None

    def follow(self, goal: Pos, t: int) -> Action:
        if self.evade is not None:
            target, blocker, until = self.evade
            if t >= until or (self.pos == target and blocker not in self.sensed_cells):
                self.evade = None
            elif self.pos == target:
                return Wait()
            else:
                return self._follow_plan(target, t)
        if self.pos != goal and self._oscillating():
            blocker = self._blocker(goal)
            if blocker is not None:
                target = self._evade_target(blocker, t)
                if target is not None:
                    self.evade = (target, blocker, t + EVADE_TIMEOUT)
                    self._recent = []
                    self.invalidate_plan()
                    return self._follow_plan(target, t)
        return self._follow_plan(goal, t)

    def _follow_plan(self, goal: Pos, t: int) -> Action:
        if (self._plan_goal != goal or self.replan_needed or self.blocked_streak >= 2
                or self._plan_conflict(t) or t >= self._replan_at
                or (len(self.plan) <= 1 and self.pos != goal)):
            self._replan(t, goal)
        if len(self.plan) > 1 and self.plan[1] != self.plan[0]:
            return Move(self.plan[1])
        if self.pos != goal and self.wait_streak >= DODGE_PATIENCE and u01(self.cfg.seed, t, self.id, 6) < 0.5:
            n = self._dodge(t)
            if n is not None:
                self.plan = [self.pos, n]
                return Move(n)
        return Wait()

    def decide(self, t: int) -> Action:
        h = self.hauler
        h.update(t)
        action: Optional[Action] = None
        if h.active():
            action = h.step(t)
        if action is None:
            if self._planned_task != self.task_idx:
                self._begin_task(t)
            if h.state == HaulState.NONE:
                proposal = self.policy.propose(self, t)
                if proposal is not None:
                    h.offer(proposal, t)
                    h.update(t)
                    if h.active():
                        action = h.step(t)
        self._hauled = action is not None
        if action is None:
            action = self.follow(self.goal, t)
        self._last_action = action
        return action

    def outgoing(self, t: int) -> List[Message]:
        cells = tuple(self.plan[: self.cfg.intent_window + 1])
        out = [Message(self.id, "INTENT", (t, self.blocked_ticks, cells), len(cells), t)]
        if self.policy.uses_gossip:
            due = t % self.cfg.gossip_period == 0 or self.hauler.holds_claim()
            if self.cfg.delta_gossip:
                if t % self.cfg.full_sync_period == 0:
                    out.append(Message(self.id, "STATE", self.belief.snapshot(), self.belief.units(), t))
                    self._last_sent = self.belief.version
                elif due:
                    delta = self.belief.delta_since(self._last_sent)
                    self._last_sent = self.belief.version
                    if delta.units() > 0:
                        out.append(Message(self.id, "STATE", delta, delta.units(), t))
            elif due:
                out.append(Message(self.id, "STATE", self.belief.snapshot(), self.belief.units(), t))
        return out

    def after_action(self, res: ActionResult, t: int) -> None:
        action = self._last_action
        self.pos = res.pos
        self.blocked_ticks = res.blocked_ticks
        self.carrying = res.carrying
        self._recent = (self._recent + [self.pos])[-EVADE_WINDOW:]
        if self._hauled:
            self.hauler.on_result(action, res, t)
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
        self.wait_streak = self.wait_streak + 1 if isinstance(action, Wait) else 0
        if not self._hauled and self.pos == self.goal:
            self.task_idx += 1
            self.replan_needed = True
            if self.task_idx >= len(self.tasks):
                self.finished = True
                self.completed_tick = t
