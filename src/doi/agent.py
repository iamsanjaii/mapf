"""RobotAgent: sense, receive, decide, act and broadcast using only the robot's own belief."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.doi.belief import CLASS_CODE, BeliefState
from src.doi.config import SimConfig
from src.doi.network import Message
from src.doi.paths import bfs_dist_map, dream_path, passable_fn
from src.doi.policies import PushPolicy, Shared, trigger
from src.doi.pushplan import candidate_plans
from src.doi.pusher import Pusher
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
    belief_blocked: FrozenSet[Pos]
    hard_blocked: FrozenSet[Pos]


@dataclass(frozen=True)
class IntentRecord:
    start_tick: int
    blocked_ticks: int
    cells: Tuple[Pos, ...]


class RobotAgent:
    def __init__(self, rid: int, scenario: Scenario, cfg: SimConfig, policy: PushPolicy,
                 shared: Shared) -> None:
        self.id = rid
        self.cfg = cfg
        self.policy = policy
        self.shared = shared
        self.grid = scenario.grid
        self.H, self.W = self.grid.height, self.grid.width
        self.U = cfg.unreachable_cost_for(self.H, self.W)
        self.tasks: List[Pos] = list(scenario.tasks[rid])
        self.belief = BeliefState(rid, dict(scenario.obstacles))
        self.pos: Pos = scenario.starts[rid]
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
        self._pushing = False
        self.pusher = Pusher(self)
        self._push_key = None
        self._last_action: Optional[Action] = None
        self._reserved_cache: Tuple[int, Set[Tuple[Pos, int]]] = (-1, set())

    @property
    def goal(self) -> Pos:
        return self.tasks[self.task_idx]

    def _snapshot_view(self) -> FrozenSet[Pos]:
        return self.belief.believed_blocked()

    def sense(self, obs: Observation, t: int) -> None:
        before = self._snapshot_view()
        b = self.belief
        kinds = dict(obs.kinds)
        for c in sorted(obs.blocked_cells):
            if b.status(c) != "confirmed":
                b.observe_cell(c, True, t, kind=kinds.get(c))
            if c in obs.clearable:
                b.cls.raise_to(c, CLASS_CODE["robot_clearable"])
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
        return passable_fn(self.grid, closed=self.belief.believed_blocked())

    def _begin_task(self, t: int) -> None:
        b = self.belief
        d = dream_path(self.grid, b.editable(), self.pos, self.goal, self.U, b.hard_blocked())
        info = TaskPlanInfo(self.id, self.task_idx, t, self.pos, self.goal, d.d_block, d.d_open,
                            tuple(sorted(d.bundle)), d.rent, b.believed_blocked(), b.hard_blocked())
        self.plan_infos.append(info)
        self.stats["rent_counted"] += info.rent
        self._planned_task = self.task_idx
        self._plan_goal = None
        self.policy.on_task_planned(self, info, t)

    def _passable_to(self, goal: Pos):
        """Passability for planning: a goal cell believed blocked is still a target, so the robot walks up to it
        (and can then see what is on it) instead of waiting where it is."""
        pf = self._passable()
        return (lambda c: pf(c) or c == goal) if goal in self.belief.believed_blocked() else pf

    def _replan(self, t: int, goal: Pos) -> None:
        pf = self._passable_to(goal)
        reserved = self._reservations(t)
        sig = (goal, self.pos, self.belief.believed_blocked(), frozenset((c, k - t) for c, k in reserved))
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
                pf2 = passable_fn(self.grid, closed=self.belief.believed_blocked() - reported)
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
        pf = passable_fn(self.grid, closed=self.belief.believed_blocked())
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
            if self.plan[1] in self.belief.believed_blocked():      # next to a goal with an obstacle on it: wait
                return Wait()
            return Move(self.plan[1])
        if self.pos != goal and self.wait_streak >= DODGE_PATIENCE and u01(self.cfg.seed, t, self.id, 6) < 0.5:
            n = self._dodge(t)
            if n is not None:
                self.plan = [self.pos, n]
                return Move(n)
        return Wait()

    def decide(self, t: int) -> Action:
        action: Optional[Action] = None
        if self.pusher.active():
            action = self.pusher.step(t)
        if action is None:
            if self._planned_task != self.task_idx:
                self._begin_task(t)
            if not self.pusher.active():
                plan = self._consider_push(t)
                if plan is not None:
                    self.pusher.start(plan, t)
                    action = self.pusher.step(t)
        self._pushing = action is not None and self.pusher.active()
        if action is None:
            action = self.follow(self.goal, t)
        self._last_action = action
        return action

    def _consider_push(self, t: int):
        """The best push plan on this robot's own route, if the arm says push rather than go round."""
        if not self.policy.pushes or self.pos == self.goal:
            return None
        b = self.belief
        key = (b.version, self.pos, self.goal, tuple(sorted(c for c, u in self.deferred_until.items() if u > t)))
        if key == self._push_key:
            return None
        self._push_key = key
        blocked = b.believed_blocked()
        d = dream_path(self.grid, b.editable(), self.pos, self.goal, self.U, b.hard_blocked())
        if not d.bundle or d.rent <= 0:
            return None
        # Only obstacles this robot has seen and knows a robot may clear (never an unverified report, never a
        # class that needs a human) are candidates.
        eligible = [c for c in d.bundle
                    if b.status(c) == "confirmed" and b.cls.get(c) == CLASS_CODE["robot_clearable"]]
        if not eligible:
            return None
        engine = self.shared.engine
        skip = frozenset(c for c, u in self.deferred_until.items() if u > t)
        plans = candidate_plans(self.grid, engine.distance, blocked, eligible, b.kind_of, self.pos, self.goal,
                                self.cfg.kappa, self.cfg.fee, self.cfg.push_max, self.U, skip)
        if not plans:
            return None
        info = TaskPlanInfo(self.id, self.task_idx, t, self.pos, self.goal, d.d_block, d.d_open,
                            tuple(sorted(d.bundle)), d.rent, blocked, b.hard_blocked())
        best = None
        for plan in plans:                     # the plan with the best score for the whole fleet, not the cheapest for me
            verdict = self.policy.assess(self, plan, info)
            if verdict is not None and (best is None or (verdict[0], -plan.total) > (best[1][0], -best[0].total)):
                best = (plan, verdict)
        if best is None:
            return None
        plan, (_score, known, price) = best
        self.shared.triggers.append(trigger(t, self, plan, known, price))
        return plan

    def outgoing(self, t: int) -> List[Message]:
        cells = tuple(self.plan[: self.cfg.intent_window + 1])
        out = [Message(self.id, "INTENT", (t, self.blocked_ticks, cells), len(cells), t)]
        if self.policy.uses_gossip:
            due = t % self.cfg.gossip_period == 0
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
        self._recent = (self._recent + [self.pos])[-EVADE_WINDOW:]
        if self._pushing:
            self.pusher.on_result(action, res, t)
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
        if not self.pusher.active() and self.pos == self.goal:
            self.task_idx += 1
            self.replan_needed = True
            if self.task_idx >= len(self.tasks):
                self.finished = True
                self.completed_tick = t
