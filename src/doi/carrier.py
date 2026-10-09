"""Carrier: the small state machine that carries out one carry or fill plan.

APPROACH  walk to the cell beside the obstacle (the ordinary follower plans the walk)
HAULING   walk, loaded, to the cell beside the slot or pit, then drop
A plan is abandoned, and the obstacle put on a short cooldown, if it is gone, the walk to it takes too long, or the
world refuses the pick-up. Once loaded, a robot does not abandon the load: if its target turns out to be full (or
the pit already filled) it picks the cheapest remaining target, and if there is none it carries on with its task
loaded until one turns up (see RobotAgent.decide).
"""
import dataclasses
from enum import Enum
from typing import Optional

from src.doi.carryplan import CarryPlan
from src.doi.world import Action, ActionResult, Drop, Pick

COOLDOWN = 25


class CarryState(Enum):
    NONE = 0
    APPROACH = 1
    HAULING = 2


class Carrier:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.state = CarryState.NONE
        self.plan: Optional[CarryPlan] = None
        self.started = 0
        self.stats = {"started": 0, "completed": 0, "aborted": 0, "retargeted": 0}

    def active(self) -> bool:
        return self.state != CarryState.NONE

    def start(self, plan: CarryPlan, t: int) -> None:
        self.plan, self.state, self.started = plan, CarryState.APPROACH, t
        self.stats["started"] += 1

    def resume(self, kind: str, mode: str, target, access, t: int) -> None:
        """A robot that is already loaded heads for a new target (no pick-up leg)."""
        self.plan = CarryPlan(obstacle=(-1, -1), kind=kind, mode=mode, approach=self.agent.pos, target=target,
                              access=access, walk_in=0.0, haul=0.0, walk_on=0.0, pick_cost=0.0, haul_cost=0.0,
                              drop_cost=0.0, before=frozenset(), after=frozenset())
        self.state, self.started = CarryState.HAULING, t
        self.stats["started"] += 1

    def _abort(self, t: int) -> None:
        a = self.agent
        a.deferred_until[self.plan.obstacle] = t + COOLDOWN
        self.state, self.plan = CarryState.NONE, None
        a.invalidate_plan()
        self.stats["aborted"] += 1

    def _target_gone(self) -> bool:
        b, plan = self.agent.belief, self.plan
        if plan.mode == "carry":
            return plan.target in b.slots_full
        return plan.target not in b.believed_blocked()          # the pit is believed filled

    def step(self, t: int) -> Optional[Action]:
        a, plan = self.agent, self.plan
        if self.state == CarryState.APPROACH:
            if plan.obstacle not in a.belief.believed_blocked() or t - self.started > 4 * max(1, int(plan.walk_in)) + 20:
                self._abort(t)
                return None
            if a.pos != plan.approach:
                return a.follow(plan.approach, t)
            return Pick(plan.obstacle)
        if self._target_gone():
            retarget = a.unload_target(t)
            if retarget is None:                                 # nowhere to put it: carry on with the task
                self.state, self.plan = CarryState.NONE, None
                a.invalidate_plan()
                return None
            self.plan = plan = dataclasses.replace(plan, mode=retarget[0], target=retarget[1], access=retarget[2])
            self.stats["retargeted"] += 1
        if a.pos != plan.access:
            return a.follow(plan.access, t)
        return Drop(plan.target)

    def on_result(self, action: Optional[Action], res: ActionResult, t: int) -> None:
        a, plan = self.agent, self.plan
        if plan is None:
            return
        if isinstance(action, Pick):
            if res.ok:
                self.state = CarryState.HAULING
                a.load = plan.kind
                a.belief.observe_cell(plan.obstacle, False, t + 1)
                a.invalidate_plan()
            else:
                self._abort(t)
        elif isinstance(action, Drop):
            if res.ok:
                if plan.mode == "carry":
                    a.belief.slots_full.add(plan.target)
                else:
                    a.belief.observe_cell(plan.target, False, t + 1)
                a.load = None
                self.state, self.plan = CarryState.NONE, None
                a.invalidate_plan()
                self.stats["completed"] += 1
            elif res.reason in ("full", "no_target"):
                if plan.mode == "carry":
                    a.belief.slots_full.add(plan.target)
                else:
                    a.belief.observe_cell(plan.target, False, t + 1)
