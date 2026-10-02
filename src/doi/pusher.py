"""Pusher: the small state machine that carries out one push plan on the robot's own route.

APPROACH  walk to the cell behind the obstacle (the ordinary follower plans the walk)
PUSHING   step into the obstacle `steps` times in a straight line
A plan is abandoned, and the obstacle put on a short cooldown, if it is no longer where the plan expects it,
the approach takes too long, or the world refuses a push.
"""
from enum import Enum
from typing import Optional

from src.doi.belief import CLASS_CODE
from src.doi.pushplan import PushPlan
from src.doi.world import Action, ActionResult, Push

COOLDOWN = 25


class PushState(Enum):
    NONE = 0
    APPROACH = 1
    PUSHING = 2


class Pusher:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.state = PushState.NONE
        self.plan: Optional[PushPlan] = None
        self.done = 0
        self.started = 0
        self.stats = {"started": 0, "completed": 0, "aborted": 0}

    def active(self) -> bool:
        return self.state != PushState.NONE

    def start(self, plan: PushPlan, t: int) -> None:
        self.plan, self.state, self.done, self.started = plan, PushState.APPROACH, 0, t
        self.stats["started"] += 1

    def _abort(self, t: int) -> None:
        a = self.agent
        a.deferred_until[self.plan.obstacle] = t + COOLDOWN
        self.state, self.plan = PushState.NONE, None
        a.invalidate_plan()
        self.stats["aborted"] += 1

    def _gone(self) -> bool:
        return self.plan.obstacle not in self.agent.belief.believed_blocked()

    def step(self, t: int) -> Optional[Action]:
        a, plan = self.agent, self.plan
        if self.state == PushState.APPROACH:
            if self._gone() or t - self.started > 4 * max(1, int(plan.walk_in)) + 20:
                self._abort(t)
                return None
            if a.pos != plan.approach:
                return a.follow(plan.approach, t)
            self.state = PushState.PUSHING
        k = self.done
        return Push((plan.obstacle[0] + k * plan.direction[0], plan.obstacle[1] + k * plan.direction[1]))

    def on_result(self, action: Optional[Action], res: ActionResult, t: int) -> None:
        if self.state != PushState.PUSHING or not isinstance(action, Push):
            return
        if res.ok:
            self.done += 1
            self.agent.belief.cls.raise_to(self.plan.landing_after(self.done), CLASS_CODE["robot_clearable"])
            if self.done >= self.plan.steps:
                self.state, self.plan = PushState.NONE, None
                self.agent.invalidate_plan()
                self.stats["completed"] += 1
        else:
            if res.reason == "needs_human":
                self.agent.belief.mark_needs_human(self.plan.obstacle)
            self._abort(t)
