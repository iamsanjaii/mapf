"""EvidenceEngine: what the ledger says a push would have saved the fleet.

The ledger holds one origin/destination record per task a robot planned (aggregated per epoch when
`record_epoch` is set), whether or not that task paid any rent. The fleet cost of a set of blocked cells is the
sum of the shortest-route lengths of all recorded tasks. The evidence for a push is the fall in that sum when the
obstacle moves from its cell to its landing cell, so an obstacle landing on a route someone used counts against it.

Two details keep the evidence honest. A past record's own start and goal cells are treated as open (a task that
is over cannot be blocked at its goal; a future one could pick any cell). The robot's own task in hand is the
exception: it is counted from where the robot stands now, with a blocked goal cell really unreachable, so a robot
whose goal has an obstacle on it always sees the full cost of leaving it there.
"""
from typing import Dict, FrozenSet, Optional, Tuple

from src.doi.belief import BeliefState
from src.doi.paths import bfs_dist_map, passable_fn
from src.environment.grid import Grid

Pos = Tuple[int, int]
MAX_CACHED_MAPS = 4000


class EvidenceEngine:
    def __init__(self, grid: Grid, unreachable: float, record_epoch: Optional[int] = None) -> None:
        self.grid = grid
        self.unreachable = unreachable
        self.record_epoch = record_epoch
        self._maps: Dict[Tuple[Pos, FrozenSet[Pos]], Dict[Pos, int]] = {}

    def distance(self, origin: Pos, dest: Pos, blocked: FrozenSet[Pos]) -> float:
        key = (origin, blocked)
        dist = self._maps.get(key)
        if dist is None:
            if len(self._maps) >= MAX_CACHED_MAPS:
                self._maps.clear()
            dist = self._maps[key] = bfs_dist_map(passable_fn(self.grid, closed=blocked), origin,
                                                  self.grid.height, self.grid.width)
        return float(dist.get(dest, self.unreachable))

    def fleet_cost(self, belief: BeliefState, blocked: FrozenSet[Pos],
                   skip: Optional[Tuple[int, int, Pos, Pos]] = None) -> float:
        """Total route length of every recorded task if `blocked` were the obstacle cells.

        `skip` = (robot, task index, start, goal) leaves that one task out (the caller counts it separately); in
        aggregated mode one traversal is taken off the latest matching entry."""
        total = 0.0
        for r in belief.records.records():
            if skip is not None and (r.robot, r.task_idx) == skip[:2]:
                continue
            total += self.distance(r.origin, r.dest, blocked - {r.origin, r.dest})
        entries = belief.agg.entries()
        latest = -1
        if skip is not None:
            latest = max((k[3] for k, _ in entries if k[:3] == (skip[0], skip[2], skip[3])), default=-1)
        for (robot, origin, dest, epoch), (_rent_sum, count) in entries:
            if skip is not None and (robot, origin, dest, epoch) == (skip[0], skip[2], skip[3], latest):
                count -= 1
            total += count * self.distance(origin, dest, blocked - {origin, dest})
        return total

    def parts(self, belief: BeliefState, before: FrozenSet[Pos], after: FrozenSet[Pos],
              own: Optional[Tuple[int, int, Pos, Pos, Pos]] = None) -> Tuple[float, float]:
        """(saving on the recorded traffic, saving on the robot's own task in hand).

        `own` = (robot, task index, position, goal, task start) counts the task in hand from the robot's current
        position instead of from the start recorded in the ledger."""
        skip = (own[0], own[1], own[4], own[3]) if own is not None else None
        ledger = self.fleet_cost(belief, before, skip) - self.fleet_cost(belief, after, skip)
        mine = 0.0
        if own is not None:
            mine = self.distance(own[2], own[3], before) - self.distance(own[2], own[3], after)
        return ledger, mine

    def evidence(self, belief: BeliefState, before: FrozenSet[Pos], after: FrozenSet[Pos],
                 own: Optional[Tuple[int, int, Pos, Pos, Pos]] = None) -> float:
        """Travel the recorded tasks would have saved had the obstacles stood on `after` instead of `before`."""
        return sum(self.parts(belief, before, after, own))
