"""Hauling finite-state machine: stagger, claim, fetch a kit, carry, apply, abort and return."""
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.doi.crdt import Claim
from src.doi.paths import bfs_dist_map, passable_fn
from src.doi.world import Action, ActionResult, Drop, Move, Pickup, Return, Wait

Pos = Tuple[int, int]
NEIGHBOURS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


class HaulState(Enum):
    NONE = 0
    PENDING = 1
    TO_DEPOT = 2
    CARRY = 3
    RETURN_BAG = 4
    AWAIT_APPROVAL = 5


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def select_haul(agent, pits: Tuple[Pos, ...], open_all: bool = False):
    """Nearest depot with stock, then the pit with the shortest carry. Returns (depot, pit, adj, d_pos_depot, d_carry)."""
    b = agent.belief
    filled = frozenset(b.filled.items())
    todo = [p for p in sorted(pits) if p not in filled]
    if open_all:
        pf = passable_fn(agent.grid, filled | frozenset(todo), closed=b.believed_blocked() - frozenset(todo))
    else:
        pf = passable_fn(agent.grid, filled, closed=b.believed_blocked())
    from_agent = bfs_dist_map(pf, agent.pos, agent.H, agent.W)
    depots = [d for d in sorted(agent.belief.stock.initial) if b.stock.remaining(d) > 0 and d in from_agent]
    depots.sort(key=lambda d: (from_agent[d], d))
    for depot in depots:
        from_depot = bfs_dist_map(pf, depot, agent.H, agent.W)
        best = None
        for pit in todo:
            for dr, dc in NEIGHBOURS:
                adj = (pit[0] + dr, pit[1] + dc)
                if adj in from_depot and adj not in pits:
                    cand = (from_depot[adj], pit, adj)
                    if best is None or cand < best:
                        best = cand
        if best is not None:
            return depot, best[1], best[2], from_agent[depot], best[0]
    return None


class Hauler:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.state = HaulState.NONE
        self.proposal = None
        self.todo: List[Pos] = []
        self.pit: Optional[Pos] = None
        self.depot: Optional[Pos] = None
        self.bag_depot: Optional[Pos] = None
        self.ticket = None
        self.offer_tick = 0
        self.ready_tick = 0
        self.claim_tick = 0
        self.cost = 0.0
        self.carried_since_pickup = 0
        self.wasted_steps = 0
        self.approval_wait = 0
        self.stats: Dict[str, int] = {"unconfirmed": 0, "issued": 0, "lost": 0, "aborts": 0}
        self.edits: List[dict] = []

    def active(self) -> bool:
        return self.state in (HaulState.TO_DEPOT, HaulState.CARRY, HaulState.RETURN_BAG)

    def holds_claim(self) -> bool:
        return self.ticket is not None

    def _finish(self, t: int) -> None:
        if self.ticket is not None:
            self.agent.belief.claims.release(self.ticket)
            self.ticket = None
        self.state = HaulState.NONE
        self.proposal = None
        self.pit = self.depot = None
        self.agent.invalidate_plan()

    def offer(self, proposal, t: int) -> None:
        a, cfg = self.agent, self.agent.cfg
        self.proposal = proposal
        self.offer_tick = t
        self.todo = [p for p in proposal.pits if p not in a.belief.filled]
        if not cfg.claim:
            self._start(t)
            return
        sel = select_haul(a, proposal.pits)
        if sel is None:
            self.proposal = None
            return
        low = select_haul(a, proposal.pits, open_all=True) or sel
        h = low[3] + cfg.kappa * low[4]
        self.ready_tick = t + min(cfg.stagger_cap, int(h // cfg.kappa))
        self.state = HaulState.PENDING

    def update(self, t: int) -> None:
        if self.state != HaulState.PENDING:
            return
        a = self.agent
        b = a.belief
        for p in self.proposal.pits:
            eff = b.claims.effective(p, t)
            if p in b.filled or (eff is not None and eff[1].hauler != a.id):
                self.state = HaulState.NONE
                self.proposal = None
                return
        if t >= self.ready_tick:
            self._start(t)

    def _start(self, t: int) -> None:
        a, cfg, b = self.agent, self.agent.cfg, self.agent.belief
        if any(b.status(p) != "confirmed" for p in self.proposal.pits):
            self.stats["unconfirmed"] += 1
            self.state = HaulState.NONE
            self.proposal = None
            return
        if cfg.gate:
            raise NotImplementedError("the approval gate is implemented in Task 15")
        self._launch(t, approval_wait=0)

    def assign(self, proposal, t: int) -> None:
        """Omniscient dispatch (central arm): no stagger, claim or approval gate."""
        self.proposal = proposal
        self.offer_tick = t
        self.todo = [p for p in proposal.pits if p not in self.agent.belief.filled]
        for p in proposal.pits:
            self.agent.belief.observe_cell(p, True, t)
        self._launch(t, approval_wait=0, use_claim=False)

    def _launch(self, t: int, approval_wait: int, use_claim: Optional[bool] = None) -> None:
        a, cfg, b = self.agent, self.agent.cfg, self.agent.belief
        use_claim = cfg.claim if use_claim is None else use_claim
        sel = select_haul(a, self.proposal.pits)
        if sel is None:
            self.state = HaulState.NONE
            self.proposal = None
            return
        if use_claim:
            self.ticket = b.next_ticket()
            b.claims.issue(self.ticket, Claim(tuple(self.proposal.pits), a.id), t + cfg.lease_ticks)
            self.stats["issued"] += 1
        self.depot, self.pit = sel[0], sel[1]
        self.claim_tick = t
        self.approval_wait = approval_wait
        self.cost = 0.0
        self.state = HaulState.TO_DEPOT

    def _abort(self, t: int, lost: bool = False) -> None:
        self.stats["aborts"] += 1
        if lost:
            self.stats["lost"] += 1
        if self.agent.carrying:
            self.wasted_steps += self.carried_since_pickup
            self.carried_since_pickup = 0
            self.state = HaulState.RETURN_BAG
            if self.ticket is not None:
                self.agent.belief.claims.release(self.ticket)
                self.ticket = None
        else:
            self._finish(t)

    def _revalidate(self, t: int) -> None:
        a, b = self.agent, self.agent.belief
        if a.cfg.claim and self.ticket is not None:
            b.claims.renew(self.ticket, t + a.cfg.lease_ticks)
        pit = self.pit
        if pit in b.filled or b.status(pit) == "refuted":
            self._abort(t)
            return
        if a.cfg.claim and self.ticket is not None:
            eff = b.claims.effective(pit, t)
            if eff is not None and eff[1].hauler != a.id and eff[0] < self.ticket:
                self._abort(t, lost=True)

    def _adjacent_target(self) -> Optional[Pos]:
        a = self.agent
        pf = passable_fn(a.grid, frozenset(a.belief.filled.items()), closed=a.belief.believed_blocked())
        dist = bfs_dist_map(pf, a.pos, a.H, a.W)
        cands = [((self.pit[0] + dr, self.pit[1] + dc)) for dr, dc in NEIGHBOURS]
        cands = [c for c in cands if c in dist and c not in (self.proposal.pits if self.proposal else ())]
        return min(cands, key=lambda c: (dist[c], c)) if cands else None

    def step(self, t: int) -> Optional[Action]:
        a = self.agent
        if self.state in (HaulState.TO_DEPOT, HaulState.CARRY):
            self._revalidate(t)
        if self.state == HaulState.TO_DEPOT:
            if a.pos == self.depot:
                return Pickup()
            return a.follow(self.depot, t)
        if self.state == HaulState.CARRY:
            if manhattan(a.pos, self.pit) == 1:
                return Drop(self.pit)
            target = self._adjacent_target()
            if target is None:
                self._abort(t)
                return self.step(t) if self.active() else None
            return a.follow(target, t)
        if self.state == HaulState.RETURN_BAG:
            if a.pos == self.bag_depot:
                return Return()
            return a.follow(self.bag_depot, t)
        return None

    def on_result(self, action: Action, res: ActionResult, t: int) -> None:
        a, cfg, b = self.agent, self.agent.cfg, self.agent.belief
        state = self.state
        carried_move = isinstance(action, Move) and res.ok and a.carrying
        if state == HaulState.RETURN_BAG:
            if carried_move:
                self.wasted_steps += 1
            if isinstance(action, Return) and res.ok:
                b.stock.give_back(self.bag_depot, a.id)
                self._finish(t)
            return
        if state in (HaulState.TO_DEPOT, HaulState.CARRY):
            self.cost += cfg.kappa if carried_move else 1.0
            if carried_move:
                self.carried_since_pickup += 1
        if state == HaulState.TO_DEPOT and isinstance(action, Pickup):
            if res.ok:
                b.stock.take(self.depot, a.id)
                self.bag_depot = self.depot
                self.carried_since_pickup = 0
                self.state = HaulState.CARRY
            elif res.reason == "empty":
                b.stock.mark_empty(self.depot, a.id)
                sel = select_haul(a, tuple(self.todo))
                if sel is None:
                    self._abort(t)
                else:
                    self.depot, self.pit = sel[0], sel[1]
        elif state == HaulState.CARRY and isinstance(action, Drop):
            if res.ok:
                self._on_drop_success(t)
            elif res.reason == "needs_human":
                b.mark_needs_human(action.pit)
                self._abort(t)
            else:
                if res.reason == "not_blocked":
                    b.observe_cell(action.pit, False, t)
                self._abort(t)

    def _on_drop_success(self, t: int) -> None:
        a, cfg, b = self.agent, self.agent.cfg, self.agent.belief
        pit = self.pit
        b.filled.add(pit)
        cost = self.cost + cfg.fee
        per_pit = self.proposal.buy_per_pit
        idx = list(self.proposal.pits).index(pit)
        b_est = per_pit[idx] if per_pit is not None else self.proposal.buy / len(self.proposal.pits)
        self.edits.append({"pit": pit, "trigger_tick": self.offer_tick, "claim_tick": self.claim_tick,
                           "fill_tick": t, "B_est": float(b_est), "B_real": float(cost),
                           "approval_wait": self.approval_wait})
        self.todo = [p for p in self.todo if p != pit]
        self.cost = 0.0
        self.carried_since_pickup = 0
        if self.todo:
            sel = select_haul(a, tuple(self.todo))
            if sel is not None:
                self.depot, self.pit = sel[0], sel[1]
                self.state = HaulState.TO_DEPOT
                return
        self._finish(t)
