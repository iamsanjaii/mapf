# Prior-art verification (spec section 12)

Date read: 2026-10-02. **Evidence level: abstracts and search-result snippets only.** Spec section 12 asks for
full reads before any claim is written; none of the papers below was read in full, and the web search
tool used here is limited (US-only, snippets). Every verdict is therefore **provisional**, and a verdict of
`distinct` means "the abstract does not describe the thing this project claims", not "the paper does not
contain it". A full read of items 1 to 5 is still owed before submission.

Verdict vocabulary: `distinct`, `overlaps on <what>`, `scooped`.

## Reading tasks (section 12, items 1 to 5)

| # | Paper | What it does (from the abstract) | Difference from this project | Verdict |
|---|---|---|---|---|
| 1a | Vainshtein, Solovey, Salzman, "Multi-Agent Terraforming" / tMAPF, arXiv [2203.10540](https://arxiv.org/abs/2203.10540) | Some agents move obstacles (pods, shelves) in automated warehouses; CBS and PBS extended to decide where and when obstacles move. Start and goal locations are given. | Centralised, offline, jointly optimised, all tasks known. No online decision, no information limits, no purchase-style cost accounting. Section 12 asks to confirm "centralised, offline": confirmed at abstract level. | distinct |
| 1b | Andreychuk, Yakovlev, "Path Finding for the Coalition of Co-operative Agents Acting in the Environment with Destructible Obstacles", arXiv [1807.00771](https://arxiv.org/abs/1807.00771) | Some agents destroy obstacles so others get shorter paths; a procedure picks which obstacles to remove inside Theta*; about 9 to 12% mission-time saving. | The abstract does not say "centralised" or "offline"; it describes one planner choosing removals for the coalition, so it is a planning-time, known-task method. The online, decentralised, information-delayed setting is absent. Re-check the full text. | distinct (provisional) |
| 2 | Wang, Sun, Beyhaghi, Lui, Hajiesmaili, Wierman, "Competitive Algorithms for Multi-Agent Ski-Rental Problems", arXiv [2507.15727](https://arxiv.org/abs/2507.15727) | Agents choose daily rental, individual purchase or a discounted group pass; agents exit over time; deterministic and randomised state-aware thresholds; ratios for overall, state-dependent and individual-rational objectives. | No space, routing, communication limits or physical execution. The abstract mentions no no-communication lower bound and no `n + 1` ratio, so P3 (isolation) is not shown as present. P1 is classical ski rental and is not claimed as new. | distinct (provisional); P3 novelty unconfirmed until the full text is read |
| 3 | Meyerson, "Online Facility Location", FOCS 2001; "The Parking Permit Problem", FOCS 2005 | Randomised online facility location with `O(log n)` competitive ratio and a matching lower bound for the adversarial order; permit purchase under changing demand. (Search snippets only.) | **Closest theory.** Centralised online algorithms with full information about arrivals; no robots, no execution cost, no information delay or loss, no per-robot views. The difference to state in the paper: here the facility (edit) is opened by a robot that must physically fetch a kit, from partial, delayed evidence. | distinct |
| 4 | Choi, Brunet, How, "Consensus-Based Decentralized Auctions for Robust Task Allocation" (CBBA), IEEE T-RO 2009 | Decentralised bundle auctions with consensus; converges to a conflict-free assignment of tasks of known value; robust to inconsistent situational awareness. | CBBA assigns tasks whose reward is known. RoF decides whether a shared task exists at all, from accumulated team rent. The hauler choice in section 4.5 is a lease with Lamport tickets, simpler than CBBA, and is not claimed as new. | distinct |
| 5 | Decentralised token-based MAPF | Found: a token-based bilateral negotiation approach (EUMAS 2021) and PRISM (arXiv 2505.08025, 2025). The 2024 Autonomous Agents and Multi-Agent Systems paper named in the spec was **not located** by search. | All concern path conflicts in a static environment; the decision variable is the path, never the map. | distinct for the two found; the 2024 paper is unverified |

## Search tasks (section 12, item 6)

| Query | What turned up | Verdict |
|---|---|---|
| "ski rental" with multi-robot, warehouse, MAPF | Nothing on ski rental in robotics; results were lifelong MAPF in warehouses (rolling-horizon collision resolution, caching-augmented MAPF). | no prior work found at this search depth |
| online with movable obstacles / permanent decentralised edits | Only decentralised collision avoidance with stationary and moving obstacles (an MIT result). Nothing about deciding to remove obstacles online. | no prior work found at this search depth |
| CRDT with multi-robot | General CRDT overviews and one bachelor thesis on a CRDT framework for autonomous robots (HAW Hamburg). | overlaps on "CRDTs in multi-robot systems in general", which spec 1.3 already says not to claim |
| human-on-the-loop with AMR | Generic human approval of machine-generated plans and explainable supervisor consoles. | overlaps on the generic idea of human approval; distinct on gating a permanent edit with numbers taken from a ledger |

## Items marked (check) in spec section 1 that were not verified

Guidance-graph optimisation for lifelong MAPF (Zhang et al. 2024); Svancara et al. AAAI 2019; market-based
MRTA (Dias et al. 2006). They were not searched individually. Their rows in spec section 1
still say "(check)".

## Pivot rules

None is triggered at this evidence level: nothing found does decentralised online permanent map edits for
robots (rule 1). Rule 1 is the claim to re-test hardest with the full texts, in particular
items 1b and 2.
