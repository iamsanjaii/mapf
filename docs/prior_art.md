# Prior-art check: removable obstacles / environment modification in MAPF, and congestion pricing

Searched 2026-09-30 with web search. Only abstracts and search snippets were read, not full papers, so
"does not do X" statements below are unverified until the full texts are checked.

## 1. Removable obstacles and environment modification in MAPF

Prior work already covers the general idea, so novelty cannot rest on "MAPF where obstacles can be removed or moved".

| Work | What it does | Relevance |
|---|---|---|
| [Multi-Agent Terraforming (Vainshtein, Solovey, Salzman, 2022)](https://arxiv.org/abs/2203.10540) | tMAPF: some agents move obstacles (warehouse pods) to open routes and relieve congestion; extends CBS and PBS; beats the best static-obstacle solution. | Closest match. Obstacle moves are planned jointly with agent collisions. |
| [Path Finding for a Coalition of Co-operative Agents with Destructible Obstacles (Andreychuk, Yakovlev, ICR 2018)](https://arxiv.org/abs/1807.00771) | Some agents destroy obstacles so others take shorter routes; obstacle-selection procedure inside Theta*; 9-12% shorter mission time. | Direct precedent for "remove an obstacle to shorten others' paths". |
| [Conflict-Based Search and Prioritized Planning for M-PAMO](https://arxiv.org/abs/2509.26050) | Multi-agent Path finding Among Movable Obstacles; fuses CBS/PP with the single-agent PAMO* planner. | Same problem family, 2025. |
| [Symbolic Planning and MAPF in Extremely Dense Environments with Movable Obstacles](https://arxiv.org/pdf/2509.01022) | Movable obstacles in dense MAPF. | Same family. |
| [Search-Based Path Planning among Movable Obstacles](https://arxiv.org/abs/2410.18333) | Single-agent PAMO (PAMO*). | Single-agent foundation. |
| NAMO literature (e.g. [NAMOSIM](https://www.theoj.org/joss-papers/joss.08816/10.21105.joss.08816.pdf), [Multi-Object Pushing into Storage Zones](https://discovery.ucl.ac.uk/id/eprint/10165039/)) | Navigation among movable obstacles, including multi-robot variants. | Robotics-side lineage. |
| [D-MAPF (dynamic MAPF)](https://easychair.org/publications/download/n21z) | Obstacles may appear, disappear or move over time. | Environment change as an input, not a decision. |
| [Multi-Robot Coordination and Layout Design for Automated Warehousing](https://arxiv.org/pdf/2305.06436) | Optimizes shelf layout to raise MAPF throughput (MAP-Elites, surrogate models). | Offline environment modification, not per-run. |
| [Strategic Infrastructure Design via Multi-Agent Congestion Games (2026)](https://arxiv.org/abs/2603.21691) | Joint placement and pricing of infrastructure as bi-level optimization. | Environment modification plus pricing together, in a non-atomic setting. |

## 2. Negotiated / priced congestion in routing

| Work | What it does | Relevance |
|---|---|---|
| [Decentralized MAPF framework with token-based negotiation (Springer, Autonomous Agents and Multi-Agent Systems, 2024)](https://link.springer.com/article/10.1007/s10458-024-09639-8) | Agents use tokens to manage negotiation and reach agreements on paths. Only the snippet was read; the page needed a login redirect. | Closest to "negotiated" resource use among path-finding agents. Read the full text before claiming a difference. |
| [Agent-based congestion pricing with heterogeneous values of travel time savings](https://www.researchgate.net/publication/303028714_Agent-based_Congestion_Pricing_and_Transport_Routing_with_Heterogeneous_Values_of_Travel_Time_Savings) | Iterative market mechanism; per-user tolls from the delay imposed on others. | Classic marginal-cost pricing in agent-based routing. |
| [Marginal Congestion Cost Pricing in a Multi-agent Simulation (Greater Berlin)](https://www.researchgate.net/publication/282908335_Marginal_Congestion_Cost_Pricing_in_a_Multi-agent_Simulation_Investigation_of_the_Greater_Berlin_Area) | Marginal-cost tolls in MATSim. | Same. |
| [Reinforcement Learning and Mechanism Design for Routing (AAMAS 2023)](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p2988.pdf) | Mechanism design for routing agents. | Mechanism-design angle. |
| [Branch-and-cut-and-price for MAPF](https://www.sciencedirect.com/science/article/abs/pii/S0305054822000946) | Column generation with pricing subproblems for MAPF. | "Pricing" in the LP sense, not tolls. Do not conflate. |
| Standard result: optimal dual variables on capacity constraints are edge prices that induce the social optimum in non-atomic routing games | Lagrangian / marginal-cost pricing theory. | Any dual-price or toll scheme in this repo would be an instance of this classical result. |

## 3. What this means for this repo

- **Do not claim** that MAPF with removable/destructible/movable obstacles is new. Terraforming MAPF (2022) and the 2018
  destructible-obstacles paper both predate `src/mapf_ro/`. Rename or reframe "MAPF-RO" as a variant of, or comparison to,
  these, and cite them.
- **Do not claim** that trading removal cost against detour cost is new by itself; that is the core of both papers.
- **Possible narrow angle, unverified:** amortizing one removal cost across the number of robots whose paths benefit
  (the traffic index), with the removal decision made by a cost agent inside a WAIT / DETOUR / REMOVE loop. Confirm against
  the full texts of the Terraforming and destructible-obstacles papers that neither does this before claiming it.
- **Claim C2 (congestion-pricing-like):** the original text of C2 was not available in this session, so it was not checked.
  If it means "agents negotiate a price/toll for shared cells or removals", the marginal-cost-pricing and token-negotiation
  work above is the prior art to distinguish from.
- **Code/README mismatch found while reading:** the README says `removal_cost = base_cost / traffic_index`, but
  `RemovalCostModel.estimate` in `src/mapf_ro/removal.py` computes `sandbag_travel + fill_cost` and only records
  `traffic_index`. The traffic index does not enter the decision. Fix one or the other before describing the mechanism.

## 4. Remaining gaps

- Full texts were not read; abstracts only.
- The 2024 Springer negotiation paper needs an authenticated or institutional read.
- Search was limited to web search. Also check Google Scholar, DBLP and the ICAPS, AAMAS, SoCS, IJCAI and IROS proceedings
  for "movable obstacles", "terraforming", "environment modification", "tMAPF", and "PAMO" follow-ups.
