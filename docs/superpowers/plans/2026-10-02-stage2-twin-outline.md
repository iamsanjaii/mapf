# Stage 2 outline: digital-twin-in-the-loop (one page, to be planned separately)

Status: outline only. It is written before any Stage 1 result exists (no real warehouse map has been run),
so everything that depends on those results is marked **pending**.

## Goal

Show the Rent-or-Fill core and the L2 layer working with real communication latency and loss and live
inference, in at least one twin scenario (spec section 9, required for submission).

## Scope

1. **Twin.** Gazebo or Isaac Sim warehouse with 6 to 12 AMRs, one aisle layout from Stage 0
   (`incidents_aisles`), obstructions spawned by a script at the same ticks as the simulator.
2. **Ledger transport.** Replace `network.py` by ROS 2 DDS topics: one topic for INTENT (traffic channel) and
   one for STATE (ledger channel). Loss and latency come from the transport (optionally shaped with `tc` or
   a DDS QoS profile), not from a model. The CRDT code in `crdt.py` and `belief.py` is reused unchanged;
   delta messages (`BeliefState.delta_since`) are the on-wire format.
3. **Live L2.** Run `llm/intake.py` live per report against the hosted model, and optionally a small local
   model on edge hardware, measuring latency and energy per report. Keep the invariant: nothing under
   `src/doi/llm/` is imported by the decision path.
4. **Supervisor.** A console process that shows `draft_request` text and returns approve or veto; a
   scripted responder for repeatable runs.
5. **Clock.** Replace the global tick with a fixed control period per robot; document skew as out of scope
   (spec section 11).

## Deliverables

A launch file, a thin ROS 2 node wrapping `RobotAgent`, a recorded run per condition (ledger on and off,
gate on and off, intake none, oracle and live), and a table of simulator-versus-twin differences.

## Pending on Stage 1 and Stage 0 results

* Which `r_comm` regime matters (decided by H2 and H6 on warehouse maps): determines the DDS range or
  topic partitioning to test. **Pending.**
* Whether the traffic layer needs the planning window and aggregated records at twin scale. **Pending.**
* Whether the hosted-model latency dominates the gate latency (H8). **Pending.**
* Open design decision: how robots localise a reported "aisle k bay b" in the twin (map lookup, as in
  `locate`, is assumed).

## Risks

DDS discovery and QoS tuning; twin physics making the one-cell shelf aisles harder than the grid; live LLM
variance (temperature 0 does not guarantee identical output).
