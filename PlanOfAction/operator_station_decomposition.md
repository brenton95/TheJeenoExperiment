# Operator-Station Decomposition Design (Phase 16 gate artifact)

This file is the accepted design for the deferred Phase 16 refactor. It does not track the
current roadmap; see [task_plan.md](task_plan.md) for phase status.

Status: **accepted and banked.** This is the hard-prerequisite design the plan requires before
*any* `operator_station.py` code moves. It defines target modules, the shared-state map, the
`TurnOrchestrator` coupling fix, and an ordered green-able extraction sequence. It does not
move code; it is the reviewed gate artifact for Phase 16.

Sequencing decision: do **not** begin operator-station de-bloat before Phase 16, including the
safe leaf extractions. The active evidence, mission-termination, search, curriculum, and
cross-substrate work must shape the kernel boundary before code moves.

Snapshot at the current repository head: `operator_station.py` is **5,923 lines and 167 methods**.
`OperatorStationSession.__init__` directly initializes 41 attributes, while additional turn and
pending fields are property-backed. `TurnState` already owns the per-turn trace fields;
`CommandAuthority`, `SideEffectAuthority`, `MissionCortex`, `TurnOrchestrator`,
`PlanReuseCache`, and `PlanCache` already exist as separate collaborators. These counts are
diagnostic only; shared ownership is the problem, not the precise file length.

## Non-negotiable constraints (every extraction step must hold these)

- Public API unchanged: `handle_utterance` / `execute_command` / `reset` / `startup` / `close`,
  and the `session.last_*` / `session.pending_*` / collaborator read attributes (~40 evals
  depend on them).
- `CommandResult` is still the return type of every turn; every recorded result carries a
  `CorticalEnvelope`; `LabelledEpisode` still projects from `command_result`.
- Phase-9 ticket gates intact: `ExecutionTicket` / `RawMotorTicket` / `MemoryWriteTicket`.
- No LLM calls in the render loop. MiniGrid golden path (`go to the red door`) green.
- `eval_master` (all suites) + `pytest -q tests` green **after every step**, not just at the end.

## The core problem: shared mutable state, not file length

The reason this is high-blast-radius is one god-object passed around. `TurnOrchestrator`
currently reaches into **61 distinct session members (23 private)**. Naively moving methods into
new classes multiplies that coupling. So the design is **state-first**: define the shared
surface explicitly, then methods follow their state.

### Shared-state map

The ~62 attributes fall into four groups. The owner column is the *post-decomposition* home.

| Group | Members | Owner | Sharing mechanism |
|---|---|---|---|
| **Runtime collaborators** (construction-time, ~immutable) | `substrate`, `sense`, `spine`, `cortex`, `cortex_session`, `capability_registry`, `planning_semantics`, `operational_context`, `domain_helper`, `intent_verifier`, `intent_cache`, `mission_cortex`, `command_authority`, `side_effect_authority`, `representation`, `knowledge_base`, `knowledge_channel`, `memory`, `plan_cache`, `request_plan_reuse_cache`, `arbitrator`, `synthesizer`, `validator`, `compiler`, `runtime_package`, `prewarm_*` | a frozen **`StationRuntime`** context object | passed by reference to every service (read-only) |
| **Per-turn trace** | the 19 `TurnState` fields (done 13A.2.4) | `TurnState` (on `StationRuntime`) | one mutable object, written through the pipeline |
| **Continuation state** | `pending_clarification`, `pending_synthesis_proposal`, `pending_primitive_definition`, `active_claims` | a **`PendingState`** object (already property-backed) | mutable, owned by the pending-flow controller |
| **Session/config + misc** | `env_id`, `seed`, `render_mode`, `max_loops`, `verbose`, `startup_prewarm_summary`, `_turn_budget_exhausted`, `context_fingerprint` | stays on the facade | facade-local |

The key insight: services should receive a **`StationRuntime`** (collaborators + `turn_state` +
`pending_state`) — *not* the `OperatorStationSession`. That converts the `TurnOrchestrator`
god-object reach-in into a typed, narrow context. `OperatorStationSession` becomes a thin
facade that owns the `StationRuntime` and delegates.

## Target modules

Grouped from the current 167 methods. Counts approximate; each module is a plain class taking
`StationRuntime`.

| Module | ~Methods | Owns | Representative methods |
|---|---|---|---|
| `OperatorStationSession` (facade, stays) | ~20 | session lifecycle, REPL I/O, display summaries, delegation | `__init__`, `startup`, `reset`, `close`, `handle_utterance`, `execute_command`, `*_summary` |
| `CommandFactory` | ~18 | turn → typed `RequestPlan`/`ApprovedCommand` (the dispatch pipeline) | `command_from_llm_intent`, `command_from_selector_readiness`, `_command_from_*`, `_arbitrate_gap`, `_local_*_plan_and_graph`, `_build_*_request_plan`, `_record_request_*` |
| `TaskExecutor` | ~18 | approved command → execution + tickets + result (the run pipeline) | `run_task`, `_run_task_with_ticket`, `_run_procedure`, `_run_sequence`, `_run_motor_*`, `_run_mission`, `_execute_*`, `_execution_ticket_*`, `_raw_motor_ticket_*`, `_record_command_result` |
| `PendingFlowController` | ~24 | clarification / synthesis / primitive-definition state machine | `handle_pending_*`, `resume_*`, `propose_primitive_definition`, `_propose_synthesis`, `_approve_primitive_definition`, `maybe_start_selector_clarification`, `_try_synthesize_*`, `clarification_prompt` |
| `GroundingService` | ~28 | target grounding + metric semantics + claim writing | `ground_target_selector`, `_base_metric_distance`, `_ensure_ranked_door_claims`, `_resolve_metric_name`, `_evaluate_metric_expression`, `_compose_*`, `_write_ranked_claims`, `_set_last_grounded_claim`, grounding/format summaries |
| `ConceptService` | ~9 | named concepts + durable knowledge writes | `teach_concept`, `forget_concept`, `concepts_summary`, `apply_knowledge_update`, `_*_from_payload`, `_memory_write_ticket_for_payload`, `store_successful_task_memory` |
| `EnvironmentTracker` | ~7 | env identity, scene fingerprint, claim validity | `_build_environment_identity`, `_update_current_environment_identity`, `_claims_valid_for_current_environment`, `_scene_state_fingerprint`, `_task_family_for_env`, `_build_scene_summary_for_arbitrator` |

`TurnOrchestrator` stays as the top-level router but is rewired to call these services via
`StationRuntime` instead of reaching into `OperatorStationSession` privates.

## Ordered, green-able extraction sequence

Leaf-first (fewest back-references first), so each step is small and the suite stays green.

0. **Introduce `StationRuntime`** holding the runtime collaborators + `turn_state` +
   `pending_state`. Session builds it in `__init__` and exposes today's attributes as
   delegating properties (same pattern as `TurnState`). No behavior change. *Probe: runtime
   holds the collaborator set; public attrs still resolve.*
1. **`EnvironmentTracker`** (7 methods, leaf — depends only on runtime + scene). Lowest risk.
2. **`ConceptService`** (9 methods — durable-knowledge writes through tickets; self-contained).
3. **`GroundingService`** (28 methods — large but cohesive; depends on runtime + `turn_state`).
4. **`TaskExecutor`** (18 methods — execution + tickets; depends on grounding + runtime).
5. **`CommandFactory`** (18 methods — planning; depends on grounding + runtime).
6. **`PendingFlowController`** (24 methods — the continuation machine; depends on factory +
   executor). Highest coupling, so last.
7. **Rewire `TurnOrchestrator`** to call services via `StationRuntime`; delete the private
   reach-ins. *Probe: orchestrator references no `station._*` privates.*
8. **Thin the facade**: `OperatorStationSession` keeps lifecycle + delegation only.

Each step: move methods, leave a delegating shim on the session if any eval calls the method
directly, run full suite, commit. A step that can't stay green is reverted and re-sliced.

## Risks and guards

- **Hidden state coupling**: a moved method mutating `turn_state`/`pending_state` is fine
  (shared object); a method mutating a *facade-local* field is a smell — flag during the step.
- **Eval reach-in**: ~40 evals read `session.last_*` and some private methods. Delegating
  shims + the `TurnState`/`StationRuntime` properties keep them green; do not change eval files.
- **Regression probe per step**: `regression_live_operator_probe` + golden path must stay green;
  add a `pipeline_station_decomposition_probe` asserting the facade no longer defines the moved
  method families (anti-drift, mirrors 13A.2 probes).
- **Stop rule**: this is orthogonal to steering (13) and substrate-independence (14–15). If a
  step stalls, park it — the facade being large is acceptable while the boundaries hold.

## Accepted decisions

1. `StationRuntime` is the seed of the eventual substrate-independent orchestration kernel,
   not merely a collaborator bag.
2. The ordered extraction sequence remains the working design, with a value/risk review after
   step 5 before taking on the highest-coupling pending-flow and orchestrator rewiring.
3. All station de-bloat, including the safe leaves, remains parked until Phase 16 unless a
   concrete regression forces an earlier boundary fix.
4. Conditional evidence execution uses the current typed control plane:
   `ExecutionTicket` admits the mission; Cortex owns the sense/evaluate/act loop and issues
   per-step `ExecutionContract`s; Spine executes those contracts. This work must not be used as
   a pretext to begin the service extraction.
