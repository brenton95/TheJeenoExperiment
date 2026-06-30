# JEENOM Architecture Blueprint

This document defines enduring architecture rules. It does not track phase status.

- Current roadmap and verification: [task_plan.md](task_plan.md)
- ORPI interface standard: [orpi_spec.md](orpi_spec.md)
- High-level architecture: [workflow_diagram.mmd](workflow_diagram.mmd)
- Detailed turn/runtime flow: [flow_of_control.mmd](flow_of_control.mmd)

## Core Thesis

JEENOM separates WHY, WHAT, and HOW:

- **WHY:** operator goals, constraints, risk, budget, authority, scope, and stopping rules.
- **WHAT:** intent, evidence needs, claims, plans, procedures, readiness, mission conditions,
  execution contracts, and explanations.
- **HOW:** sensors, actions, planners, controllers, environment calls, tools, and validation
  hooks supplied by a substrate.

The operator steers WHY. JEENOM owns WHAT. The substrate owns HOW.

JEENOM is not a MiniGrid solver. MiniGrid is the first stress substrate for a cognition layer
intended to survive robotics-like and interactive reasoning environments.

## Operating Rules

1. Work capability by capability and red-bar architectural behavior before implementation.
2. Preserve the working golden path while adding capability.
3. Delete or consolidate before introducing a new abstraction.
4. Add a schema family only when existing typed messages cannot represent an eval-backed need.
5. Keep substrate vocabulary and runtime bindings outside the generic orchestration layer.
6. Never call an LLM inside the rendered/runtime control loop.
7. Treat LLM output as untrusted typed compilation, never as executable authority.
8. Preserve operator semantics before capability matching; silent degradation is failure.
9. Build a `RequestPlan` and `ReadinessGraph` before answering, clarifying, mutating memory, or
   executing.
10. Require typed tickets for side effects.
11. Distinguish visible evidence, retained knowledge, inference, staleness, and ignorance.
12. Promote reusable procedures only after their claims, authority, preconditions,
    postconditions, verification, provenance, and failure modes are explicit.

## Canonical Blocks

Do not add a new top-level block unless an eval proves these responsibilities are insufficient.

| Block | Owns | Must not own |
|---|---|---|
| `OperatorStation` | operator I/O, session lifecycle, pending interaction, result display | mission reasoning, sensing internals, controller internals, durable storage mutation |
| `TurnOrchestrator` | typed turn routing and dispatch | substrate HOW, side-effect execution |
| `Cortex` | semantic preservation, evidence requests, procedure progress, mission decisions | substrate drivers, direct durable writes |
| `MissionCortex` | compound mission construction and lineage | runtime actuation, station UI |
| `ReadinessGraph` | executable/blocking verdicts from plans, contracts, claims, context, and authority | storage and execution |
| `Sense` | evidence requests to observation claims | task planning and durable operator truth |
| `Spine` | execution contracts to execution reports/claims | intent planning and durable operator truth |
| `KnowledgeBase` | durable claims, procedures, provenance, invalidation, snapshots | sensing/action HOW |
| `OperationalContext` | domain meaning, vocabulary, task families, frames, claim/display rules | mutable world state and execution |
| `SubstrateAdapter` | concrete sensors, actions, planners, controllers, environment calls, validation hooks | JEENOM WHAT decisions |

Every cross-block control or authority boundary uses typed messages.

## Typed Control Flow

### Operator Turn

1. Wrap the utterance in a `CorticalEnvelope`.
2. Resolve bounded controls, continuations, or exact compatibility/cache routes when applicable.
3. Route unresolved semantic input through strict-schema LLM compilation.
4. Canonicalize and validate the resulting `OperatorIntent`.
5. Run `IntentVerifier` to preserve the operator's actual semantics.
6. Build a dependency-aware `RequestPlan`.
7. Evaluate a `ReadinessGraph` against capabilities, claims, context, and authority.
8. Produce an `ApprovedCommand`.
9. Mint a typed ticket if the command has side effects.
10. Return a `CommandResult` and project a `LabelledEpisode`.

### Runtime Mission Cycle

1. An `ExecutionTicket` admits an approved task or mission.
2. Cortex requests an `EvidenceFrame`.
3. Sense uses substrate HOW and returns `OperationalEvidence`/percepts.
4. Cortex evaluates procedure and mission conditions.
5. Cortex may issue one `ExecutionContract`.
6. Spine executes the contract through substrate HOW.
7. Spine returns an `ExecutionReport`.
8. Cortex requests fresh evidence before another conditional action.
9. Claims, outcomes, provenance, and failure attribution enter the trace/representation surface.

The LLM never owns this cycle.

## Message Surface

Prefer these existing messages:

- operator/control: `CorticalEnvelope`, `OperatorIntent`, `RequestPlan`,
  `ReadinessGraph`, `ApprovedCommand`, `CommandResult`;
- authority: `ExecutionTicket`, `RawMotorTicket`, `MemoryWriteTicket`;
- sensing/execution: `EvidenceFrame`, `OperationalEvidence`, `Percepts`,
  `ExecutionContract`, `ExecutionReport`;
- task/mission: `TaskRequest`, `ProcedureRecipe`, `MissionContract`,
  `MissionExecutionPlan`;
- context/representation: `OperationalContext`, context fingerprint,
  `ClaimRecord`, `KnowledgeSnapshot`;
- trace: `LabelledEpisode`, `FailureOutcome`.

Historical internal claim/storage types may remain while boundaries stabilize. They must not become
an excuse for direct cross-block mutation.

## LLM Compiler Contract

The Operator Station defaults to the LLM compiler for unresolved semantic input.

Bounded deterministic routes may handle:

- exact controls and session lifecycle;
- pending clarification/synthesis continuations;
- exact compatibility patterns and `IntentCache` hits;
- explicit deterministic fallback after a visible LLM failure.

The model emits a strict JSON-schema decision such as `OperatorIntent`. It does not emit executable
prose or mint authority.

Deterministic code owns:

- vocabulary canonicalization;
- schema validation;
- semantic preservation through `IntentVerifier`;
- capability matching;
- request planning and readiness;
- dispatch;
- operator-facing answers, clarifications, refusals, and errors;
- ticket issuance and execution.

Fallback is a degraded availability path. Missing credentials, transport errors, truncation,
schema rejection, or unknown vocabulary may trigger it, but fallback must be visible in startup
status, turn logs, compiler history, and eval provenance. Silent fallback or regex-only semantic
coverage is an architectural regression.

The eval strategy therefore has three lanes:

- offline deterministic release gate;
- fake-transport `llm_path` parity;
- opt-in real `live_llm`.

### Why The LLM Route Is Shaped This Way

This boundary was not chosen only as a design preference. Interactive regressions proved that a
regex-heavy suite can stay green while the real semantic route is broken:

- a truncated response produced malformed JSON and silently fell back to deterministic parsing;
- a correctly classified motor sequence normalized into a task-sequence path that could not
  compile the motor steps;
- a valid visible-object query reached grounding but dead-ended because the model emitted a
  near-miss answer-field vocabulary;
- a single repeated motor action was rejected by a sequence path that unnecessarily required two
  entries.

The response was to strengthen convergence, not expand shadow NLU:

- method-specific output budgets and visible fallback provenance;
- strict schema ingress;
- canonical answer-field normalization with unknown values failing closed;
- fake-transport LLM matrices that prove the LLM route was actually used;
- deterministic parity assertions after normalization;
- operator-facing behavior driven by typed fields, never by model `reason` prose.

The detailed sequence of regressions, fixes, and probes is recorded in the Phase 13B.4 section of
[task_plan.md](task_plan.md#13b4---llm-default-tool-call-discipline-and-parity).

## Semantic Preservation

A schema-valid intent may still be wrong.

The verifier must reject or enrich compilation when the operator's utterance contains semantics
that the compiled intent lost or inverted, including:

- superlatives and ordinals;
- cardinality;
- direction;
- negation;
- metric choice;
- conditional and stopping clauses;
- requested action versus query behavior.

Examples:

- `farthest` must not become `closest`;
- `all doors` must not become one door;
- `until you see` must not collapse into an ordinary repeated motor command;
- an unsupported actuation request must not degrade into an answer-only query.

## Plans, Readiness, And Authority

`OperatorIntent` describes the request. It is not the execution plan.

`RequestPlan` decomposes the request into typed steps with:

- required capability handles;
- inputs and outputs;
- dependencies;
- evidence and steering constraints;
- memory reads/writes;
- expected side effects.

`ReadinessGraph` checks those steps against:

- `CapabilityRegistry`;
- primitive contracts;
- `OperationalContext` and environment assumptions;
- active and durable claims;
- authority and risk;
- synthesis policy;
- cache/prewarm requirements.

Answers, clarification, synthesis, and execution derive from readiness verdicts rather than
phrase branches.

Side-effect authority:

- `ExecutionTicket`: task or mission runtime entry;
- `RawMotorTicket`: explicit low-level motor action;
- `MemoryWriteTicket`: durable operator-claim mutation.

A plan, primitive name, model decision, or cached recipe is not authority by itself.

## Claims And Partial Observability

Claims carry enough information to decide whether they may be used:

- kind/status;
- authority;
- confidence;
- scope;
- provenance;
- freshness;
- invalidation policy.

Observation and operator claims are different:

- observation claims are evidence tied to a world/framing context;
- operator assertions are durable authority until retracted;
- computed analysis is not promoted to operator truth automatically;
- episodic references are neither durable claims nor current observations.

Freshness:

- `current`: presently supported;
- `unverifiable`: previously observed but currently out of frame;
- `stale`: invalidated by environment/world change;
- `unknown`: no usable current claim remains.

Looking away is not a world change. Only spatial observation claims become `unverifiable` because
of framing. Durable assertions, facts, and procedures do not.

Partial-observability rule:

- `visible_only` must answer from current evidence or ask for help;
- `search_allowed` may gather evidence only through an explicit bounded plan;
- JEENOM must never report global certainty from a local field of view.

### Why Retained Belief Is Not An Omniscient Map

The human renderer and the agent evidence surface are intentionally different. Human rendering may
show a full MiniGrid for debugging, while production partial sensing uses the native egocentric
FOV. Full-observation and partial-observation eval lanes are therefore separate.

Two naive approaches were rejected:

- **current FOV only:** turning away erases useful path and target knowledge immediately;
- **union every observed cell forever:** old passability and target beliefs become immortal and
  can survive real world changes.

The accepted model is claim persistence with freshness:

- an out-of-view target/passable cell becomes retained, `unverifiable` belief;
- it remains usable for a bounded period because it is still the best supported belief;
- re-observation refreshes it;
- step-based decay eventually turns it `unknown`;
- known environment/world mutation makes it `stale`.

This is why target evidence stops being re-emitted when out of view, and why passable-cell belief
has expiry rather than an unbounded union. The current two decay sites and their limitations are
documented under [Freshness decay](#freshness-decay--known-debt-phase-13b-claim-decay-on-the-cortex-loop).

## MissionContract

`MissionContract` is the approved mission-level authority record. It preserves why and how an
admitted mission may execute rather than flattening the request into an action string.

Current implemented fields include:

- mission identity and description;
- ordered task sequence;
- success-condition label;
- abort-on-failure policy;
- risk tier and cadence;
- validated `ProcedureRecipe`;
- bound task parameters;
- exact required capability handles.

For a conditional evidence mission, the current flow is:

```text
fresh Sense evidence
-> Cortex evaluates stop claim
-> zero or one approved Spine action
-> ExecutionReport
-> fresh Sense evidence
```

The `ExecutionTicket` verifies that its mission id, task type, and parameters match the approved
contract.

Current limitation: successful exits, interruptions, resource limits, and action-specific effects
are not yet represented by one typed termination policy. Movement blockage currently falls back to
a movement-specific `no_progress` check. The roadmap tracks the next slice; this blueprint fixes
the architectural responsibility:

- MissionContract owns the policy for why a mission continues, succeeds, or interrupts.
- Cortex evaluates mission-level conditions from fresh evidence and typed action outcomes.
- The substrate owns action feasibility/preflight.
- Spine executes approved actions and reports normalized outcomes.
- Capability contracts declare preconditions, expected effects, verification, and failure modes.
- Search/replanning is never inferred from failure; it requires explicit mission authority.

Condition precedence should be:

```text
sense
-> successful exits
-> interrupts and preflight
-> one action
-> outcome/effect verification
-> repeat or terminate
```

### Why MissionContract Became Necessary

Several superficially separate bugs had the same cause: operator intent was being flattened too
early.

- An inline derived-metric mission could become only "go to the yellow door", losing the metric,
  rank, selection reason, and parent authority.
- "Go straight until you see a blue door" could degrade into repeated motor execution and discard
  its stop condition.
- A task started after manual movement could reset to the seed pose, disconnecting approved intent
  from the world the operator had just manipulated.
- Search/spin behavior could appear purposeful without a contract saying which evidence-gathering
  actions were authorized.

The contract therefore preserves the mission-level reason, validated procedure, parameters,
required handles, lineage, and current stopping authority. It does not itself grant autonomous
search. Search/replan authority must be explicit and represented in a later bounded procedure.

The live episode rule is part of this meaning: task admission reuses the current adapter and pose;
typed `reset` is the explicit episode boundary; `Ctrl+C` is the synchronous interruption path.

## Capability Registry And ORPI

Substrate capabilities are contractual objects, not loose strings.

Each registered `PrimitiveSpec` describes:

- identity, layer, and description;
- inputs, outputs, required/produced claims;
- side effects;
- preconditions and postconditions;
- frames and units;
- safety class and authority level;
- failure modes;
- validation hooks;
- substrate fingerprint;
- runtime binding;
- ORPI mode, cadence, and invariant level.

The registry uses exact handles. No prefix relaxation or fuzzy authority matching is allowed.

ORPI v0.1 projects capabilities into:

- `sense`: reality to claims;
- `actuation`: approved command to physical effect;
- `meta`: claims to claims.

The authoritative contract/manifest/procedure/trace rules live in
[orpi_spec.md](orpi_spec.md).

Primitive contract metadata must become executable at the appropriate boundary:

- readiness checks static availability, claims, frames, risk, authority, and required hooks;
- substrate preflight checks concrete feasibility and safety;
- Spine reports the action outcome;
- Sense/postcondition checkers verify effects;
- Cortex applies mission policy.

## Execution Hierarchy

The current schema-backed hierarchy is:

| Level | Meaning | Current representation |
|---|---|---|
| L0 | substrate primitive | runtime primitive libraries + `PrimitiveSpec` |
| L1 | named command/template | command registry, `SensePlanTemplate`, `SkillPlanTemplate` |
| L2 | procedure | `ProcedureRecipe` |
| L3 | grounded task | `TaskRequest` + readiness/ticket |
| L4 | mission/goal | `MissionContract` + `MissionExecutionPlan` |

Claims and typed reports cross the level boundaries. A higher level does not bypass the authority
or verification requirements of its lower-level capabilities.

## OperationalContext

`OperationalContext` is a typed situation frame, not a hidden service.

It defines:

- object and attribute vocabulary;
- task families;
- reference and grounding semantics;
- claim and display rules;
- environment identity fields;
- frames, units, and risk policy;
- procedure hints;
- context/substrate fingerprint inputs.

Startup:

1. load `SubstrateAdapter`;
2. load `OperationalContext`;
3. build/filter `CapabilityRegistry`;
4. compute context fingerprint;
5. prewarm/cache known procedures under that fingerprint.

Only compact relevant context slices go to the LLM for compilation, repair, or synthesis.

### Object Vocabulary And Capability Authority

`OperationalContext.object_vocabulary` is the source of object meaning, but vocabulary is not
capability.

The object-parametric refactor rejected a new process-global `SUPPORTED_OBJECT_TYPES` list because
it would duplicate context and drift from the manifest. The responsibilities are:

- `OperationalContext` declares which object types make sense in the active situation;
- `PlanningSemantics` derives parsing patterns, pluralization, grounding handles, and task handles;
- the exact `CapabilityRegistry`/ORPI manifest declares which operations actually exist.

Adding `marker` or `apple` to vocabulary must not fabricate `task.go_to_object.marker` or
`task.pickup.apple`. Exact handle lookup remains the authority boundary. Legacy MiniGrid names
such as `ranked_scene_doors` survive only as compatibility debt; they do not define runtime
meaning.

## WHY / WHAT / HOW Examples

Robotics:

- WHY: deliver to the target within approved risk;
- WHAT: ground target, assess evidence/reachability, plan, execute, monitor, verify;
- HOW: detector, SLAM, Nav2/MoveIt, controller, gripper, safety monitor.

MiniGrid:

- WHY: go straight until the blue door is visible;
- WHAT: bind target predicate and allowed action, sense, evaluate, act once, verify, terminate;
- HOW: egocentric observation, turn/forward actions, grid adapter.

ARC-like environment:

- WHY: solve under an operator-selected experimentation budget;
- WHAT: represent observations as claims, compare transitions, choose bounded experiments,
  update the plan;
- HOW: game-state parser, legal-action API, replay/simulation, score/end-state feedback.

## Trace And Learning

Every executed turn should preserve:

- compiled and verified intent;
- claims consumed with confidence/freshness/provenance;
- request plan and readiness verdicts;
- authority tickets;
- actions and execution reports;
- postcondition evidence;
- failure attribution;
- steering and clarification;
- durable knowledge writes and scoped reuse.

Failed episodes are retained. A failure without typed attribution is difficult to repair or learn
from.

## Architecture Decision Register

This table keeps the most consequential decisions discoverable. The implementation narrative,
spikes, and bug history are expanded in [task_plan.md](task_plan.md).

| Decision | Accepted direction | Rejected shortcut / reason |
|---|---|---|
| Semantic routing | LLM-default unresolved semantics, strict schema, deterministic convergence | broad regex coverage hides LLM regressions |
| Execution authority | readiness plus typed tickets | model output, task strings, and cached plans are not authority |
| Partial observability | native FOV plus retained claims with freshness/decay | omniscient sensing, FOV-only amnesia, or immortal union maps |
| Missing evidence | typed `needs_evidence` and clarification | guessing, generic ambiguity, or implicit search |
| Conditional missions | Sense -> Cortex condition -> one Spine action -> fresh Sense | repeated motor loops that discard stop clauses |
| Episode continuity | live adapter reuse; explicit reset; synchronous Ctrl+C | accidental reset at task admission or premature concurrency |
| Object types | context-driven meaning plus exact manifest handles | global supported-type list or vocabulary-implies-capability |
| Primitive construction | query-only structured formulas, approval, validation, provenance | arbitrary operator code or synthesized actuation authority |
| Station decomposition | state-first `StationRuntime`, deferred to Phase 16 | method-first leaf extraction around pre-13B data shapes |
| ORPI versioning | v0.1 until a second substrate breaks/proves it | freezing an n=1 interface |
| Threat model | good-faith operator/backend until adversarial proof | claiming schema validation alone proves hostile safety |

## Operator-Station Decomposition Design

Status and execution order live in the Phase 16 section of
[task_plan.md](task_plan.md#phase-16---operational-hardening-and-station-decomposition). This
section owns the enduring target design.

`OperatorStationSession` remains a large transitional facade. At the current repository snapshot,
`operator_station.py` is **6,213 lines and 171 methods**. `OperatorStationSession.__init__`
directly initializes 41 attributes, while additional turn and pending fields are property-backed.
The counts are diagnostic only. The architectural problem is shared mutable state and implicit
ownership, not the precise file length.

`TurnOrchestrator` currently reaches into **61 distinct session members, 23 private**. Moving
methods into services while passing the entire session would preserve the station as an implicit
service locator. The design is therefore **state-first, not method-first**: define the shared
state surface, then move behavior according to ownership.

### Non-Negotiable Constraints

Every extraction must preserve:

- public `handle_utterance`, `execute_command`, `reset`, `startup`, and `close` behavior;
- compatibility reads through `session.last_*`, `session.pending_*`, and collaborator attributes
  while callers still depend on them; roughly 40 evals currently use that surface;
- `CommandResult` as the result of every turn and `CorticalEnvelope` on recorded results;
- `LabelledEpisode` projection from `CommandResult`;
- `ExecutionTicket`, `RawMotorTicket`, and `MemoryWriteTicket` gates;
- no LLM calls in the rendered/runtime control loop;
- the MiniGrid golden path and full eval/test gate after each extraction.

### `StationRuntime` Kernel Seed

Services receive a frozen `StationRuntime`, never `OperatorStationSession`.
`StationRuntime` is more than a collaborator bag: it is the seed of the eventual
substrate-independent orchestration kernel.

It exposes:

- immutable collaborators and contracts;
- one typed per-turn trace;
- one typed pending/continuation state;
- representation and authority surfaces.

`OperatorStationSession` owns the runtime and delegates. Compatibility properties may project
runtime state through the facade, but services must not reach back into session privates.

### Shared-State Map

| Group | Representative members | Post-decomposition owner | Sharing mechanism |
|---|---|---|---|
| Runtime collaborators | substrate, Sense, Spine, Cortex, registry, planning semantics, context, compiler, authorities, representation, KB, caches, arbitrator, synthesizer, validator, runtime package | frozen `StationRuntime` | read-only references passed to services |
| Per-turn trace | the `TurnState` fields | `TurnState` on `StationRuntime` | one mutable typed object used by the pipeline |
| Continuation state | pending clarification, synthesis, primitive definition, active claims | `PendingState` | mutable object owned through pending-flow control |
| Session/config | environment id, seed, render mode, loop limit, verbosity, startup summary, context fingerprint | station facade | facade-local lifecycle/configuration |

Mutation of `TurnState` and `PendingState` is explicit shared state. A service mutating a
facade-local field is a boundary smell.

### Target Services

| Module | Approx. methods | Owns | Representative methods |
|---|---:|---|---|
| `OperatorStationSession` | 20 | lifecycle, operator I/O, display summaries, delegation | `__init__`, `startup`, `reset`, `close`, `handle_utterance`, `execute_command`, summary methods |
| `CommandFactory` | 18 | turn to typed `RequestPlan`/`ApprovedCommand` | `command_from_llm_intent`, `command_from_selector_readiness`, `_command_from_*`, `_arbitrate_gap`, local plan/graph builders |
| `TaskExecutor` | 18 | approved command to ticketed execution and result | `run_task`, `_run_task_with_ticket`, `_run_procedure`, `_run_sequence`, `_run_motor_*`, `_run_mission`, ticket/result helpers |
| `PendingFlowController` | 24 | clarification, synthesis, and primitive-definition continuations | `handle_pending_*`, `resume_*`, proposal/approval handlers, selector clarification |
| `GroundingService` | 28 | target grounding, metric semantics, claim writing | `ground_target_selector`, `_base_metric_distance`, `_ensure_ranked_object_claims`, metric/compose/write helpers |
| `ConceptService` | 9 | named concepts and durable knowledge writes | `teach_concept`, `forget_concept`, `concepts_summary`, knowledge-update and memory-ticket helpers |
| `EnvironmentTracker` | 7 | environment identity, scene fingerprint, claim validity | identity refresh, `_claims_valid_for_current_environment`, `_scene_state_fingerprint`, task-family and arbitration summaries |

`TurnOrchestrator` remains the top-level typed router but calls these services through
`StationRuntime` instead of reaching into station-private methods.

### Why Safe-Leaf Extraction Was Deferred

Early cleanup discussions considered extracting `EnvironmentTracker`, `ConceptService`, and other
apparently safe leaves immediately. That was rejected because the state shapes were still moving:

- partial observation introduced FOV, unseen cells, retained belief, freshness, and decay;
- `needs_evidence` introduced typed continuation state;
- conditional missions introduced mission contracts and per-step execution contracts;
- termination and bounded search will introduce action outcomes and search authority;
- object parameterization changed grounding and task-handle inputs;
- a second substrate may change which collaborators belong in the kernel.

Extracting services before those shapes stabilize risks carving boundaries around old MiniGrid
state and repeating the work.

### Rejected Alternatives

- **Method-first extraction:** preserves hidden shared-state coupling.
- **Pass the session into every service:** renames rather than removes the god object.
- **Extract only substrate-looking methods:** confuses generic orchestration debt with the
  separate Phases 14-15 substrate proof.
- **Rewrite the station in one pass:** makes authority, trace, pending-flow, and compatibility
  regressions impossible to isolate.
- **Use conditional-mission work as the first extraction:** couples a control-plane proof to an
  unrelated structural refactor.
- **Use file length as the start gate:** optimizes appearance instead of ownership.

The accepted design is intentionally parked until Phase 16. A large facade with enforced typed
authority is preferable to several smaller services sharing state informally.

### Freshness decay — known debt (Phase 13B claim decay on the cortex loop)

The decay machine is live on the cortex loop, clocked by `world_sample.step_count`. Three
deliberate simplifications are tagged in-code with greppable markers (`grep -rn "TECH-DEBT" jeenom/`):

- `# TECH-DEBT(uniform-decay):` — every claim ages as an observation at the single
  `UNVERIFIABLE_DECAY_STEPS` rate; per-kind rates (`ttl_for_kind`) stay dormant until a
  substrate with a changing world (AI2-THOR) can falsify per-kind decay. MiniGrid cannot.
- `# TECH-DEBT(intra-task-decay):` — `cortex._claims` is rebuilt per task, so decay is
  intra-task only; mission-scope decay waits until belief moves into `memory` (mission-contract
  phase).
- `# TECH-DEBT(mission-clock-rests-on-skip-reset):` — `step_count` spans a mission only because
  the station reuses the adapter with `skip_reset=True`; that reuse is not yet a guaranteed
  contract, and mission-scope decay will depend on it.
- `# TECH-DEBT(occupancy-decay-sites):` — the spatial passable belief decays in the perception
  layer (Sense) via the same freshness TTL, parallel to the Step 2 cortex claim-decay loop rather
  than unified with it. It must be Sense-side because the planner reads `passable_positions` from
  percepts before the cortex runs. Unifying the two decay sites onto one claim store is deferred.

## Threat Model

Current assumption: good-faith operator and backend.

Typed schema decisions, semantic verification, exact capability lookup, readiness, tickets, and
no-LLM runtime loops reduce risk, but hostile prompts, prompt injection, and crafted schema-valid
tool calls have not been comprehensively proven safe.

Do not deploy JEENOM against untrusted operators or untrusted text until the adversarial phase in
[task_plan.md](task_plan.md) is complete.
