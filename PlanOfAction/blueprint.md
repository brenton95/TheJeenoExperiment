# JEENOM Architecture Blueprint

This document defines enduring architecture rules. It does not track phase status.

- Current roadmap and verification: [task_plan.md](task_plan.md)
- ORPI interface standard: [orpi_spec.md](orpi_spec.md)
- Deferred station decomposition: [operator_station_decomposition.md](operator_station_decomposition.md)
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

## Deferred Structure

`OperatorStationSession` remains a large transitional facade. The state-first decomposition is
accepted and parked for Phase 16. `StationRuntime` is intended to become the seed of the eventual
substrate-independent orchestration kernel.

Do not begin service extraction merely to reduce file size. The refactor must follow
[operator_station_decomposition.md](operator_station_decomposition.md) and preserve public session
behavior, typed tickets, traces, and runtime guarantees after every step.

## Threat Model

Current assumption: good-faith operator and backend.

Typed schema decisions, semantic verification, exact capability lookup, readiness, tickets, and
no-LLM runtime loops reduce risk, but hostile prompts, prompt injection, and crafted schema-valid
tool calls have not been comprehensively proven safe.

Do not deploy JEENOM against untrusted operators or untrusted text until the adversarial phase in
[task_plan.md](task_plan.md) is complete.
