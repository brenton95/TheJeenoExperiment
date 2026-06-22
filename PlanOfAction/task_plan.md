# JEENOM Implementation Plan

This is the authoritative roadmap and phase-status document for the repository.
Architecture rules live in [blueprint.md](blueprint.md), the ORPI interface lives in
[orpi_spec.md](orpi_spec.md), and the deferred station refactor lives in
[operator_station_decomposition.md](operator_station_decomposition.md).

## Objective

Build a steerable cognition layer in which:

- the operator steers WHY: goals, constraints, scope, risk, authority, budget, and stopping rules;
- JEENOM owns WHAT: intent, evidence needs, claims, plans, procedures, readiness, execution
  authority, and reusable mission structure;
- each substrate owns HOW: sensors, actions, planners, controllers, environment calls, and
  validation hooks;
- the same typed cognition flow can survive a second substrate without moving substrate logic
  into the orchestration kernel.

Implementation rule: delete before adding, simplify before generalizing, and introduce a new
architecture object only when a red-bar test exposes a real representational gap.

## Current State

Current phase: **Phase 13B - Partial observability and evidence-governed execution**.

Completed in Phase 13:

- **13A:** typed steering, coordinate abstraction, shared helper consolidation, typed
  `TurnState`, and the accepted Phase 16 station-decomposition design;
- **13B.1:** claim freshness (`current | unverifiable | stale | unknown`);
- **13B.2:** native MiniGrid field-of-view sensing and explicit partial/full-observation eval
  lanes;
- **13B.3:** `needs_evidence` readiness and typed `ClarificationRequest`;
- **13B.4:** LLM-default semantic routing, strict schema/tool-call discipline, visible fallback,
  and deterministic/LLM parity evals;
- **13B.5:** typed conditional evidence missions such as
  `go straight until you see a blue door`.

The current 13B.5 runtime:

1. `MissionCortex` constructs a `MissionContract` carrying a validated `ProcedureRecipe`,
   parameters, and exact required capabilities.
2. Readiness validates the sensing, action, and task handles.
3. An `ExecutionTicket` preserves the approved contract.
4. Sense produces fresh target evidence.
5. Cortex evaluates the stop claim and issues at most one `ExecutionContract`.
6. Spine executes only that contract.
7. Cortex senses again before another action.
8. Success, budget exhaustion, and movement `no_progress` terminate finitely.

This is **not** autonomous search. The procedure may execute only the operator-authorized action.

Pulled-forward Phase 14 object-parametric slice:

- `OperationalContext.object_vocabulary` and `PlanningSemantics` now drive deterministic
  parsing, LLM tool schemas, task handles, grounding handles, mission continuations, and
  station task-plan construction;
- exact manifest handles remain the capability authority, so registering an object word does
  not fabricate executable support;
- active claims expose an object-generic view while preserving the legacy
  `ranked_scene_doors` field for compatibility.

### Verification Baseline

Last verified after the object-parametric routing slice:

- `python evals/eval_master.py`: **78/78**
- `python evals/eval_master.py --suite orpi`: **10/10**
- `python evals/eval_master.py --suite cleanup`: **30/30**
- `python evals/eval_master.py --suite llm_path`: **5/5**
- `python evals/eval_master.py --suite live_llm`: **1/1** when a live backend is configured
- `python -m pytest -q tests`: **333 passed**, 1 warning, 12 subtests passed

The deterministic gate runs without the live LLM key. The `live_llm` lane is opt-in and is not
part of the offline release gate.

## Immediate Next Work

### 13B.6 - Mission Termination And Action Outcomes

Status: **next; design agreed, not implemented**.

The 13B.5 contract has one string `success_condition` plus task parameters such as
`stop_claim=target_visible`. Wall handling is currently narrower: after a movement attempt,
Cortex compares the new pose with the previous pose and reports `no_progress`.

That behavior is finite, but it is not the general contract we need:

- success and interruption are different concepts;
- `ExecutionContract.stop_conditions` are descriptive and are not evaluated by Spine;
- primitive preconditions and validation hooks are registered but are not a general runtime
  preflight system;
- unchanged pose is a reasonable movement signal but is wrong for `toggle`, `pickup`, or `drop`;
- loop limits, environment termination, controller faults, and operator cancellation need typed
  terminal outcomes.

Target semantics:

1. Gather fresh evidence.
2. Evaluate successful exit conditions.
3. Evaluate interrupt and safety conditions.
4. Run substrate-owned action preflight.
5. Authorize and execute one action.
6. Normalize the action outcome and verify its expected effects.
7. Repeat or terminate with an explicit typed result.

Likely schema responsibilities:

- a typed mission termination policy;
- typed claim predicates for successful exits;
- typed interrupt conditions and dispositions;
- substrate-owned action preflight;
- action-specific expected effects and normalized outcomes.

Exact class names are not frozen until the red bars prove the smallest useful shape.

Required examples:

**Go straight until the blue door is visible**

- success: fresh `target_visible == true`;
- interrupt: `move_forward` is blocked;
- limits: budget/runtime exhaustion;
- search/replan authority: none.

**Go straight until the blue door is visible or a wall is reached**

- success: either `target_visible == true` or forward motion is blocked;
- the wall is successful only because the operator explicitly named it as an exit.

**Toggle until the door is open**

- success: fresh `door_state == open`;
- interrupt: not toggleable or controller failure;
- progress is an object-state change, not a pose change.

Red bars before production edits:

- initial success causes zero actuation;
- visible blockage halts before environment actuation when preflight can prove it;
- a post-action blocked report halts immediately;
- a newly satisfied target condition wins before another action;
- budget, runtime limit, environment termination, and controller failure are typed;
- non-motion actions do not inherit movement-specific `no_progress`;
- partial and full observability produce equivalent terminal semantics;
- deterministic and LLM routes construct equivalent contracts.

### 13B.7 - Bounded Evidence Gathering And Deterministic Meta-Primitives

Status: **planned after 13B.6**.

Add `search_allowed` procedures that may choose evidence-gathering actions only when the operator
grants that scope. Initial deterministic meta-primitives remain:

- `searched_region`
- `reachability`
- `behind`

They infer claims from observed claims and must not invoke an LLM inside the runtime loop.

Acceptance boundary:

- `visible_only` asks for help when evidence is absent;
- `search_allowed` executes a bounded typed evidence-gathering plan;
- search actions, coverage, limits, and completion evidence are represented in the plan and trace;
- no phrase-specific search branch or unbounded spin.

### 13C - Curriculum, Knowledge Reuse, And MTBCI

Status: **planned after 13B**.

- run a partial-observability curriculum across the MiniGrid size ladder;
- measure scoped knowledge and plan reuse;
- track mean turns between clarification/intervention;
- demonstrate improved reuse without weakening evidence or authority gates.

## Standing Architecture Decisions

- The Operator Station defaults to the LLM compiler for unresolved semantic input.
- Deterministic routes are bounded to controls, continuations, exact compatibility patterns,
  cache hits, and explicit fallback.
- LLM output is a strict typed decision, never executable prose.
- LLM and deterministic outputs converge before authority through canonicalization,
  `IntentVerifier`, planning, readiness, deterministic dispatch, and tickets.
- Fallback is allowed for availability but must be visible in logs, compiler history, and eval
  provenance.
- No LLM calls are allowed inside the rendered/runtime control loop.
- `OperatorIntent` is not an execution plan.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane.
- Side effects require `ExecutionTicket`, `RawMotorTicket`, or `MemoryWriteTicket`.
- Observation claims are evidence, not durable operator truth.
- Search and replanning require explicit authority.
- Operator-station de-bloat is parked until Phase 16.

## Phase Index

| Phase | Status | Outcome |
|---|---|---|
| 0-6 | complete | MiniGrid vertical slice, typed runtime, cache/prewarm, CLI station, episodic references |
| 7-9 | complete | intent/readiness/synthesis, abstraction hierarchy, typed authority and representation gates |
| 10 | complete | first-cut context, domain, runtime-package, orchestrator, and primitive-assembly boundaries |
| 11 | complete | Cortex-owned mission flow, hostile mission ladder, architecture surgery |
| 12 | complete | MiniGrid ORPI v0.1 contract/manifest/procedure/trace boundary |
| 12D | complete | ORPI consolidation, labelled-episode completion, coupling audit |
| 13A | complete | steerable cognition core and supporting cleanup |
| 13B.1-13B.5 | complete | partial observability, evidence readiness, LLM parity, conditional mission execution |
| 13B.6 | next | typed mission termination and action outcomes |
| 13B.7 | planned | bounded evidence gathering and deterministic meta-primitives |
| 13C | planned | curriculum, scoped reuse, MTBCI |
| 14 | later | remaining cheap substrate-leak removal and AI2-THOR spike |
| 15 | later | committed second-substrate port and ORPI v1 freeze |
| 16 | later | operational hardening and station decomposition |
| 17 | later | capability stress and adversarial hardening |

## Completed Architecture Milestones

### Phases 0-6 - Working Vertical Slice

- MiniGrid adapter, render, actions, and task execution;
- Cortex/Sense/Spine runtime split;
- typed LLM compilation;
- JIT templates, cache, and prewarm;
- CLI Operator Station;
- shared live episode across preview, motor commands, sensing, and task execution;
- typed `reset` as the explicit episode boundary and `Ctrl+C` as synchronous interruption.

### Phases 7-9 - Control Plane And Authority

- `OperatorIntent`, grounding plans, scene and active claims;
- capability matching, arbitration, and safe query synthesis;
- `RequestPlan`, `ReadinessGraph`, and semantic preservation;
- `CorticalEnvelope`, `ApprovedCommand`, and `CommandResult`;
- ticketed task, motor, and durable-memory side effects;
- representation, knowledge, schema, and block boundaries.

### Phase 10 - Initial Extraction Boundaries

- `OperationalContext` and context fingerprints;
- `MiniGridDomainHelper`;
- `TurnOrchestrator`;
- injected `RuntimePackage`;
- context-driven planning semantics;
- operator-defined query primitives and inline derived metrics.

These are first-cut boundaries, not the final module decomposition.

### Phase 11 - Mission Flow And Architecture Surgery

Status: **complete**.

- `MissionCortex` owns compound mission decomposition;
- `MissionExecutionPlan` preserves mission, continuation, provenance, and child-ticket lineage;
- hostile paraphrase evals cover Sense, Spine, procedures, conditional gating, and compound
  missions;
- dispatch routes through five typed knowledge paths;
- fast-path NLU moved behind `IntentCache`;
- one readiness gate remains;
- domain vocabulary registration and capability-based eval naming are enforced.

### Phase 12 And 12D - ORPI v0.1

Status: **complete for MiniGrid**.

The authoritative interface is [orpi_spec.md](orpi_spec.md). Delivered:

- `OrpiContract`, `OrpiManifest`, `OrpiProcedure`, and `LabelledEpisode`;
- `{sense, actuation, meta}` ORPI projection over the compatibility vocabulary;
- manifest symbol mappings, frames, units, and risk policy;
- OEM bundled-procedure validation;
- scoped durable knowledge (`site | embodiment | universal`);
- named postcondition verification and failure attribution;
- ORPI conformance suite.

ORPI remains v0.1 until a second substrate validates the boundary.

### Phase 13A - Steering Core

Status: **complete**.

- typed `SteeringDirective` for budget, scope, risk, and stopping rule;
- risk withdraws authority through readiness;
- budgets cap Spine execution and surface typed exhaustion;
- active steering is recorded in `LabelledEpisode`;
- N-dimensional coordinate metrics and frame-safe geometry;
- shared fingerprint, capability-handle, and metric helpers;
- typed `TurnState`;
- accepted Phase 16 station-decomposition design.

### Phase 13B.1-13B.5 - Partial Observability And Conditional Missions

Status: **complete**.

- claim freshness distinguishes look-away from world change;
- MiniGrid production sensing uses native egocentric FOV;
- full and partial observability have separate eval lanes;
- incomplete visible evidence produces `needs_evidence`, not fabricated certainty;
- typed clarification state survives the turn trace;
- LLM-default routing and fake/live LLM eval lanes prevent regex-only confidence;
- task dispatch preserves the live adapter state;
- conditional missions execute Sense -> Cortex decision -> one Spine action -> Sense.

## Later Phases

### Phase 14 - Cheap Leak Removal And AI2-THOR Spike

Remove remaining cheap, non-curriculum substrate leaks before the exploratory second-substrate
branch:

- manifest-derived compiler prompt examples;
- injected runtime package instead of MiniGrid imports in generic station code;
- finish migration of legacy door-named claim/output compatibility fields;
- context-derived color schemas and remaining example/help text;
- stronger static architecture guards.

Structural `MiniGridSense` and `MiniGridSpine` bindings are validated by the second-substrate work,
not renamed speculatively.

The AI2-THOR branch is an exploratory requirements spike. It records every ORPI v0.1 bend or break
and feeds a targeted Phase 15 implementation.

### Phase 15 - Cross-Substrate Demonstration And ORPI v1

- run the same typed intent, plan, readiness, claims, tickets, and orchestration flow on MiniGrid
  and a second substrate;
- confine differences to substrate/context bindings;
- resolve or consciously defer spike findings;
- freeze ORPI v1 only after both substrates conform.

### Phase 16 - Operational Hardening

Execute the accepted state-first station decomposition in
[operator_station_decomposition.md](operator_station_decomposition.md):

- introduce `StationRuntime`;
- extract environment, concept, grounding, execution, command, and pending-flow services in the
  accepted order;
- rewire `TurnOrchestrator` away from private facade reach-ins;
- preserve the public session API and every ticket/trace boundary.

No Phase 16 extraction should be pulled forward merely to reduce file size.

### Phase 17 - Capability And Adversarial Stress

- harder robotics or ARC-like pressure tests;
- complete audit of every LLM-controlled dispatch field;
- hostile schema-valid tool-call and prompt-injection evals;
- prove that crafted model output cannot mint unintended side-effect authority;
- add trust boundaries if untrusted external text channels are introduced.

Until then, JEENOM assumes a good-faith operator and backend. Do not deploy it against untrusted
operators or untrusted text.

## Known Limitations

- MiniGrid is the only ORPI-conformant substrate.
- Mission termination is only partially typed; 13B.6 owns the general model.
- Registry preconditions and validation hooks are not yet a universal executable preflight
  mechanism.
- `search_allowed` does not yet construct bounded evidence-gathering behavior.
- Legacy compatibility names such as `ranked_scene_doors`, `ranked_doors`, and `other_door`
  remain even though runtime object routing is context-driven; Phase 14 owns their migration.
- Some generic-looking modules still contain MiniGrid examples or bindings tracked for Phases
  14-15.
- `OperatorStationSession` remains a large facade; its accepted decomposition is deliberately
  parked for Phase 16.
- Adversarial prompt and schema-valid side-effect containment are not yet proven.

## Stop Rules

- Red bar first for architectural changes.
- Keep the deterministic gate, focused tests, ORPI suite, and LLM-path parity green.
- Do not broaden deterministic semantic routing to bypass the LLM-default contract.
- Do not add autonomous search while implementing termination semantics.
- Do not begin station de-bloat before Phase 16 unless a concrete regression forces a boundary
  repair.
- Update this file when phase status changes; other PlanOfAction documents should link here
  instead of duplicating current status.
