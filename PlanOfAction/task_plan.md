# JEENOM Implementation Plan

This is the authoritative roadmap, phase-status, and implementation-history document. It keeps
current work and future ordering near the top, then preserves the detailed chronological record:
why work was ordered a certain way, which assumptions were challenged, what the red bars exposed,
which bugs changed the design, what landed, and what was deliberately deferred.

Enduring architecture and the operator-station target design live in
[blueprint.md](blueprint.md). The ORPI interface standard lives in
[orpi_spec.md](orpi_spec.md). The diagrams show the high-level architecture and detailed runtime
flow without owning phase status.

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

Current work item: **13B.6 - Mission Termination And Action Outcomes**.

Current work status: **in progress at the red-bar/design stage; production implementation has not
started**.

### Progress Ledger

| Work | Status | Evidence |
|---|---|---|
| Phase 13A | **complete** | steering, coordinate abstraction, consolidation spike, `TurnState`, and decomposition design landed |
| 13B.1 | **complete** | freshness state machine and live claim/map decay |
| 13B.2 | **complete** | native MiniGrid FOV plus separate full/partial eval lanes |
| 13B.3 | **complete** | `needs_evidence` and typed `ClarificationRequest` |
| 13B.4 | **complete** | LLM-default tool-call discipline and deterministic/LLM parity |
| 13B.5 | **complete** | conditional Sense/Cortex/Spine `MissionContract` execution |
| Pulled-forward Phase 14 object slice | **complete** | context-driven object routing with exact manifest authority |
| 13B.6 | **in progress** | design and required red bars defined; production schemas/runtime changes not started |
| 13B.7 | **queued** | bounded evidence gathering and deterministic meta-primitives |
| 13C | **queued** | curriculum, scoped reuse, and MTBCI |

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

The 13B.1 freshness work has since been connected to the live hot path:

- `ObservationClaim` carries freshness and `last_observed_tick`;
- Cortex treats `current` and `unverifiable` as usable retained belief, and treats `stale` and
  `unknown` as absent;
- target claims stop being refreshed when the target leaves FOV, so they can age honestly;
- passable-cell belief persists across look-away but expires through the same step-based TTL
  instead of accumulating forever.

This is a deliberate partial implementation. Uniform decay, intra-task storage, the mission clock,
and the separate Sense-side occupancy decay site remain explicit debt documented in
[blueprint.md](blueprint.md#freshness-decay--known-debt-phase-13b-claim-decay-on-the-cortex-loop)
and in the 13B.1 record below.

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

Last verified against the current Phase 13B hot path, including object-parametric routing and
claim decay:

- `python evals/eval_master.py`: **78/78**
- `python evals/eval_master.py --suite orpi`: **10/10**
- `python evals/eval_master.py --suite cleanup`: **30/30**
- `python evals/eval_master.py --suite llm_path`: **5/5**
- `python evals/eval_master.py --suite live_llm`: **1/1** when a live backend is configured
- `python -m pytest -q tests`: **355 passed**, 1 warning, 12 subtests passed

The deterministic gate runs without the live LLM key. The `live_llm` lane is opt-in and is not
part of the offline release gate.

## Immediate Next Work

### 13B.6 - Mission Termination And Action Outcomes

Status: **in progress at the red-bar/design stage; production implementation not started**.

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

## Detailed Phase Record

The records below preserve six things:

- **Pressure:** the operator outcome or architectural failure that motivated the work.
- **Decision:** the boundary or representation chosen, including rejected shortcuts.
- **Implementation:** the concrete types, modules, routes, and probes that landed.
- **Discoveries:** assumptions disproved while working against the real code.
- **Regressions fixed:** bugs that revealed a missing architectural guarantee.
- **Carry:** debt intentionally left for a later phase.

## Phases 0-6 - Establishing The Vertical Slice

### Phase 0 - MiniGrid Smoke Test

**Pressure:** prove the environment wrapper, observation, action, render, and reset loop before
claiming any cognition architecture.

**Decision:** keep the first proof deliberately narrow. MiniGrid is a stress substrate, not the
product ontology.

**Implementation:** basic environment creation, sensing, action stepping, rendering, and a simple
GoToDoor task.

**Carry:** no typed compiler boundary, no runtime separation, and no reusable execution plan yet.

### Phase 1 - Cortex, Sense, And Spine Runtime Split

**Pressure:** a single loop that both interprets and executes would make later LLM integration
unsafe and impossible to cache.

**Decision:** establish architecture-native roles early:

- Sense converts substrate observations into evidence.
- Cortex owns procedure progress and chooses what should happen next.
- Spine executes a selected skill and reports the result.

**Implementation:** typed world samples, operational evidence, percepts, execution contracts, and
execution reports.

**Carry:** the split was architectural but still MiniGrid-bound in its concrete implementations.

### Phase 2 - Typed LLM Compiler Boundary

**Pressure:** an LLM that emits executable prose or raw environment actions would collapse intent,
planning, and authority.

**Decision:** the model compiles schema objects only. Deterministic code validates and executes.
Unknown primitives fail closed or fall back visibly.

**Implementation:** typed task, procedure, sense-plan, and skill-plan compilation.

**Carry:** semantic preservation and capability-aware readiness were not yet mature.

### Phase 3 - JIT Cache And Prewarm

**Pressure:** live model calls or compile misses inside a rendered control loop are nondeterministic
and too slow.

**Decision:** compile and prewarm before motion; runtime consumes cached templates only.

**Implementation:** procedure, sense, and skill template cache keys; startup/JIT prewarm; runtime
instrumentation for LLM calls and cache misses.

**Acceptance:** the rendered golden path reports zero runtime LLM calls and zero runtime cache
misses.

### Phase 4 - Same-Task Scale Pressure

**Pressure:** confirm the architecture was not accidentally fitted to one small grid.

**Implementation:** the same `go_to_object` task family was exercised on larger GoToDoor
environments without changing the cognitive structure.

### Phase 5 - Operator Station

**Pressure:** expose a real interactive surface so status, memory, clarification, execution, and
render continuity could be tested as one session.

**Implementation:** `OperatorStationSession` and `run_operator_station.py`.

**Discovery:** the station became the fastest place to add behavior and therefore accumulated
responsibilities faster than the architecture intended. That debt later drove Phases 9-11 and the
banked Phase 16 decomposition.

### Phase 6 - Memory-Grounded References And Live Episode Continuity

**Pressure:** operators need durable facts and episodic references such as "go there again" without
recompiling the world from scratch.

**Implementation:** delivery-target knowledge, `last_target`, `last_task`, and
`last_successful_instruction`.

**Important later regression:** preview motion and task execution initially used lifecycle paths
that could reset/recreate the environment at task admission. The accepted continuity rule became:

- preview, motor commands, idle sensing, and tasks share one live adapter;
- starting a task uses the current pose and does not call `adapter.reset()`;
- typed `reset` is the explicit episode boundary;
- `Ctrl+C` is the synchronous interruption/exit mechanism.

That fix intentionally did not introduce concurrency, a new episode schema, or autonomous search.

## Phase 7 - Intent, Grounding, Readiness, And Safe Synthesis

**Pressure:** natural language requests were richer than direct task strings. The system needed to
represent selectors, relational grounding, capability gaps, and clarification without silently
choosing a nearby action.

**Decision:** split interpretation from execution control:

- `OperatorIntent` describes the request.
- `TargetSelector` and `GroundingQueryPlan` describe what must be grounded.
- `CapabilityRegistry` and `CapabilityMatcher` decide whether exact handles exist.
- `CapabilityArbitrator` decides clarify, synthesize, substitute, or refuse.
- `RequestPlan` and `ReadinessGraph` become the actual execution-control plane.

**Implementation:** typed selectors, ranked/closest queries, capability matching, arbitration,
query-only synthesis scaffolding, request-plan steps, readiness nodes, and explicit blocking
verdicts.

**Semantic-normalization discovery:** valid schemas can still invert intent. Superlatives,
ordinals, cardinality, metric choice, and negation therefore needed deterministic preservation.
`IntentVerifier` was introduced so "farthest" cannot silently become "closest" and "all" cannot
become one candidate.

**Bug-fix context:** indirect phrasing and superlatives repeatedly exposed divergence between
regex and LLM routes. The durable lesson was not to add more phrase patches; both routes must
normalize into the same typed intent and exact capability requirements.

## Phase 8 - Operational Adaptation And Abstraction Hierarchy

**Pressure:** the system needed to reason about reuse, environment change, mismatch, named
procedures, and multi-level goals rather than treating every turn as isolated text.

**Implementation:**

- environment identity and assumptions;
- plan reuse and cross-session transfer;
- stale-claim detection and mismatch repair;
- named concepts and command registry;
- typed claims and mission contracts;
- five levels: primitive, command/template, procedure, grounded task, mission/goal.

**Claims decision:** facts, beliefs, hypotheses, observations, operator assertions, execution
results, and procedures need one custody model even if their storage remains transitional.
Authority, scope, confidence, provenance, freshness, and invalidation are the meaningful axes.

**Carry:** the station and readiness paths still consumed several storage pockets directly.

## Phase 9 - Authority And Structural Enforcement

### Phase 9A - Immediate Readiness And Safety Leaks

**Pressure:** some commands could reach execution without the full readiness/authority story.

**Decision:** no command shape or model output is authority by itself.

### Phase 9B - Primitive Contracts

**Implementation:** primitive metadata expanded to preconditions, postconditions, required and
produced claims, frames, units, safety class, authority level, failure modes, validation hooks, and
substrate fingerprints.

**Reasoning:** a primitive may exist and still be unsafe or inapplicable. Readiness must check its
contract, not only its name.

### Phase 9C - Structured Selection Objectives

**Pressure:** scattered vocabulary checks could not reliably preserve ranked selection intent.

**Implementation:** `SelectionObjective` captures attribute, maximum/minimum direction, ordinal,
and metric.

### Phase 9D - Typed Command And Side-Effect Authority

**Implementation:**

- `CorticalEnvelope` wraps the turn;
- `ApprovedCommand` is the typed command decision;
- `CommandResult` preserves the trace;
- `ExecutionTicket` gates task/runtime entry;
- `RawMotorTicket` gates explicit low-level movement;
- `MemoryWriteTicket` gates durable operator-claim mutation.

**Decision:** query answers and refusals may not need side-effect tickets, but they still need the
typed plan/readiness/command chain.

### Phase 9E - Block, Schema, And Knowledge Gates

**Pressure:** typed objects existed, but direct cross-block mutation and loose dictionaries could
still bypass them.

**Decision:** enforce three gates:

1. **Block gate:** station, Cortex, Sense, Spine, readiness, knowledge, and substrate boundaries
   have explicit ownership.
2. **Schema gate:** authority and control cross boundaries through typed messages.
3. **Knowledge gate:** claims, procedures, and provenance flow through one representation surface.

**Implementation:** `ClaimRecord`, `KnowledgeSnapshot`, `RepresentationStore`, representation-
backed active claims, request-plan/readiness provenance, and probes against direct station writes
to knowledge, episodic memory, and scene state.

**Rejected shortcut:** adding a large world-model or ontology subsystem. Existing claims and
representation were kept until evals prove they are too lossy.

## Phase 10 - First-Cut Extraction Boundaries

Phase 10 was not a file-size campaign. It extracted only the boundaries required to stop
substrate and authority logic from being born inside the station.

### 10A - CommandAuthority

**Purpose:** move "what happened this turn?" construction out of the facade.

**Implementation:** `command_authority.py` owns envelope, approved-command, command-result, and
trace construction. Station compatibility methods delegate.

### 10B - SideEffectAuthority

**Purpose:** give ticket minting one owner.

**Implementation:** `side_effect_authority.py` mints execution, raw-motor, sense, and memory-write
authority from validated plans/readiness.

### 10C - SubstrateAdapter

**Purpose:** move MiniGrid HOW out of the station.

**Implementation:** architecture-level `SubstrateAdapter`, concrete
`MiniGridSubstrateAdapter`, and adapter-owned sense/spine/runtime/render/reset/prewarm paths.

**Discovery:** construction still assembled MiniGrid pieces inside the station, so an adapter
interface alone was insufficient.

### 10D - OperationalContext

**Decision:** separate HOW from MEANING:

- `SubstrateAdapter` owns sensing, action, planning, control, environment calls, rendering, and
  validation hooks.
- `OperationalContext` owns object/attribute vocabulary, task families, references, grounding
  semantics, claim/display rules, environment identity fields, and procedure hints.

**Implementation:** typed context, MiniGrid context profile, compact context slices, and stable
context fingerprint.

### 10E - Domain Helper

**Purpose:** move obvious color/object parsing, labels, metric answers, and grounding display out
of orchestration.

**Implementation:** context-bound `MiniGridDomainHelper`.

**Discovery:** extraction of helpers did not by itself make behavior parametric; deeper compiler
and station paths still contained door-shaped assumptions. That later became Phase 14 work.

### 10F - TurnOrchestrator

**Purpose:** isolate one operator turn through pending state, intent, plan, readiness, command,
ticket, and result.

**Implementation:** top-level routing and pending clarification moved to `TurnOrchestrator`.

**Discovery:** the orchestrator still reached deeply into station-private methods. Moving methods
without moving state would multiply coupling. This became the central premise of the Phase 16
state-first decomposition.

### 10G - RuntimePackage

**Purpose:** make the station injectable with a substrate/context/helper/registry bundle.

**Implementation:** `RuntimePackage`, MiniGrid factory, and a non-MiniGrid injected fixture.

**Decision:** the fixture was only the proof vehicle. The architectural outcome was
substrate-neutral construction.

### 10H - PlanningSemantics

**Pressure:** planner and verifier still constructed MiniGrid-specific grounding handles.

**Implementation:** `PlanningSemantics` derives object vocabulary, metrics, pluralization,
capability handles, ranked claim outputs, and preservation signals from `OperationalContext`.
`RequestPlanner` and `IntentVerifier` accept the same bound semantics.

**Discovery:** the LLM compiler profile and deeper station paths remained object-specific. The
control-plane boundary was clean enough to proceed, but full vocabulary parameterization remained
future work.

### 10I - Operator-Defined Query Primitive Assembly

**Pressure:** the project goal includes just-in-time capability construction. Requests such as
"define convenientDistance as min(euclidean, manhattan)" could not remain unsupported.

**Decision:** start with pure query/grounding primitives only:

- structured formulas, never arbitrary operator code;
- dependency handles checked by readiness;
- operator approval required;
- deterministic fixtures validate the callable;
- registry promotion happens only after validation;
- the new primitive cannot authorize actuation by itself.

**Implementation:** `PrimitiveDefinitionRequest`, dependency planning, proposal/approval flow,
formula assembler, validation, dynamic registry insertion, context metric registration, ticketed
knowledge record, and inline use in later ranked tasks.

**Representative pressure cases:** min/max/sum, Euclidean modulo a constant, rejected proposals,
and unsafe expressions attempting to mix movement into a query metric.

**Discovery:** inline metric-plus-task requests were compound missions. Letting the station parse,
register, resume, and dispatch them flattened mission reason and authority. That triggered Phase
11.

## Phase 11 - Mission Ownership And Architecture Surgery

### Phase 11 - Mission Flow And Architecture Surgery

Status: **complete**.

### 11A - Cortex-Owned Inline Mission Flow

**Pressure:** an inline derived metric task could collapse to "go to the yellow door" and lose why
that target was selected.

**Decision:** compound work belongs to Cortex and must preserve lineage:

1. typed primitive definition;
2. dependency/readiness work;
3. evidence/ranking;
4. ordinal/selection claim;
5. task execution;
6. final ticket carrying parent mission provenance.

**Implementation:** `MissionCortex`, expanded `MissionExecutionPlan`, mission id,
parent-request id, provenance, continuation intent/plan/graph, and child tickets.

The concrete landing sites were `jeenom/mission_cortex.py`,
`evals/claim_custody_mission_flow_probe.py`, and `tests/test_phase11_mission_flow.py`.
The probe also guards against station-local resume payloads and requires the final
`ExecutionTicket` to retain mission lineage.

### 11B - Hostile Primitive And Mission Eval Ladder

**Pressure:** the eval suite was heavily deterministic/regex-shaped and could pass while the real
LLM or paraphrased operator path regressed.

**Decision:** test architecture roles through paraphrase matrices before production fixes.

The hostile ladder covered:

- front-cell Sense questions with no motion;
- explicit low-level Spine actions with `RawMotorTicket`;
- named query procedure teaching and recall through representation;
- multi-action sequences with child lineage;
- conditional Sense-before-Spine gating;
- ranked distance paraphrases;
- compound derived-metric missions;
- negative controls for ambiguous, unsupported, or random-policy requests.

**Acceptance rule:** equivalent surface language must converge to equivalent typed intent, plan,
readiness, claims, tickets, and outcome. An LLM-path feature without a fake-transport parity probe
does not count as implemented.

The focused artifacts were `evals/synthesis_primitive_ladder_probe.py` and
`tests/test_phase11b_primitive_ladder.py`. Negative controls were as important as successful
paraphrases: ambiguous targets, unsupported pickup, and requests for random movement had to
clarify/refuse without minting task or raw-motor authority.

### 11C - Seven-Part Architecture Surgery

This was a strict sequence, each operation behind a red bar:

1. **Import partition and domain purge:** domain vocabulary registration replaced schema-level
   door comparisons; `GroundedDoorEntry` became `GroundedObjectEntry`; substrate constants moved
   behind domain registration.
2. **Dispatch extraction:** operator-intent dispatch moved to `TurnOrchestrator`.
3. **Knowledge-type routing:** the large intent-type chain collapsed into claim, procedure,
   provenance, action, and control paths.
4. **IntentCache:** bounded fast-path NLU moved into one cache/compatibility home and still passed
   through verification/dispatch.
5. **One readiness gate:** the shadow `Readiness` class was deleted; `ReadinessGraph` remained the
   real gate.
6. **Capability-based eval names:** evals were renamed around the contract they protect rather
   than archaeological phase numbers.
7. **Hardware-facing schema fields:** claim validity, postcondition checker, mission risk/cadence,
   and typed failure outcome fields were added.

**Decision:** no rewrite and no new feature work during the surgery. Preserve behavior while
removing alternative authority paths.

## Phase 12 - ORPI v0.1

The detailed interface is in [orpi_spec.md](orpi_spec.md).

### Phase 12 And 12D - ORPI v0.1

Status: **complete for MiniGrid**.

### 12A - Contract, Manifest, And Trace Boundary

**Pressure:** JEENOM needed an embodiment interface that exposes capabilities to cognition and
emits supervision/audit traces from cognition.

**Decision:** ORPI has two halves:

- inbound contract/manifest/procedure capabilities;
- outbound `LabelledEpisode` traces with intent, grounding, plan, authority, execution,
  verification, attribution, and steering.

### 12B - Procedures And Taxonomy Compatibility

**Decision:** expose OEM-vouched procedures without creating a second recipe hierarchy.
`OrpiProcedure` is an interface view over existing recipes and expands to primitive handles before
authority checks.

The ORPI classes are `sense | actuation | meta`, but v0.1 keeps a compatibility bridge from legacy
implementation layers until a second substrate proves the taxonomy.

### 12C - Knowledge Scope

**Implementation:** `site | embodiment | universal` scope, `KnowledgeChannel` writer policy,
labelled-episode KB writes, and per-scope reuse counters.

**Discovery:** `universal` is structurally supported but mostly degenerate in MiniGrid because
real plans still name concrete primitives rather than pure effect vocabulary. A second substrate
is required to prove useful universal transfer.

### 12D - Consolidation And Leak Audit

**Purpose:** close ORPI v0.1 without mixing substrate coupling and station bloat.

**Completed work:**

- unified the interface label at v0.1;
- corrected diagram trace ownership;
- wired named postcondition verification and failure-attribution mapping;
- classified substrate leaks by `{cheap | structural}` and
  `{curriculum-touching | not}`;
- moved MiniGrid grounding primitives out of the generic primitive library;
- derived ranked step ids from planning semantics.

**Crucial ordering decision:** substrate coupling is on the critical path to a second substrate;
station file size is not. Curriculum-touching leaks must be removed before the curriculum.
Non-curriculum leaks may wait for Phase 14. Station de-bloat remains Phase 16 and requires a
reviewed decomposition design first.

## Phase 13A - Constraint Steering And Supporting Cleanup

Status: **complete**.

### Phase 13A - Steering Core

Status: **complete**.

**Decision:** steering is a typed, separable constraint layer rather than prose mixed into task
intent.

`SteeringDirective` carries:

- budget;
- scope;
- risk;
- stopping rule.

The directive is attached at semantic verification only when coherent with an action intent.
Readiness treats insufficient risk authority as `needs_authorization`. Execution enforces the step
budget and emits typed `budget_exhausted`. `LabelledEpisode.steering` records the active directive.

### 13A.1 - Coordinate-System Abstraction

Status: **complete**.

**Pressure:** AI2-THOR preparation exposed integer-only 2D coordinates and hand-rolled distance
math across many files.

**Decision:** coordinate debt was distinct from station bloat and could be fixed immediately.

**Implementation:** `geometry.py`, N-dimensional Manhattan/Euclidean metrics, float-preserving
coordinate conversion, optional z coordinates, shared `.coord`/`.agent_coord`, and geometry-
backed synthesized primitive examples/validation.

**Regressions surfaced while testing:** uninitialized active steering on direct dispatch and a
stale ORPI version assertion.

### 13A.2 - Consolidation Spike

Status: **complete**.

The spike targeted the debt pattern "one concept hand-rolled in several files", not general
de-bloating.

#### 13A.2.1 - Fingerprint Helper

Status: **complete**.

Centralized canonical JSON, stable hashing, and fingerprints while preserving byte-for-byte cache
keys.

#### 13A.2.2 - Capability-Handle Grammar

Status: **complete for the scoped consolidation; remaining compiler-profile cleanup moved to
Phase 14**.

**Discovery:** the compiler's handle strings were entangled with the remaining object-vocabulary
leak. Only clean station construction sites were routed through `PlanningSemantics`; compiler
profile cleanup was deferred rather than forced into the wrong phase.

#### 13A.2.3 - Metric-Name Detection

Status: **complete for the behavior-preserving consolidation**.

Centralized behavior-preserving text detection while deliberately retaining behavior-laden default
metrics and supported lists. Euclidean-first behavior for ambiguous text was pinned rather than
accidentally changed.

#### 13A.2.4 - TurnState

Status: **complete**.

Moved the per-turn trace fields into one typed object while retaining compatibility properties.

**Architectural result:** the state map exposed that the station problem is state-first, not
method-first. This became the prerequisite design evidence for Phase 16.

### 13A.3 - Substrate Handle Routing Investigation

Status: **investigation complete; implementation folded into Phase 14**.

**Result:** effectively empty under the "routable now" constraint. Pulling more work forward would
have created churn that Phase 14 would redo. The work was folded into Phase 14.

### 13A.4 - Operator-Station Decomposition Design

Status: **complete as a design gate; implementation remains Phase 16**.

The accepted state model and service-boundary design is in
[blueprint.md](blueprint.md#operator-station-decomposition-design).

**Decision:** do not begin even the safe-leaf extractions. Partial observability, evidence
readiness, mission termination, bounded search, and second-substrate pressure must shape the
kernel first. `StationRuntime` is the seed of that kernel, not merely a collaborator bag.

## Phase 13B - Partial Observability And Evidence-Governed Execution

Status: **in progress; 13B.1-13B.5 complete, 13B.6 active**.

### 13B.1 - Claim Freshness

Status: **complete, including the later hot-path decay integration**.

**Pressure:** a single scene-fingerprint equality treated any agent motion as stale-world change.
Looking away from an observed object was indistinguishable from the world changing.

**Decision:** freshness and epistemic status stay separate. The freshness axis is:

| State | Meaning | Usable on hot path |
|---|---|---|
| `current` | supported by current observation | yes |
| `unverifiable` | previously observed, currently out of frame, world not known to have changed | yes, as retained belief |
| `stale` | invalidated by environment/world change | no |
| `unknown` | no usable claim remains after decay or absence | no |

Only spatial observation claims become `unverifiable` because of framing. Operator assertions,
facts, and procedures do not become unverifiable merely because the agent looks away.

**Transition decisions:**

- current + leaves view -> unverifiable;
- current + remains in view -> current;
- unverifiable + re-observed before decay -> current;
- unverifiable + unseen past TTL -> unknown;
- environment/world mutation -> stale;
- unknown + re-observation -> a new current grounding, not restoration of dead authority.

**Later hot-path implementation:** `ObservationClaim` gained freshness and
`last_observed_tick`; Cortex reads current/unverifiable claims and treats stale/unknown as absent.
`world_sample.step_count` drives decay. Sense stops refreshing target location when the target is
out of FOV, allowing the retained claim to age. Passable-cell belief persists with the same TTL
and expires rather than unioning forever.

**Concrete implementation surface:**

- `jeenom/claim_freshness.py` owns `UNVERIFIABLE_DECAY_STEPS`, `ClaimTTL`,
  `framing_satisfiable`, and `next_freshness`;
- `ClaimTTL` supports eternal, timed, and conditional validity, but the live decay edge currently
  uses the timed step budget;
- `ObservationClaim.last_observed_tick` records the substrate step at which supporting evidence
  was last in frame;
- `tests/test_claim_freshness.py` pins the state machine;
- `evals/claim_custody_unverifiable_freshness_probe.py` graduated from an expected red bar into
  the architecture gate;
- `tests/test_claim_freshness_hotpath.py`, `tests/test_claim_decay_on_loop.py`,
  `tests/test_target_belief_decay_on_loop.py`, and `tests/test_occupancy_belief_decay.py` protect
  the live integration and retained-map expiry.

The TTL constant is a tuning parameter, not an interface contract. Tests import the constant
rather than duplicating its current value.

**Deliberate debt:**

- uniform observation decay rate;
- intra-task Cortex claim store;
- mission clock depends on live adapter continuity;
- occupancy belief decays in Sense because planning consumes it before Cortex, leaving two decay
  sites until claim storage is unified.

### 13B.2 - Native MiniGrid FOV And Eval Lanes

Status: **complete**.

**Pressure:** human rendering could show the full grid while the agent's evidence should be
egocentric. Omniscient sensing would make partial-observability tests meaningless.

**Decision:** separate:

- full-observability evals for regression and architecture parity;
- partial-observability evals for FOV, unseen cells, evidence readiness, retained belief, and
  navigation.

Production partial sensing uses MiniGrid's native observation/FOV metadata, maps view coordinates
to global cells, and records visible/unseen sets. The agent must not answer global scene questions
from the human renderer.

**Concrete split:**

- `run_demo.build_env` creates the native partial-observation environment;
- `evals.harness.build_env` remains the legacy `FullyObsWrapper` lane;
- `evals.harness.build_partial_env` is the partial-observation lane;
- `build_minigrid_runtime_package(..., observability="full" | "partial")` makes the distinction
  explicit for station/session tests;
- `MiniGridAdapter` publishes pose, grid size, observation model, and visible-cell metadata;
- `MiniGridSense` projects local-view cells into global coordinates and creates occupancy only
  from observed cells;
- `SceneModel` and `WorldModelSample` carry the observation model and visibility boundary.

`evals/substrate_minigrid_fov_probe.py` is the boundary probe. The separate
`evals/substrate_partial_observability_navigation_probe.py` protects navigation/map behavior;
it is not interchangeable with the query/readiness probe.

**Navigation discovery:** using only the current FOV made passable knowledge disappear as the agent
turned, while unioning all observed cells forever would preserve incorrect beliefs. The accepted
model is persistence with freshness decay.

### 13B.3 - `needs_evidence` And Typed Clarification

Status: **complete**.

**Pressure:** an executable capability does not imply enough current evidence to answer or act.
Visible-only requests were either guessing or returning generic ambiguity.

**Implementation:** readiness gained `needs_evidence`; `ClarificationRequest` records prompt,
reason, resume kind, evidence scope, target, options, and provenance. Pending clarification carries
that typed request through the turn trace.

`RequestPlanStep.constraints` declares visible-only evidence dependencies, and readiness receives
a compact `evidence_state` so it can block a supported but currently unanswerable step before
execution. A new unrelated operator command cancels the pending evidence clarification instead of
being trapped by it. Raw motor movement refreshes the partial scene immediately so the next turn
uses the current point of view.

`evals/substrate_partial_observability_needs_evidence_probe.py` checks the schema, readiness
transition, pending state, operator response, and `LabelledEpisode.steering.pending_context`.

**Decision:** `visible_only` asks for help when evidence is absent. `search_allowed` is an explicit
future authority level, not permission to spin or explore by default.

### 13B.4 - LLM-Default Tool-Call Discipline And Parity

Status: **complete**.

**Pressure:** most evals exercised deterministic parsing while only a few forced the LLM path.
Regressions therefore appeared interactively even when the suite was green.

**Architecture decision:**

- unresolved semantic input defaults to `LLMCompiler`;
- the model emits a strict schema decision, never executable prose;
- deterministic controls, continuations, exact compatibility routes, and explicit fallback remain
  bounded;
- LLM and deterministic outputs converge through schema normalization, `IntentVerifier`, exact
  capability matching, planning, readiness, dispatch, and tickets;
- fallback must be visible.

**Eval lanes:**

- offline deterministic release gate;
- fake-transport `llm_path` matrix proving the LLM route and semantic parity;
- opt-in real `live_llm` probe.

`eval_master` removes `OPENROUTER_API_KEY` for every suite except `live_llm`, so the offline gate
cannot accidentally pay for or depend on a network call. The live probe skips without a key and
asserts stable structured facts: the LLM transport was used without fallback and the normalized
intent type is correct.

**Regressions that shaped this boundary:**

- a low `max_tokens` truncation caused malformed JSON and silent deterministic fallback;
- an LLM motor sequence such as "turn left twice and go forward once" was classified correctly
  but normalized into a task-sequence path that could not compile motor steps;
- a one-action motor sequence was rejected because execution incorrectly required two sequence
  entries;
- a visible door query reached a valid grounding plan but used a noncanonical answer field and
  dead-ended in result composition;
- LLM reason text was too tempting as a routing signal, so operator-facing behavior was restricted
  to structured fields.

**Fixes:** larger method budgets, visible fallback provenance, canonical answer-field aliases with
fail-closed unknown values, motor-sequence normalization, and parity probes for the real LLM route.

The truncation root cause was concrete: `compile_operator_intent` had a 256-token cap even though
the strict `OperatorIntent` schema emits a large complete object. The current method budgets are
1024 for `compile_operator_intent` and `compile_procedure`, and 768 for task, sense-plan,
skill-plan, and memory-update compilation. These are upper bounds, not requests to fill the
budget.

Answer-field repair lives at one parse chokepoint:
`schemas._ensure_canonical_answer_fields`. It maps conservative aliases such as
`distances -> distance` and `nearest -> closest`, preserves ordinal fields, and rejects unknown
values rather than letting them reach a dead-end composer. The corresponding tests are
`tests/test_answer_field_normalization.py`.

The LLM-path evidence is split deliberately:

- `evals/intent_fidelity_llm_path_parity_probe.py` proves route provenance and parity;
- `evals/intent_fidelity_llm_operator_matrix_probe.py` covers representative operator families
  and asserts that the advertised intent enum matches the schema;
- `evals/intent_fidelity_live_llm_probe.py` is the opt-in real-backend check;
- `tests/test_motor_sequence_single_step.py` pins the valid one-action repeated sequence.

**Threat-model decision:** these controls contain many model failures but are not a proof against
hostile prompts or crafted schema-valid side-effect attempts. The operator/backend remains
good-faith until Phase 17.

### Interactive Render And Episode Continuity Bug Fix

**Pressure:** after manual motor commands, starting a task could reset the environment, making
observed behavior look inconsistent and invalidating claims/pose continuity.

**Fix:** reuse the preview/task adapter with `skip_reset=True`; promote preview to task ownership;
typed `reset` closes adapters, clears episodic state, creates the fresh seeded environment, and
refreshes the scene; `Ctrl+C` closes the session cleanly.

**Scope fence:** spin/search prevention remained separate. Lifecycle repair did not add search
behavior.

### 13B.5 - Conditional Evidence MissionContract

Status: **complete**.

**Pressure:** "go straight until you see a blue door" must not become an unbounded repeated motor
command. The earlier "go to a blue door" behavior could look purposeful while actually relying on
a fixed search/spin loop and luck.

**Decision:** Cortex issues a mission contract that combines:

- a Sense primitive for the stop evidence;
- an operator-authorized Spine action primitive;
- a validated procedure that alternates fresh evidence and at most one action.

**Implemented flow:**

1. compile `conditional_sense_motor`;
2. `MissionCortex` builds a `MissionContract` with procedure, target/action/stop parameters, and
   exact capabilities;
3. readiness validates sense, action, and task handles;
4. `ExecutionTicket` admits the approved mission;
5. Sense produces fresh target evidence;
6. Cortex checks the stop claim;
7. if false, Cortex issues one `ExecutionContract`;
8. Spine executes one action;
9. Cortex senses again before another action.

The first contract uses exact handles
`sensing.find_object_by_color_type`, the configured action handle such as
`action.move_forward`, and `task.act_until_evidence`. The task contract advertises actuation
authority, a validation hook, and finite `budget_exhausted` / `no_progress` failure modes.
`tests/test_conditional_mission_contract.py` protects the contract, readiness, zero-actuation
initial match, one-action cadence, live-pose continuity, and typed stuck result.

**Acceptance:** initial match causes zero actuation; false evidence permits one action; newly
visible target stops before the next action; deterministic and LLM routes build equivalent
contracts; current pose is preserved; budget and `no_progress` terminate finitely.

**Limit discovered:** unchanged pose is a movement-specific blockage signal, not a general action
outcome. Toggle, pickup, drop, controller faults, environment termination, and operator-defined
wall exits need typed mission termination and action outcomes. That is the next 13B slice.

### Pulled-Forward Phase 14 Slice - Parametric Object Types

Status: **complete for the scoped object-parametric routing slice**.

**Pressure:** door/key literals were scattered through generic compiler and station paths. Adding
another object type required edits across the kernel.

**Rejected proposal:** a new global `SUPPORTED_OBJECT_TYPES`. The repository already had the
correct sources of truth:

- `OperationalContext.object_vocabulary` for meaning;
- `PlanningSemantics` for handles/pluralization;
- exact `CapabilityRegistry`/manifest declarations for executable support.

A global list would create a second source of truth, and vocabulary registration alone must never
fabricate a capability.

**Implementation:** context-driven deterministic parsing, LLM tool schemas, task/grounding handles,
mission continuation, station plan construction, ranked display, mismatch checks, and generic
active-claim views. Exact manifest handles remain authoritative.

**Compatibility debt retained deliberately:** `ranked_scene_doors`, `ranked_doors`, `other_door`,
MiniGrid manifest declarations, validator fixtures, and illustrative prompts. Renaming those is a
cross-contract migration, not a string replacement.

## Phase 13C - Curriculum, Reuse, And MTBCI

Status: **planned after 13B**.

Planned longitudinal proof:

- partial-observability curriculum over increasing MiniGrid sizes;
- scoped knowledge and plan reuse;
- mean turns between clarification/intervention;
- proof that reuse improves intervention rate without weakening evidence or authority gates.

## Phase 14 - Remaining Leak Removal And AI2-THOR Spike

Status: **planned; the object-parametric routing slice was completed early**.

The remaining work is intentionally split:

- cheap generic leaks: prompt examples, color schemas, compatibility names, generic-station
  MiniGrid imports/defaults, and stronger static guards;
- structural bindings such as concrete Sense/Spine roles: validate through the second substrate
  rather than rename speculatively.

The AI2-THOR branch is a requirements-discovery spike, not the committed port. Every ORPI bend or
break becomes a concrete spec issue and Phase 15 requirement.

## Phase 15 - Cross-Substrate Proof And ORPI v1

Status: **planned**.

The committed second-substrate port must preserve the same typed intent, plan, readiness, claims,
tickets, and orchestration flow. Differences should be confined to context/substrate bindings.
Only after both substrates conform may ORPI freeze to v1.

## Phase 16 - Operational Hardening And Station Decomposition

Status: **later; design accepted and banked**.

Execute the state-first design in
[blueprint.md](blueprint.md#operator-station-decomposition-design). File size is not the
motivation. The target is explicit shared ownership, a `StationRuntime` kernel seed, service
boundaries, and removal of `TurnOrchestrator` private facade reach-ins while preserving public API
and every authority/trace guarantee after each extraction.

Entry criteria:

- mission termination/action outcomes have stable typed shapes;
- bounded evidence-gathering authority is represented or consciously deferred behind a stable
  boundary;
- the second-substrate work has identified the runtime collaborators the kernel actually needs;
- the public session API and eval reach-ins are inventoried again;
- deterministic, ORPI, LLM-path, and focused pytest gates are green;
- a decomposition probe is ready to enforce each moved responsibility.

Ordered extraction:

0. Introduce frozen `StationRuntime` with collaborators, `TurnState`, and `PendingState`; retain
   compatibility properties on the facade.
1. Extract `EnvironmentTracker`.
2. Extract `ConceptService`.
3. Extract `GroundingService`.
4. Extract `TaskExecutor`.
5. Extract `CommandFactory`, then review value/risk before the highest-coupling work.
6. Extract `PendingFlowController`.
7. Rewire `TurnOrchestrator` through services and remove private facade reach-ins.
8. Thin `OperatorStationSession` to lifecycle, I/O, summaries, and delegation.

After every step:

- the facade delegates through the new service;
- no new service receives `OperatorStationSession`;
- public compatibility properties remain;
- tickets, `CommandResult`, `LabelledEpisode`, and no-LLM runtime guarantees remain unchanged;
- the full gate runs before continuing.

Risks and guards:

- mutation of `TurnState` or `PendingState` is explicit shared state; mutation of facade-local
  fields from a service is a boundary smell;
- delegating shims remain while tests or external callers use a moved method;
- `regression_live_operator_probe`, the golden path, and a decomposition anti-drift probe protect
  every extraction;
- a step that cannot stay green is re-sliced or parked rather than forcing the whole refactor.

## Phase 17 - Capability And Adversarial Stress

Status: **planned**.

Use harder robotics/ARC-like pressure tests to expose missing contracts, claims, evidence,
decomposition, and steering. Complete the audit of all LLM-controlled dispatch fields and prove
that hostile or crafted schema-valid output cannot mint unintended side-effect authority.

## Historical Documentation Lesson

The cleanup that made the roadmap concise also removed too much of this reasoning. The standing
documentation rule is now:

- streamline duplicate status;
- retain rationale, red bars, discoveries, bug context, rejected alternatives, and acceptance
  evidence;
- rewrite stale prose as history instead of deleting the architectural memory.

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
- Update this file when phase status, phase history, verification, or ordering changes.
- Preserve rationale, spikes, regressions, rejected alternatives, and acceptance evidence when
  streamlining; rewrite stale prose as history instead of deleting architectural memory.
