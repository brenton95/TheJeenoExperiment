# ORPI v0.1 — Open Robotics Primitive Interface

**Status: v0.1, explicitly unstable.** Freeze to v1 only after the interface survives the
Phase 15 cross-substrate port (the validation event). Standards extracted from n=1 substrates
ossify the wrong abstractions; v0.1 exists to be broken by the second substrate, deliberately.

This document is the authoritative contract/manifest/procedure/trace reference for ORPI. It is built
and versioned alongside the code. The implementation roadmap lives in
[task_plan.md](task_plan.md) Phase 12.

---

## 1. What ORPI Is

ORPI is to robot cognition what MCP is to LLM tooling: a typed interface standard between a
cognition layer (JEENOM) and an embodiment (MiniGrid, Jackal, UR5). It has **two halves**:

1. **Inbound — what a substrate exposes to cognition.** Every capability of a robot is published
   as a *primitive* with a machine-readable contract. Cognition never sees hardware, drivers, or
   policies; it sees contracts.
2. **Outbound — what a deployment emits to learning.** Every executed turn produces a *labelled
   episode trace* in a standard format: what was intended, what was grounded, which contracts were
   checked, what executed, what the postcondition verification said, and — on failure — which
   component's contract was violated. This is the supervision artifact that self-improving
   deployment loops consume. No other part of the robotics stack produces this by construction;
   ORPI does.

The grounding obligation is discharged **at the primitive boundary**. Primitives convert reality
into typed claims and intents into physical effects. The cognition layer above provides grounding
*accounting* (custody, validity, arbitration, composition), never grounding itself. A primitive is
swappable precisely when its grounding obligation is fully discharged at its contract.

## 2. Scope and Non-Goals

**In scope:** the contract schema, the manifest schema, the trace schema, registration/conformance
rules, versioning policy.

**Out of scope:** how primitives are implemented (policies, planners, controllers, models —
substrate's business); transport/serialization beyond "JSON-serializable dataclasses" (v0 is
in-process Python; wire protocol is a v1+ question); tasks whose success conditions don't compress
into checkable object-centric predicates (contact-rich, deformable, aesthetic — explicitly fenced
out).

## 3. Primitive Classes

Every primitive declares exactly one class:

| Class | Role | Examples | Cadence home |
|---|---|---|---|
| `sense` | reality → claims | `detect_door`, `get_pose`, `gripper_contact_check` | perception rate |
| `actuation` | approved command → physical effect | `navigate_to`, `open_door`, `pick_up` | control rate (Spine) |
| `meta` | claims → claims (computation over the operational model) | spatial relations (`ahead_of`, `visible`), metric ranking, claim queries | deliberation or perception rate |

`meta` primitives further declare `mode: deterministic | deliberative`:
- **deterministic** meta-primitives may be referenced inside compiled plans and run during execution
  (e.g., egocentric frame transforms, manhattan-distance ranking, derived-claim inference).
- **deliberative** meta-primitives invoke the LLM compiler (recompilation, repair, synthesis). They
  are exception handlers that pause execution. **Compiled plans may never reference deliberative
  meta-primitives.** This is the enforcement point for the no-LLM-in-the-loop invariant.

These three classes are the ORPI taxonomy. `OrpiContract.primitive_type` always returns one of
`{sense, actuation, meta}` via `orpi_primitive_type_for()`.

**Compatibility bridge (v0):** `schemas.PrimitiveSpec.primitive_type` still accepts both the legacy
values (`task | grounding | sensing | action | claims`) and the ORPI values (`sense | actuation |
meta`). The mapping is: `sensing → sense`, `action → actuation`, `task / grounding / claims →
meta`. MiniGrid primitives are authored with legacy values; `OrpiContract` projects them into the
ORPI taxonomy. The hard schema-level remap — where primitives are authored with ORPI types directly
— is deferred until the contract/manifest/trace boundary survives the Phase 15 cross-substrate
port. This is the right sequencing: standardise the interface before forcing every primitive
author to use the new vocabulary.

## 4. The Contract

The contract is the existing `schemas.PrimitiveSpec`, serialized. Fields marked **NEW** are v0
additions; everything else already exists in the repo.

| Field | Meaning | Notes |
|---|---|---|
| `name`, `primitive_type`, `layer`, `description` | identity | `OrpiContract.primitive_type` ∈ {sense, actuation, meta} (mapped from legacy values by `orpi_primitive_type_for()`); `layer` ∈ {sense, cortex, spine} |
| `inputs` / `outputs` | typed parameters | continuous params (grasp pose, force threshold) live here — the symbol layer passes them through, never chooses them |
| `preconditions` | claims that must hold, with min confidence | evaluated by ReadinessGraph |
| `postconditions` | **object-centric state deltas** (Δg), not action descriptions | "door(d).state: closed→open", not "arm moved" — this is what makes procedures retargetable across embodiments |
| `postcondition_primitive` | the `sense` primitive that verifies the postcondition | a postcondition is a proposition *plus its checker*; closes the contract system over the primitive vocabulary. `None` = implicit/free (MiniGrid degenerate case) |
| `required_claims` / `produced_claims` | claim kinds consumed/emitted | |
| `units`, `frame_id`, `required_frames` | dimensional and frame contracts | |
| `safety_class` | risk category | drives required claim confidence and verification tier via the manifest risk policy |
| `authority_level` | who may invoke | ticket system input |
| `failure_modes` | **typed** failure outcomes this primitive can emit | each maps to a `FailureOutcome.category`; failures are states, not just unmet postconditions |
| `validation_hooks` | preflight checks before execution | a world-model shadow rollout is a validation hook — this is the ConsequencePredictor socket |
| `substrate_fingerprint` | binds the contract to a substrate version | staleness detection for cached plans |
| `mode` **NEW** | `deterministic \| deliberative` (meta only) | see §3 |
| `cadence` **NEW** | declared execution rate class: `control \| perception \| deliberation` | substrate enforces: nothing at deliberation cadence sits on a control-cadence path |
| `invariant_level` **NEW** | which invariant the primitive preserves: `pose \| contact \| object_state \| intent` | ReadinessGraph metadata for primitive substitution across embodiments |

## 5. The Manifest

One per substrate, registered at adapter init. Extends the existing `register_domain_vocabulary` /
`OperationalContext` pattern:

- `substrate_id`, `substrate_fingerprint`, `orpi_version`
- `object_vocabulary` — the registered object types (exists today)
- `symbol_mappings` — domain constants the substrate owns (e.g., MiniGrid IDX_TO_COLOR/IDX_TO_OBJECT —
  moved here per the Phase 11C partition; partly done today via `register_domain_index_maps`)
- `frames` and `units` registries
- `risk_policy` — table mapping `safety_class` → required claim confidence, required verification
  tier, required validation hooks. **Policy is auditable manifest data, not buried thresholds.**
- `primitives` — list of contracts (§4)
- `bundled_procedures` — optional OEM-vouched procedure contracts (§6)

Registration is **fail-closed in spirit**: validation is permissive only before any manifest is
registered, and a conformance probe must assert each adapter registers at init, so the permissive
window is provably never live.

## 6. Procedures

ORPI v0.1 includes `OrpiProcedure` for vouched-for recipes that a substrate or embodiment vendor
can expose as first-class interface objects. A procedure is not a second recipe hierarchy; it is the
serialized interface view over the existing recipe/plan-cache record.

Serialized fields:

- `name`
- `steps` — ordered primitive references by name plus effect/postcondition metadata
- `declared_postconditions`
- `declared_preconditions`
- `provenance` — `oem | synthesized | operator`
- `safety_class` — max risk class across constituent primitives
- `authority_level` — max authority requirement across constituent primitives
- `substrate_fingerprint`

`OrpiManifest.bundled_procedures` is reserved for OEM-vouched procedures only. Manifest validation
rejects any bundled entry whose provenance is not `oem`, and rejects any procedure step that does
not reference a primitive present in the same manifest. Synthesized and operator procedures may be
recorded and traced, but they are not allowed to masquerade as substrate-bundled OEM capability.

Planner parity rule: a bundled procedure is selectable by declared postcondition alongside primitive
contracts. Once selected, it expands to primitive handles before ticket issuance and readiness/
authority checks, so procedure selection never bypasses per-primitive gates.

## 7. The Trace (Outbound Half)

Every turn emits a `LabelledEpisode`. It is a thin aggregator/serializer over existing trace
machinery (`OperatorIntent`, `RequestPlan`, `ReadinessGraph`, `CommandResult`, `FailureOutcome`,
`CorticalEnvelope`) — not a parallel type tree.

```
intent          — OperatorIntent as compiled (+ verifier verdict)
grounding       — claims consumed, with provenance/confidence/freshness at read time
plan            — RequestPlan + ReadinessGraph verdicts per step
                  + candidate kind/provenance (primitive vs procedure)
authority       — tickets issued, scope, issuer
execution       — per-step CommandResult, incl. typed FailureOutcome
verification    — postcondition_primitive results vs. predicted postconditions  [*]
attribution     — on failure: which contract was violated                        [*]
                  (stale_claim | miscompiled_intent | unmet_postcondition |
                   missing_authority | substrate_fault)
steering        — any operator interventions in this episode (Phase 13 ask-for-help)
                  + KB writes and per-scope KB reuse counters
```

**[*] Closed in Phase 12D.** `verification` and `attribution` were the highest-value fields for
component-level credit assignment — and the most incomplete in v0.

- `verification` now invokes the contract's named `postcondition_primitive` (
  `sensing.parse_grid_objects` for MiniGrid action primitives) and records the checker name. The
  degenerate case (`postcondition_primitive=None`) is labelled explicitly as `"degenerate_boolean"`.
- `attribution` maps `FailureOutcome.category` through the ORPI taxonomy:
  `stuck | progress → unmet_postcondition`, `blocking_claim → stale_claim`,
  `timeout → substrate_fault`. The raw category is preserved alongside.

Rich Δg verification (invoking the checker against a re-observed state rather than the execution
result already in `final_state`) is Phase 13 work — it requires the derived-claim layer.

The trace is the product's audit story ("what did the operator tell it, when, and did the robot
honour it") and the learning story (component-level credit assignment for deployment loops) in one
artifact. Failed episodes are **kept**, with attribution — failures are future skills if properly
labelled.

## 8. Knowledge Scope

Durable knowledge records carry a transfer scope. This is the falsifiable replacement for an
informal information/knowledge/wisdom ladder:

| Scope | Invalidated by | Default for |
|---|---|---|
| `episodic` | scene change | claims only; not stored in `KnowledgeBase` |
| `site` | site/map change | operator-taught facts and constraints |
| `embodiment` | substrate fingerprint change | OEM procedures and recipes naming morphology-specific primitives |
| `universal` | task ontology change | recipes expressed purely in postcondition/effect vocabulary |

`KnowledgeBase` rejects `episodic` records. `derive_scope(record, manifest)` deterministically
derives broader scopes from ORPI contracts: direct substrate-fingerprinted primitive references are
`embodiment`; effect-only recipes whose steps stay inside the manifest effect vocabulary can be
`universal`; OEM bundled procedures are always `embodiment`.

`KnowledgeChannel` is the gated write/read surface used by station/orchestration paths. It enforces
writer identity by scope, emits durable KB writes into `LabelledEpisode.steering.knowledge`, and
records per-scope reuse counters for Phase 13 transfer metrics.

## 9. Conformance

A substrate is ORPI-v0.1 conformant when:
1. Every capability is registered through a contract; no side-channel capabilities.
2. The manifest is registered at init (probe-enforced).
3. All postconditions are object-centric deltas; all actuation primitives with `safety_class ≠ query`
   name a `postcondition_primitive`.
4. Cadence declarations are honoured: no actuation contract has non-control cadence; no sense
   contract has non-perception cadence; no control-cadence contract is deliberative. (v0 probe:
   data check over manifest contracts. Full AST static check — tracing Spine call paths — is
   deferred to v1 when the second substrate makes it worth the investment.)
5. Every executed turn emits a `LabelledEpisode`.
6. No compiled plan references a `deliberative` meta-primitive. This invariant is enforced at
   runtime in `CortexSession.plan` (raises `SchemaValidationError` on violation) and covered by
   a probe. (v0 probe: exercises the enforcement path with a synthetic deliberative plan. Full
   AST static check of all plan-build paths is deferred to v1.)
7. Any bundled procedure is OEM-provenance only and references only primitive contracts present in
   the same manifest.
8. Durable KB writes use the scope-gated knowledge channel; `episodic` records are not persisted in
   the KB.

## 10. Versioning Policy

- v0: in-process, Python dataclasses, MiniGrid is the only conformant substrate. Breaking changes
  allowed freely.
- v0.1: adds `OrpiProcedure`, optional `bundled_procedures`, and scoped knowledge traces without
  changing the primitive contract compatibility bridge.
- The Phase 15 port (second substrate) is the validation event. Every place v0.1 bends or breaks during
  the port is recorded as a spec issue.
- v1 freeze happens only after the second substrate is conformant. From v1: additive changes only;
  breaking changes require a major version.

## 11. Spike Findings (Phase 14 — AI2-THOR boundary)

Deviations surfaced while wiring a second substrate (AI2-THOR). Per §10, every
place v0.1 bends or breaks is recorded here. These feed the Phase 15 port and the
v1 freeze decision.

### F1 — Domain registration is last-writer-wins; substrates cannot coexist in-process (kernel)

`register_domain_vocabulary` / `register_domain_index_maps` /
`register_open_state_passable` / `register_traverse_to_adjacent` are
**module-level global registrations** in `schemas.py` / `sense.py`. They are not
keyed by substrate, so the last substrate to register clobbers the previous one.
Two substrates (MiniGrid + AI2-THOR) therefore cannot be live in the same process.

- **Surfaced by:** `Ai2thorOperationalContext` deliberately does **not** call these
  registrars (see its docstring), to avoid clobbering MiniGrid's registration. The
  boundary test passes only because of this omission.
- **Severity:** kernel-level, not adapter-level. This is a genuine
  substrate-independence failure — the registration mechanism is substrate-exclusive.
- **Triage for Phase 15:** key these registrations by `substrate_id` (per-context
  maps) instead of process globals, or move them onto `OperationalContext` itself.
- **Status:** open. Do not resolve as part of the boundary spike (kernel edit;
  escalate per `AGENTS.md`).

### F2 — `SceneObject.x/y` are `int`; AI2-THOR coordinates are continuous floats (schema)

`SceneObject.x` / `.y` are `int`-typed, but AI2-THOR positions are continuous 3D
floats `(x, z)`. The spike's decided mitigation is to project `(x, z)` onto a 2D
occupancy grid, but the schema's `int` typing forces quantization at the boundary
and will lose sub-cell precision.

- **Surfaced by:** plan 001, Finding #1 (pre-filed before sense work).
- **Severity:** schema-level. Not exercised by the boundary test (no live sense
  loop yet); becomes blocking the moment `ai2thor_sense.py` maps
  `event.metadata["objects"]` → `SceneModel`.
- **Triage for Phase 15:** decide adapter-side quantization (keep `int`, document
  precision loss) vs. widening `SceneObject` coords to `float` (kernel/schema edit).
- **Status:** **RESOLVED upstream** (`origin/master` `5b89f0f`, "changed hardcoded
  2D into scalable coordinate system"). The kernel chose the second option:
  `SceneObject.x/y` and `agent_x/y` are now `float`, with an optional
  `z: float | None` (absent on 2D substrates, set on 3D ones). A new pure
  `jeenom/geometry.py` provides N-dimensional `manhattan`/`euclidean` and an
  `as_coord` coercer (keeps integral grid coords as `int` for MiniGrid display,
  preserves genuine floats). MiniGrid behaviour preserved. The adapter must now
  route coords through `geometry.as_coord` instead of `int(round(...))` — see
  plan 006.

### F3 — Success determination is coupled to gym `reward` signal (spine)

`MiniGridSpine._execute_env_action` (`spine.py:231–237`) determines task success
via `done and reward > 0`. AI2-THOR has no reward signal — success must come from
a postcondition/sense check (e.g. "am I near the target object?"). The AI2-THOR
spine therefore returns `status="running"` for all dispatched motor actions; it
never fakes `succeeded`.

- **Surfaced by:** plan 003 spine implementation. Verified at `spine.py:232`.
- **Severity:** spine-level. Not a blocker for motor dispatch, but blocks
  end-to-end task completion detection on AI2-THOR.
- **Triage for Phase 15:** frame against ORPI's existing `postcondition_primitive`
  concept (§9.3). Success should be determined by a postcondition sense check, not
  a reward signal. Consider making success-determination pluggable per substrate.
- **Status:** resolved (adapter-side, plans 005 → 007). Navigation now uses
  closed-loop execution: each action checks `lastActionSuccess` from AI2-THOR
  metadata; a blocked move triggers a single re-plan from the agent's actual
  position (excluding the blocked cell). Postcondition proximity check
  (`euclidean(agent_floor, target_floor) <= grid_size*1.5`) confirms success
  after the path completes. No reward signal used. The kernel concept of a
  pluggable `postcondition_primitive` (§9.3) remains a Phase 15 target.
- **Design note (plan 007):** "re-plan once" is underspecified for a substrate
  with no obstacle map — a bare BFS from the same node over the same reachable
  set reproduces the blocked path (no-op). The adapter resolves this by
  excluding only the just-blocked cell for the single re-plan (one-shot drop,
  not a persistent obstacle model — Dev Rule 11). Phase 15 should decide whether
  ORPI's nav contract names a substrate-provided "blocked cell" signal, or
  leaves obstacle-tracking entirely adapter-side.

### F4 — `primitive_library.py` motor primitives are MiniGrid-shaped (primitive library)

`ACTION_PRIMITIVES` in `primitive_library.py` uses `int` `runtime_value` fields
(0–6 mapping to MiniGrid gym action indices) and MiniGrid-worded descriptions
(e.g. "Turn the MiniGrid agent left"). Despite being billed as the
substrate-agnostic source of truth, these are substrate-specific.

- **Surfaced by:** plan 003, confirmed by reading `primitive_library.py:248–332`.
- **Severity:** primitive-library-level. The AI2-THOR spine works around this by
  maintaining its own `AI2THOR_ACTIONS` map in `ai2thor_spine.py` and never
  importing `ACTION_PRIMITIVES`. But the leak means `primitive_library.py` cannot
  serve as a true substrate-agnostic registry.
- **Triage for Phase 15:** either split `primitive_library.py` into
  substrate-agnostic primitives + substrate-specific bindings, or move motor
  bindings onto the substrate adapter (where AI2-THOR's already live).
- **Status:** open. *Update (plan 007):* AI2-THOR spine previously hardcoded
  substrate defaults (`gridSize=0.25`, `rotateStepDegrees=90`); now grid spacing
  is **derived** from `GetReachablePositions` and rotation step is **injected
  config**. Fixed rotation step is a substrate artifact, not a kernel concept —
  the kernel reasons about "reach the target," angle-agnostic; real robots turn
  continuously via heading feedback.

### F5 — `SceneModel.grid_width/grid_height` are non-optional; AI2-THOR has no grid (schema)

`SceneModel` requires `grid_width: int` and `grid_height: int` as non-optional
constructor arguments. AI2-THOR operates in continuous 3D space with no
occupancy grid — there is no meaningful width/height. The AI2-THOR sense adapter
stubs these as `(0, 0)` via `WorldModelSample.grid_size=None`, which
`from_world_model_sample` falls back to `(0, 0)`.

- **Surfaced by:** plan 004 sense implementation (`ai2thor_sense.py`).
- **Severity:** schema-level. The stub works for the spike, but downstream code
  that reads `grid_width/height` (e.g. `_navigation_goals` bounds checking in
  `spine.py:288`) will behave incorrectly with `(0, 0)`.
- **Triage for Phase 15:** either make `grid_width/height` optional in
  `SceneModel` (kernel edit), or define a projected occupancy grid for AI2-THOR
  derived from `GetReachablePositions` (adapter-side, plan 005 territory).
- **Status:** confirmed non-blocking (plans 005 → 007). AI2-THOR nav uses
  `GetReachablePositions` as the reachability source and BFS over that set — it
  never reads `grid_width/height` or does bounds checking. The `(0,0)` stub is
  fine with no kernel edit. *Update (plan 007):* grid spacing is now **derived**
  from `GetReachablePositions` (minimum nonzero pairwise distance), eliminating
  the hardcoded `GRID_SIZE=0.25`. The schema-level leak remains a Phase 15
  cleanup target (make `grid_width/height` optional or substrate-adaptive).

### F6 — Coordinate frame is world-absolute; manipulation will need height + body-relative commands (kernel convention)

The kernel reasons in **absolute world coordinates**: `MiniGridSense` computes
distance by subtracting agent world-pose from target world-pose
(`sense.py:383–384`) and the spine plans navigation over those same world cells.
AI2-THOR metadata is natively world-frame too, so the adapter passes coords
through with no frame conversion — this is fine **for navigation**.

It breaks down for **manipulation**. AI2-THOR agents can have an arm; reaching for
an object needs (a) the object's **height** — now capturable via `SceneObject.z`
(F2) — and (b) plausibly **body-relative / egocentric** arm commands ("0.4m
forward, 0.3m up from the shoulder"), not world coordinates. The current
world-frame convention has no place for an egocentric command target.

- **Surfaced by:** plan 006 design discussion (sense → true-3D). Verified against
  the world-frame distance calc at `sense.py:383` and the floor-only AI2-THOR
  motor set (`MoveAhead`/`RotateLeft`/`RotateRight`) in `ai2thor_spine.py`.
- **Severity:** kernel-convention-level, but **not blocking the spike**. The
  golden path is `go_to_object` (navigation, floor-plane, world-frame — works).
  `task.pickup` is unsupported even on MiniGrid, so manipulation is out of scope
  to build here. This is filed as a forward bend, not resolved.
- **Triage for Phase 15+:** if/when manipulation lands, decide whether the kernel
  stays world-frame with the adapter/spine doing world→body-relative conversion
  internally (preferred — keeps the kernel substrate-independent, mirrors how the
  MiniGrid spine already converts world `target_location` to motor primitives), or
  whether an egocentric command frame becomes a first-class kernel concept.
- **Status:** open (forward finding; do not resolve in the boundary spike).
