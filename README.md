# JEENOM

Just-In-Time Embodied Execution Networked Operational Model

JEENOM is an experiment in substrate-independent cognition. The core idea is
simple: a substrate exposes Sense, Cortex, and Spine primitives; JEENOM turns
operator steering into claims, plans, readiness decisions, and safe execution.

The broader JEENO vision is an externalized, queryable, and auditable cognition
layer for robots. This repository is the JEENOM prototype of that idea, currently
tested in MiniGrid.

## Current Status

JEENOM is an architecture prototype in MiniGrid. The current work is **Phase
13B: partial observability, evidence gathering, and ask-for-help**. Completed
13B slices include claim freshness, native MiniGrid FOV, typed
`needs_evidence`, deterministic/LLM-path routing discipline, and continuous
interactive episode ownership. Typed Cortex-owned conditional evidence missions
are now implemented for requests such as "go straight until you see a blue
door." The next work is broader bounded `search_allowed` evidence gathering and
deterministic meta-primitives.

The project has eval-backed enforcement for the core architecture boundaries:

- canonical blocks: `OperatorStation`, `Cortex`, `ReadinessGraph`, `Sense`,
  `Spine`, `KnowledgeBase`, and `SubstrateAdapter`
- typed messages between those blocks
- one knowledge surface for claims, procedures, and provenance
- station-owned runtime context, substrate package, and context-driven
  planner/verifier semantics

Phase 10 extraction is complete for now. `OperatorStationSession` is still large,
but the blocking boundaries are explicit: runtime package injection,
operational context, substrate adapter, domain helper, turn orchestrator,
side-effect authority, command authority, and context-driven planning semantics.
The baseline now includes a live operator regression probe because structural
boundary checks alone were not enough to prove user-visible progress.
Phase 10I added operator-defined primitive assembly, so commands like defining
`convenientDistance = min(manhattan, euclidean)` become typed, validated,
reusable query primitives instead of unsupported text. Phase 11 moved inline
compound metric missions into `MissionCortex`, with typed mission plans and
mission-linked execution tickets.

Phase 11B, Phase 12/12D, and Phase 13A are complete. Operator-station
decomposition is designed and banked for Phase 16; it remains deliberately
parked while Phase 13B establishes the evidence and execution boundaries that
the later kernel must own.

The guiding split is:

- **WHY**: operator/mission steering, constraints, authority, risk, budget, and
  stopping rules.
- **WHAT**: JEENOM's architecture-level intent, evidence needs, claims, plans,
  procedures, readiness, and execution authority.
- **WHERE/MEANING**: `OperationalContext`, the typed situation frame describing
  objects, task families, references, grounding semantics, claim rules, and
  display rules for the current domain.
- **HOW**: substrate/tool bindings such as camera frames, lidar scans, MiniGrid
  observations, ARC game states, path planners, controllers, `env.step`, and
  policies.

Concrete HOW belongs behind substrate adapters.
Domain meaning belongs behind `OperationalContext`.

Implemented so far:

- Natural-language operator station for task, memory, status, concept, sequence,
  motor, and mission requests.
- Typed LLM compiler boundary: LLMs emit schema objects; runtime validates and
  executes deterministic primitives.
- The Operator Station defaults to the LLM compiler. Exact controls,
  continuations, bounded `IntentCache` patterns, and the existing exact
  command/status compatibility surface may resolve deterministically; otherwise
  unresolved semantic input uses strict JSON-schema `OperatorIntent`
  compilation. New semantic features should not be added as broad regex routes.
- The current OpenRouter transport implements this tool-call boundary with
  strict structured output (`response_format=json_schema`), not direct
  provider-side function execution. The model chooses a typed decision; JEENOM
  validates, authorizes, and executes it.
- LLM and deterministic fallback decisions converge through canonical schema
  normalization, `IntentVerifier`, capability/readiness checks, deterministic
  dispatch, and typed tickets. Fallback is logged and route-provenance tested;
  silent fallback is considered a regression.
- Capability registry and command registry over task, grounding, sensing, action,
  and claims primitives.
- Intent verification, capability matching, and arbitration to avoid silent
  capability degradation.
- Scene model, grounding claims, durable operator claims, episodic memory, and
  named concepts.
- Typed `RequestPlan` and `ReadinessGraph` for decomposing requests before
  clarification, synthesis, answer generation, memory updates, or execution.
- `CorticalEnvelope`, `ApprovedCommand`, `ExecutionTicket`, `RawMotorTicket`,
  `MemoryWriteTicket`, and `CommandResult` as the current execution-authority
  and trace objects.
- `MissionContract` can carry a validated conditional `ProcedureRecipe` that
  binds target sensing to one approved Spine action. Cortex re-senses and checks
  the stop claim before every action; no-progress and budget exhaustion terminate
  as typed failures.
- Plan reuse, environment assumptions, stale-claim detection, mismatch detection,
  and early repair-loop scaffolding.
- JIT procedure/sense/skill template caching with the invariant that rendered
  runtime execution should not make LLM calls.
- Safe synthesis scaffolding for pure grounding/query primitives, with validation
  before registration.
- Operator-defined query metric assembly for pure ranked-door grounding
  primitives, with approval, validation, registry/context update, and ticketed
  provenance recording, including inline derived metrics inside task requests.
- Explicit 5-level abstraction hierarchy: primitive, command, procedure, task,
  and goal/mission.

Current architectural debt:

- `OperatorStationSession` still owns too many internals:
  conversation flow, deterministic fast paths, LLM intent routing, readiness
  dispatch, synthesis/repair, memory writes, and domain-specific formatting.
- Phase 9E added a thin `RepresentationStore`, `ClaimRecord`, and
  `KnowledgeSnapshot`; deeper station extraction still needs to move more
  orchestration behind clean modules.
- Phase 10A moved command/result trace construction into `CommandAuthority`;
  Phase 10B moved execution, raw-motor, and memory-write ticket minting into
  `SideEffectAuthority`; Phase 10C introduced `SubstrateAdapter` and moved the
  first MiniGrid env/runtime HOW paths into `MiniGridSubstrateAdapter`; Phase
  10D added the typed `OperationalContext` and `MiniGridOperationalContext`;
  Phase 10E added a context-bound `MiniGridDomainHelper`; Phase 10F added
  `TurnOrchestrator` for top-level turn routing; Phase 10G added
  `RuntimePackage` injection; Phase 10H added `PlanningSemantics` so planner
  and verifier handles derive from `OperationalContext`.
- `CapabilityRegistry.minigrid_default()` is the only real substrate manifest.
- Deeper station branches, primitive validation fixtures, and the LLM compiler
  profile still carry MiniGrid-shaped assumptions.
- Contract preflight is represented and gated, but not yet a general executable
  proof system for arbitrary robot stacks.
- Robotics-like and ARC-style substrate pressure remain future work.

Current evals prove many control-plane invariants, including the completed Phase
11B hostile paraphrase ladder and deterministic/LLM routing parity. They still
do not prove generalized cognition, cross-substrate transfer, or adversarial
prompt safety.

## Architecture Invariants

- WHY is steered; WHAT is JEENOM architecture; HOW is substrate/tool binding.
- Sense, Spine, and Cortex are architecture-native roles. Their concrete
  sensors, actions, controllers, planners, and tool APIs are substrate bindings.
- Blocks communicate through typed messages only. A direct field or dict is not
  authority.
- Claims, provenance, procedures, and memory are the representation. They must
  be accessed through an enforced representation surface, not ad hoc station or
  memory dictionaries. Add new schema families only when an eval proves the
  current representation is lossy.
- LLM compiler outputs are typed schema objects only.
- The runtime validates and executes; unknown primitives are rejected or routed
  through explicit repair/synthesis paths.
- No LLM calls are allowed inside the rendered control loop.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane, not raw
  operator text.
- Side-effectful actions require approved tickets:
  - `ExecutionTicket` for task/runtime entry.
  - `RawMotorTicket` for explicit low-level motor action.
  - `MemoryWriteTicket` for durable operator-claim mutation.
- Grounding claims are session-scoped and track current, unverifiable, stale,
  and unknown freshness; looking away is not treated as a world mutation.
- Operator claims are durable and invalidated only by explicit operator action.
- Invalid or stale plan/claim reuse must never execute silently.
- JEENOM must distinguish known, visible, inferred, stale, unknown, searchable,
  and impossible-to-know. It should not claim global certainty from local
  visibility.
- Substrate primitives are contractual objects, not string handles. Readiness
  must check preconditions, effects, required/produced claims, frames/units,
  safety class, authority, failure modes, validation hooks, and substrate
  fingerprint assumptions.

## Known Limitations

- **Threat model: good-faith operator assumed.** JEENOM currently assumes a non-adversarial
  operator and a non-adversarial LLM backend. The architecture is *designed* to contain a
  misbehaving model for the dangerous cases — the LLM only emits typed tool-call decisions
  (its decision fields are enum-validated), unknown primitives are rejected, every side effect
  is gated by station-minted tickets plus the `IntentVerifier` and `ReadinessGraph`, and no LLM
  runs in the rendered control loop. **However, this containment is not yet hardened or proven
  against hostile prompts, prompt injection, or jailbreaks.** The previously known
  `grounding_query_plan.answer_fields` vocabulary gap is fixed by canonicalization and
  fail-closed validation, but the complete LLM-controlled dispatch vocabulary and
  side-effect-authority path have not yet received a hostile-input proof. Adversarial robustness
  and a hostile tool-call/prompt-injection eval suite are deferred to **Phase 17** of the plan.
  Do not run JEENOM for untrusted operators, or feed it untrusted text, until that work lands.
- **Single substrate.** Only MiniGrid is ORPI-conformant today; substrate-independence is
  validated later (AI2-THOR spike, Phases 14–15).

## Environment

JEENOM currently runs on MiniGrid through Gymnasium. A future robot or simulator
port should start by registering the substrate's actual primitive contracts and
bindings, then reusing the same intent, readiness, claims, and execution-control
layers.

## Running

Install the basic dependencies:

```bash
pip install gymnasium minigrid
```

Run the operator station:

```bash
python run_operator_station.py
```

The operator station defaults to `--compiler llm`. For live LLM compilation,
set an OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Without an API key, the Operator Station uses the deterministic smoke-test
compiler as a visible degraded-availability fallback. At runtime, the station
prints whether the LLM is live or whether that fallback is active.

## Verification

Run the eval suite:

```bash
python evals/eval_master.py
```

Run the Phase 9 cleanup suite:

```bash
python evals/eval_master.py --suite cleanup
```

List the manifest-selected probes without running them:

```bash
python evals/eval_master.py --list
```

Run the project-local tests:

```bash
python -m pytest -q tests
```

Avoid treating whole-repo `pytest` as the primary project signal right now,
because the local `Minigrid/` tree can introduce unrelated dependency noise.

Current green baseline:

- `python evals/eval_master.py`: 78/78 passing.
- `python evals/eval_master.py --suite orpi`: 10/10 passing.
- `python evals/eval_master.py --suite cleanup`: 30/30 passing.
- `python evals/eval_master.py --suite llm_path`: 5/5 passing.
- `python -m pytest -q tests`: 322 passed, 1 warning, 12 subtests passed.
- `python evals/eval_master.py --suite live_llm`: opt-in, 1/1 with a configured key.

Roadmap:

- Phase 13B: current. Conditional evidence missions are complete; bounded
  `search_allowed` evidence gathering and deterministic meta-primitives remain.
- Phase 13C: curriculum, knowledge reuse, and intervention metrics.
- Phase 14: cheap substrate-leak removal plus an AI2-THOR discovery spike.
- Phase 15: committed cross-substrate demonstration and ORPI v1 freeze.
- Phase 16: operational hardening and the banked station decomposition.
- Phase 17: capability stress and adversarial/hostile-input hardening.
