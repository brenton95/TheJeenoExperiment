# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install gymnasium minigrid

# Set LLM API key (required for live LLM paths; evals fall back to smoke-test compiler without it)
export OPENROUTER_API_KEY="your-api-key-here"

# Run the interactive operator station
python run_operator_station.py

# Run the full eval suite (passes --allow-fallback automatically when applicable)
python evals/eval_master.py

# Run a single eval probe
python evals/eval_golden.py
python evals/capability_matcher_probe.py
python evals/intent_verifier_probe.py

# Run unit tests
python -m pytest -q tests/

# Run a single test class or filter
python -m pytest -q tests/test_jeenom_minigrid.py -k "TestStationActiveClaims"
python -m pytest -q tests/test_jeenom_schemas.py tests/test_jeenom_minigrid.py
```

## Golden Path (must never break)

```
instruction: "go to the red door"
compiler: llm
render/prewarm enabled
expected:
  task_complete=True
  runtime_llm_calls_during_render=0
  cache_miss_during_render=0
  final skill_plan=['done']
```

Run `python evals/eval_golden.py` to verify. If a change breaks this, fix it before adding anything else.

## Architecture

JEENO translates natural-language operator instructions into validated, executable intents — without making any LLM calls during the rendered control loop. All LLM calls happen at compile/prewarm time; the render loop uses only cached deterministic primitives.

### 5-Level Abstraction Hierarchy

| Level | Name      | Motor                           | Sensory                        | Key schema                     |
|-------|-----------|---------------------------------|--------------------------------|--------------------------------|
| L0    | primitive | `move_forward`, `turn_right`, … | `parse_grid_objects`, …        | `ACTION_PRIMITIVES` dict       |
| L1    | command   | `navigate_to_object`            | `locate_object`                | `MotorCommandTemplate` / `SensoryCommandTemplate` |
| L2    | procedure | `go_to_object` step sequence    | sense procedure                | `ProcedureContract`            |
| L3    | task      | `go_to_object` with params      | task with grounding applied    | `TaskContract`                 |
| L4    | goal      | multi-task mission              | abort-on-failure               | `MissionContract`              |

### Module Responsibilities

- **`operator_station.py`** — `OperatorStationSession` is the top-level session owner. It drives the READY prompt, owns `OperationalMemory`, `PlanCache`, `StationActiveClaims`, `PendingClarification`, and orchestrates every other component. All routing decisions (clarify, execute, refuse, synthesize) happen here.
- **`schemas.py`** — All typed data contracts. Single source of truth for every schema object (`OperatorIntent`, `TargetSelector`, `GroundingQueryPlan`, `SceneModel`, `StationActiveClaims`, `ArbitrationDecision`, etc.). All `from_dict()` methods validate strictly; `SchemaValidationError` is the failure mode.
- **`llm_compiler.py`** — `CompilerBackend` ABC with `SmokeTestCompiler` (deterministic, no API key) and `LLMCompiler` (OpenRouter). Compiles operator utterances → typed `OperatorIntent`, and task requests → `ProcedureRecipe` / `SensePlanTemplate` / `SkillPlanTemplate`. LLM output is always validated against schemas before use.
- **`primitive_library.py`** — Source of truth for runtime primitives (`TASK_PRIMITIVES`, `SENSING_PRIMITIVES`, `ACTION_PRIMITIVES`, `GROUNDING_PRIMITIVES`). `CapabilityRegistry` is derived from this; they must not drift.
- **`capability_registry.py`** — Typed index over `primitive_library.py`. Exposes `lookup(handle)` for exact handle matching. Never subsumes or weakens: `grounding.closest_door` does NOT satisfy `grounding.ranked_doors`.
- **`capability_matcher.py`** — `CapabilityMatcher.match(intent, registry)` deterministically checks `required_capabilities` against the registry. Overrides the LLM's `capability_status` field — the matcher always wins.
- **`intent_verifier.py`** — Sits between the LLM compiler output and `CapabilityMatcher`. Proactively extracts semantic signals (SUPERLATIVE, CARDINALITY, ORDINAL) from the utterance text and injects the correct `required_capabilities`. Pure class, no substrate imports. This is the hard stop for intent inversion and silent degradation.
- **`capability_arbitrator.py`** — When `CapabilityMatcher` detects a gap, `CapabilityArbitrator.arbitrate()` decides: refuse / clarify / substitute / synthesize. `SmokeTestArbitrator` is rule-based; `LLMArbitrator` calls the LLM. Hard rule: `refuse` and `synthesize` decisions are never `safe_to_execute=True`.
- **`cortex.py`** — Task execution brain. Runs the sense→decide→act loop. Owns per-task `ObservationClaim` store (separate from `StationActiveClaims`). Issues `MotorSkillRequest` to `MiniGridSpine` and `EvidenceFrame` requests to `MiniGridSense`.
- **`sense.py`** — `MiniGridSense` executes sensory primitives against the live adapter. Projects `WorldModelSample` → `SceneModel` after every tick. `sense_idle_scene()` populates `SceneModel` between tasks.
- **`spine.py`** — `MiniGridSpine` executes motor primitives (A\* navigation, turn, move).
- **`plan_cache.py`** — Caches `ProcedureRecipe`, `SensePlanTemplate`, `SkillPlanTemplate`. Prewarmed before render; zero cache misses during the rendered control loop is a hard invariant.
- **`memory.py`** — `OperationalMemory` owns episodic memory (last target, last task) and durable knowledge (`delivery_target`). Persisted to disk. Distinct from `StationActiveClaims` (session-scoped, not durable).
- **`knowledge_base.py`** — `KnowledgeBase` + `NamedConcept` store operator-asserted durable claims. Concept teaching (`concept_teach`) and recall (`concept_recall`) live here.
- **`request_planner.py`** / **`readiness_graph.py`** — `build_request_plan()` decomposes operator requests into dependency-aware `RequestPlan` steps. `evaluate_request_plan()` returns a `ReadinessGraph` with per-step verdicts. The `ReadinessGraph` arbitrates the plan, not the raw utterance.
- **`primitive_synthesizer.py`** — On-demand synthesis of missing pure grounding primitives. Generated code is restricted to `math` and `typing` imports only. `SmokeTestSynthesizer` always refuses; `LLMSynthesizer` calls OpenRouter.
- **`primitive_validator.py`** — Validates synthesized primitives against deterministic `ValidationFixture` scenes before registration. Includes a critical `distance_is_euclidean_not_manhattan` fixture to catch Manhattan masquerading as Euclidean.
- **`minigrid_adapter.py`** / **`minigrid_envs.py`** — MiniGrid-specific gym wrapper. All substrate-specific code is isolated here.

### Claims Architecture

All facts the station holds are Claims, differing in scope and invalidation policy:
- **`StationActiveClaims`** — grounding results (ranked doors, last grounded target). Session-scoped, fingerprinted to `(agent_x, agent_y, step_count)`. Cleared on reset and at task start. Never written to durable memory.
- **`KnowledgeBase` / `OperationalMemory.knowledge`** — operator-asserted durable claims. Persist across session restarts. Invalidated only by explicit operator retraction.
- **`Cortex.claims`** (`ObservationClaim`)  — per-task sensory evidence. Separate from inter-turn claims; populated during task execution only.

### Intent Processing Pipeline (per operator turn)

```
Utterance
  → SmokeTestCompiler fast path (trivial commands only)
  → LLMCompiler.compile_operator_intent() → OperatorIntent (validated)
  → IntentVerifier.enrich() — injects missing required_capabilities from utterance signals
  → CapabilityMatcher.match() — overrides LLM's capability_status
  → if gap: CapabilityArbitrator.arbitrate() → refuse/clarify/synthesize
  → if executable: RequestPlan + ReadinessGraph → execution
  → Cortex.run() → MiniGridSense / MiniGridSpine (no LLM calls here)
```

## Development Rules (from `.codex` / `blueprint.md`)

1. **No LLM calls in the rendered control loop.** Compile and prewarm before render. The runtime loop uses cached templates only.
2. **LLM compiler outputs are schema objects only.** The runtime validates and executes. Unknown primitives must be rejected or corrected.
3. **`primitive_library.py` is the source of truth** for implemented runtime primitives. `CapabilityRegistry` is derived from it — never create an independent manifest that can drift.
4. **No capability weakening.** `grounding.closest_door` does NOT satisfy `grounding.ranked_doors`. `CapabilityMatcher.lookup()` is exact — no subsumption, no fuzzy matching.
5. **Intent inversion is a hard stop.** "farthest" must not silently navigate to "closest". `IntentVerifier` is the enforcement layer.
6. **`refuse` and `synthesize` arbitration decisions must always have `safe_to_execute=False`.**
7. **CapabilityMatcher, IntentVerifier, and CapabilityArbitrator must have no substrate imports** (no `minigrid`, `gymnasium`, `sense`, `spine`). This is AST-verified by their respective probes.
8. **Synthesized primitives are validated before registration.** A validation failure returns an honest operator message; nothing is registered or executed.
9. Work capability by capability. For every implementation task: state the phase, state the capability, state files touched, state success criteria, add or update a regression test.
10. If a change breaks the golden path, stop and fix that before adding new features.

## Current Phase

**Phase 8 — General Object Handling.** Expanding beyond door navigation to general objects. Prior phases 0–7.95 are complete; their regression probes live in `evals/`.

### Phase 8 progress

| Stage | Status | Notes |
|-------|--------|-------|
| 8.1 Environment identity + stale claim safety | done | `phase8_environment_change_stale_claim_probe.py` |
| 8.2 Explicit environment assumptions | done | `phase8_environment_assumption_probe.py` |
| 8.3 Conservative RequestPlan reuse | done | `phase8_plan_reuse_probe.py` |
| 8.3.5 Named Concept Knowledge Base | done | `phase835_knowledge_base_probe.py` |
| 8.4 Mismatch detection | done | `phase84_mismatch_detection_probe.py` |
| 8.4.5–8.5 Sequential/motor intents, mission contract, typed claims, command registry | done | multiple probes |
| **Key object type** | **done** | `phase8_key_object_probe.py` — 8/8 PASS |
| `task.go_to_object.key` execution | **done** | Agent navigates to key; `task.pickup.key` stays `unsupported` |
| Phase 8.5 — Parametric object type refactor | planned | See `TODOS.md`; door/key branches should be driven by config |
| Phase 9 — Repair loop | planned | `jeenom/repair_loop.py` not yet started |

### Eval baseline (post go_to_object.key)
- `eval_master.py`: **32/33 passing** (phase91 pre-existing failure, repair loop not yet implemented)
- `pytest tests/`: **153 passing**, 7 pre-existing failures unrelated to object types

### Known follow-ups (see `TODOS.md`)
- Parametric object type handling — remove explicit `door`/`key` branches across registry, operator_station, sense
- Generalise `SmokeTestCompiler` door/key regex to build dynamically from `OPERATOR_OBJECT_TYPES`
