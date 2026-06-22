# Plan 010 — AI2-THOR episode runner: make `run_task_episode` real

**Status:** ready for implementation · **Planner:** Opus · **Implementer:** Sonnet
**Date:** 2026-06-22
**Prereq reading:** `AGENTS.md` (§1.1 voice, §2.2 simplicity, Dev Rule 11
substrate-honest), `CLAUDE.md` ("Current Phase", Dev Rules 1/7/11), plans
[005](005-...), [006](006-...), [007](007-...), this file.
**Blocks:** plan [009](009-...) (the golden eval) — 009 cannot be written until
this lands. 009 is correctly demoted to "blocked by 010".
**Why this plan exists:** Sonnet hit two real blockers writing 009 (both verified
on the merged tree, 2026-06-22):
1. `Ai2thorDomainHelper` has no `parse_go_to_object_utterance` — the kernel calls
   it at `operator_station.py:5181, 5284` → `AttributeError`.
2. `Ai2thorSubstrateAdapter.run_task_episode` (`ai2thor_substrate_adapter.py:127`)
   is a spike stub returning `{success, task_complete, error}` with **no
   `final_state` key**. The kernel reads `last_result["final_state"]
   ["task_complete"]` (`operator_station.py:5305, 5308`) → `KeyError`.
The 009 reconciliation pass verified the `build_ai2thor_runtime_package`
*signature* but never read the `run_task_episode` *body*. That was the miss; 009's
"seam exists, just consume it, zero edits" premise was false. This plan fixes it.

## Goal

Make the AI2-THOR substrate run one task episode end-to-end, adapter-side, with
**zero kernel edits**:

```
instruction: "go to the red apple"
→ task_complete=True, runtime_llm_calls_during_render=0
```

Two artifacts:
- **A1:** `Ai2thorDomainHelper.parse_go_to_object_utterance(utterance)` — small
  parser, modeled on `minigrid_domain_helper.py:115`.
- **A2:** `Ai2thorSubstrateAdapter.run_task_episode(...)` — the episode loop that
  threads `Cortex` + `Ai2thorSense` + `Ai2thorSpine`, modeled structurally on
  `run_demo.run_episode` (`run_demo.py:412`).

## The design decision (already made — Option A, do NOT revisit)

We mirror MiniGrid's **loop structure** in an AI2-THOR-local runner; we do NOT
generalise `run_demo.run_episode` into a shared substrate-parametric loop.
- **Why A, not B:** `run_demo.run_episode` is kernel-adjacent (MiniGrid's adapter
  calls into it; 372 tests depend on it). Unifying both substrates onto one loop
  is a *kernel edit* — forbidden on this spike branch, and it's Phase 15's job
  (committed port + ORPI v1 freeze), a decision owned by the kernel owner.
- **The spike's deliverable is proof + findings, not the clean merged port.**
  Structural duplication of the loop here is the *correct* cost: every place the
  AI2-THOR loop must diverge from MiniGrid's is a requirements-discovery signal to
  FILE in `orpi_spec.md`, not to paper over.
- **FILE, don't fix:** "MiniGrid's `run_episode` is the natural shared loop but is
  kernel-adjacent; unifying both substrates (Option B) is a Phase 15 kernel edit"
  → add as a finding (F8) in `orpi_spec.md` §11 when A2 lands. That's the bend
  this plan surfaces.

## Cortex ports unchanged — verified

`cortex.py` has **no** `import minigrid` (only one cosmetic log string mentioning
"MiniGrid mission reward"). Cortex is substrate-agnostic: it consumes
`OperationalEvidence` and issues skill requests against whatever sense/spine it is
handed. So A2 constructs `Cortex(memory, compiler, plan_cache=...)` exactly as
`run_demo.py:442` does, and threads it to the AI2-THOR sense/spine. **If Sonnet
finds cortex reaching into a MiniGrid-only attribute, STOP and escalate — that is
a kernel coupling and becomes a filed bend, not an adapter edit.**

## A1 — `parse_go_to_object_utterance`

- Location: `jeenom/ai2thor_domain_helper.py`. Adapter-side. ~15–25 lines.
- Template: `minigrid_domain_helper.py:115`. Read it first; match its return shape
  exactly (`dict[str, str] | None` with the same keys the kernel expects —
  `object_type`, `color`).
- Drive object types from the helper's existing `self.object_types()` /
  `self.default_object_type()` (parametric — do NOT hardcode "apple"). This is the
  post-refactor convention: vocabulary is per-substrate from context.
- Colour handling: the AI2-THOR context vocabulary is minimal (apple). Keep colour
  parsing as permissive as MiniGrid's but do not invent colours the context
  doesn't declare. If colour normalisation diverges from MiniGrid, that's fine —
  match what `ai2thor_operational_context.py` actually declares.
- Return `None` for utterances that are not go-to-object (the kernel branches on
  `is not None`).

## A2 — `run_task_episode`

- Location: `jeenom/ai2thor_substrate_adapter.py` (replace the stub at `:127`).
- **Signature must match the protocol** (`substrate_adapter.py:67`) and the call
  site (`operator_station.py:5292`): keyword args `instruction, compiler_name,
  compiler, seed, max_loops, memory, plan_cache, progress_callback, task_override,
  procedure_override, step_budget`. Read the MiniGrid impl
  (`minigrid_substrate_adapter.py:138`) for the exact set.
- **Return contract (non-negotiable):** the dict MUST contain `final_state` with at
  least `task_complete: bool` (the kernel reads `last_result["final_state"]
  ["task_complete"]` at `operator_station.py:5308`, and `.get("final_state", {})
  .get("budget_exhausted")` at `:5305`). Mirror the *key shape* MiniGrid's
  `run_episode` returns — read its return statement and replicate the keys 009 and
  the kernel consume (`final_state`, `runtime_llm_calls_during_render`,
  `cache_miss_during_render`, success/status). Do NOT invent a new result schema.

### The loop (structure to mirror from `run_demo.run_episode`)
Thread these, in order — the *substrate-independent* spine of the loop:
1. Resolve compiler/memory/plan_cache (passed in; default like `run_demo.py:438`).
2. Construct `cortex = Cortex(...)`, `sense = Ai2thorSense(...)`,
   `spine = Ai2thorSpine(controller=self._controller, ...)`. Use the adapter's
   already-injected controller — do NOT construct a controller here (it's injected
   per the spike's testability rule; the mock controller in
   `tests/test_ai2thor_substrate_boundary.py` must drive this).
3. Compile/lookup task → procedure (honour `task_override`/`procedure_override`
   like `run_demo.py:468–508`; prewarm so `runtime_llm_calls_during_render=0`).
4. `cortex.onboard_task(task, procedure)`; store procedure to cache on executable.
5. Sense→decide→act loop up to `max_loops`/`step_budget`: `sense.tick(...)` →
   project to cortex → cortex decides skill → `spine.tick(...)` executes →
   success via the spine's existing `_postcondition_check` (plan 007 — proximity,
   NO reward signal; F3). Repeat until success/abort/budget.
6. `cortex.finalize()`; assemble the `final_state` result dict.

### Substrate-honest — what NOT to port (Dev Rule 11)
The MiniGrid runner carries machinery AI2-THOR does NOT have; do NOT copy it:
- **`_probe_requested_target(env_id=...)`** — MiniGrid grid-probe keyed to
  `env_id`. AI2-THOR has no `env_id` grid; target presence comes from
  `event.metadata["objects"]` via `Ai2thorSense`. If a target-absent preflight is
  wanted, derive it from sensed metadata, not an env probe.
- **`observability` partial/full split + `build_full_env`/`build_env`** — MiniGrid
  occupancy concept. AI2-THOR uses `GetReachablePositions` (plan 005). Skip.
- **`render_adapter` / `keep_render_open` / window handoff** — JEENOM never needs
  the RGB frame (`renderImage=False`); the adapter reads metadata only. Skip.
- **Progress-checker / partial-travel guards** — `MoveAhead` is all-or-nothing
  (Dev Rule 11 example). The spine already handles blocked cells (plan 007). Don't
  add drift guards the substrate can't produce.
Operational test for each line you're tempted to port: *can AI2-THOR produce the
failure this guards?* If no, don't port it — and if MiniGrid has it but AI2-THOR
genuinely needs a different mechanism, that divergence is a finding to FILE.

## Tests
- Extend `tests/test_ai2thor_substrate_boundary.py` (currently 37 green) with a
  case that drives `run_task_episode` through the **existing `_NavMockController`**
  seeded with a reachable grid + an "apple" in metadata a few cells from the agent,
  and asserts `result["final_state"]["task_complete"] is True` and
  `runtime_llm_calls_during_render == 0`. This is the de-risking artifact 009's
  tier-2a will then build on. Do NOT invent a new test harness — reuse the mock.
- Add a focused unit test for `parse_go_to_object_utterance` (apple parses;
  non-go-to returns `None`).

## Verify (definition of done)
```bash
python -m pytest -q tests/test_ai2thor_substrate_boundary.py   # was 37; now +cases, all green
python -m pytest -q tests/                                      # no regressions (2 pre-existing
                                                               # partial-observability failures on
                                                               # origin/master are NOT ours)
```
Live Unity validation is NOT part of this plan's DoD — that's 009 tier-2b on Colab.

## Out of scope (do NOT do)
- Any kernel edit (`operator_station.py`, `cortex.py`, `schemas.py`,
  `substrate_adapter.py` protocol). Consume the contract; don't change it. If you
  can't satisfy the kernel without editing it → STOP, that's a filed bend.
- Option B (generalising `run_demo.run_episode`). Phase 15, kernel-owner's call.
- The eval file itself — that's plan 009, which unblocks after this.
- `LLMCompiler` path. Spike uses SmokeTest.
- Fixing the 2 pre-existing `test_partial_observability_map_persistence` failures —
  they fail on clean `origin/master`; flag to the repo owner, don't touch.

## Per AGENTS.md
- A1 + A2 are adapter-layer, NOT kernel — Sonnet may implement, Opus reviews the
  diff before it lands (it defines the AI2-THOR episode contract).
- File F8 (Option B / shared-loop unification is a Phase 15 kernel edit) in
  `orpi_spec.md` §11 as part of this work — surfacing the bend IS the spike's job.
- PEP 8, type hints, `from __future__ import annotations`.
