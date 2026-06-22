# Plan 009 — `eval_golden_ai2thor.py`: live AI2-THOR golden path

**Status:** BLOCKED by plan [010](010-...) · **Planner:** Opus · **Implementer:** Sonnet
**Date:** 2026-06-21 · **Revised:** 2026-06-22 (reconciled to origin/master
parametric refactor `33105b0`; demoted — see block note)

> **2026-06-22 BLOCK.** Sonnet found this plan's premise false: it claimed "the
> seam exists, consume it, zero edits", but `Ai2thorSubstrateAdapter.run_task_episode`
> is a stub (no `final_state` → KeyError in the kernel) and the AI2-THOR domain
> helper lacks `parse_go_to_object_utterance` (→ AttributeError). The episode
> runner was never wired. Plan **010** builds it (adapter-side, no kernel edits).
> This eval is unblocked only after 010 lands. The reconciliation note below is
> still valid; the "architecture is already proven — consume the seam" section was
> over-stated and is superseded by 010.
**Prereq reading:** `AGENTS.md` (§2.2 simplicity, §2.4 golden path = done),
`CLAUDE.md` ("Golden path target for this spike", "Testing without a live Unity
process", Dev Rule 11), plans [007](007-...) + [008](008-...), this file.
**Depends on:** 008 (landed — apple compiles) **and** the merge of origin/master
`33105b0` into this branch (landed — the kernel is now parametric over object
type; "door" is no longer hardcoded). This is the LAST artifact before the live
Colab run.

> **2026-06-22 reconciliation note.** Three things in the original plan are now
> stale after the friend's parametric refactor landed:
> 1. **F7 is dead.** "Object types hardcoded" was fixed upstream — do NOT file or
>    reference F7. The kernel reads object vocabulary per-substrate from context.
> 2. **Line numbers shifted** (see updated "injection seam" section below).
> 3. **Apple match is exact-equality** (verified `ai2thor_sense.py:89/103`) — the
>    AI2-THOR context's `object_type` must equal the lowercased `objectType`.
> The two-tier design, "no kernel edits", golden utterance, and Colab bootstrap
> are all unchanged and still correct.

## Goal

Write `evals/eval_golden_ai2thor.py` — the end-to-end golden-path eval for the
AI2-THOR substrate:

```
instruction: "go to the red apple"
compiler:    SmokeTestCompiler (no API key)
expected:    task_complete=True, runtime_llm_calls_during_render=0
```

This is the spike's definition of done (CLAUDE.md). It is the **first** artifact
that drives the full kernel pipeline (compile → match → plan → cortex → spine)
against a **live** AI2-THOR controller end-to-end.

## The architecture is already proven — DO NOT change the kernel

`OperatorStationSession.__init__` takes `runtime_package: RuntimePackage | None`
(`operator_station.py:353`, post-merge) and only defaults to MiniGrid when none is
passed (`:368`). Pass `build_ai2thor_runtime_package(controller=...)` and the
*entire* pipeline runs on AI2-THOR with **zero kernel edits**. The injection seam is
`build_ai2thor_runtime_package(*, controller=None, scene_id="FloorPlan1")`
(`ai2thor_substrate_adapter.py:151`, post-merge — note both args are keyword-only
and `scene_id` already defaults to `"FloorPlan1"`). **This is the whole point of
the spike — the eval consumes the seam, it does not add new wiring.** If you find
yourself editing `operator_station.py` or any kernel file, STOP and escalate.
(Line numbers verified against the merged tree on 2026-06-22; if they have drifted
again, grep for `runtime_package` / `build_ai2thor_runtime_package` rather than
trusting the digits.)

## Two-tier design (REQUIRED — this is what makes the async run-loop work)

The eval CANNOT run here (WSL2 Unity socket broken); it runs on Colab. The user
pulls, runs, and pastes results back. For that loop to be debuggable, a Colab
failure must be *unambiguously* environment/substrate — never "our harness is
wrong." So the eval has two tiers sharing one assertion core:

### Tier 1 — `run_golden(controller) -> dict` (the shared core)
- Pure function: takes an *injected* controller, builds the AI2-THOR runtime
  package, constructs `OperatorStationSession(runtime_package=...)`, runs
  `session.handle_utterance("go to the red apple")`, returns a result dict
  (`task_complete`, `runtime_llm_calls_during_render`, `cache_miss_during_render`,
  final status, reason-if-failed). NO controller construction inside.

### Tier 2a — local mock validation (`main()` default, RUNS HERE)
- Build the existing nav mock from
  `tests/test_ai2thor_substrate_boundary.py` (`_NavMockController`) — or a thin
  eval-local equivalent — seeded with a reachable grid + an "apple" object in
  metadata positioned a few cells from the agent.
- Call `run_golden(mock)`; assert `task_complete=True`,
  `runtime_llm_calls_during_render=0`. **This must pass locally** (`python
  evals/eval_golden_ai2thor.py`). It proves the wiring + assertions are correct
  BEFORE Colab. This is the de-risking tier.

### Tier 2b — live controller (`--live` flag, RUNS ON COLAB)
- When `--live` is passed: construct a real
  `ai2thor.controller.Controller(...)` (CloudRendering platform, `renderImage=
  False`, a scene that contains an apple — see "Scene" below), then call the SAME
  `run_golden(controller)`. Identical assertions.
- Because tier 2a already proved the harness, any tier-2b failure is necessarily
  Unity/scene/headless-setup — which is exactly the diagnosable signal we want
  from a pasted Colab log.

## Scene + object (VERIFY against AI2-THOR docs before hardcoding)
- The golden utterance needs a scene that actually contains an apple. iTHOR
  kitchen FloorPlans (e.g. `FloorPlan1`) contain an `Apple` object —
  **CONFIRM the exact scene_id and that `objectType == "Apple"`** against the
  AI2-THOR object/scene docs or `controller.last_event.metadata["objects"]` on
  Colab. Do not assume the FloorPlan number from memory.
- Note the case: AI2-THOR `objectType` is `"Apple"` (capitalized);
  `ai2thor_sense.py:89` lowercases via `obj.get("objectType", "").lower()` and
  `:103` matches by **exact equality** (`obj_type == target_object_type`) against
  the JEENO `object_type="apple"`. (Verified post-merge — the parametric refactor
  did not change this path.) Because it is exact equality, the AI2-THOR context's
  `object_type` must be exactly `"apple"`. Verify the lowercasing path fires so
  "Apple" metadata matches. (This is the most likely substrate bend the eval will
  surface — if it mismatches, FILE it in `orpi_spec.md`, don't hack the kernel.)

## Colab bootstrap (in the eval's docstring / a sibling `.agent/` cell — NOT repo scaffolding)
Per `feedback_follow_repo_conventions`, no environment-specific install scaffolding
in the committed repo. Put the bootstrap as a documented block in the eval's module
docstring so "git pull and run" is self-explanatory:
```
# On Colab (one-time):
!pip install ai2thor
!python -c "from ai2thor.controller import Controller; Controller(platform='CloudRendering')"  # warms headless build
# Then:
!python evals/eval_golden_ai2thor.py --live
```
The exact headless incantation (CloudRendering vs. Xvfb/`startx`) is **Colab-live
debugging**, done when we get there — do not over-specify it in the plan.

## Tests
- The local tier (2a) IS the test — it runs in the eval itself and must be green
  here. Additionally add a lightweight `tests/` case if the existing eval-probe
  convention has one (check how `regression_golden_probe.py` is or isn't tested);
  if probes aren't unit-tested, the self-running tier-2a assertion is sufficient
  and matches repo convention. **Do not invent a new test harness.**

## Verify (definition of done)
```bash
python evals/eval_golden_ai2thor.py        # tier 2a (mock) GREEN here, no Unity
python -m pytest -q tests/                  # no regressions
python evals/eval_master.py                 # AI2-THOR eval registered if the suite expects it; no regressions
```
Live tier (2b) is verified on Colab by the user and is NOT part of local
definition-of-done. Then flip 009 → `done` in `.agent/plans/README.md`.

## Out of scope (do NOT do)
- Any kernel edit (`operator_station.py`, `schemas.py`, `cortex.py`). The seam
  exists; consume it.
- The closed-loop blocked-move path (plan 007). A clean golden path won't trigger
  a collision; exercising 007 live is a SEPARATE future scenario, not this eval.
- `LLMCompiler` (spike uses SmokeTest — no API key path needed for this eval).
- Real headless-rendering setup code in the repo (Colab-live, gitignored notes
  only).
- Fixing any substrate bend the live run surides — FILE it in `orpi_spec.md`
  (that's the spike's job), bring it back to an Opus session, don't hack around it.

## Per AGENTS.md
- Eval file is adapter/test-tier, NOT kernel — Sonnet may implement, Opus reviews
  diff before it lands (touches the golden-path definition).
- Two-tier or it's not done: the local mock tier must be green here.
- PEP 8, type hints, `from __future__ import annotations`.
