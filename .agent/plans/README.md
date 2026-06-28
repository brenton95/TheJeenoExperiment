# Agent Plans Index

Gitignored working plans for JEENO. **Sonnet: read this index first** to find the
right plan, then open the numbered file.

Naming: `NNN-YYYY-MM-DD-slug.md` — ordinal (reading order) · date (when written) ·
slug (what it is). Status: `draft` → `active` → `done` / `superseded`.

| # | Plan | Date | Status | What it is |
|---|------|------|--------|------------|
| 001 | [ai2thor-spike-overview](001-2026-06-15-ai2thor-spike-overview.md) | 2026-06-15 | active | Why + full sequence of the AI2-THOR substrate work (framework-generality test) |
| 002 | [ai2thor-boundary-test](002-2026-06-15-ai2thor-boundary-test.md) | 2026-06-15 | done | Mock-controller boundary test + thin adapter skeletons (11/11 green on WSL2). Findings F1/F2 filed in `orpi_spec.md` §11. |
| 003 | [ai2thor-spine-motor-dispatch](003-2026-06-16-ai2thor-spine-motor-dispatch.md) | 2026-06-16 | done | `ai2thor_spine.py` motor-dispatch slice (primitive → `controller.step`), 18/18 tests green. Findings F3/F4 filed in `orpi_spec.md` §11. |
| 004 | [ai2thor-sense-up-to-coords](004-2026-06-16-ai2thor-sense-up-to-coords.md) | 2026-06-16 | superseded by 006 | `ai2thor_sense.py` adapter-side implemented (23 pass + 1 xfail). Coord cast + coord test were pending F2 — F2 now landed upstream (`5b89f0f`); follow-through moved to plan 006. F5 filed in `orpi_spec.md` §11. |
| 005 | [ai2thor-spine-navigation](005-2026-06-17-ai2thor-spine-navigation.md) | 2026-06-17 | done | Spine navigation (BFS over `GetReachablePositions`) + success detection (resolves F3). 32/32 boundary tests green, 322/322 full suite green. F3 resolved adapter-side; F5 confirmed: nav uses reachable-set membership, not grid bounds — `(0,0)` stub is fine with no kernel edit. Live golden-path run deferred to GPU/native Linux. |
| 006 | [ai2thor-sense-true-3d-coords](006-2026-06-17-ai2thor-sense-true-3d-coords.md) | 2026-06-17 | done | Finishes sense now F2 landed: float, true-3D coords via `geometry.as_coord` + the AI2-THOR→JEENO axis cross-map; xfail removed, 24/24 green. |
| 007 | [ai2thor-spine-closed-loop-nav](007-2026-06-18-ai2thor-spine-closed-loop-nav.md) | 2026-06-18 | done | Closed-loop nav: per-action `lastActionSuccess`, single re-plan (excluding blocked cell) on blocked move, iteration cap, derived grid spacing, injected rotation step. 37/37 boundary tests, 327/327 full suite green. No progress-checker/recovery/deadband (Dev Rule 11). |
| 008 | [smoketest-compiler-apple](008-2026-06-20-smoketest-compiler-apple.md) | 2026-06-20 | done (superseded by upstream refactor) | Widened `SmokeTestCompiler` `door\|key` → `door\|key\|apple` so "go to the red apple" compiles. Landed `8cc5fc3`. **Superseded:** origin/master `33105b0` made object types parametric per-substrate (apple now lives in the AI2-THOR context, MiniGrid is door-only). The plan-008 MiniGrid apple/key compile tests were removed as obsolete. F7 (hardcoded object types) is now **resolved upstream** — do not file it. |
| 009 | [ai2thor-golden-eval](009-2026-06-21-ai2thor-golden-eval.md) | 2026-06-21 | done | End-to-end golden-path eval `eval_golden_ai2thor.py` ("go to the red apple", SmokeTest, two-tier: mock tier 2a here / `--live` tier 2b on Colab). Required domain helper surface (4 methods) + manifest field fixes (F11, F12 filed in `orpi_spec.md`). Tier 2a green, 78/78 evals, 377/379 tests (2 pre-existing partial-obs failures). |
| 010 | [ai2thor-episode-runner](010-2026-06-22-ai2thor-episode-runner.md) | 2026-06-22 | ready for implementation | Makes `run_task_episode` real: AI2-THOR-local episode loop threading Cortex + Ai2thorSense + Ai2thorSpine (Option A — mirror MiniGrid's loop, do NOT unify `run_demo.run_episode` = Phase 15 kernel work) + `parse_go_to_object_utterance` on the domain helper. Adapter-side, **no kernel edits**. Files F8 (shared-loop unification is a Phase 15 kernel bend). Unblocks 009. → Opus plans / Sonnet implements / Opus reviews diff. |
| 011 | [ai2thor-nearest-farthest](011-2026-06-25-ai2thor-nearest-farthest.md) | 2026-06-25 (rev 06-27) | BLOCKED on prereq 011a (scope RESOLVED) | Multi-candidate ranking + intent inversion: "nearest apple" / "farthest apple" over a SHARED two-apple scene. **DoD = specific-target assertion** (nearest→A, farthest→B), NOT `task_complete` (vacuous under F9). **✓ Scope fork RESOLVED 2026-06-27 by running:** "nearest apple" never reaches the ranked path — AI2-THOR helper `parse_exact_go_to_object_utterance` mis-parses the superlative as a color (`operator_station.py:302`), short-circuiting to plain go-to → nearest/farthest collapse. Routing is shared/correct → **adapter-only fix, NOT kernel STOP.** Prereq **011a (adapter, Sonnet-able + Opus review):** (a) reject superlatives in `parse_exact_go_to_object_utterance`; (b) add 4 missing `ai2thor_domain_helper` methods (`color_reference_in_utterance`, `task_utterance_for_entry`, `entry_target_dict`, `entry_label`); (c) register a **euclidean** ranked grounding primitive (registry has zero `grounding.*`; compiler emits `manhattan` — wrong metric). Build 011a → then this eval. |

## Conventions
- Plans live here, gitignored (`.agent/`), never enter a PR.
- The repo's authoritative phase plan is `PlanOfAction/task_plan.md` — these are
  *implementation breakdowns* of it, not a replacement.
- Working agreement (how we collaborate) is tracked in `AGENTS.md`.
