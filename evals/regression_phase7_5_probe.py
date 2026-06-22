"""Consolidated Phase 7.5 eval — Scene model, active claims, capability registry.

Covers:
- Phase 7.55: CapabilityRegistry manifest, readiness for selector/task, help query.
- Phase 7.57: Persistent SceneModel after task/idle sense, grounding from SceneModel.
- Phase 7.58: ActiveClaims lifecycle, staleness, fingerprinting, compact summary.

Migrated from: capability_registry_probe.py, scene_model_probe.py, active_claims_probe.py.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any
from unittest.mock import patch
from harness import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.minigrid_adapter import MiniGridAdapter
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_library import (
    ACTION_PRIMITIVES,
    GROUNDING_PRIMITIVES,
    SENSING_PRIMITIVES,
    TASK_PRIMITIVES,
)
from jeenom.schemas import GroundedObjectEntry, SceneModel, StationActiveClaims






def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


def _names(prefix: str, source: dict[str, Any]) -> set[str]:
    return {f"{prefix}.{name}" for name in source}


# ── Phase 7.55: Capability Registry ──────────────────────────────────────────

def _check_registry(checks: dict[str, bool]) -> None:
    registry = CapabilityRegistry.minigrid_default()
    summary = registry.compact_summary()
    registry_names = set(registry.primitive_names())

    task_names = _names("task", TASK_PRIMITIVES)
    sensing_names = _names("sensing", SENSING_PRIMITIVES)
    action_names = _names("action", ACTION_PRIMITIVES)
    grounding_names = _names("grounding", GROUNDING_PRIMITIVES)

    help_session = OperatorStationSession(
        compiler=LLMCompiler(api_key="test-key"),
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    help_response = help_session.handle_utterance("what can you do")

    plan_grid_path = registry.primitive("action.plan_grid_path")
    euclidean = registry.readiness_for_selector(
        {
            "object_type": "door",
            "color": None,
            "exclude_color": None,
            "relation": "closest",
            "distance_metric": "euclidean",
            "distance_reference": "agent",
        }
    )
    pickup = registry.readiness_for_task(task_type="pickup", object_type="key")

    checks["registry_name"] = summary["name"] == "minigrid_primitive_registry_v1"
    checks["lists_all_task_primitives"] = task_names.issubset(registry_names)
    checks["lists_all_sensing_primitives"] = sensing_names.issubset(registry_names)
    checks["lists_all_action_primitives"] = action_names.issubset(registry_names)
    checks["lists_grounding_primitives"] = grounding_names.issubset(registry_names)
    checks["compact_summary_has_all_layers"] = {"task", "grounding", "sensing", "action"}.issubset(
        set(summary["primitives"])
    )
    checks["exposes_consumes_produces"] = bool(plan_grid_path) \
        and plan_grid_path.inputs == ["agent_pose", "target_location", "occupancy_grid"] \
        and plan_grid_path.outputs == ["planned_action_names", "path"]
    checks["exposes_runtime_binding"] = bool(plan_grid_path) \
        and plan_grid_path.runtime_binding == {"kind": "python", "value": "plan_grid_path"}
    checks["euclidean_is_synthesizable"] = euclidean["status"] == "synthesizable_missing_primitive"
    checks["pickup_key_is_unsupported_task"] = pickup["status"] == "unsupported" \
        and pickup["primitive"] == "task.pickup.key"
    checks["help_query_uses_registry"] = help_response.startswith("CAPABILITIES") \
        and "task.go_to_object.door" in help_response \
        and "sensing.parse_grid_objects" in help_response \
        and "action.plan_grid_path" in help_response \
        and "grounding.closest_door.euclidean.agent" in help_response


# ── Phase 7.57: Scene Model ──────────────────────────────────────────────────

def _check_scene_model(checks: dict[str, bool]) -> None:
    # 1. SceneModel populated after task
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()
        session.handle_utterance("go to the red door")
    scene_after_task: SceneModel | None = session.memory.scene_model
    checks["scene_model_populated_after_task"] = scene_after_task is not None
    checks["scene_model_source_task_sense"] = (
        scene_after_task is not None and scene_after_task.source == "task_sense"
    )
    checks["scene_model_has_agent_pose"] = scene_after_task is not None and (
        isinstance(scene_after_task.agent_x, int)
        and isinstance(scene_after_task.agent_y, int)
    )
    checks["scene_model_has_door_objects"] = scene_after_task is not None and len(
        scene_after_task.find(object_type="door")
    ) > 0

    # 2. Idle sense before any task
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session2 = _make_session()
        session2._ensure_scene_model()
    scene_idle = session2.memory.scene_model
    checks["idle_sense_populates_scene_model"] = scene_idle is not None
    checks["idle_sense_source_is_idle_sense"] = (
        scene_idle is not None and scene_idle.source == "idle_sense"
    )

    # 3. Grounding uses SceneModel — no adapter.reset() call
    reset_calls: list[Any] = []
    original_reset = MiniGridAdapter.reset

    def tracking_reset(self_adapter, seed=None):
        reset_calls.append(seed)
        return original_reset(self_adapter, seed=seed)

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session3 = _make_session()
        session3.handle_utterance("go to the red door")
        reset_calls.clear()
        with patch.object(MiniGridAdapter, "reset", tracking_reset):
            grounded_closest = session3.ground_target_selector({
                "object_type": "door",
                "color": None,
                "exclude_color": None,
                "relation": "closest",
                "distance_metric": "manhattan",
                "distance_reference": "agent",
            })

    checks["closest_grounding_ok"] = grounded_closest.get("ok", False)
    checks["closest_grounding_returns_distance"] = grounded_closest.get("distance") is not None
    checks["grounding_did_not_call_adapter_reset"] = reset_calls == []

    # 4. Unique door grounding returns distance
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session4 = _make_session()
        session4.handle_utterance("go to the red door")
        grounded_unique = session4.ground_target_selector({
            "object_type": "door",
            "color": "red",
            "exclude_color": None,
            "relation": "unique",
            "distance_metric": None,
            "distance_reference": None,
        })

    checks["unique_grounding_ok"] = grounded_unique.get("ok", False)
    checks["unique_grounding_returns_distance"] = grounded_unique.get("distance") is not None

    # 5. Scene summary includes agent pose and source
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session5 = _make_session()
        session5.handle_utterance("go to the red door")
        scene_text = session5.status_summary(query="scene")

    checks["scene_summary_has_agent"] = "agent=" in scene_text
    checks["scene_summary_has_source"] = "source=" in scene_text
    checks["scene_summary_has_doors"] = "doors=" in scene_text

    # 6. Scene model replaced with a fresh seeded idle scene on explicit reset
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session6 = _make_session()
        session6.handle_utterance("go to the red door")
        previous_scene = session6.memory.scene_model
        session6.reset()
        reset_scene = session6.memory.scene_model

    checks["scene_model_replaced_on_reset"] = (
        previous_scene is not None
        and reset_scene is not None
        and reset_scene is not previous_scene
    )
    checks["scene_model_refreshed_after_reset"] = (
        reset_scene is not None
        and reset_scene.source == "idle_sense"
        and reset_scene.step_count == 0
    )


# ── Phase 7.58: Active Claims ────────────────────────────────────────────────

def _check_active_claims(checks: dict[str, bool]) -> None:
    # 1. active_claims is None before any grounding
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s1 = _make_session()
        s1.handle_utterance("go to the red door")
    checks["active_claims_none_before_grounding"] = s1.active_claims is None

    # 2. Closest grounding writes StationActiveClaims
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s2 = _make_session()
        s2.handle_utterance("go to the red door")
        s2.handle_utterance("which door is closest by manhattan distance")
    checks["active_claims_written_after_closest"] = s2.active_claims is not None
    checks["active_claims_is_correct_type"] = isinstance(s2.active_claims, StationActiveClaims)
    checks["ranked_scene_doors_non_empty"] = (
        s2.active_claims is not None and len(s2.active_claims.ranked_scene_doors) > 0
    )
    checks["last_grounded_target_is_entry"] = (
        s2.active_claims is not None
        and isinstance(s2.active_claims.last_grounded_target, GroundedObjectEntry)
    )
    checks["last_grounded_rank_is_zero"] = (
        s2.active_claims is not None and s2.active_claims.last_grounded_rank == 0
    )

    # 3. next_closest resolves from claims
    reset_calls: list = []
    original_reset = MiniGridAdapter.reset

    def tracking_reset(self_adapter, seed=None):
        reset_calls.append(seed)
        return original_reset(self_adapter, seed=seed)

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s3 = _make_session()
        s3.handle_utterance("go to the red door")
        s3.handle_utterance("which door is closest by manhattan distance")
        reset_calls.clear()
        with patch.object(MiniGridAdapter, "reset", tracking_reset):
            result_next = s3.handle_utterance("next closest door")

    checks["next_closest_resolves_ok"] = "CLAIM" in result_next.upper() or "door" in result_next.lower()
    checks["next_closest_no_adapter_reset"] = reset_calls == []

    # 4. other_door resolves from claims
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s_other = _make_session()
        s_other.handle_utterance("go to the red door")
        s_other.handle_utterance("which door is closest by manhattan distance")
        result_other = s_other.handle_utterance("the other door")
    checks["other_door_resolves"] = result_other is not None and len(result_other) > 0

    # 5. claims cleared on reset
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s5 = _make_session()
        s5.handle_utterance("go to the red door")
        s5.handle_utterance("which door is closest by manhattan distance")
        had_claims = s5.active_claims is not None
        s5.reset()
        cleared = s5.active_claims is None

    checks["claims_cleared_on_reset"] = had_claims and cleared

    # 5. stale claims fail gracefully
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s6 = _make_session()
        s6.handle_utterance("go to the red door")
        s6.handle_utterance("which door is closest by manhattan distance")
        stale = dataclasses.replace(s6.active_claims, scene_fingerprint=(-1, -1, -1))
        s6.active_claims = stale
        stale_result = s6._resolve_claim_reference("next_closest")

    checks["stale_claims_fail_gracefully"] = not stale_result.get("ok", True)

    # 6. compact_summary shape
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s7 = _make_session()
        s7.handle_utterance("go to the red door")
        s7.handle_utterance("which door is closest by manhattan distance")
        summary = s7.active_claims.compact_summary()

    checks["compact_summary_is_dict"] = isinstance(summary, dict)
    checks["compact_summary_has_last_grounded_target"] = "last_grounded_target" in summary
    checks["compact_summary_has_ranked_doors"] = "ranked_doors" in summary
    checks["compact_summary_has_last_rank"] = "last_rank" in summary

    # 7. is_valid_for checks fingerprint
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s8 = _make_session()
        s8.handle_utterance("go to the red door")
        s8.handle_utterance("which door is closest by manhattan distance")
        scene = s8.memory.scene_model
        valid_for_current = s8.active_claims.is_valid_for(scene)
        fake_scene = dataclasses.replace(scene, agent_x=scene.agent_x + 99, step_count=9999)
        invalid_for_fake = not s8.active_claims.is_valid_for(fake_scene)

    checks["is_valid_for_current_scene"] = valid_for_current
    checks["is_invalid_for_different_scene"] = invalid_for_fake


def main() -> int:
    checks: dict[str, bool] = {}

    print("CONSOLIDATED EVAL: PHASE 7.5 (Registry + SceneModel + ActiveClaims)\n")

    print("── Phase 7.55: Capability Registry ──")
    _check_registry(checks)

    print("── Phase 7.57: Scene Model ──")
    _check_scene_model(checks)

    print("── Phase 7.58: Active Claims ──")
    _check_active_claims(checks)

    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
