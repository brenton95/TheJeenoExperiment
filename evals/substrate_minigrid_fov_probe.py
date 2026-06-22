"""Phase 13B: MiniGrid sensing is field-of-view based, not omniscient.

This probe exercises the substrate boundary directly. It does not ask the station to
solve a task; it verifies that the MiniGrid adapter/sense path can project a partial
egocentric observation into global visible-cell claims without falling back to
FullyObsWrapper-style whole-grid parsing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from harness import build_partial_env, emit_result

from jeenom import run_demo
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import MiniGridAdapter
import jeenom.minigrid_operational_context  # noqa: F401  # registers MiniGrid vocabulary
from jeenom.sense import MiniGridSense


ENV = "MiniGrid-GoToDoor-8x8-v0"
SEED = 42


def _visible_global_cells(env: Any) -> set[tuple[int, int]]:
    raw = env.unwrapped
    _, vis_mask = raw.gen_obs_grid()
    cells: set[tuple[int, int]] = set()
    for x in range(raw.width):
        for y in range(raw.height):
            rel = raw.relative_coords(x, y)
            if rel is not None and bool(vis_mask[int(rel[0]), int(rel[1])]):
                cells.add((x, y))
    return cells


def _non_empty_global_cells(env: Any) -> set[tuple[int, int]]:
    raw = env.unwrapped
    cells: set[tuple[int, int]] = set()
    for x in range(raw.width):
        for y in range(raw.height):
            cell = raw.grid.get(x, y)
            if cell is not None:
                cells.add((x, y))
    return cells


def _run(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    built = run_demo.build_env(ENV, "none")
    try:
        metrics["build_env_uses_partial_observation"] = not isinstance(built, FullyObsWrapper)
    finally:
        built.close()

    harness_env = build_partial_env(ENV, "none")
    try:
        metrics["partial_eval_harness_uses_partial_observation"] = not isinstance(
            harness_env, FullyObsWrapper
        )
    finally:
        harness_env.close()

    env = gym.make(ENV)
    adapter = MiniGridAdapter(env)
    try:
        observation = adapter.reset(seed=SEED)
        image = observation.raw.get("image")
        raw = env.unwrapped
        details["raw_image_shape"] = tuple(int(v) for v in image.shape[:2])
        details["full_grid_shape"] = (int(raw.width), int(raw.height))
        details["adapter_observation_model"] = observation.raw.get("_jeenom_observation_model")
        metrics["adapter_marks_agent_fov"] = (
            observation.raw.get("_jeenom_observation_model") == "agent_fov"
            and tuple(image.shape[:2]) != (raw.width, raw.height)
        )

        sense = MiniGridSense(
            OperationalMemory(root=Path(tempfile.mkdtemp())),
            SmokeTestCompiler(),
        )
        try:
            scene = sense.sense_idle_scene(observation, env_id=ENV, seed=SEED)
        except Exception as exc:  # pragma: no cover - red-bar until FOV parser lands
            details["sense_error"] = f"{type(exc).__name__}: {exc}"
            scene = None

        visible = _visible_global_cells(env)
        non_empty = _non_empty_global_cells(env)
        expected_object_cells = visible & non_empty
        details["visible_cell_count"] = len(visible)
        details["expected_object_cell_count"] = len(expected_object_cells)

        if scene is not None:
            agent_pos = tuple(int(v) for v in raw.agent_pos)
            scene_object_cells = {
                (int(obj.x), int(obj.y))
                for obj in scene.objects
            }
            details["scene_agent"] = (scene.agent_x, scene.agent_y, scene.agent_dir)
            details["scene_object_cells"] = sorted(scene_object_cells)
            metrics["scene_uses_full_grid_dimensions"] = (
                scene.grid_width == raw.width and scene.grid_height == raw.height
            )
            metrics["scene_agent_pose_is_global"] = (
                (int(scene.agent_x), int(scene.agent_y), int(scene.agent_dir))
                == (agent_pos[0], agent_pos[1], int(raw.agent_dir))
            )
            metrics["scene_objects_are_visible_global_cells"] = (
                scene_object_cells == expected_object_cells
            )
            metrics["scene_records_unseen_cells"] = (
                hasattr(scene, "unseen_cells")
                and len(scene.unseen_cells) == raw.width * raw.height - len(visible)
            )
        else:
            metrics["scene_uses_full_grid_dimensions"] = False
            metrics["scene_agent_pose_is_global"] = False
            metrics["scene_objects_are_visible_global_cells"] = False
            metrics["scene_records_unseen_cells"] = False
    finally:
        adapter.close()


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        _run(metrics, details)
    except Exception as exc:  # pragma: no cover - emitted as eval detail
        details["error"] = f"{type(exc).__name__}: {exc}"
    for key in (
        "build_env_uses_partial_observation",
        "partial_eval_harness_uses_partial_observation",
        "adapter_marks_agent_fov",
        "scene_uses_full_grid_dimensions",
        "scene_agent_pose_is_global",
        "scene_objects_are_visible_global_cells",
        "scene_records_unseen_cells",
    ):
        metrics.setdefault(key, False)
    metrics["minigrid_fov_boundary_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="minigrid_fov_boundary_holds")


if __name__ == "__main__":
    raise SystemExit(main())
