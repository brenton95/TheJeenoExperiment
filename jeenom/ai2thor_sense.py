from __future__ import annotations

from typing import Any

from . import geometry
from .schemas import (
    OperationalEvidence,
    Percepts,
    SceneModel,
    WorldModelSample,
)


def _project_coords(pos: dict[str, Any]) -> tuple[float, float, float]:
    """AI2-THOR position {x, y, z} -> JEENO (x, y, z).

    AI2-THOR y is vertical; its (x, z) is the floor plane.
    JEENO floor is (x, y), JEENO z is height.
    So: JEENO.y <- AI2THOR.z, JEENO.z <- AI2THOR.y.
    """
    jx = geometry.as_coord(pos.get("x", 0.0))
    jy = geometry.as_coord(pos.get("z", 0.0))
    jz = geometry.as_coord(pos.get("y", 0.0))
    return jx, jy, jz


class Ai2thorSense:
    """Sensory adapter for AI2-THOR: parses event.metadata into WorldModelSample/SceneModel.

    No template machinery — AI2-THOR sensing is a fixed metadata parse, not a
    compile-time sense plan. No domain global registrars (F1). No ai2thor import
    at module level.
    """

    def __init__(self, memory: Any, compiler: Any, plan_cache: Any = None) -> None:
        self.memory = memory
        self.compiler = compiler
        self.plan_cache = plan_cache

    def tick(
        self,
        observation: Any,
        evidence_frame: Any,
        execution_context: Any,
        loop_index: int = 0,
        allow_llm_compile: bool = True,
    ) -> tuple[OperationalEvidence, Percepts, WorldModelSample, list[str], dict[str, Any]]:
        metadata = observation.metadata if hasattr(observation, "metadata") else {}
        target_object_type = (
            execution_context.params.get("object_type")
            if hasattr(execution_context, "params") else None
        )
        if target_object_type is None and hasattr(evidence_frame, "context"):
            target_object_type = evidence_frame.context.get("object_type")

        sample = self._parse_metadata(metadata, target_object_type)
        evidence = self.project_to_cortex(sample)
        percepts = self.project_to_spine(sample)

        plan = ["parse_ai2thor_metadata"]
        cache_meta: dict[str, Any] = {
            "cache": "disabled",
            "source": "direct_parse",
            "cache_key": "ai2thor_sense",
            "compiler_backend": "ai2thor_sense",
            "runtime_compiler_call": False,
        }
        return evidence, percepts, sample, plan, cache_meta

    def _parse_metadata(
        self,
        metadata: dict[str, Any],
        target_object_type: str | None,
    ) -> WorldModelSample:
        objects_raw = metadata.get("objects", [])
        agent_raw = metadata.get("agent", {})
        agent_pos = agent_raw.get("position", {})
        agent_rot = agent_raw.get("rotation", {})

        agent_x, agent_y, agent_z = _project_coords(agent_pos)
        agent_dir = int(round(agent_rot.get("y", 0.0))) % 360

        grid_objects: list[dict[str, Any]] = []
        target_location: tuple[float, float] | None = None
        target_object: dict[str, Any] | None = None
        target_visible = False

        for obj in objects_raw:
            obj_type = obj.get("objectType", "").lower()
            obj_pos = obj.get("position", {})
            ox, oy, oz = _project_coords(obj_pos)

            grid_obj: dict[str, Any] = {
                "type": obj_type,
                "color": None,
                "x": ox,
                "y": oy,
                "z": oz,
                "state": None,
            }
            grid_objects.append(grid_obj)

            if target_object_type and obj_type == target_object_type:
                target_visible = True
                target_location = (ox, oy)
                target_object = grid_obj

        return WorldModelSample(
            direction=agent_dir,
            step_count=0,
            grid_size=None,
            grid_objects=grid_objects,
            passable_positions=set(),
            agent_pose={"x": agent_x, "y": agent_y, "z": agent_z, "dir": agent_dir},
            target_visible=target_visible,
            target_location=target_location,
            target_object=target_object,
        )

    def project_to_cortex(self, sample: WorldModelSample) -> OperationalEvidence:
        claims = {
            "mission": sample.mission,
            "agent_pose": sample.agent_pose,
            "target_visible": sample.target_visible,
            "target_location": sample.target_location,
            "target_object": sample.target_object,
        }
        return OperationalEvidence(claims=claims, confidence=1.0, source="sense")

    def project_to_spine(self, sample: WorldModelSample) -> Percepts:
        cues = {
            "mission": sample.mission,
            "agent_pose": sample.agent_pose,
            "target_location": sample.target_location,
            "target_object": sample.target_object,
            "passable_positions": sample.passable_positions,
            "grid_size": sample.grid_size,
        }
        return Percepts(cues=cues, source="sense")

    def sense_idle_scene(
        self,
        observation: Any,
        *,
        env_id: str | None = None,
        seed: int | None = None,
    ) -> SceneModel:
        metadata = observation.metadata if hasattr(observation, "metadata") else {}
        sample = self._parse_metadata(metadata, target_object_type=None)
        # SceneModel requires grid_width/grid_height (non-optional ints).
        # AI2-THOR has no grid — we stub 0,0.  Candidate finding: see F5 note.
        model = SceneModel.from_world_model_sample(
            sample, source="idle_sense", env_id=env_id, seed=seed
        )
        if self.memory is not None:
            self.memory.update_scene_model(model)
        return model
