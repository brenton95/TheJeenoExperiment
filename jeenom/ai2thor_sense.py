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


def _target_ref_object_id(target_ref: Any) -> Any:
    """Extract the adapter-minted object_id from a target_ref hint, if present.

    Mirrors MiniGridSense's helper (sense.py:29-37). target_ref is the small
    substrate-neutral dict the kernel stamps when it has disambiguated among
    description-identical objects. Treated as an opaque key.
    """
    if not isinstance(target_ref, dict):
        return None
    return target_ref.get("object_id")


def _target_ref_coord(target_ref: Any) -> tuple[int, int] | None:
    """Extract the disambiguating (x, y) from a target_ref hint, if present.

    Mirrors MiniGridSense's helper (sense.py:40-52). DEAD on AI2-THOR by
    construction (float coords vs int()-truncated stamp) — kept for shape-parity.
    """
    if not isinstance(target_ref, dict):
        return None
    coord = target_ref.get("coord")
    if coord is None:
        return None
    return tuple(coord)  # type: ignore[return-value]


def _target_ref_attributes(target_ref: Any) -> dict[str, Any] | None:
    """Extract the parse-time attribute criterion from a target_ref hint, if present.

    Plan 014 Step 2. Unlike object_id/coord (kernel-selected identity, F13), this
    is packed by the ADAPTER PARSER at parse time — the kernel forwards it opaquely
    (compose_known_task, committed 352a806) without evaluating it.
    """
    if not isinstance(target_ref, dict):
        return None
    attrs = target_ref.get("attributes")
    if not isinstance(attrs, dict) or not attrs:
        return None
    return attrs


class Ai2thorSense:
    """Sensory adapter for AI2-THOR: parses event.metadata into WorldModelSample/SceneModel.

    No template machinery — AI2-THOR sensing is a fixed metadata parse, not a
    compile-time sense plan. No domain global registrars (F1). No ai2thor import
    at module level.
    """

    def __init__(
        self,
        memory: Any,
        compiler: Any,
        plan_cache: Any = None,
        adjacency_threshold: float = 0.375,
    ) -> None:
        self.memory = memory
        self.compiler = compiler
        self.plan_cache = plan_cache
        # The episode runner injects the spine's live reach_threshold
        # (grid_size * 1.5) so sense and spine agree on "arrived". The 0.375
        # default is a degenerate fallback only (0.25 default grid * 1.5) for
        # callers that construct sense without the spine; it is wrong for any
        # non-default grid spacing. See orpi_spec F9.
        self._adjacency_threshold = adjacency_threshold

    def tick(
        self,
        observation: Any,
        evidence_frame: Any,
        execution_context: Any,
        loop_index: int = 0,
        allow_llm_compile: bool = True,
    ) -> tuple[OperationalEvidence, Percepts, WorldModelSample, list[str], dict[str, Any]]:
        metadata = observation.metadata if hasattr(observation, "metadata") else {}
        # Merge the two carriers the same way MiniGridSense does (sense.py:186-200):
        # execution_context.params overlaid with the per-tick evidence_frame.context.
        # This is the channel through which the kernel's F13 target_ref reaches sense.
        merged_context: dict[str, Any] = {}
        if hasattr(execution_context, "params") and execution_context.params:
            merged_context.update(execution_context.params)
        if hasattr(evidence_frame, "context") and evidence_frame.context:
            merged_context.update(evidence_frame.context)

        target_object_type = merged_context.get("object_type")
        target_ref = merged_context.get("target_ref")

        sample = self._parse_metadata(metadata, target_object_type, target_ref)
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
        target_ref: dict[str, Any] | None = None,
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
                # Rule 11: hand up AI2-THOR's own state fields opaquely, do not
                # interpret them. Plan 014 Step 1 — was hardcoded None.
                "state": {
                    k: obj[k]
                    for k in ("isOpen", "isToggled", "isSliced", "isPickedUp")
                    if k in obj
                },
                # A1 (link 1): hand up AI2-THOR's native objectId opaquely (Rule 11:
                # substrate-honest — no minting). The kernel carries it as the object's
                # identity so F13 disambiguation survives colourless re-description.
                "object_id": obj.get("objectId"),
            }
            grid_objects.append(grid_obj)

            if target_object_type and obj_type == target_object_type:
                target_visible = True
                target_location = (ox, oy)
                target_object = grid_obj

        # A2 (link 5): F13 target_ref-first resolution, mirroring
        # MiniGridSense._find_object_by_color_type (sense.py:539-566). When the kernel
        # already disambiguated among description-identical objects it stamps the chosen
        # object's identity into target_ref; prefer that object over scan order. Match on
        # the adapter-minted object_id (the substrate-neutral identity); the coord branch
        # is kept for shape-parity with his code but is DEAD on AI2-THOR (coords are
        # floats, the stamped coord is int()-truncated — see plan 013 constraint). Fall
        # through to the description match above only if the chosen object is not
        # currently observable, rather than fabricating it.
        chosen_id = _target_ref_object_id(target_ref)
        chosen_coord = _target_ref_coord(target_ref)
        chosen_attrs = _target_ref_attributes(target_ref)
        if chosen_id is not None or chosen_coord is not None or chosen_attrs is not None:
            for grid_obj in grid_objects:
                if target_object_type and grid_obj["type"] != target_object_type:
                    continue
                id_match = (
                    chosen_id is not None and grid_obj.get("object_id") == chosen_id
                )
                coord_match = (
                    chosen_id is None
                    and chosen_coord is not None
                    and (grid_obj["x"], grid_obj["y"]) == chosen_coord
                )
                # Plan 014 Step 2: a one-turn attribute criterion ("the open
                # fridge") has no id/coord — the kernel never selected among
                # candidates, it only forwarded what the parser packed. Match
                # only when EVERY declared attribute equals the object's
                # populated state (Step 1).
                attrs_match = chosen_id is None and chosen_coord is None and (
                    chosen_attrs is not None
                    and all(
                        grid_obj.get("state", {}).get(field) == value
                        for field, value in chosen_attrs.items()
                    )
                )
                if id_match or coord_match or attrs_match:
                    target_visible = True
                    target_location = (grid_obj["x"], grid_obj["y"])
                    target_object = grid_obj
                    break

        adjacency = False
        if target_location is not None:
            dist = geometry.euclidean(
                (float(agent_x), float(agent_y)),
                (float(target_location[0]), float(target_location[1])),
            )
            adjacency = dist <= self._adjacency_threshold

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
            adjacency_to_target=adjacency,
        )

    def project_to_cortex(self, sample: WorldModelSample) -> OperationalEvidence:
        claims = {
            "mission": sample.mission,
            "agent_pose": sample.agent_pose,
            "target_visible": sample.target_visible,
            "target_location": sample.target_location,
            "target_object": sample.target_object,
            "adjacency_to_target": sample.adjacency_to_target,
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
