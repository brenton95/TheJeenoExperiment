from __future__ import annotations

from .plan_cache import sense_key
from .primitive_library import SENSING_PRIMITIVES
from .schemas import (
    OperationalEvidence,
    Percepts,
    PrimitiveCall,
    SceneModel,
    SensePlanTemplate,
    WorldModelSample,
)

# Domain index maps — registered by the domain adapter at init.
_IDX_TO_OBJECT: dict[int, str] = {}
_IDX_TO_COLOR: dict[int, str] = {}
# Object types whose state==0 means "open and passable" (domain-specific passability rule).
_OPEN_STATE_PASSABLE: frozenset[str] = frozenset()
# Object types where adjacency means distance==1 (stand next to, not on) and
# navigation targets a neighbour cell rather than the object cell itself.
_TRAVERSE_TO_ADJACENT: frozenset[str] = frozenset()


def register_domain_index_maps(
    object_index: dict[int, str],
    color_index: dict[int, str],
) -> None:
    global _IDX_TO_OBJECT, _IDX_TO_COLOR
    _IDX_TO_OBJECT = object_index
    _IDX_TO_COLOR = color_index


def register_open_state_passable(types: frozenset[str]) -> None:
    global _OPEN_STATE_PASSABLE
    _OPEN_STATE_PASSABLE = types


def register_traverse_to_adjacent(types: frozenset[str]) -> None:
    global _TRAVERSE_TO_ADJACENT
    _TRAVERSE_TO_ADJACENT = types


class MiniGridSense:
    def __init__(self, memory, compiler, plan_cache=None):
        self.memory = memory
        self.compiler = compiler
        self.plan_cache = plan_cache

    def tick(
        self,
        observation,
        evidence_frame,
        execution_context,
        loop_index: int = 0,
        allow_llm_compile: bool = True,
    ):
        template, cache_meta = self._resolve_template(
            evidence_frame=evidence_frame,
            execution_context=execution_context,
            loop_index=loop_index,
            allow_llm_compile=allow_llm_compile,
        )
        plan = self.instantiate_template(template, evidence_frame, execution_context)
        sample = self.execute_plan(observation, plan)
        evidence = self.project_to_cortex(sample)
        percepts = self.project_to_spine(sample)
        return evidence, percepts, sample, plan, cache_meta

    def _resolve_template(
        self,
        evidence_frame,
        execution_context,
        loop_index: int,
        allow_llm_compile: bool = True,
    ):
        cache_key = sense_key(
            evidence_frame=evidence_frame,
            execution_context=execution_context,
            resolved_task_params=evidence_frame.context,
        )
        if self.plan_cache is not None:
            entry = self.plan_cache.lookup(cache_key)
            if entry is not None:
                try:
                    self._validate_template(evidence_frame, entry.template)
                except RuntimeError:
                    self.plan_cache.invalidate(cache_key)
                else:
                    return entry.template, {
                        "cache": "hit",
                        "source": "cache",
                        "compiler_source": entry.source,
                        "cache_key": cache_key,
                        "compiler_backend": entry.compiler_backend,
                        "runtime_compiler_call": False,
                    }

        compile_backend = self.compiler
        if not allow_llm_compile and getattr(self.compiler, "name", None) == "llm_compiler":
            compile_backend = getattr(self.compiler, "fallback", self.compiler)
            self.compiler.log(
                "Human render cache miss for sense plan; using local fallback instead of llm_compiler."
            )

        template = compile_backend.compile_sense_plan(
            evidence_frame=evidence_frame,
            execution_context=execution_context,
            available_sensing_primitives=SENSING_PRIMITIVES,
            memory=self.memory,
        )
        try:
            self._validate_template(evidence_frame, template)
        except RuntimeError as exc:
            fallback = getattr(self.compiler, "fallback", None)
            if fallback is None or compile_backend is fallback:
                raise
            self.compiler.log(f"corrected invalid sense template via fallback: {exc}")
            template = fallback.compile_sense_plan(
                evidence_frame=evidence_frame,
                execution_context=execution_context,
                available_sensing_primitives=SENSING_PRIMITIVES,
                memory=self.memory,
            )
            self._validate_template(evidence_frame, template)

        cache_status = "disabled"
        if self.plan_cache is not None and self.plan_cache.enabled:
            self.plan_cache.store(
                key=cache_key,
                template_type="sense",
                template=template,
                created_at_loop=loop_index,
            )
            cache_status = "miss"

        return template, {
            "cache": cache_status,
            "source": template.source,
            "compiler_source": template.source,
            "cache_key": cache_key,
            "compiler_backend": template.compiler_backend,
            "runtime_compiler_call": True,
        }

    def instantiate_template(self, template: SensePlanTemplate, evidence_frame, execution_context):
        merged_context = dict(execution_context.params)
        merged_context.update(evidence_frame.context)
        plan: list[PrimitiveCall] = []
        for primitive in template.primitives:
            if primitive == "find_object_by_color_type":
                plan.append(
                    PrimitiveCall(
                        name=primitive,
                        params={
                            "color": merged_context.get("color"),
                            "object_type": merged_context.get("object_type"),
                        },
                    )
                )
            elif primitive == "check_adjacency":
                plan.append(
                    PrimitiveCall(
                        name=primitive,
                        params={"object_type": merged_context.get("object_type")},
                    )
                )
            else:
                plan.append(PrimitiveCall(name=primitive))
        return plan

    def execute_plan(self, observation, plan):
        self._validate_plan(plan)
        sample = WorldModelSample(
            mission=observation.raw.get("mission"),
            direction=int(observation.raw.get("direction"))
            if observation.raw.get("direction") is not None
            else None,
            step_count=observation.step_count,
            raw_image=observation.raw.get("image"),
        )

        for step in plan:
            if step.name == "parse_grid_objects":
                self._parse_grid_objects(sample)
            elif step.name == "build_occupancy_grid":
                self._build_occupancy_grid(sample)
            elif step.name == "find_object_by_color_type":
                self._find_object_by_color_type(sample, step.params)
            elif step.name == "get_agent_pose":
                self._get_agent_pose(sample)
            elif step.name == "check_adjacency":
                self._check_adjacency(sample, step.params)
            else:
                raise RuntimeError(f"Unknown sensing primitive at runtime: {step.name}")

        self.memory.update_episodic_memory("last_world_sample", sample.summary())
        if sample.target_location is not None:
            self.memory.update_episodic_memory("known_target_location", sample.target_location)
        if sample.grid_objects is not None and sample.agent_pose is not None:
            self.memory.update_scene_model(
                SceneModel.from_world_model_sample(sample, source="task_sense")
            )
        return sample

    def project_to_cortex(self, sample: WorldModelSample):
        claims = {
            "mission": sample.mission,
            "agent_pose": sample.agent_pose,
            "target_visible": sample.target_visible,
            "target_location": sample.target_location,
            "target_object": sample.target_object,
            "adjacency_to_target": sample.adjacency_to_target,
            "occupancy_grid": sample.occupancy_grid,
        }
        return OperationalEvidence(claims=claims, confidence=1.0, source="sense")

    def project_to_spine(self, sample: WorldModelSample):
        cues = {
            "mission": sample.mission,
            "agent_pose": sample.agent_pose,
            "target_location": sample.target_location,
            "target_object": sample.target_object,
            "adjacency_to_target": sample.adjacency_to_target,
            "occupancy_grid": sample.occupancy_grid,
            "passable_positions": sample.passable_positions,
            "grid_size": sample.grid_size,
            "grid_objects": sample.grid_objects,
        }
        return Percepts(cues=cues, source="sense")

    def sense_idle_scene(
        self,
        observation,
        *,
        env_id: str | None = None,
        seed: int | None = None,
    ) -> SceneModel:
        """Build a SceneModel from a current observation without task context.

        Uses the same parse_grid_objects primitive as task sense — no separate
        sensing path. Stores the result in memory.scene_model.
        """
        sample = WorldModelSample(
            mission=observation.raw.get("mission"),
            direction=int(observation.raw.get("direction", 0))
            if observation.raw.get("direction") is not None
            else None,
            step_count=observation.step_count,
            raw_image=observation.raw.get("image"),
        )
        self._parse_grid_objects(sample)
        self._get_agent_pose(sample)
        model = SceneModel.from_world_model_sample(
            sample, source="idle_sense", env_id=env_id, seed=seed
        )
        self.memory.update_scene_model(model)
        return model

    def _validate_template(self, evidence_frame, template: SensePlanTemplate) -> None:
        for primitive in template.primitives:
            if primitive not in SENSING_PRIMITIVES:
                raise RuntimeError(f"Unknown sensing primitive at runtime: {primitive}")

        needs = set(evidence_frame.needs)
        required_primitives = []
        if "object_location" in needs:
            required_primitives.append("find_object_by_color_type")
        if "agent_pose" in needs:
            required_primitives.append("get_agent_pose")
        if "occupancy_grid" in needs:
            required_primitives.append("build_occupancy_grid")
        if "adjacency_to_target" in needs:
            required_primitives.append("check_adjacency")

        missing = [primitive for primitive in required_primitives if primitive not in template.primitives]
        if missing:
            raise RuntimeError(
                "Sense template missing required primitives for evidence frame: "
                + ", ".join(missing)
            )

    def _validate_plan(self, plan) -> None:
        for step in plan:
            if step.name not in SENSING_PRIMITIVES:
                raise RuntimeError(f"Unknown sensing primitive at runtime: {step.name}")

    def _ensure_parsed_grid(self, sample: WorldModelSample) -> None:
        if sample.grid_size is None:
            self._parse_grid_objects(sample)

    def _parse_grid_objects(self, sample: WorldModelSample) -> None:
        image = sample.raw_image
        if image is None:
            raise RuntimeError("MiniGridSense requires an observation image for parsing.")

        width, height = image.shape[0], image.shape[1]
        sample.grid_size = (width, height)
        sample.grid_objects = []
        sample.agent_pose = None

        for x in range(width):
            for y in range(height):
                object_idx, color_idx, state_idx = [int(v) for v in image[x][y]]
                object_type = _IDX_TO_OBJECT.get(object_idx, "unknown")
                color = _IDX_TO_COLOR.get(color_idx)
                state = state_idx

                if object_type == "agent":
                    sample.agent_pose = {"x": x, "y": y, "dir": state}
                    continue

                if object_type in {"empty", "unseen"}:
                    continue

                sample.grid_objects.append(
                    {
                        "x": x,
                        "y": y,
                        "type": object_type,
                        "color": color,
                        "state": state,
                    }
                )

    def _build_occupancy_grid(self, sample: WorldModelSample) -> None:
        self._ensure_parsed_grid(sample)
        if sample.raw_image is None or sample.grid_size is None:
            raise RuntimeError("Cannot build occupancy grid before parsing the grid.")

        width, height = sample.grid_size
        occupancy_grid = [[False for _ in range(width)] for _ in range(height)]
        passable_positions: set[tuple[int, int]] = set()

        for x in range(width):
            for y in range(height):
                object_idx, _, state_idx = [int(v) for v in sample.raw_image[x][y]]
                object_type = _IDX_TO_OBJECT.get(object_idx, "unknown")

                passable = object_type in {"empty", "floor", "goal", "agent"}
                if object_type in _OPEN_STATE_PASSABLE and int(state_idx) == 0:
                    passable = True

                occupancy_grid[y][x] = passable
                if passable:
                    passable_positions.add((x, y))

        sample.occupancy_grid = occupancy_grid
        sample.passable_positions = passable_positions

    def _find_object_by_color_type(self, sample: WorldModelSample, params) -> None:
        self._ensure_parsed_grid(sample)
        target_color = params.get("color") or self.memory.knowledge.get("target_color")
        target_type = params.get("object_type") or self.memory.knowledge.get("target_type")

        sample.target_visible = False
        sample.target_location = self.memory.episodic_memory.get("known_target_location")
        sample.target_object = None

        for obj in sample.grid_objects:
            if target_type and obj["type"] != target_type:
                continue
            if target_color and obj["color"] != target_color:
                continue

            sample.target_visible = True
            sample.target_location = (obj["x"], obj["y"])
            sample.target_object = obj
            return

    def _get_agent_pose(self, sample: WorldModelSample) -> None:
        self._ensure_parsed_grid(sample)
        if sample.agent_pose is None:
            raise RuntimeError("Agent pose missing after parse_grid_objects.")

    def _check_adjacency(self, sample: WorldModelSample, params) -> None:
        self._get_agent_pose(sample)
        if sample.target_location is None:
            self._find_object_by_color_type(sample, params)
        if sample.agent_pose is None or sample.target_location is None:
            sample.adjacency_to_target = False
            return

        distance = abs(sample.agent_pose["x"] - sample.target_location[0]) + abs(
            sample.agent_pose["y"] - sample.target_location[1]
        )
        target_type = params.get("object_type")
        if target_type is None and sample.target_object is not None:
            target_type = sample.target_object.get("type")

        if target_type in _TRAVERSE_TO_ADJACENT:
            sample.adjacency_to_target = distance == 1
        elif target_type == "key":
            sample.adjacency_to_target = distance == 1    
        else:
            sample.adjacency_to_target = distance == 0
