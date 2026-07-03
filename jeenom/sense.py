from __future__ import annotations

from .claim_freshness import (
    FRESHNESS_UNVERIFIABLE,
    OBSERVATION_KIND,
    ttl_for_kind,
)
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


def _target_ref_object_id(target_ref):
    """Extract the adapter-minted object_id from a target_ref hint, if present.

    This is the primary identity the kernel stamps when it has disambiguated among
    description-identical objects. Sense treats it as an opaque key.
    """
    if not isinstance(target_ref, dict):
        return None
    return target_ref.get("object_id")


def _target_ref_coord(target_ref) -> tuple[int, int] | None:
    """Extract the disambiguating (x, y) from a target_ref hint, if present.

    target_ref is a small substrate-neutral dict stamped by the kernel when it has
    already chosen among description-identical objects. On coordinate substrates it
    carries ``{"coord": (x, y)}``; other substrates may key on an object id, which a
    later reader handles alongside this branch.
    """
    if not isinstance(target_ref, dict):
        return None
    coord = target_ref.get("coord")
    if coord is None:
        return None
    try:
        x, y = coord
    except (TypeError, ValueError):
        return None
    return (int(x), int(y))


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
        # Per-cell passable belief: cell -> step_count when last observed passable.
        # Intra-task (a fresh MiniGridSense is built per task episode); cells decay
        # out via the freshness TTL rather than accumulating forever.
        self._passable_belief: dict[tuple[int, int], int] = {}

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
                            # F13: the kernel's chosen-target identity, when it had to
                            # disambiguate description-identical objects. Absent for the
                            # ordinary unique-match case.
                            "target_ref": merged_context.get("target_ref"),
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
        sample = self._sample_from_observation(observation)

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
            "adjacency_to_target": sample.adjacency_to_target,
            "occupancy_grid": sample.occupancy_grid,
        }
        # in_view-aware target belief: only re-emit target_location/target_object
        # while the target is actually in the field of view. When it leaves view we
        # stop refreshing it, so the cortex's prior claim ages on the loop via the
        # freshness decay machine (current -> unverifiable -> unknown) instead of
        # being blanket-refreshed from the known_target_location memory fallback.
        if sample.target_visible:
            claims["target_location"] = sample.target_location
            claims["target_object"] = sample.target_object
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
        sample = self._sample_from_observation(observation)
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

    def _sample_from_observation(self, observation) -> WorldModelSample:
        raw = observation.raw
        grid_size = self._tuple2(raw.get("_jeenom_grid_size"))
        visible_cells, view_to_global = self._visibility_from_raw(raw)
        unseen_cells: set[tuple[int, int]] = set()
        if grid_size is not None and visible_cells:
            unseen_cells = {
                (x, y)
                for x in range(grid_size[0])
                for y in range(grid_size[1])
            } - visible_cells
        agent_pose = self._agent_pose_from_raw(raw)
        return WorldModelSample(
            mission=raw.get("mission"),
            direction=int(raw.get("direction"))
            if raw.get("direction") is not None
            else None,
            step_count=observation.step_count,
            raw_image=raw.get("image"),
            grid_size=grid_size,
            observation_model=str(raw.get("_jeenom_observation_model") or "unknown"),
            visible_cells=visible_cells,
            unseen_cells=unseen_cells,
            view_to_global=view_to_global,
            agent_pose=agent_pose,
        )

    @staticmethod
    def _tuple2(value) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return None

    @staticmethod
    def _agent_pose_from_raw(raw) -> dict[str, int] | None:
        pos = raw.get("_jeenom_agent_pos")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
            return None
        direction = raw.get("_jeenom_agent_dir", raw.get("direction", 0))
        return {"x": int(pos[0]), "y": int(pos[1]), "dir": int(direction)}

    @staticmethod
    def _visibility_from_raw(raw) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
        visible_cells: set[tuple[int, int]] = set()
        view_to_global: dict[tuple[int, int], tuple[int, int]] = {}
        records = raw.get("_jeenom_visible_cells") or []
        if not isinstance(records, list):
            return visible_cells, view_to_global
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                view = (int(record["view_x"]), int(record["view_y"]))
                global_cell = (int(record["x"]), int(record["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            view_to_global[view] = global_cell
            visible_cells.add(global_cell)
        return visible_cells, view_to_global

    def _observed_cells(self, sample: WorldModelSample):
        image = sample.raw_image
        if image is None:
            raise RuntimeError("MiniGridSense requires an observation image for parsing.")
        if sample.view_to_global:
            items = sorted(sample.view_to_global.items(), key=lambda item: item[1])
        else:
            width, height = image.shape[0], image.shape[1]
            items = [
                ((x, y), (x, y))
                for x in range(width)
                for y in range(height)
            ]
            if not sample.visible_cells:
                sample.visible_cells = {global_cell for _, global_cell in items}
        for (view_x, view_y), (global_x, global_y) in items:
            if not (0 <= view_x < image.shape[0] and 0 <= view_y < image.shape[1]):
                continue
            object_idx, color_idx, state_idx = [int(v) for v in image[view_x][view_y]]
            yield {
                "view_x": view_x,
                "view_y": view_y,
                "x": global_x,
                "y": global_y,
                "object_type": _IDX_TO_OBJECT.get(object_idx, "unknown"),
                "color": _IDX_TO_COLOR.get(color_idx),
                "state": state_idx,
            }

    def _parse_grid_objects(self, sample: WorldModelSample) -> None:
        image = sample.raw_image
        if image is None:
            raise RuntimeError("MiniGridSense requires an observation image for parsing.")

        width, height = sample.grid_size or (image.shape[0], image.shape[1])
        sample.grid_size = (int(width), int(height))
        sample.grid_objects = []
        metadata_agent_pose = sample.agent_pose
        sample.agent_pose = metadata_agent_pose

        for cell in self._observed_cells(sample):
            object_type = cell["object_type"]
            color = cell["color"]
            state = cell["state"]

            if object_type == "agent":
                sample.agent_pose = {"x": cell["x"], "y": cell["y"], "dir": state}
                continue

            if object_type in {"empty", "unseen"}:
                continue

            sample.grid_objects.append(
                {
                    "x": cell["x"],
                    "y": cell["y"],
                    "type": object_type,
                    "color": color,
                    "state": state,
                    "visible": True,
                    # Adapter-minted opaque identity. MiniGrid has no native object
                    # ids, and position is its only stable per-object identity (the
                    # world is static), so the id is derived from the cell. The kernel
                    # treats this as an opaque handle and never parses it; a substrate
                    # with native ids (AI2-THOR) supplies its own here instead. This is
                    # what lets the kernel disambiguate two description-identical
                    # objects (two red doors, two apples) by identity, not attributes.
                    "object_id": self._mint_object_id(cell["x"], cell["y"]),
                }
            )

    @staticmethod
    def _mint_object_id(x: int, y: int) -> str:
        """Opaque, stable per-object id for MiniGrid (derived from the static cell)."""
        return f"mg-cell:{int(x)}:{int(y)}"

    def _build_occupancy_grid(self, sample: WorldModelSample) -> None:
        self._ensure_parsed_grid(sample)
        if sample.raw_image is None or sample.grid_size is None:
            raise RuntimeError("Cannot build occupancy grid before parsing the grid.")

        width, height = sample.grid_size
        occupancy_grid = [[False for _ in range(width)] for _ in range(height)]
        observed_passable: set[tuple[int, int]] = set()

        for cell in self._observed_cells(sample):
            x = int(cell["x"])
            y = int(cell["y"])
            if not (0 <= x < width and 0 <= y < height):
                continue
            object_type = cell["object_type"]
            state = int(cell["state"])
            passable = object_type in {"empty", "floor", "goal", "agent"}
            if object_type in _OPEN_STATE_PASSABLE and state == 0:
                passable = True

            occupancy_grid[y][x] = passable
            if passable:
                observed_passable.add((x, y))
            else:
                # Observed non-passable now: drop any stale passable belief here.
                self._passable_belief.pop((x, y), None)

        sample.occupancy_grid = occupancy_grid
        sample.passable_positions = self._accumulate_passable_belief(
            observed_passable, sample.step_count
        )

    def _accumulate_passable_belief(
        self, observed_passable: set[tuple[int, int]], tick: int
    ) -> set[tuple[int, int]]:
        """Return the believed-passable set: observed cells plus cells seen recently
        enough not to have decayed out of belief.

        Observed-this-tick cells refresh to ``current``; cells unseen for fewer than
        ``UNVERIFIABLE_DECAY_STEPS`` ticks remain ``unverifiable`` (still believed);
        cells unseen at/after the threshold decay to ``unknown`` and are dropped.

        # TECH-DEBT(occupancy-decay-sites): claim unification merged the cortex belief
        # store into the single mission-scoped memory.claims store, but this spatial
        # passable belief still decays here in the perception layer via the same
        # freshness TTL, as a separate per-cell store. It stays Sense-side because the
        # planner reads passable_positions from percepts before the cortex runs, and
        # folding per-cell occupancy onto ClaimRecord would change per-cell decay
        # semantics and add hundreds of validated records per tick. This second decay
        # site is deliberately deferred (not closed by the claim merge).
        """
        for cell in observed_passable:
            self._passable_belief[cell] = tick

        ttl = ttl_for_kind(OBSERVATION_KIND, freshness=FRESHNESS_UNVERIFIABLE)
        accumulated: set[tuple[int, int]] = set()
        expired: list[tuple[int, int]] = []
        for cell, last_tick in self._passable_belief.items():
            steps_unseen = max(0, tick - last_tick)
            if ttl.expired(steps_unseen=steps_unseen):
                expired.append(cell)
            else:
                accumulated.add(cell)
        for cell in expired:
            del self._passable_belief[cell]
        return accumulated

    def _find_object_by_color_type(self, sample: WorldModelSample, params) -> None:
        self._ensure_parsed_grid(sample)
        target_color = params.get("color") or self.memory.knowledge.get("target_color")
        target_type = params.get("object_type") or self.memory.knowledge.get("target_type")

        sample.target_visible = False
        sample.target_location = self.memory.episodic_memory.get("known_target_location")
        sample.target_object = None

        # F13: when the kernel already disambiguated among description-identical
        # objects, it stamps the chosen object's identity into target_ref. Prefer the
        # object with that identity so Sense grounds the one the brain picked instead
        # of the first description match in scan order. Match on the adapter-minted
        # object_id (the real, substrate-neutral identity); fall back to the coord for
        # paths where an id is unavailable. Sense treats both as opaque keys.
        chosen_id = _target_ref_object_id(params.get("target_ref"))
        chosen_coord = _target_ref_coord(params.get("target_ref"))
        if chosen_id is not None or chosen_coord is not None:
            for obj in sample.grid_objects:
                if target_type and obj["type"] != target_type:
                    continue
                if target_color and obj["color"] != target_color:
                    continue
                id_match = chosen_id is not None and obj.get("object_id") == chosen_id
                coord_match = (
                    chosen_id is None
                    and chosen_coord is not None
                    and (obj["x"], obj["y"]) == chosen_coord
                )
                if id_match or coord_match:
                    sample.target_visible = True
                    sample.target_location = (obj["x"], obj["y"])
                    sample.target_object = obj
                    return
            # The chosen object is not currently observable (out of FOV, or the world
            # changed): fall through to a description match rather than fabricating it.

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
        else:
            sample.adjacency_to_target = distance == 0
