from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    description: str = ""
    safety_notes: str | None = None
    side_effects: tuple[str, ...] = ()
    implementation_status: str = "implemented"
    safe_to_synthesize: bool = False
    required_action_primitives: tuple[str, ...] = ()
    runtime_kind: str | None = None
    runtime_value: int | str | None = None
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    required_claims: tuple[str, ...] = ()
    produced_claims: tuple[str, ...] = ()
    units: dict[str, Any] = field(default_factory=dict)
    frame_id: str | None = None
    required_frames: tuple[str, ...] = ()
    safety_class: str = "query"
    authority_level: str = "none"
    failure_modes: tuple[str, ...] = ()
    validation_hooks: tuple[str, ...] = ()
    substrate_fingerprint: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


TASK_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "locate_object": PrimitiveSpec(
        name="locate_object",
        consumes=("object_location",),
        produces=("target_location_candidate",),
        description="Find a target object in the current world model.",
        safety_notes="Requires a grounded object query; do not hallucinate target locations.",
    ),
    "navigate_to_object": PrimitiveSpec(
        name="navigate_to_object",
        consumes=("agent_pose", "object_location", "occupancy_grid"),
        produces=("adjacency_to_target",),
        description="Move the agent toward the target using available navigation primitives.",
        safety_notes="Must only use validated action primitives; never call env.step directly from the compiler.",
        required_action_primitives=(
            "plan_grid_path",
            "execute_next_path_action",
            "turn_left",
            "turn_right",
            "move_forward",
        ),
    ),
    "verify_adjacent": PrimitiveSpec(
        name="verify_adjacent",
        consumes=("adjacency_to_target",),
        produces=("target_verified",),
        description="Check whether the agent reached the required completion position.",
        safety_notes="Only succeed when the runtime evidence indicates adjacency.",
    ),
    "pickup_object": PrimitiveSpec(
        name="pickup_object",
        consumes=("adjacency_to_target",),
        produces=("item_picked_up",),
        description="Pick up the target item once adjacent.",
        safety_notes="Must only fire after adjacency to the correct target object is verified. Never pick up an unintended item.",
        required_action_primitives=("pickup",),
        implementation_status="planned",
    ),
    "done": PrimitiveSpec(
        name="done",
        consumes=("adjacency_to_target",),
        produces=("task_complete",),
        description="Issue the completion action once the target condition is met.",
        safety_notes="Must not fire before the runtime verifies the completion condition.",
        required_action_primitives=("done",),
    ),
    "ask_operator": PrimitiveSpec(
        name="ask_operator",
        consumes=(),
        produces=("clarification",),
        description="Request clarification from the operator.",
        safety_notes="Use only when the task is underspecified.",
        implementation_status="planned",
    ),
    "replan": PrimitiveSpec(
        name="replan",
        consumes=("error_state",),
        produces=("new_procedure",),
        description="Recompose the plan when the current one is blocked.",
        safety_notes="Use after a validated failure, not preemptively.",
        implementation_status="planned",
    ),
    "abort": PrimitiveSpec(
        name="abort",
        consumes=(),
        produces=("aborted",),
        description="Stop execution when the task cannot be completed safely.",
        safety_notes="Use only for impossible or unsafe states.",
        implementation_status="planned",
    ),
}


GROUNDING_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "claims.last_grounded_target": PrimitiveSpec(
        name="claims.last_grounded_target",
        consumes=("active_claims.last_grounded_target",),
        produces=("grounded_target",),
        description=(
            "Resolve the most recent grounded target from session-scoped ActiveClaims. "
            "Used for operator references such as 'that' after a grounding answer."
        ),
        runtime_kind="claims",
        runtime_value="last_grounded_target",
    ),
}


SENSING_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "parse_grid_objects": PrimitiveSpec(
        name="parse_grid_objects",
        consumes=("raw_image",),
        produces=("grid_objects", "agent_pose"),
        description="Parse the fully observable MiniGrid image into symbolic objects.",
        safety_notes="The runtime parser owns the actual decoding logic.",
    ),
    "find_object_by_color_type": PrimitiveSpec(
        name="find_object_by_color_type",
        consumes=("grid_objects",),
        produces=("object_location", "target_visible", "target_object"),
        description="Select the target object by color and type from parsed objects.",
    ),
    "get_agent_pose": PrimitiveSpec(
        name="get_agent_pose",
        consumes=("grid_objects",),
        produces=("agent_pose",),
        description="Recover the agent position and orientation from the parsed grid.",
    ),
    "check_adjacency": PrimitiveSpec(
        name="check_adjacency",
        consumes=("agent_pose", "object_location"),
        produces=("adjacency_to_target",),
        description="Check whether the agent is in the completion position relative to the target.",
    ),
    "build_occupancy_grid": PrimitiveSpec(
        name="build_occupancy_grid",
        consumes=("grid_objects", "raw_image"),
        produces=("occupancy_grid", "passable_positions", "grid_size"),
        description="Build a passability map for navigation from parsed grid content.",
    ),
}


ACTION_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "plan_grid_path": PrimitiveSpec(
        name="plan_grid_path",
        consumes=("agent_pose", "target_location", "occupancy_grid"),
        produces=("planned_action_names", "path"),
        description="Plan a path in the MiniGrid occupancy map.",
        safety_notes="Runtime path planner must reject unreachable targets.",
        side_effects=("computes_path",),
        runtime_kind="python",
        runtime_value="plan_grid_path",
    ),
    "execute_next_path_action": PrimitiveSpec(
        name="execute_next_path_action",
        consumes=("planned_action_names",),
        produces=("executed_action",),
        description="Execute the first action from the current planned path.",
        side_effects=("moves_agent",),
        runtime_kind="python",
        runtime_value="execute_next_path_action",
        preconditions=("planned_action_names is non-empty",),
        postconditions=("agent pose may change",),
        required_claims=("planned_action_names",),
        produced_claims=("executed_action",),
        frame_id="grid",
        required_frames=("grid",),
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("blocked", "invalid_action"),
        validation_hooks=("minigrid_path_action_preflight",),
    ),
    "turn_left": PrimitiveSpec(
        name="turn_left",
        description="Turn the MiniGrid agent left.",
        side_effects=("moves_agent",),
        runtime_kind="env_action",
        runtime_value=0,
        postconditions=("agent direction changes",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "turn_right": PrimitiveSpec(
        name="turn_right",
        description="Turn the MiniGrid agent right.",
        side_effects=("moves_agent",),
        runtime_kind="env_action",
        runtime_value=1,
        postconditions=("agent direction changes",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "move_forward": PrimitiveSpec(
        name="move_forward",
        description="Move the MiniGrid agent forward.",
        side_effects=("moves_agent",),
        runtime_kind="env_action",
        runtime_value=2,
        preconditions=("front cell is passable",),
        postconditions=("agent position may change",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("blocked",),
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "pickup": PrimitiveSpec(
        name="pickup",
        description="Pick up an item in MiniGrid.",
        side_effects=("changes_environment", "changes_inventory"),
        runtime_kind="env_action",
        runtime_value=3,
        preconditions=("agent is adjacent to item",),
        postconditions=("inventory may change",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("no_item", "inventory_full"),
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "drop": PrimitiveSpec(
        name="drop",
        description="Drop the currently carried item in MiniGrid.",
        side_effects=("changes_environment", "changes_inventory"),
        runtime_kind="env_action",
        runtime_value=4,
        preconditions=("agent is carrying an item",),
        postconditions=("inventory may change",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("empty_inventory", "blocked"),
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "toggle": PrimitiveSpec(
        name="toggle",
        description="Toggle the front cell interaction in MiniGrid.",
        side_effects=("changes_environment",),
        runtime_kind="env_action",
        runtime_value=5,
        preconditions=("front cell is toggleable",),
        postconditions=("environment state may change",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("not_toggleable",),
        validation_hooks=("minigrid_env_action_preflight",),
    ),
    "done": PrimitiveSpec(
        name="done",
        description="Emit the MiniGrid done action.",
        runtime_kind="env_action",
        runtime_value=6,
        preconditions=("completion condition is verified",),
        postconditions=("episode may terminate",),
        frame_id="grid",
        safety_class="actuation",
        authority_level="operator",
        failure_modes=("premature_done",),
        validation_hooks=("minigrid_done_preflight",),
    ),
}


CLAIMS_FILTER_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "filter.threshold.euclidean": PrimitiveSpec(
        name="filter.threshold.euclidean",
        consumes=("active_claims.ranked_scene_doors", "condition"),
        produces=("filtered_entries",),
        description=(
            "Filter euclidean-ranked GroundedObjectEntry claims by a distance threshold. "
            "Parametric: fn(entries, condition) where condition carries threshold (float) "
            "and comparison ('above'|'below'|'within'|'at_least'|'at_most'). "
            "Only operates on typed ActiveClaims entries — never accesses SceneModel or env."
        ),
        implementation_status="synthesizable",
        safe_to_synthesize=True,
    ),
    "filter.threshold.manhattan": PrimitiveSpec(
        name="filter.threshold.manhattan",
        consumes=("active_claims.ranked_scene_doors", "condition"),
        produces=("filtered_entries",),
        description=(
            "Filter manhattan-ranked GroundedObjectEntry claims by a distance threshold. "
            "Parametric: fn(entries, condition) where condition carries threshold (float) "
            "and comparison ('above'|'below'|'within'|'at_least'|'at_most'). "
            "Only operates on typed ActiveClaims entries — never accesses SceneModel or env."
        ),
        implementation_status="synthesizable",
        safe_to_synthesize=True,
    ),
}


def library_payload(library: dict[str, PrimitiveSpec]) -> list[dict[str, Any]]:
    return [library[name].payload() for name in sorted(library)]


def primitive_names(library: dict[str, PrimitiveSpec]) -> list[str]:
    return sorted(library)


def produced_evidence(library: dict[str, PrimitiveSpec]) -> set[str]:
    outputs: set[str] = set()
    for spec in library.values():
        outputs.update(spec.produces)
    return outputs
