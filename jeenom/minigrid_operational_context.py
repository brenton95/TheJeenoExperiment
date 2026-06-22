from __future__ import annotations

from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from .planning_semantics import PlanningSemantics, register_default_planning_semantics
from .schemas import OperationalContext, register_domain_vocabulary
from .sense import (
    register_domain_index_maps,
    register_open_state_passable,
    register_traverse_to_adjacent,
)

# Register MiniGrid domain vocabulary and behavioural rules when this module is imported.
register_domain_vocabulary(("door",))
register_domain_index_maps(
    object_index=dict(IDX_TO_OBJECT),
    color_index=dict(IDX_TO_COLOR),
)
register_open_state_passable(frozenset({"door"}))
register_traverse_to_adjacent(frozenset({"door"}))


class MiniGridOperationalContext(OperationalContext):
    """MiniGrid situation frame for the current GoToDoor stress substrate."""

    @classmethod
    def default(
        cls,
        *,
        env_id: str = "MiniGrid-GoToDoor-8x8-v0",
    ) -> "MiniGridOperationalContext":
        return cls(
            context_id="minigrid.goto-door",
            substrate_id="minigrid",
            version="1",
            object_vocabulary=["door"],
            attribute_vocabulary=[
                "color",
                "position",
                "distance",
                "agent_pose",
                "visibility",
            ],
            task_families=[
                {
                    "task_type": "go_to_object",
                    "canonical_pattern": "go to the {color} {object_type}",
                    "object_types": ["door"],
                    "required_attributes": ["color", "object_type"],
                    "capability_handle": "task.go_to_object.door",
                }
            ],
            reference_semantics={
                "closest": {
                    "requires_metric": True,
                    "default_metric": "manhattan",
                    "reference": "agent",
                    "tie_policy": "clarify",
                },
                "farthest": {
                    "requires_metric": True,
                    "default_metric": "manhattan",
                    "reference": "agent",
                    "tie_policy": "display",
                },
                "same": {"source": "episodic.last_target"},
                "other": {"source": "active_claims.other_doors"},
                "delivery_target": {"source": "operator_claim.delivery_target"},
            },
            grounding_semantics={
                "visibility_model": "fully_observed_currently",
                "object_types": ["door"],
                "motor_object_terms": ["ball", "box", "door", "goal", "key", "target"],
                "attribute_values": {
                    "color": ["red", "green", "blue", "yellow", "purple", "grey"],
                },
                "attribute_aliases": {
                    "color": {"gray": "grey"},
                },
                "distance_metrics": ["manhattan", "euclidean"],
                "distance_references": ["agent"],
                "ranked_claims_output": "active_claims.ranked_scene_doors",
                "capability_handles": {
                    "ranked": "grounding.all_doors.ranked.{metric}.agent",
                    "closest": "grounding.closest_door.{metric}.{reference}",
                    "unique": "grounding.unique_door.color_filter",
                    "filter_threshold": "claims.filter.threshold.{metric}",
                    "task_go_to_object": "task.go_to_object.{object_type}",
                },
                "rankable_relations": ["closest", "farthest"],
                "tie_policy": "clarify_or_display",
            },
            claim_rules={
                "grounding_scope": "session",
                "grounding_fingerprint": "agent_pose_and_step_count",
                "operator_claim_scope": "durable",
                "promotion_requires_operator_assertion": True,
            },
            display_rules={
                "target_label": "{color} {object_type}@({x},{y})",
                "grounding_answer_header": "GROUNDING ANSWER",
                "coordinate_frame": "minigrid_cell",
                "scene_object_key": "doors",
            },
            environment_identity_fields=[
                "env_id",
                "seed",
                "grid_width",
                "grid_height",
                "mission",
                "task_family",
            ],
            procedure_hints={
                "go_to_object": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ]
            },
            metadata={
                "env_id": env_id,
                "description": "MiniGrid GoToDoor operational context.",
                "substrate_fingerprint": f"minigrid:{env_id}:v1",
                "symbol_mappings": {
                    "object_index": dict(IDX_TO_OBJECT),
                    "color_index": dict(IDX_TO_COLOR),
                    "actions": {
                        "turn_left": 0,
                        "turn_right": 1,
                        "move_forward": 2,
                        "pickup": 3,
                        "drop": 4,
                        "toggle": 5,
                        "done": 6,
                    },
                },
                "frames": {
                    "grid": {
                        "type": "discrete_2d",
                        "origin": "MiniGrid cell coordinates",
                    }
                },
                "units": {
                    "distance": "cell",
                    "angle": "quarter_turn",
                },
                "risk_policy": {
                    "query": {
                        "min_confidence": 0.0,
                        "verification_tier": "none",
                        "required_validation_hooks": [],
                    },
                    "memory": {
                        "min_confidence": 1.0,
                        "verification_tier": "operator",
                        "required_validation_hooks": [],
                    },
                    "actuation": {
                        "min_confidence": 0.8,
                        "verification_tier": "postcondition",
                        "required_validation_hooks": ["minigrid_env_action_preflight"],
                    },
                    "hazardous": {
                        "min_confidence": 1.0,
                        "verification_tier": "manual",
                        "required_validation_hooks": ["operator_authorization"],
                    },
                },
            },
        )


register_default_planning_semantics(
    lambda: PlanningSemantics(MiniGridOperationalContext.default())
)
