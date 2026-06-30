from __future__ import annotations

from .schemas import OperationalContext

# Single source of truth for the object types this substrate can navigate to.
# Both the operational-context vocabulary and the adapter manifest derive from
# this list so they cannot drift (CLAUDE.md rule 3). Each type listed here must
# be a single-instance, floor-reachable object in the scene (or placed onto a
# reachable cell at episode setup) — see plan 012.
AI2THOR_GO_TO_OBJECT_TYPES = ["apple", "tomato", "mug"]


class Ai2thorOperationalContext(OperationalContext):
    """AI2-THOR situation frame for the boundary-test spike.

    NOTE: deliberately does NOT call register_domain_vocabulary /
    register_domain_index_maps / register_open_state_passable /
    register_traverse_to_adjacent. Those are module-level globals in
    schemas.py / sense.py — last writer wins, so calling them here at
    import time would clobber MiniGridOperationalContext's registration
    (see plan 002 "watch out" note). Filed as an ORPI/kernel finding:
    this registration mechanism is substrate-exclusive, not
    substrate-independent. Not needed for OperationalContext construction
    or RuntimePackage wiring, so the skeleton omits it.
    """

    @classmethod
    def default(cls, *, scene_id: str = "FloorPlan1") -> "Ai2thorOperationalContext":
        return cls(
            context_id="ai2thor.goto-object",
            substrate_id="ai2thor",
            version="1",
            object_vocabulary=list(AI2THOR_GO_TO_OBJECT_TYPES),
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
                    "canonical_pattern": "go to the {object_type}",
                    "object_types": list(AI2THOR_GO_TO_OBJECT_TYPES),
                    "required_attributes": ["object_type"],
                    "capability_handle": "task.go_to_object.{object_type}",
                }
            ],
            reference_semantics={
                "closest": {
                    "requires_metric": True,
                    "default_metric": "euclidean",
                    "reference": "agent",
                    "tie_policy": "clarify",
                },
                "farthest": {
                    "requires_metric": True,
                    "default_metric": "euclidean",
                    "reference": "agent",
                    "tie_policy": "display",
                },
                "same": {"source": "episodic.last_target"},
            },
            grounding_semantics={
                "visibility_model": "fully_observed_currently",
                "object_types": list(AI2THOR_GO_TO_OBJECT_TYPES),
                "attribute_values": {},
                "attribute_aliases": {},
                "distance_metrics": ["euclidean"],
                "distance_references": ["agent"],
                "ranked_claims_output": "active_claims.ranked_scene_objects",
                "capability_handles": {
                    "task_go_to_object": "task.go_to_object.{object_type}",
                    "ranked": "grounding.all_{object_type_plural}.ranked.euclidean.agent",
                    "closest": "grounding.closest_{object_type}.euclidean.{reference}",
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
                "target_label": "{object_type}@({x},{y})",
                "grounding_answer_header": "GROUNDING ANSWER",
                "coordinate_frame": "ai2thor_projected_2d",
            },
            environment_identity_fields=[
                "scene_id",
                "seed",
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
                "scene_id": scene_id,
                "description": "AI2-THOR go_to_object boundary-test context.",
                "substrate_fingerprint": f"ai2thor:{scene_id}:v1",
                "frames": {
                    "grid": {
                        "type": "projected_2d",
                        "origin": "AI2-THOR (x, z) projected to 2D occupancy grid",
                    }
                },
                "units": {
                    "distance": "meter",
                    "angle": "degree",
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
                        "required_validation_hooks": ["ai2thor_env_action_preflight"],
                    },
                    "hazardous": {
                        "min_confidence": 1.0,
                        "verification_tier": "manual",
                        "required_validation_hooks": ["operator_authorization"],
                    },
                },
            },
        )
