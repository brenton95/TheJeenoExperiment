from __future__ import annotations

from dataclasses import asdict

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_domain_helper import MiniGridDomainHelper
from jeenom.mismatch import MismatchDetector
from jeenom.mission_cortex import InlineMetricMissionRequest, parse_inline_metric_request
from jeenom.operator_station import OperatorStationSession
from jeenom.planning_semantics import PlanningSemantics
from jeenom.turn_state import TurnState
from jeenom.schemas import (
    GroundedObjectEntry,
    OperationalContext,
    OperatorIntent,
    PrimitiveManifest,
    RequestPlan,
    RequestPlanStep,
    SceneModel,
    SceneObject,
    StationActiveClaims,
    operator_intent_json_schema,
)


def _context() -> OperationalContext:
    return OperationalContext(
        context_id="probe.objects",
        substrate_id="probe",
        object_vocabulary=["token", "marker"],
        attribute_vocabulary=["color", "distance"],
        task_families=[
            {
                "task_type": "go_to_object",
                "canonical_pattern": "go to the {color} {object_type}",
                "object_types": ["token", "marker"],
                "required_attributes": ["color", "object_type"],
            }
        ],
        grounding_semantics={
            "attribute_values": {"color": ["red", "blue"]},
            "distance_metrics": ["manhattan", "euclidean"],
            "distance_references": ["agent"],
            "capability_handles": {
                "ranked": "grounding.all_{object_type_plural}.ranked.{metric}.agent",
                "closest": "grounding.closest_{object_type}.{metric}.{reference}",
                "unique": "grounding.unique_{object_type}.color_filter",
                "task_go_to_object": "task.go_to_object.{object_type}",
            },
        },
        reference_semantics={"closest": {"default_metric": "manhattan"}},
    )


def _registry() -> CapabilityRegistry:
    def spec(
        *,
        name: str,
        primitive_type: str,
        layer: str,
        description: str,
        outputs: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "primitive_type": primitive_type,
            "layer": layer,
            "description": description,
            "inputs": [],
            "outputs": list(outputs or []),
            "side_effects": [],
            "implementation_status": "implemented",
            "safe_to_synthesize": False,
            "runtime_binding": None,
        }

    primitives = []
    for object_type in ("token", "marker"):
        primitives.extend(
            [
                spec(
                    name=f"task.go_to_object.{object_type}",
                    primitive_type="task",
                    layer="task",
                    description=f"Go to a grounded {object_type}.",
                ),
                spec(
                    name=f"grounding.unique_{object_type}.color_filter",
                    primitive_type="grounding",
                    layer="grounding",
                    description=f"Ground one {object_type} by color.",
                ),
                spec(
                    name=f"grounding.closest_{object_type}.manhattan.agent",
                    primitive_type="grounding",
                    layer="grounding",
                    description=f"Ground the closest {object_type}.",
                ),
            ]
        )
    for metric in ("manhattan", "euclidean"):
        primitives.append(
            spec(
                name=f"grounding.all_markers.ranked.{metric}.agent",
                primitive_type="grounding",
                layer="grounding",
                description=f"Rank markers by {metric} distance.",
                outputs=["ranked_object_list"],
            )
        )
    return CapabilityRegistry(
        PrimitiveManifest.from_dict(
            {"name": "parametric_object_registry", "primitives": primitives}
        )
    )


def test_domain_parser_accepts_every_context_object_type() -> None:
    helper = MiniGridDomainHelper(_context())

    token = helper.parse_exact_go_to_object_utterance("go to the red token")
    marker = helper.parse_exact_go_to_object_utterance("go to the blue marker")

    assert token == {"verb": "go to", "color": "red", "object_type": "token"}
    assert marker == {"verb": "go to", "color": "blue", "object_type": "marker"}


def test_smoke_compiler_uses_bound_context_vocabulary_and_handles() -> None:
    semantics = PlanningSemantics(_context())
    compiler = SmokeTestCompiler()
    compiler.bind_planning_semantics(semantics)

    task = compiler.compile_task(
        "go to the blue marker",
        available_task_primitives={},
        memory=OperationalMemory(),
    )
    intent = compiler.compile_operator_intent(
        "go to the blue marker",
        memory=OperationalMemory(),
    )

    assert task.params["object_type"] == "marker"
    assert intent.target == {"color": "blue", "object_type": "marker"}
    assert intent.required_capabilities == ["task.go_to_object.marker"]


def test_claim_followup_keeps_the_context_object_type() -> None:
    compiler = SmokeTestCompiler()
    compiler.bind_planning_semantics(PlanningSemantics(_context()))

    intent = compiler.compile_operator_intent(
        "show their distances",
        memory=OperationalMemory(),
        active_claims_summary={
            "object_type": "marker",
            "ranked_objects": ["blue marker@3", "red marker@5"],
            "last_grounded_target": "blue marker @ distance 3",
        },
    )

    plan = intent.grounding_query_plan
    assert plan is not None
    assert plan["object_type"] == "marker"
    assert plan["primitive_handle"] == "grounding.all_markers.ranked.manhattan.agent"


def test_capability_registry_routes_by_exact_parametric_handle() -> None:
    registry = _registry()

    task = registry.readiness_for_task(
        task_type="go_to_object",
        object_type="marker",
    )
    selector = registry.readiness_for_selector(
        {
            "object_type": "marker",
            "color": "blue",
            "exclude_colors": [],
            "relation": "unique",
            "distance_metric": None,
            "distance_reference": None,
        }
    )

    assert task["status"] == "executable"
    assert task["primitive"] == "task.go_to_object.marker"
    assert selector["status"] == "executable"
    assert selector["primitive"] == "grounding.unique_marker.color_filter"


def test_llm_tool_schema_uses_bound_object_vocabulary() -> None:
    schema = operator_intent_json_schema(object_types=("token", "marker"))
    properties = schema["properties"]

    assert properties["target"]["properties"]["object_type"]["enum"] == [
        "token",
        "marker",
        None,
    ]
    assert properties["target_selector"]["properties"]["object_type"]["enum"] == [
        "token",
        "marker",
        None,
    ]
    assert properties["grounding_query_plan"]["properties"]["object_type"]["enum"] == [
        "token",
        "marker",
        None,
    ]


def test_llm_compiler_accepts_context_object_type_without_global_registration() -> None:
    captured_request: dict[str, object] = {}
    response = OperatorIntent(
        intent_type="task_instruction",
        canonical_instruction="go to the blue marker",
        task_type="go_to_object",
        target={"color": "blue", "object_type": "marker"},
        capability_status="executable",
        required_capabilities=["task.go_to_object.marker"],
        confidence=1.0,
        reason="Context-bound marker task.",
    )

    def transport(request: dict[str, object]) -> dict[str, object]:
        captured_request.update(request)
        return asdict(response)

    compiler = LLMCompiler(api_key="test-key", transport=transport)
    compiler.bind_planning_semantics(PlanningSemantics(_context()))

    intent = compiler.compile_operator_intent(
        "go to the blue marker",
        memory=OperationalMemory(),
        capability_manifest=_registry().compact_summary(),
    )

    object_enum = captured_request["schema"]["properties"]["target"]["properties"][
        "object_type"
    ]["enum"]
    assert object_enum == ["token", "marker", None]
    assert intent.target == {"color": "blue", "object_type": "marker"}
    assert intent.required_capabilities == ["task.go_to_object.marker"]
    assert compiler.call_history[-1]["success"] is True
    assert compiler.call_history[-1]["used_fallback"] is False


def test_station_task_intent_uses_context_task_handle() -> None:
    session = OperatorStationSession.__new__(OperatorStationSession)
    session.domain_helper = MiniGridDomainHelper(_context())
    session.planning_semantics = PlanningSemantics(_context())
    session.turn_state = TurnState()

    intent = session._task_intent_for_instruction("go to the blue marker")

    assert intent.target == {"color": "blue", "object_type": "marker"}
    assert intent.required_capabilities == ["task.go_to_object.marker"]


def test_active_claims_offer_an_object_generic_view() -> None:
    marker = GroundedObjectEntry(
        color="blue",
        x=2,
        y=1,
        distance=3,
        object_type="marker",
    )
    claims = StationActiveClaims(
        scene_fingerprint=(0, 0, 0),
        ranked_scene_doors=[marker],
        last_grounded_target=marker,
        last_grounded_rank=0,
        last_grounding_query={"object_type": "marker"},
    )

    assert claims.ranked_objects == [marker]
    assert claims.other_objects() == []
    assert claims.compact_summary()["ranked_objects"] == ["blue marker@3"]


def test_ranked_display_uses_the_grounded_object_type() -> None:
    helper = MiniGridDomainHelper(_context())
    marker = GroundedObjectEntry(
        color="blue",
        x=2,
        y=1,
        distance=3,
        object_type="marker",
    )

    rendered = helper.format_ranked_objects_from_entries(
        [marker],
        metric="manhattan",
    )

    assert rendered.startswith("MARKERS RANKED BY MANHATTAN DISTANCE FROM AGENT")
    assert "navigate to any specific marker" in rendered
    assert "door" not in rendered.lower()


def test_inline_metric_mission_uses_planning_context_object_type() -> None:
    semantics = PlanningSemantics(_context())
    parsed = parse_inline_metric_request(
        "go to the third farthest marker based on the sum of euclidean and manhattan distance",
        _registry(),
        planning_semantics=semantics,
    )

    assert isinstance(parsed, InlineMetricMissionRequest)
    plan = parsed.continuation_intent.grounding_query_plan
    assert plan is not None
    assert plan["object_type"] == "marker"
    assert "task.go_to_object.marker" in parsed.continuation_intent.required_capabilities


def test_mismatch_detector_does_not_invent_an_object_type() -> None:
    plan = RequestPlan(
        request_id="parametric-object-plan",
        original_utterance="find the red object",
        objective_type="query",
        objective_summary="Object type was not specified.",
        steps=[
            RequestPlanStep(
                step_id="ground",
                layer="grounding",
                operation="select",
                constraints={"color": "red"},
            )
        ],
        expected_response="answer_query",
    )
    scene = SceneModel(
        agent_x=0,
        agent_y=0,
        agent_dir=0,
        grid_width=4,
        grid_height=4,
        objects=[SceneObject(object_type="marker", color="red", x=1, y=1)],
        source="test",
    )

    mismatches = MismatchDetector().detect(
        plan,
        registry=_registry(),
        active_claims=None,
        scene_model=scene,
    )

    assert not any(item.mismatch_type == "REQUIRED_ENTITY_ABSENT" for item in mismatches)
