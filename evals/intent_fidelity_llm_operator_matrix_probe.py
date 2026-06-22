"""LLM-path operator-intent parity matrix.

The broad eval suite exercises many station behaviors through the deterministic
smoke compiler. This spike forces the same semantic families through
`LLMCompiler` with a fake transport and compares the normalized station outcome
against the smoke path.

The fake transport is offline, but it still runs the live compiler surface:
prompt payload, JSON-schema parser, OperatorIntent validation, IntentVerifier,
request planning, readiness, dispatch, and execution.
"""
from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Deterministic parity probe: the compiler decision is supplied by a fake transport, so the
# arbitrator/synthesizer must not reach the live LLM either (they would otherwise make a real
# OpenRouter call on the unsupported path, making the probe flaky). Forcing them to the smoke
# fallback keeps this probe in the always-green deterministic gate. Genuine live-LLM coverage
# belongs in the opt-in `live_llm` suite, not here.
os.environ.pop("OPENROUTER_API_KEY", None)

from harness import emit_result, first_line

from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.operator_station import ApprovedCommand, OperatorStationSession
from jeenom.schemas import OPERATOR_INTENT_TYPES


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"
SEED = 8


def _base_operator_intent(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent_type": "unsupported",
        "canonical_instruction": None,
        "task_type": None,
        "target": None,
        "knowledge_update": None,
        "reference": None,
        "status_query": None,
        "claim_reference": None,
        "control": None,
        "target_selector": None,
        "grounding_query_plan": None,
        "primitive_definition": None,
        "capability_status": "executable",
        "required_capabilities": [],
        "clear_memory": False,
        "confidence": 1.0,
        "reason": "llm operator matrix fixture",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": None,
        "action_name": None,
        "repeat_count": None,
        "mission_steps": None,
        "selection_objective": None,
        "steering_directive": None,
    }
    payload.update(overrides)
    return payload


def _task_payload(color: str) -> dict[str, Any]:
    return _base_operator_intent(
        intent_type="task_instruction",
        canonical_instruction=f"go to the {color} door",
        task_type="go_to_object",
        target={"color": color, "object_type": "door"},
        required_capabilities=["task.go_to_object.door"],
        reason=f"LLM parsed direct navigation to the {color} door.",
    )


def _distance_plan(color: str) -> dict[str, Any]:
    return {
        "object_type": "door",
        "operation": "answer",
        "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
        "metric": "manhattan",
        "reference": "agent",
        "order": "ascending",
        "ordinal": None,
        "color": color,
        "exclude_colors": [],
        "distance_value": None,
        "comparison": None,
        "tie_policy": "display",
        "answer_fields": ["distance"],
        "required_capabilities": ["grounding.all_doors.ranked.manhattan.agent"],
        "preserved_constraints": [color, "door", "distance", "manhattan"],
    }


def _fake_transport(
    payloads_by_utterance: dict[str, dict[str, Any]]
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def transport(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request.get("method_name") != "compile_operator_intent":
            raise AssertionError(f"unexpected LLM method: {request.get('method_name')}")
        utterance = str(request.get("user_payload", {}).get("utterance", ""))
        if utterance not in payloads_by_utterance:
            raise AssertionError(f"no fake LLM payload for utterance: {utterance!r}")
        return copy.deepcopy(payloads_by_utterance[utterance])

    return transport, calls


@dataclass(frozen=True)
class MatrixCase:
    name: str
    utterance: str
    llm_payloads: dict[str, dict[str, Any]]
    expected_primary_intent: str
    expected_command_kind: str
    response_contains: str
    compare_pose: bool = False
    compare_plan_type: bool = True
    compare_graph_status: bool = True
    expect_extra_compile: bool = False
    primary_intent_may_normalize_to: tuple[str, ...] = ()


def _session(compiler: Any, compiler_name: str) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=compiler,
        compiler_name=compiler_name,
        env_id=ENV_ID,
        seed=SEED,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def _snapshot(session: OperatorStationSession, command: ApprovedCommand, response: str) -> dict[str, Any]:
    scene = session.memory.scene_model
    concept = session.knowledge_channel.recall("scout")
    return {
        "response": response,
        "response_first_line": first_line(response),
        "command_kind": command.kind,
        "intent_type": (
            session.last_operator_intent.intent_type
            if session.last_operator_intent is not None
            else None
        ),
        "plan_type": (
            session.last_request_plan.objective_type
            if session.last_request_plan is not None
            else None
        ),
        "graph_status": (
            session.last_readiness_graph.graph_status
            if session.last_readiness_graph is not None
            else None
        ),
        "pose": (
            (scene.agent_x, scene.agent_y, scene.agent_dir)
            if scene is not None
            else None
        ),
        "scout_utterance": concept.utterance if concept is not None else None,
    }


def _run_llm_case(case: MatrixCase) -> dict[str, Any]:
    transport, calls = _fake_transport(case.llm_payloads)
    compiler = LLMCompiler(api_key="test-key", transport=transport)
    session = _session(compiler, "llm")
    command = session.command_from_llm_intent(case.utterance)
    response = session.turn_orchestrator.execute_command(session, command)
    snap = _snapshot(session, command, str(response))
    operator_calls = [
        call for call in calls
        if call.get("method_name") == "compile_operator_intent"
    ]
    snap["calls"] = calls
    snap["operator_calls"] = operator_calls
    snap["all_call_methods"] = [call.get("method_name") for call in calls]
    snap["operator_call_utterances"] = [
        call.get("user_payload", {}).get("utterance")
        for call in operator_calls
    ]
    snap["supported_intents"] = (
        operator_calls[0].get("user_payload", {}).get("supported", {}).get("intent_types", [])
        if operator_calls
        else []
    )
    snap["used_fallback"] = any(
        call.get("method_name") == "compile_operator_intent" and call.get("used_fallback")
        for call in compiler.call_history
    )
    return snap


def _run_smoke_case(case: MatrixCase) -> dict[str, Any]:
    session = _session(SmokeTestCompiler(), "smoke")
    command = session.command_from_llm_intent(case.utterance)
    response = session.turn_orchestrator.execute_command(session, command)
    return _snapshot(session, command, str(response))


def _primary_call_used_llm(case: MatrixCase, llm: dict[str, Any]) -> bool:
    return (
        bool(llm["operator_calls"])
        and llm["operator_call_utterances"][0] == case.utterance
    )


def _intent_matches(case: MatrixCase, llm: dict[str, Any]) -> bool:
    allowed = (case.expected_primary_intent, *case.primary_intent_may_normalize_to)
    return llm["intent_type"] in allowed


def _case_metrics(case: MatrixCase, llm: dict[str, Any], smoke: dict[str, Any]) -> dict[str, bool]:
    metrics = {
        f"{case.name}_primary_call_used_llm": _primary_call_used_llm(case, llm),
        f"{case.name}_no_llm_fallback": not llm["used_fallback"],
        f"{case.name}_intent_family_ok": _intent_matches(case, llm),
        f"{case.name}_command_kind_matches_smoke": (
            llm["command_kind"] == case.expected_command_kind
            and smoke["command_kind"] == case.expected_command_kind
        ),
        # Parity: the deterministic statement appears in BOTH paths (LLM decides, the
        # deterministic formatter writes the operator-facing text the same way each time).
        f"{case.name}_response_shape_ok": (
            case.response_contains in llm["response"]
            and case.response_contains in smoke["response"]
        ),
    }
    if case.compare_plan_type:
        metrics[f"{case.name}_plan_type_matches_smoke"] = (
            llm["plan_type"] == smoke["plan_type"]
        )
    if case.compare_graph_status:
        metrics[f"{case.name}_graph_status_matches_smoke"] = (
            llm["graph_status"] == smoke["graph_status"]
        )
    if case.compare_pose:
        metrics[f"{case.name}_pose_matches_smoke"] = llm["pose"] == smoke["pose"]
    if case.expect_extra_compile:
        metrics[f"{case.name}_extra_operator_compile_accounted_for"] = (
            len(llm["operator_calls"]) > 1
        )
    else:
        metrics[f"{case.name}_single_operator_compile"] = len(llm["operator_calls"]) == 1
    return metrics


def main() -> int:
    cases = [
        MatrixCase(
            name="motor_command",
            utterance="go straight for 3 steps",
            llm_payloads={
                "go straight for 3 steps": _base_operator_intent(
                    intent_type="motor_command",
                    action_name="move_forward",
                    repeat_count="3",
                    reason="LLM parsed direct motor command with string count.",
                )
            },
            expected_primary_intent="motor_command",
            expected_command_kind="motor_execute",
            response_contains="MOTOR COMPLETE",
            compare_pose=True,
        ),
        MatrixCase(
            name="motor_sequence",
            utterance="can you turn left twice and go forward once",
            llm_payloads={
                "can you turn left twice and go forward once": _base_operator_intent(
                    intent_type="motor_sequence",
                    utterance_steps=["turn_left:2", "move_forward:1"],
                    reason="LLM emitted the explicit motor_sequence shape.",
                )
            },
            expected_primary_intent="motor_sequence",
            expected_command_kind="motor_sequence_execute",
            response_contains="MOTOR SEQUENCE",
            compare_pose=True,
        ),
        MatrixCase(
            name="motor_sequence_misclassified",
            utterance="turn left twice then go forward once",
            llm_payloads={
                "turn left twice then go forward once": _base_operator_intent(
                    intent_type="sequence_instruction",
                    utterance_steps=["turn left twice", "go forward once"],
                    reason="LLM emitted task sequence shape for an all-motor chain.",
                )
            },
            expected_primary_intent="sequence_instruction",
            expected_command_kind="motor_sequence_execute",
            response_contains="MOTOR SEQUENCE",
            compare_pose=True,
        ),
        MatrixCase(
            name="task_instruction",
            utterance="go to the red door",
            llm_payloads={"go to the red door": _task_payload("red")},
            expected_primary_intent="task_instruction",
            expected_command_kind="task_instruction",
            response_contains="COMPLETE",
            compare_pose=True,
        ),
        MatrixCase(
            name="grounded_distance_query",
            utterance="what is the distance to the red door using manhattan distance",
            llm_payloads={
                "what is the distance to the red door using manhattan distance": _base_operator_intent(
                    intent_type="status_query",
                    status_query="ground_target",
                    grounding_query_plan=_distance_plan("red"),
                    required_capabilities=["grounding.all_doors.ranked.manhattan.agent"],
                    reason="LLM emitted a color-specific grounding answer plan.",
                )
            },
            expected_primary_intent="status_query",
            expected_command_kind="clarification",
            response_contains="NEEDS EVIDENCE",
            compare_pose=False,
        ),
        MatrixCase(
            name="ambiguous_navigation",
            utterance="go to the door",
            llm_payloads={
                "go to the door": _base_operator_intent(
                    intent_type="ambiguous",
                    confidence=0.85,
                    reason="Ambiguous target: no color specified. Please clarify which door.",
                )
            },
            expected_primary_intent="ambiguous",
            expected_command_kind="clarification",
            response_contains="Ambiguous",
        ),
        MatrixCase(
            name="unsupported_pickup",
            utterance="pick up the red key",
            llm_payloads={
                "pick up the red key": _base_operator_intent(
                    intent_type="unsupported",
                    capability_status="unsupported",
                    required_capabilities=["task.pickup.key"],
                    reason="Pickup key task is unsupported.",
                )
            },
            expected_primary_intent="unsupported",
            expected_command_kind="missing_skills",
            # Deterministic statement owned by cap_match.operator_message(), not LLM prose.
            response_contains="MISSING SKILLS",
        ),
        MatrixCase(
            name="conditional_motor",
            utterance="if the cell ahead is empty, go forward once",
            llm_payloads={
                "if the cell ahead is empty, go forward once": _base_operator_intent(
                    intent_type="conditional_sense_motor",
                    capability_status="needs_clarification",
                    reason="Conditional motor command requires Sense evidence before actuation.",
                )
            },
            expected_primary_intent="conditional_sense_motor",
            expected_command_kind="clarification",
            response_contains="Conditional motor command requires Sense evidence",
            compare_graph_status=False,
        ),
        MatrixCase(
            name="conditional_until_visible_mission",
            utterance="go straight until you see a blue door",
            llm_payloads={
                "go straight until you see a blue door": _base_operator_intent(
                    intent_type="conditional_sense_motor",
                    target={"color": "blue", "object_type": "door"},
                    action_name="move_forward",
                    required_capabilities=[
                        "sensing.find_object_by_color_type",
                        "action.move_forward",
                        "task.act_until_evidence",
                    ],
                    steering_directive={
                        "budget": {
                            "max_steps": 32,
                            "max_clarifications": None,
                        },
                        "scope": "visible_only",
                        "risk": "operator_authorized",
                        "stopping_rule": "first_match",
                    },
                    reason="LLM preserved the until-visible stop condition.",
                )
            },
            expected_primary_intent="conditional_sense_motor",
            expected_command_kind="conditional_mission_execute",
            response_contains="RUN",
            compare_pose=True,
        ),
        MatrixCase(
            name="concept_teach",
            utterance="when i say scout, you need to go to the red door",
            llm_payloads={
                "when i say scout, you need to go to the red door": _base_operator_intent(
                    intent_type="concept_teach",
                    concept_name="scout",
                    concept_utterance="go to the red door",
                    reason="LLM parsed natural-language concept teaching.",
                ),
                "go to the red door": _task_payload("red"),
            },
            expected_primary_intent="concept_teach",
            expected_command_kind="concept_teach",
            response_contains="CONCEPT STORED",
            compare_graph_status=False,
            expect_extra_compile=True,
        ),
    ]

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    first_supported_intents: list[str] = []
    for case in cases:
        llm = _run_llm_case(case)
        smoke = _run_smoke_case(case)
        first_supported_intents = first_supported_intents or list(llm["supported_intents"])
        metrics.update(_case_metrics(case, llm, smoke))
        if case.name == "concept_teach":
            metrics["concept_teach_stores_same_expansion"] = (
                llm["scout_utterance"] == smoke["scout_utterance"] == "go to the red door"
            )
        details[case.name] = {
            "llm": {
                "response_first_line": llm["response_first_line"],
                "intent_type": llm["intent_type"],
                "command_kind": llm["command_kind"],
                "plan_type": llm["plan_type"],
                "graph_status": llm["graph_status"],
                "pose": llm["pose"],
                "operator_call_utterances": llm["operator_call_utterances"],
                "all_call_methods": llm["all_call_methods"],
            },
            "smoke": {
                "response_first_line": smoke["response_first_line"],
                "intent_type": smoke["intent_type"],
                "command_kind": smoke["command_kind"],
                "plan_type": smoke["plan_type"],
                "graph_status": smoke["graph_status"],
                "pose": smoke["pose"],
            },
        }

    metrics["llm_supported_intents_match_schema_enum"] = (
        set(first_supported_intents) == set(OPERATOR_INTENT_TYPES)
    )
    metrics["llm_supported_intents_cover_newer_runtime_intents"] = all(
        intent in set(first_supported_intents)
        for intent in {
            "primitive_definition",
            "concept_forget",
            "conditional_sense_motor",
            "metric_query",
            "steering_directive",
        }
    )
    metrics["llm_operator_matrix_holds"] = all(metrics.values())
    details["advertised_intents"] = first_supported_intents

    return emit_result(metrics, details, pass_metric="llm_operator_matrix_holds")


if __name__ == "__main__":
    raise SystemExit(main())
