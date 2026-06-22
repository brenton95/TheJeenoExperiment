"""LLM-path semantic parity probe.

The regular smoke/regex evals prove the deterministic compiler path. This probe
proves the same operator-facing behavior when the utterance actually routes
through `LLMCompiler` with a fake transport. The transport is offline, but the
code path is the live compiler path: prompt payload, strict parser,
IntentVerifier, dispatch, readiness, and execution.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harness import emit_result

from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


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
        "reason": "llm_path parity fixture",
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


def _run_llm_case(utterance: str, payload: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def transport(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request["method_name"] != "compile_operator_intent":
            raise AssertionError(f"unexpected LLM method: {request['method_name']}")
        return payload

    session = OperatorStationSession(
        compiler=LLMCompiler(api_key="test-key", transport=transport),
        compiler_name="llm",
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    response = session.handle_utterance(utterance)
    scene_text = session.scene_summary()
    scene = session.memory.scene_model
    return {
        "response": str(response),
        "scene_text": scene_text,
        "pose": (
            (scene.agent_x, scene.agent_y, scene.agent_dir)
            if scene is not None
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
        "calls": calls,
    }


def _run_smoke_case(utterance: str) -> dict[str, Any]:
    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    response = session.handle_utterance(utterance)
    scene_text = session.scene_summary()
    scene = session.memory.scene_model
    return {
        "response": str(response),
        "scene_text": scene_text,
        "pose": (
            (scene.agent_x, scene.agent_y, scene.agent_dir)
            if scene is not None
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
    }


def main() -> int:
    details: dict[str, Any] = {}

    forward_utterance = "can you go forward two steps"
    llm_forward = _run_llm_case(
        forward_utterance,
        _base_operator_intent(
            intent_type="motor_command",
            action_name="move_forward",
            repeat_count=2,
            reason="LLM parsed a direct motor command.",
        ),
    )
    smoke_forward = _run_smoke_case(forward_utterance)

    sequence_utterance = "can you turn left twice and go forward once"
    llm_sequence = _run_llm_case(
        sequence_utterance,
        _base_operator_intent(
            intent_type="sequence_instruction",
            utterance_steps=["turn left twice", "go forward once"],
            reason="LLM misclassified an all-motor chain as sequence_instruction.",
        ),
    )
    smoke_sequence = _run_smoke_case(sequence_utterance)

    details["forward"] = {
        "llm_response_first_line": llm_forward["response"].splitlines()[0],
        "smoke_response_first_line": smoke_forward["response"].splitlines()[0],
        "llm_pose": llm_forward["pose"],
        "smoke_pose": smoke_forward["pose"],
        "llm_calls": [call["method_name"] for call in llm_forward["calls"]],
    }
    details["sequence"] = {
        "llm_response_first_line": llm_sequence["response"].splitlines()[0],
        "smoke_response_first_line": smoke_sequence["response"].splitlines()[0],
        "llm_pose": llm_sequence["pose"],
        "smoke_pose": smoke_sequence["pose"],
        "llm_plan_type": llm_sequence["plan_type"],
        "smoke_plan_type": smoke_sequence["plan_type"],
        "llm_calls": [call["method_name"] for call in llm_sequence["calls"]],
    }

    metrics = {
        "forward_utterance_used_llm_path": len(llm_forward["calls"]) == 1,
        "forward_llm_matches_smoke_pose": (
            "MOTOR COMPLETE" in llm_forward["response"]
            and "MOTOR COMPLETE" in smoke_forward["response"]
            and llm_forward["pose"] == smoke_forward["pose"]
        ),
        "forward_llm_uses_motor_plan": llm_forward["plan_type"] == "motor_control",
        "sequence_utterance_used_llm_path": len(llm_sequence["calls"]) == 1,
        "sequence_llm_misclassification_was_normalized": (
            "MOTOR SEQUENCE" in llm_sequence["response"]
            and "SEQUENCE ERROR" not in llm_sequence["response"]
            and llm_sequence["plan_type"] == "motor_control"
        ),
        "sequence_llm_matches_smoke_pose": (
            "MOTOR SEQUENCE" in llm_sequence["response"]
            and "MOTOR SEQUENCE" in smoke_sequence["response"]
            and llm_sequence["pose"] == smoke_sequence["pose"]
        ),
        "sequence_llm_readiness_matches_smoke": (
            llm_sequence["graph_status"] == "executable"
            and smoke_sequence["graph_status"] == "executable"
        ),
    }
    metrics["llm_path_semantic_parity_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="llm_path_semantic_parity_holds")


if __name__ == "__main__":
    raise SystemExit(main())
