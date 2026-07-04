"""LLM-path heterogeneous-decomposition parity probe.

The LLM (the semantic-normalization owner) is responsible for decomposing a
compound multi-intent utterance — mixing motor actions, tasks, and scene
questions — into an ordered `sequence_instruction`. The station then orchestrates
each atomic step through its own authorized command path.

This verifies, without a network model, that:

- the compiler prompt actually instructs the model to decompose mixed compounds
  into `sequence_instruction` (and that question/observation steps are allowed);
- when the LLM emits such a heterogeneous `sequence_instruction`, the station runs
  every step (motor + motor + query) in order, each with its own authority, rather
  than erroring or collapsing it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from harness import build_env as _build_env, emit_result

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession

COMPOUND = "turn left and go forward twice and tell me what you see around you"
STEPS = ["turn left", "go forward twice", "what do you see around you"]


def _base() -> dict[str, Any]:
    return {
        "intent_type": "sequence_instruction",
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
        "reason": "decomposition parity probe",
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


def _payload_for(utterance: str) -> dict[str, Any]:
    u = utterance.strip()
    if u == "turn left":
        return {**_base(), "intent_type": "motor_command", "action_name": "turn_left", "repeat_count": 1}
    if u == "go forward twice":
        return {**_base(), "intent_type": "motor_command", "action_name": "move_forward", "repeat_count": 2}
    if u == "what do you see around you":
        return {**_base(), "intent_type": "status_query", "status_query": "scene"}
    # The full compound: decompose into a heterogeneous sequence.
    return {**_base(), "intent_type": "sequence_instruction", "utterance_steps": list(STEPS)}


def main() -> int:
    calls: list[dict[str, Any]] = []

    def transport(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request["method_name"] != "compile_operator_intent":
            raise AssertionError(f"unexpected LLM method: {request['method_name']}")
        return _payload_for(request.get("user_payload", {}).get("utterance", ""))

    compiler = LLMCompiler(api_key="test-key", transport=transport)
    session = OperatorStationSession(
        compiler_name="llm",
        compiler=compiler,
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        response = str(session.handle_utterance(COMPOUND))

    system_prompt = str(calls[0].get("system_prompt", "")) if calls else ""
    metrics = {
        "prompt_advertises_heterogeneous_decomposition": (
            "decompose" in system_prompt.lower()
            and "sequence_instruction" in system_prompt
            and "question" in system_prompt.lower()
        ),
        "compound_compiled_by_llm": (
            bool(calls)
            and calls[0].get("user_payload", {}).get("utterance", "") == COMPOUND
        ),
        "decomposed_sequence_runs_without_error": (
            "PROCEDURE COMPLETE" in response and "SEQUENCE ERROR" not in response
        ),
        "both_motor_steps_executed": response.count("MOTOR COMPLETE") == 2,
        "query_step_executed": "SCENE" in response,
        "not_collapsed_to_single_query": response.count("MOTOR COMPLETE") >= 1,
    }
    details = {
        "response_first_line": response.splitlines()[0] if response else "",
        "motor_complete_count": response.count("MOTOR COMPLETE"),
        "llm_calls": len(calls),
    }
    metrics["llm_decompose_sequence_probe_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="llm_decompose_sequence_probe_holds")


if __name__ == "__main__":
    raise SystemExit(main())
