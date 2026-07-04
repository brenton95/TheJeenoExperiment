"""LLM-path mixed-sequence parity probe.

Closes the coverage hole behind the "cannot resolve handle for 'turn right'"
regression: the sequence probes only ever exercised all-task or all-motor chains,
so a sequence mixing a motor step and a task step was never tested on either the
deterministic or the LLM route.

This verifies the live-compiler path shape without a network model:

- a mixed utterance routes through `LLMCompiler.compile_operator_intent`;
- when the LLM emits a `sequence_instruction` whose `utterance_steps` mix a motor
  action and a task, the station executes it heterogeneously (each step through
  its own command/authority path) rather than erroring or collapsing it into the
  all-motor `motor_sequence` normalization.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from harness import build_env as _build_env, emit_result

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession


def _base_intent_payload() -> dict[str, Any]:
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
        "reason": "Mixed motor+task sequence emitted as sequence_instruction.",
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


def _mixed_sequence_payload() -> dict[str, Any]:
    return {**_base_intent_payload(), "utterance_steps": ["turn right", "go to the red door"]}


def _motor_step_payload() -> dict[str, Any]:
    # A real LLM compiling the bare step "turn right" returns a motor command, not
    # a sequence. (The task step "go to the red door" resolves deterministically and
    # never reaches the compiler.)
    return {
        **_base_intent_payload(),
        "intent_type": "motor_command",
        "action_name": "turn_right",
        "repeat_count": 1,
        "reason": "single motor action",
    }


def main() -> int:
    calls: list[dict[str, Any]] = []

    def transport(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request["method_name"] != "compile_operator_intent":
            raise AssertionError(f"unexpected LLM method: {request['method_name']}")
        utterance = request.get("user_payload", {}).get("utterance", "")
        if utterance.strip() == "turn right":
            return _motor_step_payload()
        return _mixed_sequence_payload()

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
        response = session.handle_utterance("can you turn right and head over to the red door")

    response = str(response)
    first_utterance = calls[0].get("user_payload", {}).get("utterance", "") if calls else ""
    metrics = {
        "mixed_utterance_compiled_by_llm": (
            len(calls) >= 1
            and first_utterance == "can you turn right and head over to the red door"
        ),
        "mixed_sequence_runs_without_error": (
            "PROCEDURE COMPLETE" in response and "SEQUENCE ERROR" not in response
        ),
        "motor_step_executed_with_authority": "MOTOR COMPLETE" in response,
        "task_step_executed": ("RUN COMPLETE" in response or "TASK COMPLETE" in response),
        "not_collapsed_to_motor_sequence": "MOTOR SEQUENCE" not in response,
    }
    details = {
        "response_first_line": response.splitlines()[0] if response else "",
        "llm_calls": len(calls),
        "first_utterance": first_utterance,
    }
    metrics["llm_mixed_sequence_probe_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="llm_mixed_sequence_probe_holds")


if __name__ == "__main__":
    raise SystemExit(main())
