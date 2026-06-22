"""LLM-path motor-sequence regression probe.

This probe verifies the live-compiler path shape without calling a network model:

- the utterance must route through `LLMCompiler.compile_operator_intent`, not a
  regex/deterministic shortcut;
- the LLM prompt must advertise `motor_sequence` as the semantic target;
- if an LLM still emits an all-motor `sequence_instruction`, the station must
  semantically normalize those steps into `motor_sequence_execute` rather than
  sending them to task-sequence execution.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harness import emit_result

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession


def _llm_sequence_instruction_payload() -> dict[str, Any]:
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
        "reason": "LLM-shaped all-motor sequence misclassified as sequence_instruction.",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": ["turn left twice", "go forward once"],
        "action_name": None,
        "repeat_count": None,
        "mission_steps": None,
        "selection_objective": None,
        "steering_directive": None,
    }


def main() -> int:
    calls: list[dict[str, Any]] = []

    def transport(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request["method_name"] != "compile_operator_intent":
            raise AssertionError(f"unexpected LLM method: {request['method_name']}")
        return _llm_sequence_instruction_payload()

    compiler = LLMCompiler(api_key="test-key", transport=transport)
    session = OperatorStationSession(
        compiler_name="llm",
        compiler=compiler,
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )

    response = session.handle_utterance("can you turn left twice and go forward once")
    first_call = calls[0] if calls else {}
    supported = (
        first_call.get("user_payload", {})
        .get("supported", {})
        .get("intent_types", [])
    )
    system_prompt = str(first_call.get("system_prompt", ""))
    plan = session.last_request_plan
    graph = session.last_readiness_graph

    metrics = {
        "utterance_routed_through_llm_compiler": len(calls) == 1,
        "llm_prompt_advertises_motor_sequence": "motor_sequence" in supported,
        "llm_prompt_separates_motor_sequence_from_task_sequence": (
            "Do not emit sequence_instruction for all-motor sequences" in system_prompt
        ),
        "llm_sequence_instruction_semantically_normalized": (
            "MOTOR SEQUENCE" in str(response)
            and "SEQUENCE ERROR" not in str(response)
        ),
        "motor_sequence_executes_all_steps": (
            "turn left" in str(response)
            and "move forward" in str(response)
            and str(response).count("MOTOR COMPLETE") == 2
        ),
        "request_plan_is_motor_control": (
            plan is not None
            and plan.objective_type == "motor_control"
            and graph is not None
            and graph.graph_status == "executable"
        ),
    }
    details = {
        "response_first_line": str(response).splitlines()[0] if response else "",
        "supported_intents": supported,
        "request_plan": plan.as_dict() if plan is not None else None,
        "readiness_graph": graph.as_dict() if graph is not None else None,
    }
    metrics["llm_motor_sequence_probe_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="llm_motor_sequence_probe_holds")


if __name__ == "__main__":
    raise SystemExit(main())
