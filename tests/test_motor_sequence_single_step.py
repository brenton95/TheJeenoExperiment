"""A single-action repeat phrased as a sequence ("go forward twice") must execute.

The LLM legitimately classifies "go forward twice" as a motor_sequence with one canonical
step `move_forward:2`. The dispatch parsed that step fine but rejected the whole command with
"Could not parse motor sequence steps." because of a `< 2` step-count guard — so a valid
single-action repeat dead-ended. A non-empty sequence must execute; only a sequence with zero
parseable steps is unparseable. (Mirrors the sequence_instruction path, which has no such guard.)
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession


def _motor_sequence_intent(steps):
    return {
        "intent_type": "motor_sequence",
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
        "reason": "test motor sequence",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": steps,
        "action_name": None,
        "repeat_count": None,
        "mission_steps": None,
        "selection_objective": None,
        "steering_directive": None,
    }


def _session(intent):
    def transport(request):
        assert request.get("method_name") == "compile_operator_intent"
        return copy.deepcopy(intent)

    return OperatorStationSession(
        compiler=LLMCompiler(api_key="test-key", transport=transport),
        compiler_name="llm",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def test_single_step_motor_sequence_executes():
    session = _session(_motor_sequence_intent(["move_forward:2"]))
    try:
        command = session.command_from_llm_intent("go forward twice")
        assert command.kind == "motor_sequence_execute"
        assert command.payload["sequence"] == [{"action": "move_forward", "count": 2}]
    finally:
        session.close()


def test_multi_step_motor_sequence_still_executes():
    session = _session(_motor_sequence_intent(["turn_left:2", "move_forward:1"]))
    try:
        command = session.command_from_llm_intent("turn left twice and go forward")
        assert command.kind == "motor_sequence_execute"
        assert len(command.payload["sequence"]) == 2
    finally:
        session.close()


def test_empty_motor_sequence_is_rejected():
    # No parseable steps → genuinely unparseable → clarification, not execution.
    session = _session(_motor_sequence_intent(["not a step"]))
    try:
        command = session.command_from_llm_intent("do the thing")
        assert command.kind != "motor_sequence_execute"
    finally:
        session.close()
