"""Heterogeneous sequences: motor + task + query steps in one chain.

The task-sequence path historically composed *every* step as a `go_to_object`
task (`compose_known_task`), so a raw motor step like "turn right" or a query
step like "what do you see" inside a sequence failed with
"cannot resolve handle". Pure-task and pure-motor sequences each had their own
working path; a *mixed* sequence fell into the task-only builder and errored.

These pins require sequence execution to be kind-aware per step — each step runs
through its proper command path and authority (RawMotorTicket / ExecutionTicket /
query) — while still validating every step before executing any (no partial
execution when a step cannot be compiled).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVALS = ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from harness import build_env as _build_env, make_session as _make_session


def _task_completed(text: str) -> bool:
    return "RUN COMPLETE" in text or "TASK COMPLETE" in text


def test_motor_then_task_sequence_runs():
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("turn right then go to the red door")
    assert "SEQUENCE ERROR" not in resp
    assert "PROCEDURE COMPLETE" in resp
    assert "MOTOR COMPLETE" in resp          # the motor step ran
    assert _task_completed(resp)             # the task step ran


def test_task_then_motor_sequence_runs():
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("go to the red door then turn right")
    assert "SEQUENCE ERROR" not in resp
    assert "PROCEDURE COMPLETE" in resp
    assert "MOTOR COMPLETE" in resp
    assert _task_completed(resp)


def test_query_then_motor_sequence_runs():
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("what do you see around you then turn right")
    assert "SEQUENCE ERROR" not in resp
    assert "PROCEDURE COMPLETE" in resp
    assert "MOTOR COMPLETE" in resp          # the motor step still ran after the query


def test_pure_task_sequence_still_runs():
    """Regression: the original pure-task sequence path must keep working."""
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("go to the red door then go to the green door")
    assert "SEQUENCE ERROR" not in resp
    assert "PROCEDURE COMPLETE" in resp


def test_uncompilable_step_aborts_before_any_execution():
    """No partial execution: an uncompilable step fails the whole sequence and
    the earlier (motor) step must not have executed."""
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        result = sess._run_sequence(
            ["turn right", "florble glorp xyzzy"],
            "turn right then florble glorp xyzzy",
        )
    assert "SEQUENCE ERROR" in result
    assert "MOTOR COMPLETE" not in result
    assert sess.last_raw_motor_ticket is None   # motor step never authorized


def test_nested_sequence_step_rejected_before_execution():
    """A step that is itself a sequence is rejected, and earlier steps do not run."""
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        result = sess._run_sequence(
            ["turn right", "go to the red door then go to the green door"],
            "nested",
        )
    assert "SEQUENCE ERROR" in result
    assert sess.last_raw_motor_ticket is None


def test_forward_motor_step_in_sequence_executes_deterministically():
    """A forward-motion step ('go straight twice') inside a sequence must run.

    Regression: re-compiling such a step through the LLM/dispatch path classified it
    as a motor_sequence and a plan-readiness evidence gate demoted it to a
    clarification, breaking the sequence. Motor steps are now resolved deterministically
    to a motor_execute command."""
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        result = sess._run_sequence(
            ["go straight twice", "what do you see around you"],
            "go straight twice and tell me what you see",
        )
    assert "SEQUENCE ERROR" not in result
    assert "PROCEDURE COMPLETE" in result
    assert "MOTOR COMPLETE" in result
    assert sess.last_raw_motor_ticket is not None  # motor authorized + executed


def test_motor_sequence_step_is_accepted_in_sequence():
    """A step the compiler classifies as a motor_sequence (e.g. 'go left two steps')
    is a valid executable sequence step (runs via RawMotorTicket), not a rejected
    nested kind."""
    from jeenom.llm_compiler import LLMCompiler
    from jeenom.operator_station import OperatorStationSession, _SEQUENCE_STEP_KINDS

    assert "motor_sequence_execute" in _SEQUENCE_STEP_KINDS

    def transport(request):
        # The lone step 'go left two steps' compiles to a motor_sequence.
        base = {
            "intent_type": "motor_sequence",
            "capability_status": "executable",
            "required_capabilities": [],
            "confidence": 1.0,
            "reason": "valid motor sequence",
            "utterance_steps": ["turn_left:2"],
        }
        return base

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = OperatorStationSession(
            compiler_name="llm",
            compiler=LLMCompiler(api_key="test-key", transport=transport),
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        result = sess._run_sequence(["go left two steps"], "go left two steps")
    assert "SEQUENCE ERROR" not in result
    assert "PROCEDURE COMPLETE" in result


def test_sequence_instruction_with_grounding_query_plan_routes_to_sequence():
    """If a sequence_instruction also carries a grounding_query_plan (the model filled
    one for a query sub-clause), the structural intent must win: dispatch routes it to
    sequence execution, not single-result grounding composition (which dead-ended as
    'ambiguous / could not compose a result from the semantic query plan')."""
    from jeenom.schemas import OperatorIntent

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        intent = OperatorIntent(
            intent_type="sequence_instruction",
            utterance_steps=["turn left twice", "what do you see"],
            grounding_query_plan={"object_type": "door", "operation": "answer"},
            capability_status="executable",
            confidence=1.0,
            reason="",
        )
        cmd = sess.turn_orchestrator.dispatch(sess, intent, "turn left twice and tell me what do you see")
        assert cmd.kind == "sequence_execute"
        assert cmd.payload.get("steps") == ["turn left twice", "what do you see"]

        # And it orchestrates end-to-end: motor step + query step both run.
        resp = sess.turn_orchestrator.execute_command(sess, cmd)
    assert "could not compose" not in resp.lower()
    assert "PROCEDURE COMPLETE" in resp
    assert "MOTOR COMPLETE" in resp   # the motor step ran
    assert "SCENE" in resp            # the query step ran


def test_fallback_on_multi_intent_clarifies_instead_of_silently_dropping_steps():
    """When the LLM compile falls back and the utterance is clearly multi-step, the
    deterministic path must not silently answer just one clause (silent degradation
    is failure) — it must fail loud and ask the operator to split/rephrase."""
    from jeenom.llm_compiler import LLMCompiler
    from jeenom.operator_station import OperatorStationSession

    def boom(_request):
        raise RuntimeError("forced LLM failure")

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = OperatorStationSession(
            compiler_name="llm",
            compiler=LLMCompiler(api_key="test-key", transport=boom),
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        resp = sess.handle_utterance(
            "turn left and go straight twice and tell me what door do you see"
        )

    # Must NOT silently collapse to a single scene answer.
    assert "doors=" not in resp
    assert "SCENE" not in resp
    # Must fail loud about the multi-step request.
    lowered = resp.lower()
    assert "step" in lowered
    # No motor side-effect fired (we clarified, not executed a partial guess).
    assert sess.last_raw_motor_ticket is None
