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
