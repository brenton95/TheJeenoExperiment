"""Phase 8.4.9 probe: Multi-Step Utterance Intent (sequence_instruction).

Verifies that the station can handle sequential execution of raw task utterances
(not just named KB concepts) via the sequence_instruction intent type.

Residual gap addressed: procedure_recall with concept_steps only works when every
step is a named KB concept. This phase adds sequence_instruction with utterance_steps
for novel multi-step raw utterances like 'go to the red door then go to the green door'.

Architecture mapping:
  concept_recall + concept_name    → single named concept → KB lookup
  procedure_recall + concept_steps → named concepts in sequence → KB lookup per step
  sequence_instruction + utterance_steps → raw utterances in sequence → direct compile

Checks:
  sequence_instruction_in_intent_types    — 'sequence_instruction' in OPERATOR_INTENT_TYPES
  operator_intent_has_utterance_steps     — OperatorIntent has utterance_steps field
  operator_intent_utterance_steps_default — default value is None
  smoke_compile_raw_x_then_y             — SmokeTestCompiler emits sequence_instruction for
                                           'go to the red door then go to the green door'
  smoke_compile_navigate_then            — same for 'navigate to red door then green door'
  smoke_utterance_steps_ordered          — utterance_steps preserves left-to-right order
  procedure_recall_not_regression        — single-word 'bingo then scout' still → procedure_recall
  schema_roundtrip_sequence_instruction  — OperatorIntent.from_dict round-trips correctly
  schema_validates_steps_required        — < 2 steps raises SchemaValidationError
  sequence_instruction_routes_to_execute — command_from_operator_intent returns sequence_execute
  sequence_instruction_motor_steps_route_to_motor_sequence — LLM-shaped motor steps don't
                                           fall into task-sequence execution
  handle_utterance_raw_seq_runs          — handle_utterance('go to red door then green door')
                                           returns PROCEDURE COMPLETE
  handle_utterance_last_result_set       — last_result is populated after sequence execution
  try_natural_sequence_raw_returns_cmd   — _try_natural_sequence returns ApprovedCommand for
                                           multi-word sequential utterance
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch
from harness import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OPERATOR_INTENT_TYPES, OperatorIntent, SchemaValidationError






def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ─────────────────────────────────────────────────────────
    metrics["sequence_instruction_in_intent_types"] = (
        "sequence_instruction" in OPERATOR_INTENT_TYPES
    )

    field_names = {f.name for f in fields(OperatorIntent)}
    metrics["operator_intent_has_utterance_steps"] = "utterance_steps" in field_names

    dummy = OperatorIntent(intent_type="unsupported", confidence=0.0, reason="")
    metrics["operator_intent_utterance_steps_default"] = dummy.utterance_steps is None

    # ── SmokeTestCompiler emits sequence_instruction for multi-word parts ─────
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    intent_rg = compiler.compile_operator_intent(
        "go to the red door then go to the green door", memory=memory
    )
    metrics["smoke_compile_raw_x_then_y"] = intent_rg.intent_type == "sequence_instruction"

    intent_nav = compiler.compile_operator_intent(
        "navigate to the red door then the green door", memory=memory
    )
    metrics["smoke_compile_navigate_then"] = intent_nav.intent_type == "sequence_instruction"

    metrics["smoke_utterance_steps_ordered"] = (
        intent_rg.utterance_steps is not None
        and "red door" in intent_rg.utterance_steps[0]
        and "green door" in intent_rg.utterance_steps[1]
    )

    # ── Named concept sequences still go to procedure_recall (no regression) ──
    intent_pr = compiler.compile_operator_intent("bingo then scout", memory=memory)
    metrics["procedure_recall_not_regression"] = intent_pr.intent_type == "procedure_recall"

    # ── Schema round-trip ─────────────────────────────────────────────────────
    raw = {
        "intent_type": "sequence_instruction",
        "utterance_steps": ["go to the red door", "go to the green door"],
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
        "capability_status": "executable",
        "required_capabilities": [],
        "clear_memory": False,
        "confidence": 0.75,
        "reason": "test",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": ["go to the red door", "go to the green door"],
    }
    restored = OperatorIntent.from_dict(raw)
    metrics["schema_roundtrip_sequence_instruction"] = (
        restored.intent_type == "sequence_instruction"
        and restored.utterance_steps == ["go to the red door", "go to the green door"]
    )

    bad = dict(raw, utterance_steps=["go to the red door"])
    try:
        OperatorIntent.from_dict(bad)
        metrics["schema_validates_steps_required"] = False
    except SchemaValidationError:
        metrics["schema_validates_steps_required"] = True

    # ── Station dispatch: sequence_instruction → sequence_execute ─────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        from jeenom.operator_station import ApprovedCommand
        cmd = sess.turn_orchestrator.dispatch(sess, 
            OperatorIntent.from_dict(raw),
            "go to the red door then go to the green door",
        )
        metrics["sequence_instruction_routes_to_execute"] = (
            isinstance(cmd, ApprovedCommand)
            and cmd.kind == "sequence_execute"
            and cmd.payload.get("steps") == ["go to the red door", "go to the green door"]
        )
        motor_cmd = sess.turn_orchestrator.dispatch(
            sess,
            OperatorIntent(
                intent_type="sequence_instruction",
                utterance_steps=["turn left twice", "go forward once"],
                capability_status="executable",
                confidence=1.0,
                reason="LLM-shaped motor sequence emitted as sequence_instruction.",
            ),
            "can you turn left twice and go forward once",
        )
        metrics["sequence_instruction_motor_steps_route_to_motor_sequence"] = (
            isinstance(motor_cmd, ApprovedCommand)
            and motor_cmd.kind == "motor_sequence_execute"
            and motor_cmd.payload.get("sequence") == [
                {"action": "turn_left", "count": 2},
                {"action": "move_forward", "count": 1},
            ]
        )

    # ── End-to-end: handle_utterance with raw sequential utterance ────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess2 = _make_session()
        resp = sess2.handle_utterance("go to the red door then go to the green door")
        metrics["handle_utterance_raw_seq_runs"] = "PROCEDURE COMPLETE" in resp
        metrics["handle_utterance_last_result_set"] = sess2.last_result is not None

    # ── _try_natural_sequence returns ApprovedCommand for multi-word parts ────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        cmd3 = sess3._try_natural_sequence("go to the red door then go to the green door")
        metrics["try_natural_sequence_raw_returns_cmd"] = (
            cmd3 is not None
            and cmd3.kind == "sequence_execute"
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
