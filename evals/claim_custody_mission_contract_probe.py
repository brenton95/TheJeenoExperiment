"""Phase 8.11 probe: Mission Contract.

Verifies MissionContract (L4 goal) — an ordered task sequence with
abort-on-failure semantics, distinct from sequence_instruction (L2 procedure).

Checks:
  mission_contract_in_intent_types    — 'mission_contract' in OPERATOR_INTENT_TYPES
  mission_contract_dataclass          — MissionContract importable from jeenom.schemas
  mission_contract_fields             — has mission_id, description, task_sequence,
                                        success_condition, abort_on_failure
  operator_intent_has_mission_steps   — OperatorIntent has mission_steps field
  mission_steps_default_none          — default value is None
  schema_roundtrip                    — OperatorIntent.from_dict round-trips correctly
  schema_validates_steps_required     — fewer than 2 steps raises SchemaValidationError
  smoke_compile_mission_syntax        — SmokeTestCompiler detects "mission: X; Y"
  smoke_mission_steps_correct         — compiled steps match input
  smoke_mission_step_count            — 3-step mission compiles to 3 steps
  routes_to_mission_execute           — command_from_operator_intent returns mission_execute
  handle_utterance_mission_runs       — handle_utterance('mission: X; Y') returns MISSION result
  mission_complete_on_success         — 'MISSION COMPLETE' in response when tasks succeed
  mission_abort_on_failure            — 'MISSION ABORTED' when first step fails
  conditional_contract_has_procedure  — conditional mission carries validated ProcedureRecipe
  conditional_contract_binds_roles    — mission binds Sense, Spine, and composite task handles
  conditional_ticket_carries_contract — execution authority carries the approved MissionContract
  task_instruction_no_regression      — 'go to the red door' still returns RUN COMPLETE
  sequence_instruction_no_regression  — 'go to the red door then go to the green door' still
                                        returns PROCEDURE COMPLETE
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
from jeenom.mission_cortex import MissionCortex
from jeenom.operator_station import ApprovedCommand, OperatorStationSession
from jeenom.planning_semantics import default_planning_semantics
from jeenom.capability_registry import CapabilityRegistry
from jeenom.side_effect_authority import SideEffectAuthority
from jeenom.schemas import (
    MissionContract,
    OPERATOR_INTENT_TYPES,
    OperatorIntent,
    SchemaValidationError,
)






def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ──────────────────────────────────────────────────────────
    metrics["mission_contract_in_intent_types"] = "mission_contract" in OPERATOR_INTENT_TYPES
    metrics["mission_contract_dataclass"] = MissionContract is not None

    mc_field_names = {f.name for f in fields(MissionContract)}
    metrics["mission_contract_fields"] = {
        "mission_id", "description", "task_sequence", "success_condition", "abort_on_failure"
    }.issubset(mc_field_names)

    intent_field_names = {f.name for f in fields(OperatorIntent)}
    metrics["operator_intent_has_mission_steps"] = "mission_steps" in intent_field_names

    dummy = OperatorIntent(intent_type="unsupported", confidence=0.0, reason="")
    metrics["mission_steps_default_none"] = dummy.mission_steps is None

    # ── Schema round-trip ──────────────────────────────────────────────────────
    raw = {
        "intent_type": "mission_contract",
        "mission_steps": ["go to the red door", "go to the blue door"],
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
        "confidence": 0.9,
        "reason": "test",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": None,
        "action_name": None,
        "repeat_count": None,
    }
    restored = OperatorIntent.from_dict(raw)
    metrics["schema_roundtrip"] = (
        restored.intent_type == "mission_contract"
        and restored.mission_steps == ["go to the red door", "go to the blue door"]
    )

    bad = dict(raw, mission_steps=["go to the red door"])  # only 1 step
    try:
        OperatorIntent.from_dict(bad)
        metrics["schema_validates_steps_required"] = False
    except SchemaValidationError:
        metrics["schema_validates_steps_required"] = True

    # ── SmokeTestCompiler detection ────────────────────────────────────────────
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    intent2 = compiler.compile_operator_intent(
        "mission: go to the red door; go to the blue door", memory=memory
    )
    metrics["smoke_compile_mission_syntax"] = intent2.intent_type == "mission_contract"
    metrics["smoke_mission_steps_correct"] = intent2.mission_steps == [
        "go to the red door", "go to the blue door"
    ]

    intent3 = compiler.compile_operator_intent(
        "mission: go to the red door; go to the blue door; go to the yellow door",
        memory=memory,
    )
    metrics["smoke_mission_step_count"] = (
        intent3.intent_type == "mission_contract" and len(intent3.mission_steps or []) == 3
    )

    # ── Conditional Sense/Cortex/Spine mission construction ──────────────────
    conditional = compiler.compile_operator_intent(
        "go straight until you see a blue door",
        memory=memory,
    )
    mission_cortex = MissionCortex(
        planning_semantics=default_planning_semantics(),
        registry=CapabilityRegistry.minigrid_default(),
    )
    conditional_plan = mission_cortex.plan_conditional_evidence_action(
        conditional,
        utterance="go straight until you see a blue door",
        active_claims=None,
        claims_valid=False,
        environment_identity=None,
    )
    conditional_contract = conditional_plan.mission_contract
    metrics["conditional_contract_has_procedure"] = (
        conditional_contract is not None
        and conditional_contract.procedure is not None
        and conditional_contract.procedure.validated
        and conditional_contract.procedure.steps == ["act_until_evidence"]
    )
    metrics["conditional_contract_binds_roles"] = (
        conditional_contract is not None
        and conditional_contract.required_capabilities
        == [
            "sensing.find_object_by_color_type",
            "action.move_forward",
            "task.act_until_evidence",
        ]
        and conditional_plan.readiness_graph.graph_status == "executable"
    )
    conditional_ticket = SideEffectAuthority().issue_execution_ticket(
        instruction=conditional_contract.description,
        task_type=conditional_contract.procedure.task_type,
        params=conditional_contract.params,
        request_plan=conditional_plan.request_plan,
        readiness_graph=conditional_plan.readiness_graph,
        mission_id=conditional_contract.mission_id,
        mission_contract=conditional_contract,
    )
    metrics["conditional_ticket_carries_contract"] = (
        conditional_ticket.mission_contract is conditional_contract
    )

    # ── Station routing ────────────────────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        cmd = sess.turn_orchestrator.dispatch(sess, 
            OperatorIntent.from_dict(raw),
            "mission: go to the red door; go to the blue door",
        )
        metrics["routes_to_mission_execute"] = (
            isinstance(cmd, ApprovedCommand)
            and cmd.kind == "mission_execute"
            and cmd.payload.get("steps") == ["go to the red door", "go to the blue door"]
        )

    # ── End-to-end: handle_utterance ───────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess2 = _make_session()
        resp = sess2.handle_utterance(
            "mission: go to the red door; go to the blue door"
        )
        metrics["handle_utterance_mission_runs"] = (
            "MISSION COMPLETE" in resp or "MISSION ABORTED" in resp
        )
        metrics["mission_complete_on_success"] = "MISSION COMPLETE" in resp

    # ── Abort-on-failure: first step bogus → aborted ──────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        # "go to the purple door" — GoToDoor-8x8 has no purple door; task won't complete
        resp3 = sess3.handle_utterance(
            "mission: go to the purple door; go to the green door"
        )
        metrics["mission_abort_on_failure"] = "MISSION ABORTED" in resp3

    # ── Regression checks ─────────────────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess4 = _make_session()
        resp4 = sess4.handle_utterance("go to the red door")
        metrics["task_instruction_no_regression"] = (
            "RUN COMPLETE" in resp4 or "TASK COMPLETE" in resp4
        )

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess5 = _make_session()
        resp5 = sess5.handle_utterance("go to the red door then go to the green door")
        metrics["sequence_instruction_no_regression"] = "PROCEDURE COMPLETE" in resp5

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
