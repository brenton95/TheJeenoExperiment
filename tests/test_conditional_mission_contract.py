from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile

from jeenom.capability_registry import CapabilityRegistry
from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import MiniGridAdapter
from jeenom.mission_cortex import MissionCortex
from jeenom.operator_station import OperatorStationSession
from jeenom.planning_semantics import default_planning_semantics
from jeenom.run_demo import build_env
from jeenom.schemas import (
    ExecutionReport,
    MissionContract,
    OperationalEvidence,
    OperatorIntent,
    ProcedureRecipe,
    SchemaValidationError,
    SteeringDirective,
    TaskRequest,
)
from jeenom.side_effect_authority import SideEffectAuthority


def _intent() -> OperatorIntent:
    return OperatorIntent(
        intent_type="conditional_sense_motor",
        target={"color": "blue", "object_type": "door"},
        action_name="move_forward",
        capability_status="executable",
        required_capabilities=[
            "sensing.find_object_by_color_type",
            "action.move_forward",
            "task.act_until_evidence",
        ],
        steering_directive=SteeringDirective(
            budget={"max_steps": 8, "max_clarifications": None},
            scope="visible_only",
            risk="operator_authorized",
            stopping_rule="first_match",
        ),
        confidence=1.0,
        reason="Move forward until the blue door is visible.",
    )


def _mission_plan():
    mission_cortex = MissionCortex(
        planning_semantics=default_planning_semantics(),
        registry=CapabilityRegistry.minigrid_default(),
    )
    return mission_cortex.plan_conditional_evidence_action(
        _intent(),
        utterance="go straight until you see a blue door",
        active_claims=None,
        claims_valid=False,
        environment_identity=None,
    )


def _runtime():
    memory = OperationalMemory()
    cortex = Cortex(memory, SmokeTestCompiler())
    task = TaskRequest(
        instruction="go straight until you see a blue door",
        task_type="act_until_evidence",
        params={
            "color": "blue",
            "object_type": "door",
            "action_name": "move_forward",
            "stop_claim": "target_visible",
            "stop_value": True,
        },
        source="mission_cortex",
    )
    procedure = ProcedureRecipe(
        task_type="act_until_evidence",
        steps=["act_until_evidence"],
        source="mission_cortex",
        compiler_backend="deterministic",
        validated=True,
        rationale="Sense the target, evaluate the stop claim, then authorize one action.",
    )
    cortex.onboard_task(task, procedure)
    return cortex


def _evidence(*, visible: bool, pose: tuple[int, int, int]) -> OperationalEvidence:
    return OperationalEvidence(
        claims={
            "target_visible": visible,
            "target_location": (4, 0) if visible else None,
            "target_object": (
                {"type": "door", "color": "blue", "x": 4, "y": 0}
                if visible
                else None
            ),
            "agent_pose": {"x": pose[0], "y": pose[1], "dir": pose[2]},
        },
        source="sense",
    )


def test_mission_cortex_constructs_typed_sense_spine_procedure():
    plan = _mission_plan()

    assert isinstance(plan.mission_contract, MissionContract)
    contract = plan.mission_contract
    assert contract.procedure.task_type == "act_until_evidence"
    assert contract.procedure.steps == ["act_until_evidence"]
    assert contract.params == {
        "color": "blue",
        "object_type": "door",
        "action_name": "move_forward",
        "stop_claim": "target_visible",
        "stop_value": True,
    }
    assert contract.required_capabilities == [
        "sensing.find_object_by_color_type",
        "action.move_forward",
        "task.act_until_evidence",
    ]
    assert plan.readiness_graph.graph_status == "executable"
    assert plan.readiness_graph.next_action == "execute_task"
    assert [step.required_handle for step in plan.request_plan.steps] == [
        "sensing.find_object_by_color_type",
        "action.move_forward",
        "task.act_until_evidence",
    ]


def test_execution_ticket_carries_the_approved_mission_contract():
    mission_plan = _mission_plan()
    contract = mission_plan.mission_contract

    ticket = SideEffectAuthority().issue_execution_ticket(
        instruction=contract.description,
        task_type=contract.procedure.task_type,
        params=contract.params,
        request_plan=mission_plan.request_plan,
        readiness_graph=mission_plan.readiness_graph,
        mission_id=contract.mission_id,
        mission_contract=contract,
    )

    assert ticket.mission_contract is contract
    assert ticket.task_type == "act_until_evidence"
    assert ticket.params["action_name"] == "move_forward"


def test_execution_ticket_rejects_params_that_do_not_match_mission_contract():
    mission_plan = _mission_plan()
    contract = mission_plan.mission_contract

    try:
        SideEffectAuthority().issue_execution_ticket(
            instruction=contract.description,
            task_type=contract.procedure.task_type,
            params={**contract.params, "action_name": "turn_left"},
            request_plan=mission_plan.request_plan,
            readiness_graph=mission_plan.readiness_graph,
            mission_id=contract.mission_id,
            mission_contract=contract,
        )
    except SchemaValidationError:
        pass
    else:
        raise AssertionError("mismatched mission parameters must not receive authority")


def test_cortex_stops_without_actuation_when_initial_evidence_matches():
    cortex = _runtime()

    frame = cortex.make_evidence_frame()
    assert frame.active_step == "act_until_evidence"
    assert "object_location" in frame.needs

    cortex.update_from_evidence(_evidence(visible=True, pose=(5, 4, 3)))

    assert cortex.choose_execution_contract() is None
    assert cortex.execution_state["task_complete"] is True
    assert not [
        event for event in cortex.trace if event.event == "execution_contract_issued"
    ]


def test_cortex_senses_then_issues_one_spine_contract_and_stops_on_new_evidence():
    cortex = _runtime()

    cortex.make_evidence_frame()
    cortex.update_from_evidence(_evidence(visible=False, pose=(5, 4, 3)))
    contract = cortex.choose_execution_contract()

    assert contract is not None
    assert contract.skill == "move_forward"
    assert contract.stop_conditions == ["target_visible"]
    assert contract.source == "cortex"

    cortex.update_from_report(
        ExecutionReport(
            status="running",
            progress={"executed_action": "move_forward", "step_count": 1},
            source="spine",
        )
    )
    next_frame = cortex.make_evidence_frame()
    assert next_frame.active_step == "act_until_evidence"

    cortex.update_from_evidence(_evidence(visible=True, pose=(5, 3, 3)))

    assert cortex.choose_execution_contract() is None
    assert cortex.execution_state["task_complete"] is True
    issued = [
        event.payload
        for event in cortex.trace
        if event.event == "execution_contract_issued"
    ]
    assert [payload["skill"] for payload in issued] == ["move_forward"]


def test_cortex_fails_finitely_when_the_authorized_action_makes_no_progress():
    cortex = _runtime()

    cortex.make_evidence_frame()
    cortex.update_from_evidence(_evidence(visible=False, pose=(5, 4, 3)))
    contract = cortex.choose_execution_contract()
    assert contract is not None
    cortex.update_from_report(
        ExecutionReport(
            status="running",
            progress={"executed_action": "move_forward", "step_count": 1},
            source="spine",
        )
    )

    cortex.make_evidence_frame()
    cortex.update_from_evidence(_evidence(visible=False, pose=(5, 4, 3)))

    assert cortex.choose_execution_contract() is None
    assert cortex.execution_state["task_complete"] is False
    assert cortex.execution_state["task_failed"] is True
    assert cortex.execution_state["failure_reason"] == "no_progress"


def test_smoke_compiler_preserves_until_condition_instead_of_motor_sequence():
    compiler = SmokeTestCompiler()
    intent = compiler.compile_operator_intent(
        "go straight until you see a blue door",
        memory=OperationalMemory(),
    )

    assert intent.intent_type == "conditional_sense_motor"
    assert intent.target == {"color": "blue", "object_type": "door"}
    assert intent.action_name == "move_forward"
    assert intent.steering_directive is not None
    assert intent.steering_directive.stopping_rule == "first_match"
    assert intent.steering_directive.budget["max_steps"] > 0
    assert "task.act_until_evidence" in intent.required_capabilities


def test_mission_contract_is_serializable_with_its_procedure():
    contract = _mission_plan().mission_contract
    payload = asdict(contract)

    assert payload["procedure"]["steps"] == ["act_until_evidence"]
    assert payload["params"]["stop_claim"] == "target_visible"


def test_station_executes_one_step_then_stops_when_blue_door_enters_fov():
    env = build_env("MiniGrid-GoToDoor-16x16-v0", "none")
    adapter = MiniGridAdapter(env)
    adapter.reset(seed=8)
    env.unwrapped.agent_pos = (5, 3)
    env.unwrapped.agent_dir = 0
    adapter.obs = adapter._annotate_observation(env.unwrapped.gen_obs())

    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke_test",
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    session.substrate.task_adapter = adapter
    try:
        command = session.command_from_llm_intent(
            "go straight until you see a blue door"
        )
        assert command.kind == "conditional_mission_execute"

        response = session.execute_command(command)

        assert "RUN COMPLETE" in response
        assert session.last_execution_ticket is not None
        assert session.last_execution_ticket.mission_contract is not None
        actions = [
            record["action"]
            for record in session.last_result["loop_records"]
            if record["action"] is not None
        ]
        assert actions == ["move_forward"]
        assert session.last_result["final_claims"]["target_visible"] is True
        assert session.last_result["final_state"]["task_complete"] is True
        assert env.unwrapped.agent_pos == (6, 3)
    finally:
        session.close()


def test_station_surfaces_no_progress_as_a_typed_stuck_failure():
    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke_test",
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    try:
        result = session.handle_utterance(
            "go straight until you see a blue door"
        )

        assert "reason=no_progress" in result
        assert result.failure_outcome is not None
        assert result.failure_outcome.category == "stuck"
    finally:
        session.close()
