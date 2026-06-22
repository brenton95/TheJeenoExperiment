"""Phase 11 mission-flow architecture probe.

This probe is intentionally hostile. It fails if compound mission ownership
drifts back into OperatorStationSession or if a live compound metric task
executes without mission lineage.
"""
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
from typing import Any

from harness import ROOT, emit_result, first_line, make_session


INLINE_TASK = "go to the third farthest door based on the sum of euclidean and manhattan distance"
EXPECTED_HANDLE = "grounding.all_doors.ranked.sum_euclidean_manhattan.agent"


def _source(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _class_method_names(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
    return set()


def _module_function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _step_ids(plan: Any) -> list[str]:
    if plan is None:
        return []
    return [getattr(step, "step_id", "") for step in getattr(plan, "steps", [])]


def _run_static_checks(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    station_source = _source("jeenom/operator_station.py")
    station_methods = _class_method_names(station_source, "OperatorStationSession")
    details["station_forbidden_methods_present"] = sorted(
        name
        for name in ("_primitive_definition_plan", "_resume_inline_metric_request")
        if name in station_methods
    )
    metrics["station_no_longer_owns_primitive_definition_plan"] = (
        "_primitive_definition_plan" not in station_methods
    )
    metrics["station_no_longer_owns_inline_metric_resume"] = (
        "_resume_inline_metric_request" not in station_methods
    )
    metrics["station_does_not_use_resume_payload_authority"] = "resume_payload" not in station_source

    mission_path = ROOT / "jeenom/mission_cortex.py"
    metrics["mission_cortex_module_exists"] = mission_path.exists()
    if mission_path.exists():
        mission_source = mission_path.read_text(encoding="utf-8")
        details["mission_cortex_functions"] = sorted(_module_function_names(mission_source))
        metrics["mission_cortex_defines_owner"] = "class MissionCortex" in mission_source
        metrics["mission_cortex_instantiates_mission_execution_plan"] = (
            "MissionExecutionPlan(" in mission_source
        )
    else:
        metrics["mission_cortex_defines_owner"] = False
        metrics["mission_cortex_instantiates_mission_execution_plan"] = False

    try:
        from jeenom.schemas import ExecutionTicket, MissionExecutionPlan

        ticket_fields = {field.name for field in fields(ExecutionTicket)}
        mission_fields = {field.name for field in fields(MissionExecutionPlan)}
        details["execution_ticket_fields"] = sorted(ticket_fields)
        details["mission_execution_plan_fields"] = sorted(mission_fields)
        metrics["execution_ticket_has_mission_lineage"] = {
            "mission_id",
            "parent_request_id",
            "provenance",
        }.issubset(ticket_fields)
        metrics["mission_execution_plan_has_typed_flow_fields"] = {
            "mission_contract",
            "primitive_definition",
            "continuation_intent",
            "continuation_request_plan",
            "continuation_readiness_graph",
            "provenance",
            "child_tickets",
        }.issubset(mission_fields)
    except Exception as exc:  # pragma: no cover - emitted as eval detail
        details["schema_error"] = f"{type(exc).__name__}: {exc}"
        metrics["execution_ticket_has_mission_lineage"] = False
        metrics["mission_execution_plan_has_typed_flow_fields"] = False

    task_plan = _source("PlanOfAction/task_plan.md")
    blueprint = _source("PlanOfAction/blueprint.md")
    workflow = _source("PlanOfAction/workflow_diagram.mmd")
    flow = _source("PlanOfAction/flow_of_control.mmd")
    docs_blob = "\n".join([task_plan, blueprint, workflow, flow])
    metrics["planofaction_records_mission_flow_complete_before_phase12"] = (
        "| 12D | complete |" in task_plan
        and "Current phase: **Phase 13B" in task_plan
        and "### Phase 12 And 12D - ORPI v0.1" in task_plan
        and "Status: **complete for MiniGrid**." in task_plan
    )
    metrics["task_plan_phase11_numbering_not_corrupted"] = (
        "claims and pri\n- Decide whether repo liposuction" not in task_plan
        and task_plan.find("### Phase 11 - Mission Flow And Architecture Surgery")
        < task_plan.find("### Phase 12 And 12D - ORPI v0.1")
        < task_plan.find("### Phase 13A - Steering Core")
    )
    metrics["planofaction_documents_share_mission_flow"] = all(
        token in docs_blob
        for token in (
            "MissionContract",
            "Cortex",
            "Sense",
            "Spine",
            "ExecutionTicket",
        )
    )
    metrics["planofaction_no_stale_phase10c_next_step"] = (
        "Current Repo Shape After Phase 10C" not in blueprint
        and "next architecture step is Phase 10D" not in blueprint
    )


def _run_behavior_checks(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
    proposal = session.handle_utterance(INLINE_TASK)
    details["proposal_first_line"] = first_line(proposal)
    pending = getattr(session, "pending_primitive_definition", None)
    pending_mission = getattr(pending, "mission_plan", None)
    metrics["compound_task_creates_pending_mission_plan"] = (
        "PRIMITIVE DEFINITION PROPOSAL" in str(proposal)
        and pending_mission is not None
        and pending_mission.__class__.__name__ == "MissionExecutionPlan"
    )
    metrics["pending_mission_keeps_original_request"] = (
        pending_mission is not None
        and getattr(pending_mission, "primitive_definition", None) is not None
        and pending_mission.primitive_definition.proposed_handle == EXPECTED_HANDLE
        and getattr(pending_mission, "mission_contract", None) is not None
        and INLINE_TASK in pending_mission.mission_contract.description
    )
    metrics["pending_definition_has_no_resume_payload"] = not hasattr(
        pending,
        "resume_payload",
    )

    approval = session.handle_utterance("yes")
    details["approval_first_line"] = first_line(approval)
    details["approval_contains_yellow_target"] = "go to the yellow door" in str(approval)
    mission_plan = getattr(session, "last_mission_execution_plan", None)
    ticket = getattr(session, "last_execution_ticket", None)
    details["continuation_steps"] = _step_ids(
        getattr(mission_plan, "continuation_request_plan", None)
    )
    details["ticket_mission_id"] = getattr(ticket, "mission_id", None)
    details["ticket_parent_request_id"] = getattr(ticket, "parent_request_id", None)
    details["ticket_provenance"] = getattr(ticket, "provenance", None)

    metrics["approval_registers_query_only_metric"] = (
        EXPECTED_HANDLE in session.capability_registry.primitive_names()
        and session.capability_registry.lookup(EXPECTED_HANDLE).safety_class == "query"
    )
    steps = set(_step_ids(getattr(mission_plan, "continuation_request_plan", None)))
    metrics["continuation_plan_keeps_rank_select_execute"] = {
        "rank_scene_doors",
        "select_grounded_target",
        "execute_task",
    }.issubset(steps)
    metrics["final_ticket_carries_mission_lineage"] = (
        mission_plan is not None
        and ticket is not None
        and ticket.mission_id == mission_plan.mission_id
        and ticket.parent_request_id == mission_plan.request_plan.request_id
        and ticket.provenance.get("original_utterance") == INLINE_TASK
        and ticket.provenance.get("primitive_handle") == EXPECTED_HANDLE
    )
    metrics["mission_plan_records_child_ticket"] = (
        mission_plan is not None
        and ticket is not None
        and ticket in mission_plan.child_tickets
    )
    metrics["compound_task_does_not_flatten_without_provenance"] = (
        "go to the yellow door" in str(approval)
        and metrics["final_ticket_carries_mission_lineage"]
    )
    metrics["runtime_execution_remains_llm_free"] = (
        session.last_result is not None
        and session.last_result.get("runtime_llm_calls_during_render") == 0
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        _run_static_checks(metrics, details)
        _run_behavior_checks(metrics, details)
    except Exception as exc:  # pragma: no cover - emitted as eval detail
        details["error"] = f"{type(exc).__name__}: {exc}"
        for key in (
            "station_no_longer_owns_primitive_definition_plan",
            "station_no_longer_owns_inline_metric_resume",
            "station_does_not_use_resume_payload_authority",
            "mission_cortex_module_exists",
            "mission_cortex_defines_owner",
            "mission_cortex_instantiates_mission_execution_plan",
            "execution_ticket_has_mission_lineage",
            "mission_execution_plan_has_typed_flow_fields",
            "planofaction_records_mission_flow_complete_before_phase12",
            "task_plan_phase11_numbering_not_corrupted",
            "planofaction_documents_share_mission_flow",
            "planofaction_no_stale_phase10c_next_step",
            "compound_task_creates_pending_mission_plan",
            "pending_mission_keeps_original_request",
            "pending_definition_has_no_resume_payload",
            "approval_registers_query_only_metric",
            "continuation_plan_keeps_rank_select_execute",
            "final_ticket_carries_mission_lineage",
            "mission_plan_records_child_ticket",
            "compound_task_does_not_flatten_without_provenance",
            "runtime_execution_remains_llm_free",
        ):
            metrics.setdefault(key, False)
    metrics["phase11_mission_flow_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase11_mission_flow_holds")


if __name__ == "__main__":
    raise SystemExit(main())
