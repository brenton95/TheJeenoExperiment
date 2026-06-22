"""Phase 13B: partial-observation gaps become typed needs_evidence.

MiniGrid FOV means some query/task plans are well-formed and supported, but not
currently answerable from visible evidence. That is not a missing skill and not
stale claims; readiness should expose a distinct needs_evidence status and the
station should be able to carry a typed clarification request for the operator.
"""
from __future__ import annotations

from typing import Any

from harness import emit_result, make_session

from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
from jeenom.readiness_graph import evaluate_request_plan
from jeenom.schemas import RequestPlan, RequestPlanStep, SchemaValidationError
import jeenom.schemas as schemas


def _schema_round_trip(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    request_cls = getattr(schemas, "ClarificationRequest", None)
    metrics["clarification_request_schema_exists"] = request_cls is not None
    if request_cls is None:
        return
    payload = {
        "request_type": "needs_evidence",
        "prompt": "I need more visible evidence before ranking doors.",
        "reason": "No visible door evidence is available.",
        "resume_kind": "status_query",
        "evidence_scope": "visible_only",
        "target": {"object_type": "door"},
        "options": ["search_allowed", "visible_only"],
        "provenance": {"source": "substrate_partial_observability_needs_evidence_probe"},
    }
    try:
        request = request_cls.from_dict(payload)
        details["clarification_request"] = request.as_dict()
        metrics["clarification_request_round_trips"] = request.as_dict() == payload
    except Exception as exc:  # pragma: no cover - red-bar detail
        details["clarification_request_error"] = f"{type(exc).__name__}: {exc}"
        metrics["clarification_request_round_trips"] = False


def _readiness_status(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    plan = RequestPlan(
        request_id="request:needs-evidence-probe",
        original_utterance="how far are all the doors from you",
        objective_type="query",
        objective_summary="Rank visible doors by distance.",
        steps=[
            RequestPlanStep(
                step_id="rank_scene_doors",
                layer="grounding",
                operation="rank",
                required_handle="grounding.all_doors.ranked.manhattan.agent",
                inputs={"object_type": "door"},
                outputs=["active_claims.ranked_scene_doors"],
                constraints={
                    "metric": "manhattan",
                    "evidence_scope": "visible_only",
                    "requires_visible_objects": ["door"],
                },
            ),
            RequestPlanStep(
                step_id="answer_query",
                layer="answer",
                operation="answer",
                depends_on=["rank_scene_doors"],
                memory_reads=["active_claims.ranked_scene_doors"],
            ),
        ],
        expected_response="answer_query",
    )
    try:
        graph = evaluate_request_plan(
            plan,
            registry=build_minigrid_runtime_package(
                env_id="MiniGrid-GoToDoor-8x8-v0",
                render_mode="none",
            ).resolve_capability_registry(),
            evidence_state={
                "observation_model": "agent_fov",
                "visible_objects": [],
                "unseen_cell_count": 44,
            },
        )
        details["readiness_graph"] = graph.as_dict()
        metrics["readiness_uses_needs_evidence"] = (
            graph.graph_status == "needs_evidence"
            and graph.blocking_step_id == "rank_scene_doors"
        )
        metrics["needs_evidence_asks_clarification"] = graph.next_action == "ask_clarification"
    except TypeError as exc:
        details["readiness_error"] = f"{type(exc).__name__}: {exc}"
        metrics["readiness_accepts_evidence_state"] = False
    except SchemaValidationError as exc:
        details["readiness_error"] = f"{type(exc).__name__}: {exc}"
        metrics["readiness_accepts_evidence_state"] = False
    except Exception as exc:  # pragma: no cover - red-bar detail
        details["readiness_error"] = f"{type(exc).__name__}: {exc}"
        metrics["readiness_accepts_evidence_state"] = False
    else:
        metrics["readiness_accepts_evidence_state"] = True


def _station_request(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    session = make_session(
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        observability="partial",
    )
    response = session.handle_utterance("what is the distance of all the doors from you")
    pending = session.pending_clarification
    request = getattr(pending, "clarification_request", None)
    steering = (
        response.labelled_episode.steering
        if response.labelled_episode is not None
        else {}
    )
    pending_context = steering.get("pending_context") if isinstance(steering, dict) else None
    details["station_response_first_line"] = str(response).splitlines()[0]
    details["station_graph"] = (
        session.last_readiness_graph.as_dict()
        if session.last_readiness_graph is not None
        else None
    )
    details["station_clarification_request"] = (
        request.as_dict() if request is not None else None
    )
    details["station_steering"] = steering
    metrics["station_returns_needs_evidence"] = str(response).startswith("NEEDS EVIDENCE")
    metrics["station_graph_is_needs_evidence"] = (
        session.last_readiness_graph is not None
        and session.last_readiness_graph.graph_status == "needs_evidence"
        and session.last_readiness_graph.next_action == "ask_clarification"
    )
    metrics["station_pending_has_typed_request"] = (
        request is not None
        and request.request_type == "needs_evidence"
        and request.evidence_scope == "visible_only"
    )
    metrics["labelled_episode_records_clarification_request"] = (
        isinstance(pending_context, dict)
        and isinstance(pending_context.get("clarification_request"), dict)
        and pending_context["clarification_request"].get("request_type") == "needs_evidence"
    )


def _pending_followup_boundary(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    session = make_session(
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        observability="partial",
    )
    session.handle_utterance("what is the distance of all the doors from you")
    motor_response = session.handle_utterance("can you go forward two steps")

    details["motor_followup_first_line"] = str(motor_response).splitlines()[0]
    metrics["needs_evidence_new_motor_intent_cancels_pending"] = (
        "MOTOR COMPLETE" in str(motor_response)
        and session.pending_clarification is None
    )


def _motor_refreshes_visible_evidence(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    session = make_session(
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        observability="partial",
    )
    initial = session.scene_summary()
    motor_response = session.handle_utterance("turn left twice")
    refreshed = session.scene_summary()
    scene = session.memory.scene_model

    details["motion_initial_scene"] = initial
    details["motion_response"] = str(motor_response).splitlines()[0]
    details["motion_refreshed_scene"] = refreshed
    metrics["motor_command_refreshes_current_fov_scene"] = (
        "MOTOR COMPLETE" in str(motor_response)
        and scene is not None
        and scene.agent_dir == 3
        and any(
            obj.object_type == "door"
            and obj.color == "purple"
            and obj.x == 4
            and obj.y == 0
            for obj in scene.objects
        )
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    _schema_round_trip(metrics, details)
    _readiness_status(metrics, details)
    _station_request(metrics, details)
    _pending_followup_boundary(metrics, details)
    _motor_refreshes_visible_evidence(metrics, details)
    metrics["partial_observability_needs_evidence_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="partial_observability_needs_evidence_holds")


if __name__ == "__main__":
    raise SystemExit(main())
