from __future__ import annotations

from typing import Any

from .capability_registry import CapabilityRegistry
from .schemas import (
    EnvironmentAssumption,
    EnvironmentIdentity,
    KnowledgeSnapshot,
    ReadinessGraph,
    ReadinessNode,
    RequestPlan,
    RequestPlanStep,
    STEERING_RISK_ALLOWED_SAFETY,
    StationActiveClaims,
)


def _steering_risk_block(step: RequestPlanStep, safety_class: str) -> str | None:
    """Phase 13A: a risk directive that does not authorize a step's safety class is an
    authorization withdrawal. Returns a reason when blocked, else None. Reuses the
    existing `needs_authorization` status — no new readiness value."""
    risk = step.constraints.get("steering_risk")
    if risk is None:
        return None
    allowed = STEERING_RISK_ALLOWED_SAFETY.get(risk)
    if allowed is None or safety_class in allowed:
        return None
    return (
        f"Steering risk '{risk}' withdraws authority for a '{safety_class}' step "
        f"({step.step_id})."
    )


def _status_for_primitive(
    step: RequestPlanStep,
    registry: CapabilityRegistry,
) -> tuple[str, str]:
    handle = step.required_handle
    if handle is None:
        if step.operation in {"answer", "read", "update", "reset", "refuse", "select"}:
            return "executable", "No primitive handle required for this plan step."
        if (
            step.layer == "action"
            and step.operation == "execute"
            and bool(step.constraints.get("raw_motor"))
        ):
            # Raw motor is actuation — a query/reversible-only risk directive withdraws it.
            blocked = _steering_risk_block(step, "actuation")
            if blocked is not None:
                return "needs_authorization", blocked
            return "executable", "Raw motor primitive is explicitly authorized by plan."
        return "needs_clarification", "Plan step is missing a required primitive handle."

    spec = registry.lookup(handle)
    if spec is None:
        return "missing_skills", f"Primitive '{handle}' is not present in the registry."
    if spec.implementation_status == "implemented":
        blocked = _steering_risk_block(step, spec.safety_class)
        if blocked is not None:
            return "needs_authorization", blocked
        if (
            spec.authority_level in {"restricted", "admin"}
            and not bool(step.constraints.get("authority_granted"))
        ):
            return (
                "needs_authorization",
                f"Primitive '{handle}' requires {spec.authority_level} authority.",
            )
        if spec.safety_class in {"actuation", "hazardous"} and not spec.validation_hooks:
            return (
                "validation_required",
                f"Primitive '{handle}' requires a validation hook before execution.",
            )
        return "executable", spec.description
    if spec.implementation_status == "synthesizable" or spec.safe_to_synthesize:
        return "synthesizable", spec.description
    if spec.implementation_status in {"planned", "missing"}:
        return "missing_skills", spec.description
    return "unsupported", spec.description


def _claims_status(
    step: RequestPlanStep,
    *,
    active_claims: StationActiveClaims | None,
    claims_valid: bool,
    claims_produced_by_dependency: bool = False,
    manifest_min_confidence: float | None = None,
) -> tuple[str | None, str | None]:
    if not step.scene_fingerprint_required:
        return None, None
    if claims_produced_by_dependency:
        return None, None
    if active_claims is None:
        return "stale_claims", "Step needs ActiveClaims, but none are available yet."
    if not claims_valid:
        return "stale_claims", "ActiveClaims exist but do not match the current scene."
    min_confidence = step.constraints.get("min_claim_confidence", manifest_min_confidence)
    if min_confidence is not None:
        try:
            required_confidence = float(min_confidence)
        except (TypeError, ValueError):
            required_confidence = 1.0
        if active_claims.confidence < required_confidence:
            return (
                "claim_contract_failed",
                (
                    "ActiveClaims confidence is below the plan requirement: "
                    f"{active_claims.confidence} < {required_confidence}."
                ),
            )
    required_frame = step.constraints.get("required_frame_id")
    if required_frame is not None and active_claims.frame_id != required_frame:
        return (
            "claim_contract_failed",
            (
                "ActiveClaims frame does not match the plan requirement: "
                f"{active_claims.frame_id!r} != {required_frame!r}."
            ),
        )
    return None, None


def _evidence_status(
    step: RequestPlanStep,
    *,
    evidence_state: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if evidence_state is None:
        return None, None
    required = step.constraints.get("requires_visible_objects") or []
    if isinstance(required, str):
        required = [required]
    required_types = {str(item) for item in required if item is not None}
    if not required_types:
        return None, None
    evidence_scope = str(step.constraints.get("evidence_scope") or "visible_only")
    if evidence_scope != "visible_only":
        return None, None
    if evidence_state.get("observation_model") != "agent_fov":
        return None, None
    try:
        unseen_count = int(evidence_state.get("unseen_cell_count") or 0)
    except (TypeError, ValueError):
        unseen_count = 0
    if unseen_count <= 0:
        return None, None

    visible_objects = evidence_state.get("visible_objects") or []
    visible_types: set[str] = set()
    if isinstance(visible_objects, list):
        for item in visible_objects:
            if isinstance(item, dict):
                obj_type = item.get("object_type", item.get("type"))
                if obj_type is not None:
                    visible_types.add(str(obj_type))
            elif item is not None:
                visible_types.add(str(item))
    if visible_types.intersection(required_types):
        return None, None
    return (
        "needs_evidence",
        (
            "Visible-only evidence is insufficient for this step: "
            f"requires one of {sorted(required_types)}, but none are visible "
            f"and {unseen_count} cells remain unseen."
        ),
    )


def _actual_for_assumption(
    assumption: EnvironmentAssumption,
    environment_identity: EnvironmentIdentity | None,
) -> dict[str, object] | None:
    if environment_identity is None:
        return None
    if assumption.kind == "env_id":
        return {"env_id": environment_identity.env_id}
    if assumption.kind == "seed":
        return {"seed": environment_identity.seed}
    if assumption.kind == "grid_size":
        return {
            "grid_width": environment_identity.grid_width,
            "grid_height": environment_identity.grid_height,
        }
    if assumption.kind == "task_family":
        return {"task_family": environment_identity.task_family}
    if assumption.kind == "substrate_fingerprint":
        return {"substrate_fingerprint": environment_identity.substrate_fingerprint}
    if assumption.kind == "environment_fingerprint":
        return {"fingerprint": environment_identity.fingerprint()}
    if assumption.kind == "layout_summary":
        return {"summary": dict(environment_identity.summary)}
    return None


def _evaluate_assumptions(
    step: RequestPlanStep,
    plan: RequestPlan,
    environment_identity: EnvironmentIdentity | None,
) -> tuple[list[str], list[str], str]:
    if environment_identity is None:
        return [], [], ""
    assumptions = {
        assumption.assumption_id: assumption
        for assumption in plan.environment_assumptions
    }
    violated: list[str] = []
    diagnostic: list[str] = []
    reasons: list[str] = []
    for assumption_id in step.environment_assumption_ids:
        assumption = assumptions.get(assumption_id)
        if assumption is None:
            continue
        actual = _actual_for_assumption(assumption, environment_identity)
        if actual != assumption.expected:
            if assumption.required:
                violated.append(assumption_id)
                reasons.append(
                    f"{assumption_id} expected {assumption.expected} got {actual}"
                )
            else:
                diagnostic.append(assumption_id)
    return violated, diagnostic, "; ".join(reasons)


def evaluate_request_plan(
    plan: RequestPlan,
    *,
    registry: CapabilityRegistry,
    active_claims: StationActiveClaims | None = None,
    claims_valid: bool = False,
    environment_identity: EnvironmentIdentity | None = None,
    knowledge_snapshot: KnowledgeSnapshot | None = None,
    risk_policy: dict[str, object] | None = None,
    evidence_state: dict[str, Any] | None = None,
) -> ReadinessGraph:
    if knowledge_snapshot is not None:
        if active_claims is None:
            active_claims = knowledge_snapshot.active_claims
        if not claims_valid:
            claims_valid = knowledge_snapshot.claims_valid
        if environment_identity is None:
            environment_identity = knowledge_snapshot.environment_identity

    nodes: list[ReadinessNode] = []
    statuses_by_step: dict[str, str] = {}
    outputs_by_step: dict[str, set[str]] = {}
    graph_violated_assumptions: list[str] = []
    graph_diagnostic_assumptions: list[str] = []

    for step in plan.steps:
        violated_assumptions: list[str] = []
        diagnostic_assumptions: list[str] = []
        blocking_dependencies = [
            dep
            for dep in step.depends_on
            if statuses_by_step.get(dep) not in {None, "executable"}
        ]
        if blocking_dependencies:
            status = "blocked_by_dependency"
            reason = "Blocked by non-executable dependency."
        else:
            violated_assumptions, diagnostic_assumptions, assumption_reason = (
                _evaluate_assumptions(step, plan, environment_identity)
            )
            graph_violated_assumptions.extend(violated_assumptions)
            graph_diagnostic_assumptions.extend(diagnostic_assumptions)
            if violated_assumptions:
                status = "environment_assumption_failed"
                reason = f"Environment assumption failed: {assumption_reason}"
            else:
                dependency_outputs: set[str] = set()
                for dep in step.depends_on:
                    if statuses_by_step.get(dep) == "executable":
                        dependency_outputs.update(outputs_by_step.get(dep, set()))
                claims_produced_by_dependency = any(
                    read in dependency_outputs
                    for read in step.memory_reads
                )
                manifest_min_confidence = None
                if risk_policy is not None and step.required_handle is not None:
                    spec = registry.lookup(step.required_handle)
                    policy = (
                        risk_policy.get(spec.safety_class)
                        if spec is not None
                        else None
                    )
                    if isinstance(policy, dict) and "min_confidence" in policy:
                        try:
                            manifest_min_confidence = float(policy["min_confidence"])
                        except (TypeError, ValueError):
                            manifest_min_confidence = None
                claims_status, claims_reason = _claims_status(
                    step,
                    active_claims=active_claims,
                    claims_valid=claims_valid,
                    claims_produced_by_dependency=claims_produced_by_dependency,
                    manifest_min_confidence=manifest_min_confidence,
                )
                if claims_status is not None:
                    status = claims_status
                    reason = claims_reason or ""
                else:
                    status, reason = _status_for_primitive(step, registry)
                    evidence_status, evidence_reason = _evidence_status(
                        step,
                        evidence_state=evidence_state,
                    )
                    if status == "executable" and evidence_status is not None:
                        status = evidence_status
                        reason = evidence_reason or ""

        statuses_by_step[step.step_id] = status
        outputs_by_step[step.step_id] = set(step.outputs or [])
        nodes.append(
            ReadinessNode(
                step_id=step.step_id,
                status=status,
                layer=step.layer,
                operation=step.operation,
                required_handle=step.required_handle,
                reason=reason,
                blocking_dependencies=blocking_dependencies,
                violated_assumption_ids=violated_assumptions if not blocking_dependencies else [],
                diagnostic_assumption_ids=diagnostic_assumptions if not blocking_dependencies else [],
            )
        )

    graph_status = "executable"
    blocking = next((node for node in nodes if node.status != "executable"), None)
    if blocking is not None:
        graph_status = blocking.status

    next_action = _next_action(plan, graph_status, blocking)
    explanation = (
        "RequestPlan is ready."
        if blocking is None
        else f"{blocking.step_id}: {blocking.reason}"
    )
    return ReadinessGraph(
        request_id=plan.request_id,
        nodes=nodes,
        graph_status=graph_status,
        next_action=next_action,
        blocking_step_id=blocking.step_id if blocking is not None else None,
        explanation=explanation,
        violated_assumption_ids=list(dict.fromkeys(graph_violated_assumptions)),
        diagnostic_assumption_ids=list(dict.fromkeys(graph_diagnostic_assumptions)),
    )


def _next_action(
    plan: RequestPlan,
    graph_status: str,
    blocking: ReadinessNode | None,
) -> str:
    if blocking is None:
        if plan.expected_response == "execute_task":
            return "execute_task"
        if plan.expected_response == "execute_motor":
            return "execute_motor"
        if plan.expected_response == "update_memory":
            return "update_memory"
        if plan.expected_response == "answer_query":
            return "answer_query"
        if plan.expected_response == "propose_definition":
            return "propose_definition"
        if plan.expected_response == "refuse":
            return "refuse"
        return "answer_query"
    if graph_status == "needs_clarification":
        return "ask_clarification"
    if graph_status == "needs_evidence":
        return "ask_clarification"
    if graph_status == "needs_authorization":
        return "ask_authorization"
    if graph_status == "validation_required":
        return "run_validation"
    if graph_status == "synthesizable":
        return "propose_synthesis"
    if graph_status in {"stale_claims", "claim_contract_failed"}:
        return "refresh_claims"
    return "refuse"
