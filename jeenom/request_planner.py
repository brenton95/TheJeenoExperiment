from __future__ import annotations

import re
from typing import Any

from .planning_semantics import PlanningSemantics, default_planning_semantics
from .schemas import (
    EnvironmentAssumption,
    EnvironmentIdentity,
    OperatorIntent,
    RequestPlan,
    RequestPlanStep,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# Phase 13A: layers whose steps actuate / execute and therefore carry steering teeth.
_STEERABLE_LAYERS = frozenset({"task", "action"})


def _fold_steering(steps: list[RequestPlanStep], directive: Any) -> None:
    """Fold a SteeringDirective into the constraints of actuating steps (in place).

    Readiness reads `steering_risk` to gate side effects; the Spine loop reads
    `steering_budget` to cap execution. Mirrors how SelectionObjective fields land in
    RequestPlanStep.constraints — typed values, no vocabulary scanning downstream.
    """
    if directive is None:
        return
    for step in steps:
        if step.layer not in _STEERABLE_LAYERS:
            continue
        if directive.risk is not None:
            step.constraints["steering_risk"] = directive.risk
        if directive.budget:
            step.constraints["steering_budget"] = dict(directive.budget)
        if directive.stopping_rule is not None:
            step.constraints["steering_stopping_rule"] = directive.stopping_rule
        if directive.scope is not None:
            step.constraints["steering_scope"] = directive.scope


def _objective_type(intent: OperatorIntent) -> str:
    if intent.intent_type in {"task_instruction", "conditional_sense_motor"}:
        return "task"
    if intent.intent_type == "motor_command":
        return "motor_control"
    if intent.intent_type == "knowledge_update":
        return "knowledge_update"
    if intent.intent_type in {"status_query", "claim_reference", "cache_query"}:
        return "query"
    if intent.intent_type == "primitive_definition":
        return "primitive_definition"
    if intent.intent_type in {"reset", "quit", "accept_proposal", "reject_proposal"}:
        return "control"
    return "unsupported"


def _expected_response(intent: OperatorIntent) -> str:
    if intent.capability_status == "needs_clarification":
        return "ask_clarification"
    if intent.capability_status == "synthesizable":
        return "propose_synthesis"
    if intent.intent_type == "primitive_definition":
        return "propose_definition"
    if intent.intent_type in {"task_instruction", "conditional_sense_motor"}:
        return "execute_task"
    if intent.intent_type == "motor_command":
        return "execute_motor"
    if intent.intent_type == "knowledge_update":
        return "update_memory"
    if intent.intent_type in {"status_query", "claim_reference", "cache_query"}:
        return "answer_query"
    return "refuse"


def _comparison_from_text_or_plan(text: str, plan: dict[str, Any] | None) -> str | None:
    if plan is not None:
        c = plan.get("comparison")
        return str(c) if c else None
    normalized = _normalize(text)
    if "at least" in normalized:
        return "at_least"
    if "at most" in normalized or "within" in normalized:
        return "at_most"
    if "above" in normalized or "greater than" in normalized or "over " in normalized:
        return "above"
    if "below" in normalized or "less than" in normalized or "under " in normalized:
        return "below"
    return None


def _distance_value_from_text_or_plan(text: str, plan: dict[str, Any] | None) -> int | None:
    if plan is not None:
        dv = plan.get("distance_value")
        return int(dv) if dv is not None else None
    match = re.search(r"\b(\d+)\b", text)
    if match is None:
        return None
    return int(match.group(1))


def build_environment_assumptions(
    environment_identity: EnvironmentIdentity | None,
) -> list[EnvironmentAssumption]:
    if environment_identity is None:
        return []
    assumptions: list[EnvironmentAssumption] = []
    if environment_identity.env_id is not None:
        assumptions.append(
            EnvironmentAssumption(
                assumption_id="env.env_id",
                kind="env_id",
                expected={"env_id": environment_identity.env_id},
                required=True,
                description="Plan was recorded for this environment id.",
            )
        )
    if (
        environment_identity.grid_width is not None
        and environment_identity.grid_height is not None
    ):
        assumptions.append(
            EnvironmentAssumption(
                assumption_id="env.grid_size",
                kind="grid_size",
                expected={
                    "grid_width": environment_identity.grid_width,
                    "grid_height": environment_identity.grid_height,
                },
                required=True,
                description="Plan assumes this MiniGrid size.",
            )
        )
    if environment_identity.task_family is not None:
        assumptions.append(
            EnvironmentAssumption(
                assumption_id="env.task_family",
                kind="task_family",
                expected={"task_family": environment_identity.task_family},
                required=True,
                description="Plan assumes this task family.",
            )
        )
    if environment_identity.substrate_fingerprint is not None:
        assumptions.append(
            EnvironmentAssumption(
                assumption_id="env.substrate_fingerprint",
                kind="substrate_fingerprint",
                expected={
                    "substrate_fingerprint": environment_identity.substrate_fingerprint,
                },
                required=True,
                description=(
                    "Plan assumes this substrate/controller/tool/calibration fingerprint."
                ),
            )
        )
    assumptions.append(
        EnvironmentAssumption(
            assumption_id="env.seed",
            kind="seed",
            expected={"seed": environment_identity.seed},
            required=False,
            description="Diagnostic seed captured when the plan was recorded.",
        )
    )
    assumptions.append(
        EnvironmentAssumption(
            assumption_id="env.fingerprint",
            kind="environment_fingerprint",
            expected={"fingerprint": environment_identity.fingerprint()},
            required=False,
            description="Diagnostic stable environment fingerprint.",
        )
    )
    assumptions.append(
        EnvironmentAssumption(
            assumption_id="env.layout_summary",
            kind="layout_summary",
            expected={"summary": dict(environment_identity.summary)},
            required=False,
            description="Diagnostic object/layout summary captured with the plan.",
        )
    )
    return assumptions


def _assumption_ids_for_environment(
    assumptions: list[EnvironmentAssumption],
) -> list[str]:
    return [assumption.assumption_id for assumption in assumptions]


def _operation_from_query_plan(plan: dict[str, Any] | None) -> str:
    if plan is None:
        return "answer"
    operation = str(plan.get("operation") or "answer")
    if operation in {"list", "rank"}:
        return "rank"
    if operation in {"filter", "select", "answer"}:
        return operation
    return "answer"


def build_request_plan(
    utterance: str,
    intent: OperatorIntent,
    *,
    active_claims_summary: dict[str, Any] | None = None,
    environment_identity: EnvironmentIdentity | None = None,
    planning_semantics: PlanningSemantics | None = None,
) -> RequestPlan:
    """Build a typed, dependency-aware request plan from validated operator intent.

    This is intentionally side-effect free. The station and readiness graph decide
    whether the plan can execute, clarify, synthesize, or refuse.
    """

    request_id = f"request:{abs(hash((utterance, intent.intent_type))) % 1_000_000}"
    objective_type = _objective_type(intent)
    expected_response = _expected_response(intent)
    steps: list[RequestPlanStep] = []
    semantics = planning_semantics or default_planning_semantics()
    environment_assumptions = build_environment_assumptions(environment_identity)
    environment_assumption_ids = _assumption_ids_for_environment(environment_assumptions)
    plan = intent.grounding_query_plan
    metric = semantics.metric_from_text_or_plan(utterance, plan)
    comparison = _comparison_from_text_or_plan(utterance, plan)
    distance_value = _distance_value_from_text_or_plan(utterance, plan)
    ranked_claims_output = semantics.ranked_claims_output
    steering_directive = getattr(intent, "steering_directive", None)
    steering_payload = steering_directive.as_dict() if steering_directive is not None else None

    if objective_type == "motor_control":
        steps.append(
            RequestPlanStep(
                step_id="execute_raw_motor",
                layer="action",
                operation="execute",
                inputs={
                    "action_name": intent.action_name,
                    "repeat_count": intent.repeat_count or 1,
                },
                outputs=["motor_result"],
                constraints={"raw_motor": True},
            )
        )
        _fold_steering(steps, steering_directive)
        return RequestPlan(
            request_id=request_id,
            original_utterance=utterance,
            objective_type=objective_type,
            objective_summary=intent.reason or "Explicit low-level motor command.",
            steps=steps,
            preservation_signals=semantics.preservation_signals(utterance),
            expected_response=expected_response,
            steering=steering_payload,
        )

    if intent.intent_type == "conditional_sense_motor":
        target = dict(intent.target or {})
        action_name = str(intent.action_name or "")
        steps.extend(
            [
                RequestPlanStep(
                    step_id="sense_stop_condition",
                    layer="sensing",
                    operation="execute",
                    required_handle="sensing.find_object_by_color_type",
                    inputs={"target": target},
                    outputs=["target_visible", "target_location", "target_object"],
                    constraints={
                        "fresh_evidence_required": True,
                        "evidence_scope": "visible_only",
                    },
                    environment_assumption_ids=environment_assumption_ids,
                ),
                RequestPlanStep(
                    step_id="execute_conditional_action",
                    layer="action",
                    operation="execute",
                    required_handle=f"action.{action_name}",
                    inputs={"action_name": action_name},
                    outputs=["execution_report"],
                    depends_on=["sense_stop_condition"],
                    constraints={
                        "stop_claim": "target_visible",
                        "stop_value": True,
                    },
                    environment_assumption_ids=environment_assumption_ids,
                ),
                RequestPlanStep(
                    step_id="execute_conditional_mission",
                    layer="task",
                    operation="execute",
                    required_handle="task.act_until_evidence",
                    inputs={
                        "target": target,
                        "action_name": action_name,
                        "stop_claim": "target_visible",
                        "stop_value": True,
                    },
                    outputs=["task_result"],
                    depends_on=[
                        "sense_stop_condition",
                        "execute_conditional_action",
                    ],
                    memory_writes=[
                        "episodic.last_request_plan",
                        "episodic.last_task",
                        "episodic.last_result",
                    ],
                    environment_assumption_ids=environment_assumption_ids,
                ),
            ]
        )
        _fold_steering(steps, steering_directive)
        return RequestPlan(
            request_id=request_id,
            original_utterance=utterance,
            objective_type=objective_type,
            objective_summary=intent.reason or "Conditional evidence mission.",
            steps=steps,
            preservation_signals=semantics.preservation_signals(utterance),
            expected_response=expected_response,
            environment_assumptions=environment_assumptions,
            steering=steering_payload,
        )

    if objective_type == "control":
        steps.append(
            RequestPlanStep(
                step_id="control",
                layer="control",
                operation="reset" if intent.intent_type == "reset" else "answer",
                outputs=["control_response"],
            )
        )
        return RequestPlan(
            request_id=request_id,
            original_utterance=utterance,
            objective_type=objective_type,
            objective_summary=intent.reason or intent.intent_type,
            steps=steps,
            preservation_signals=semantics.preservation_signals(utterance),
            expected_response=expected_response,
        )

    if objective_type == "primitive_definition":
        definition = intent.primitive_definition
        if definition is not None:
            for idx, handle in enumerate(definition.dependency_handles):
                metric = (
                    definition.dependencies[idx]
                    if idx < len(definition.dependencies)
                    else handle
                )
                steps.append(
                    RequestPlanStep(
                        step_id=f"dependency_{idx + 1}",
                        layer="grounding",
                        operation="rank",
                        required_handle=handle,
                        inputs={"object_type": semantics.default_object_type},
                        outputs=[ranked_claims_output],
                        constraints={
                            "metric": metric,
                            "definition_dependency": True,
                        },
                        environment_assumption_ids=environment_assumption_ids,
                    )
                )
            steps.append(
                RequestPlanStep(
                    step_id="propose_primitive_definition",
                    layer="control",
                    operation="answer",
                    outputs=["operator_response"],
                    depends_on=[step.step_id for step in steps],
                    constraints={
                        "definition_type": definition.definition_type,
                        "proposed_handle": definition.proposed_handle,
                        "safety_class": definition.safety_class,
                    },
                    environment_assumption_ids=environment_assumption_ids,
                )
            )
        return RequestPlan(
            request_id=request_id,
            original_utterance=utterance,
            objective_type=objective_type,
            objective_summary=(
                intent.reason
                or (
                    f"Define primitive {intent.primitive_definition.proposed_handle}"
                    if intent.primitive_definition is not None
                    else "Define primitive"
                )
            ),
            steps=steps,
            preservation_signals=semantics.preservation_signals(utterance),
            expected_response=expected_response,
            environment_assumptions=environment_assumptions,
        )

    if objective_type == "unsupported" and intent.required_capabilities:
        for idx, handle in enumerate(intent.required_capabilities):
            layer = handle.split(".", 1)[0] if "." in handle else "control"
            steps.append(
                RequestPlanStep(
                    step_id=f"unsupported_capability_{idx + 1}",
                    layer=layer,
                    operation="refuse",
                    required_handle=handle,
                    outputs=["operator_response"],
                    environment_assumption_ids=environment_assumption_ids,
                )
            )

    if objective_type == "knowledge_update":
        steps.append(
            RequestPlanStep(
                step_id="update_knowledge",
                layer="memory",
                operation="update",
                inputs={"knowledge_update": intent.knowledge_update or {}},
                outputs=["durable_knowledge"],
                memory_writes=["knowledge.delivery_target"],
            )
        )

    if intent.reference == "delivery_target":
        steps.append(
            RequestPlanStep(
                step_id="read_delivery_target",
                layer="memory",
                operation="read",
                outputs=["target_fact"],
                memory_reads=["knowledge.delivery_target"],
            )
        )
    if intent.reference in {"last_target", "last_task"}:
        steps.append(
            RequestPlanStep(
                step_id=f"read_{intent.reference}",
                layer="memory",
                operation="read",
                outputs=[intent.reference],
                memory_reads=[f"episodic.{intent.reference}"],
            )
        )

    ranked_handle = None
    rank_step_id: str | None = None
    if plan is not None:
        ranked_handle = plan.get("primitive_handle")
        required = list(plan.get("required_capabilities") or [])
        if ranked_handle is None:
            ranked_handle = next(
                (handle for handle in required if ".ranked." in handle),
                semantics.ranked_handle(metric, object_type=plan.get("object_type")),
            )
        if plan.get("operation") in {"rank", "list", "answer", "filter", "select"}:
            _obj_type = plan.get("object_type") or semantics.default_object_type or "object"
            rank_step_id = f"rank_scene_{semantics.pluralize(_obj_type)}"
            steps.append(
                RequestPlanStep(
                    step_id=rank_step_id,
                    layer="grounding",
                    operation="rank",
                    required_handle=ranked_handle,
                    inputs={"object_type": _obj_type},
                    outputs=[ranked_claims_output],
                    constraints={
                        "metric": metric,
                        "reference": plan.get("reference") or "agent",
                        "evidence_scope": "visible_only",
                        "requires_visible_objects": [_obj_type],
                    },
                    environment_assumption_ids=environment_assumption_ids,
                )
            )
    elif intent.target_selector is not None:
        handle = semantics.target_handle(intent)
        selector_object_type = (
            intent.target_selector.get("object_type")
            if isinstance(intent.target_selector, dict)
            else None
        )
        steps.append(
            RequestPlanStep(
                step_id="ground_target",
                layer="grounding",
                operation="ground",
                required_handle=handle,
                inputs={"target_selector": intent.target_selector},
                outputs=["grounded_target"],
                constraints={
                    **dict(intent.target_selector),
                    "evidence_scope": "visible_only",
                    "requires_visible_objects": [
                        selector_object_type or semantics.default_object_type
                    ],
                },
                environment_assumption_ids=environment_assumption_ids,
            )
        )

    if plan is not None and distance_value is not None:
        depends = [rank_step_id] if rank_step_id is not None else []
        steps.append(
            RequestPlanStep(
                step_id="filter_distance_threshold",
                layer="claims",
                operation="filter",
                required_handle=semantics.filter_handle(metric),
                inputs={"entries": ranked_claims_output},
                outputs=["filtered_candidates"],
                depends_on=depends,
                constraints={
                    "metric": metric,
                    "comparison": comparison,
                    "threshold": distance_value,
                },
                memory_reads=[ranked_claims_output],
                scene_fingerprint_required=True,
                environment_assumption_ids=environment_assumption_ids,
            )
        )

    if plan is not None and plan.get("operation") not in {"rank", "list"} and (
        plan.get("ordinal") is not None
        or plan.get("order") is not None
        or any(str(field).endswith(("closest", "farthest")) for field in plan.get("answer_fields", []))
    ):
        source = (
            "filtered_candidates"
            if any(s.step_id == "filter_distance_threshold" for s in steps)
            else ranked_claims_output
        )
        depends = [
            steps[-1].step_id
        ] if steps else []
        steps.append(
            RequestPlanStep(
                step_id="select_grounded_target",
                layer="claims",
                operation="select",
                inputs={"entries": source},
                outputs=["grounded_target"],
                depends_on=depends,
                constraints={
                    "order": plan.get("order"),
                    "ordinal": plan.get("ordinal"),
                    "answer_fields": list(plan.get("answer_fields") or []),
                    "metric": metric,
                },
                tie_policy=plan.get("tie_policy") or "clarify",
                memory_reads=[source],
                scene_fingerprint_required=True,
                environment_assumption_ids=environment_assumption_ids,
            )
        )

    if objective_type == "task":
        task_dep = []
        if any(s.outputs and "grounded_target" in s.outputs for s in steps):
            task_dep = [steps[-1].step_id]
        target = intent.target or {}
        object_type = target.get("object_type") or (intent.target_selector or {}).get("object_type")
        if object_type is None and intent.task_type == "go_to_object":
            object_type = semantics.default_object_type
        task_handle = semantics.task_handle(intent.task_type, object_type)
        steps.append(
            RequestPlanStep(
                step_id="execute_task",
                layer="task",
                operation="execute",
                required_handle=task_handle,
                inputs={"task_type": intent.task_type, "object_type": object_type},
                outputs=["task_result"],
                depends_on=task_dep,
                memory_writes=[
                    "episodic.last_request_plan",
                    "episodic.last_grounded_target",
                    "episodic.last_task",
                    "episodic.last_result",
                ],
                environment_assumption_ids=environment_assumption_ids,
            )
        )

    if objective_type == "query":
        if not steps and intent.status_query in {"status", "help", "cache", "scene"}:
            steps.append(
                RequestPlanStep(
                    step_id=f"answer_{intent.status_query or 'status'}",
                    layer="answer",
                    operation="answer",
                    outputs=["operator_response"],
                    memory_reads=["knowledge.delivery_target", "episodic.last_result"],
                )
            )
        else:
            depends = [steps[-1].step_id] if steps else []
            steps.append(
                RequestPlanStep(
                    step_id="answer_query",
                    layer="answer",
                    operation="answer",
                    outputs=["operator_response"],
                    depends_on=depends,
                    memory_reads=[ranked_claims_output]
                    if active_claims_summary is not None
                    else [],
                )
            )

    if not steps:
        steps.append(
            RequestPlanStep(
                step_id="refuse",
                layer="control",
                operation="refuse",
                outputs=["operator_response"],
            )
        )

    _fold_steering(steps, steering_directive)
    return RequestPlan(
        request_id=request_id,
        original_utterance=utterance,
        objective_type=objective_type,
        objective_summary=intent.reason or intent.canonical_instruction or intent.intent_type,
        steps=steps,
        preservation_signals=semantics.preservation_signals(utterance),
        expected_response=expected_response,
        environment_assumptions=environment_assumptions,
        steering=steering_payload,
    )
