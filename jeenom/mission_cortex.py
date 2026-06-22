from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import fingerprint as _fp
from .capability_registry import CapabilityRegistry
from .planning_semantics import PlanningSemantics, default_planning_semantics
from .readiness_graph import evaluate_request_plan
from .request_planner import build_request_plan
from .schemas import (
    MissionContract,
    MissionExecutionPlan,
    OperatorIntent,
    ProcedureRecipe,
    PrimitiveDefinitionRequest,
    ReadinessGraph,
    RequestPlan,
    SelectionObjective,
    StationActiveClaims,
)
from .semantic_normalizer import infer_direction_from_utterance, normalize_distance_ordinal


_ACTION_LEAK_TERMS = (
    "move",
    "moves",
    "moving",
    "turn",
    "turns",
    "pickup",
    "pick up",
    "grab",
    "toggle",
    "open",
    "unlock",
    "navigate",
    "go forward",
    "env step",
    "env.step",
    "controller",
    "motor",
    "actuate",
)


@dataclass(frozen=True)
class InlineMetricMissionRequest:
    """Typed parse result for an operator-defined metric embedded in a mission."""

    mission_id: str
    mission_contract: MissionContract
    primitive_definition: PrimitiveDefinitionRequest
    continuation_intent: OperatorIntent
    provenance: dict[str, Any]


def _normalize_utterance(utterance: str) -> str:
    text = " ".join(utterance.lower().strip().split())
    text = re.sub(r"[?!]+$", "", text)
    text = re.sub(r"[.,;:]+", " ", text)
    text = " ".join(text.split())
    while True:
        stripped = re.sub(
            r"^(?:ok|okay|alright|right|so|well|now|first|then)\s+",
            "",
            text,
        )
        if stripped == text:
            return text
        text = stripped


def _normalize_metric_name(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _metric_dependencies(formula: str, known_metrics: list[str]) -> list[str]:
    """Scan formula for known metric names. known_metrics comes from the registry."""
    normalized = _normalize_utterance(formula)
    dependencies: list[str] = []
    for metric in known_metrics:
        if re.search(rf"\b{re.escape(metric)}\b", normalized) and metric not in dependencies:
            dependencies.append(metric)
    if (
        not dependencies
        and re.search(r"\bboth\s+(?:distance\s+)?metrics?\b", normalized)
    ):
        dependencies.extend(known_metrics)
    return dependencies


def _parse_metric_expression(formula: str, known_metrics: list[str]) -> dict[str, Any] | None:
    normalized = _normalize_utterance(formula)
    dependencies = _metric_dependencies(formula, known_metrics)
    if any(term in normalized for term in _ACTION_LEAK_TERMS):
        return {
            "op": "unsafe",
            "reason": "Metric definitions must be query-only and cannot include actuation.",
            "metrics": dependencies,
        }
    if not dependencies:
        return None

    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", normalized)
    constant = float(number_match.group(1)) if number_match else None

    if len(dependencies) >= 2 and (
        "sum" in normalized
        or "total" in normalized
        or "combined" in normalized
        or "plus" in normalized
        or "+" in formula
    ):
        return {"op": "sum", "metrics": dependencies}
    if "minimum" in normalized or "min(" in normalized or "min of" in normalized:
        return {"op": "min", "metrics": dependencies}
    if "maximum" in normalized or "max(" in normalized or "max of" in normalized:
        return {"op": "max", "metrics": dependencies}
    if "mod" in normalized or "modulo" in normalized:
        if constant is None:
            return None
        return {"op": "mod", "metric": dependencies[0], "constant": constant}
    if "abs" in normalized and "-" in formula and len(dependencies) >= 2:
        expression: dict[str, Any] = {
            "op": "abs_diff",
            "metrics": dependencies[:2],
        }
        if constant is not None and ("+" in formula or " plus " in normalized):
            expression["op"] = "abs_diff_plus"
            expression["constant"] = constant
        return expression
    if " plus " in normalized or "+" in formula:
        if constant is None:
            return None
        return {"op": "add", "metric": dependencies[0], "constant": constant}
    if " minus " in normalized or "-" in formula:
        if constant is None:
            return None
        return {"op": "subtract", "metric": dependencies[0], "constant": constant}
    if len(dependencies) == 1:
        return {"op": "alias", "metric": dependencies[0]}
    return None


def _metric_expression_name(expression: dict[str, Any]) -> str:
    op = str(expression.get("op") or "metric")
    if op in {"sum", "min", "max"}:
        metrics = [
            _normalize_metric_name(str(metric))
            for metric in expression.get("metrics", [])
        ]
        return _normalize_metric_name("_".join([op, *metrics]))
    if op in {"alias", "mod", "add", "subtract"}:
        metric = _normalize_metric_name(str(expression.get("metric") or "metric"))
        suffix = ""
        if expression.get("constant") is not None:
            suffix = "_" + str(expression["constant"]).replace(".", "_")
        return _normalize_metric_name(f"{op}_{metric}{suffix}")
    if op in {"abs_diff", "abs_diff_plus"}:
        metrics = [
            _normalize_metric_name(str(metric))
            for metric in expression.get("metrics", [])
        ]
        return _normalize_metric_name("_".join([op, *metrics]))
    return _normalize_metric_name(op)


def _mission_id_for(utterance: str, handle: str) -> str:
    return f"mission:{_fp.stable_hash(f'{utterance}|{handle}', length=12)}"


def _continuation_intent(
    *,
    mission_id: str,
    utterance: str,
    request: PrimitiveDefinitionRequest,
    expression: dict[str, Any],
    planning_semantics: PlanningSemantics,
) -> OperatorIntent | str:
    normalized = _normalize_utterance(utterance)
    object_type = (
        planning_semantics.object_type_from_text(normalized)
        or planning_semantics.default_object_type
    )
    if object_type is None:
        return "The operational context does not declare an object type for this mission."
    object_type_plural = planning_semantics.pluralize(object_type)
    wants_task = bool(
        re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
    )
    ordinal_semantics = normalize_distance_ordinal(normalized)
    direction = (
        ordinal_semantics.order
        if ordinal_semantics is not None
        else infer_direction_from_utterance(normalized)
    )
    ordinal = ordinal_semantics.ordinal if ordinal_semantics is not None else None
    if direction is not None and ordinal is None:
        ordinal = 1
    is_rank_query = bool(
        re.search(r"\b(rank|list|show)\b", normalized)
        or re.search(
            rf"\b(all|every|each)\b.{{0,30}}\b"
            rf"(?:{re.escape(object_type)}|{re.escape(object_type_plural)})\b",
            normalized,
        )
    )
    if wants_task and direction is None:
        return (
            "I can derive that metric, but the task does not say how to select "
            "a target from it. Use wording like 'third farthest' or 'closest'."
        )
    operation = "select" if wants_task else "rank" if is_rank_query and direction is None else "answer"
    required = [request.proposed_handle]
    if wants_task:
        task_handle = planning_semantics.task_handle("go_to_object", object_type)
        if task_handle is None:
            return f"No go_to_object task handle is declared for {object_type}."
        required.append(task_handle)
    answer_fields = (
        ["ranked_doors", "distance"]
        if operation in {"rank", "list"}
        else ["target", "distance"]
    )
    selection_objective = None
    if order := direction:
        if ordinal is not None:
            selection_objective = SelectionObjective(
                attribute="distance",
                direction="maximum" if order == "descending" else "minimum",
                ordinal=int(ordinal),
                metric=request.normalized_name,
            )
    return OperatorIntent(
        intent_type="task_instruction" if wants_task else "status_query",
        status_query=None if wants_task else "ground_target",
        task_type="go_to_object" if wants_task else None,
        grounding_query_plan={
            "object_type": object_type,
            "operation": operation,
            "primitive_handle": request.proposed_handle,
            "metric": request.normalized_name,
            "reference": "agent",
            "order": direction,
            "ordinal": ordinal,
            "color": None,
            "exclude_colors": [],
            "distance_value": None,
            "comparison": None,
            "tie_policy": "first" if wants_task else "display",
            "answer_fields": answer_fields,
            "required_capabilities": required,
            "preserved_constraints": [
                "inline_metric",
                request.normalized_name,
                *request.dependencies,
            ],
            "metric_dependencies": list(request.dependencies),
            "derived_metric": True,
            "mission_id": mission_id,
        },
        primitive_definition=request,
        capability_status="executable",
        required_capabilities=required,
        selection_objective=selection_objective,
        confidence=1.0,
        reason=(
            f"MissionCortex continuation for mission_id={mission_id}; "
            f"approved inline derived metric {request.normalized_name}."
        ),
    )


def parse_inline_metric_request(
    text: str,
    registry: CapabilityRegistry,
    *,
    planning_semantics: PlanningSemantics | None = None,
) -> InlineMetricMissionRequest | str | None:
    semantics = planning_semantics or default_planning_semantics()
    normalized = _normalize_utterance(text)
    # Gate on navigation/query intent — no substrate primitive name check here.
    if not (
        re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
        or re.search(r"\b(rank|list|show|what|which|find)\b", normalized)
    ):
        return None

    # "create X = formula and go to the farthest X door" — inline definition + task.
    _def_task_m = re.match(
        r"^(?:create|define|make|build)?\s*(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<formula>.+?)\s+(?:and\s+)?(?:go to|go the|reach|get to|head to|navigate to)\b",
        text.strip(),
        re.IGNORECASE,
    )
    if _def_task_m:
        ranked_handles = registry.ranked_metric_handles()
        known_metrics = list(ranked_handles.keys())
        _def_formula = _def_task_m.group("formula").strip()
        _def_name = _def_task_m.group("name")
        expression = _parse_metric_expression(_def_formula, known_metrics)
        if expression is not None and expression.get("op") not in {None, "unsafe", "alias"}:
            if expression.get("op") == "unsafe":
                return (
                    "REFUSE\n"
                    "Metric definitions must be query-only. I will not build a metric "
                    "that contains actuation, movement, controller, or motor side effects."
                )
            normalized_name = _metric_expression_name(expression)
            dependencies = list(dict.fromkeys(
                expression.get("metrics") or _metric_dependencies(_def_formula, known_metrics)
            ))
            request = PrimitiveDefinitionRequest(
                definition_type="distance_metric",
                name=_def_name,
                normalized_name=normalized_name,
                expression=expression,
                dependencies=dependencies,
                dependency_handles=[registry.ranked_handle_for(metric) for metric in dependencies],
                proposed_handle=registry.ranked_handle_for(normalized_name),
                safety_class="query",
                authority_level="operator",
                provenance={"operator_utterance": text, "formula": _def_formula, "inline_request": True},
            )
            mission_id = _mission_id_for(text, request.proposed_handle)
            continuation = _continuation_intent(
                mission_id=mission_id, utterance=text, request=request, expression=expression,
                planning_semantics=semantics,
            )
            if isinstance(continuation, str):
                return continuation
            contract = MissionContract(
                mission_id=mission_id,
                description=f"Compound operator mission: {text}",
                task_sequence=[text],
                success_condition="approved_metric_then_original_request_complete",
                abort_on_failure=True,
            )
            return InlineMetricMissionRequest(
                mission_id=mission_id,
                primitive_definition=request,
                continuation_intent=continuation,
                mission_contract=contract,
                provenance={
                    "original_utterance": text,
                    "formula": _def_formula,
                    "expression": dict(expression),
                    "primitive_handle": request.proposed_handle,
                },
            )

    formula_match = re.search(
        r"\b(?:based on|according to|using|by)\s+(?:the\s+)?(?P<formula>.+)$",
        text.strip(),
        re.IGNORECASE,
    )
    if formula_match is None:
        return None
    formula = formula_match.group("formula").strip()

    ranked_handles = registry.ranked_metric_handles()
    known_metrics = list(ranked_handles.keys())

    # "their sum/max/min" — resolve metric deps from full utterance context
    _their_op_m = re.match(
        r"^their\s+(sum|max|min|maximum|minimum|total)$",
        formula.lower(),
    )
    if _their_op_m:
        op_word = _their_op_m.group(1)
        op = {
            "sum": "sum", "total": "sum",
            "max": "max", "maximum": "max",
            "min": "min", "minimum": "min",
        }[op_word]
        deps = _metric_dependencies(normalized, known_metrics)
        expression: dict[str, Any] | None = (
            {"op": op, "metrics": deps} if len(deps) >= 2 else None
        )
    else:
        expression = _parse_metric_expression(formula, known_metrics)

    if expression is None:
        return None
    if expression.get("op") == "alias":
        return None
    if expression.get("op") == "unsafe":
        return (
            "REFUSE\n"
            "Metric definitions must be query-only. I will not build a metric "
            "that contains actuation, movement, controller, or motor side effects."
        )

    dependencies = list(dict.fromkeys(
        expression.get("metrics") or _metric_dependencies(formula, known_metrics)
    ))
    normalized_name = _metric_expression_name(expression)
    request = PrimitiveDefinitionRequest(
        definition_type="distance_metric",
        name=normalized_name,
        normalized_name=normalized_name,
        expression=expression,
        dependencies=dependencies,
        dependency_handles=[registry.ranked_handle_for(metric) for metric in dependencies],
        proposed_handle=registry.ranked_handle_for(normalized_name),
        safety_class="query",
        authority_level="operator",
        provenance={
            "operator_utterance": text,
            "formula": formula,
            "inline_request": True,
        },
    )
    mission_id = _mission_id_for(text, request.proposed_handle)
    continuation = _continuation_intent(
        mission_id=mission_id,
        utterance=text,
        request=request,
        expression=expression,
        planning_semantics=semantics,
    )
    if isinstance(continuation, str):
        return continuation
    contract = MissionContract(
        mission_id=mission_id,
        description=f"Compound operator mission: {text}",
        task_sequence=[text],
        success_condition="approved_metric_then_original_request_complete",
        abort_on_failure=True,
    )
    return InlineMetricMissionRequest(
        mission_id=mission_id,
        mission_contract=contract,
        primitive_definition=request,
        continuation_intent=continuation,
        provenance={
            "original_utterance": text,
            "formula": formula,
            "expression": dict(expression),
            "primitive_handle": request.proposed_handle,
        },
    )


class MissionCortex:
    """Cortex-owned control-plane helper for compound operator missions."""

    def __init__(
        self,
        *,
        planning_semantics: PlanningSemantics,
        registry: CapabilityRegistry,
    ) -> None:
        self.planning_semantics = planning_semantics
        self.registry = registry

    def _definition_intent(
        self,
        request: PrimitiveDefinitionRequest,
        *,
        mission_id: str | None = None,
    ) -> OperatorIntent:
        reason = f"Define primitive {request.proposed_handle}."
        if mission_id is not None:
            reason = f"MissionCortex definition step for mission_id={mission_id}: {reason}"
        return OperatorIntent(
            intent_type="primitive_definition",
            primitive_definition=request,
            capability_status="executable",
            required_capabilities=list(request.dependency_handles),
            confidence=1.0,
            reason=reason,
        )

    def plan_primitive_definition(
        self,
        request: PrimitiveDefinitionRequest,
        *,
        utterance: str,
        active_claims: StationActiveClaims | None,
        claims_valid: bool,
        environment_identity: Any,
        mission_id: str | None = None,
    ) -> tuple[RequestPlan, ReadinessGraph]:
        intent = self._definition_intent(request, mission_id=mission_id)
        active_summary = (
            active_claims.compact_summary()
            if active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
        return plan, graph

    def plan_conditional_evidence_action(
        self,
        intent: OperatorIntent,
        *,
        utterance: str,
        active_claims: StationActiveClaims | None,
        claims_valid: bool,
        environment_identity: Any,
    ) -> MissionExecutionPlan:
        """Build the typed mission that binds one Sense primitive to one Spine command."""
        if intent.intent_type != "conditional_sense_motor":
            raise ValueError("Conditional evidence planning requires conditional_sense_motor")
        target = dict(intent.target or {})
        action_name = str(intent.action_name or "")
        if not target.get("color") or not target.get("object_type") or not action_name:
            raise ValueError("Conditional evidence mission requires target and action")

        procedure = ProcedureRecipe(
            task_type="act_until_evidence",
            steps=["act_until_evidence"],
            source="mission_cortex",
            compiler_backend="deterministic",
            validated=True,
            rationale=(
                "Sense target visibility, let Cortex evaluate first_match, and issue "
                "one approved Spine action only while the condition is false."
            ),
        )
        params = {
            "color": target["color"],
            "object_type": target["object_type"],
            "action_name": action_name,
            "stop_claim": "target_visible",
            "stop_value": True,
        }
        required = [
            "sensing.find_object_by_color_type",
            f"action.{action_name}",
            "task.act_until_evidence",
        ]
        mission_id = _mission_id_for(utterance, "|".join(required))
        contract = MissionContract(
            mission_id=mission_id,
            description=utterance,
            task_sequence=[utterance],
            success_condition="target_visible",
            abort_on_failure=True,
            risk_tier="operator_authorized",
            cadence="control",
            procedure=procedure,
            params=params,
            required_capabilities=required,
        )
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=None,
            environment_identity=environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
        return MissionExecutionPlan(
            mission_id=mission_id,
            description=utterance,
            request_plan=plan,
            readiness_graph=graph,
            mission_contract=contract,
            provenance={
                "original_utterance": utterance,
                "sense_primitive": "sensing.find_object_by_color_type",
                "action_primitive": f"action.{action_name}",
                "stop_claim": "target_visible",
            },
        )

    def plan_inline_metric_request(
        self,
        mission_request: InlineMetricMissionRequest,
        *,
        active_claims: StationActiveClaims | None,
        claims_valid: bool,
        environment_identity: Any,
    ) -> MissionExecutionPlan:
        utterance = mission_request.provenance["original_utterance"]
        plan, graph = self.plan_primitive_definition(
            mission_request.primitive_definition,
            utterance=utterance,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
            mission_id=mission_request.mission_id,
        )
        # Pre-build the continuation plan from the structured continuation_intent so
        # that resume_after_approval can reuse it rather than reconstructing from the
        # original utterance text.  The registry doesn't have the new primitive yet so
        # the readiness graph will be non-executable here — that is expected and
        # corrected in resume_after_approval after registration.
        active_summary = (
            active_claims.compact_summary()
            if active_claims is not None and claims_valid
            else None
        )
        continuation_intent = mission_request.continuation_intent
        cont_plan = build_request_plan(
            utterance,
            continuation_intent,
            active_claims_summary=active_summary,
            environment_identity=environment_identity,
            planning_semantics=self.planning_semantics,
        )
        cont_graph = evaluate_request_plan(
            cont_plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
        return MissionExecutionPlan(
            mission_id=mission_request.mission_id,
            description=mission_request.mission_contract.description,
            request_plan=plan,
            readiness_graph=graph,
            mission_contract=mission_request.mission_contract,
            primitive_definition=mission_request.primitive_definition,
            continuation_intent=continuation_intent,
            continuation_request_plan=cont_plan,
            continuation_readiness_graph=cont_graph,
            provenance=dict(mission_request.provenance),
        )

    def resume_after_approval(
        self,
        mission_plan: MissionExecutionPlan,
        *,
        active_claims: StationActiveClaims | None,
        claims_valid: bool,
        environment_identity: Any,
    ) -> MissionExecutionPlan:
        if mission_plan.continuation_intent is None:
            raise ValueError("MissionExecutionPlan requires continuation_intent")
        if mission_plan.continuation_request_plan is not None:
            # Reuse the pre-built plan structure; only re-evaluate readiness now that
            # the primitive has been registered and the registry has changed.
            graph = evaluate_request_plan(
                mission_plan.continuation_request_plan,
                registry=self.registry,
                active_claims=active_claims,
                claims_valid=claims_valid,
                environment_identity=environment_identity,
            )
            mission_plan.continuation_readiness_graph = graph
            return mission_plan
        # Fallback: no pre-built plan — reconstruct from the continuation intent.
        active_summary = (
            active_claims.compact_summary()
            if active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            mission_plan.provenance.get("original_utterance", mission_plan.description),
            mission_plan.continuation_intent,
            active_claims_summary=active_summary,
            environment_identity=environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
        mission_plan.continuation_request_plan = plan
        mission_plan.continuation_readiness_graph = graph
        return mission_plan
