from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping

from . import fingerprint as _fp
from . import geometry


SENSE_TEMPLATE_ALLOWED_INPUTS = (
    "observation",
    "mission",
    "direction",
    "color",
    "object_type",
    "target_location",
    "agent_pose",
    "occupancy_grid",
    "grid_objects",
    "known_target_location",
    "execution_context",
)
SENSE_TEMPLATE_ALLOWED_OUTPUTS = (
    "world_sample",
    "operational_evidence",
    "percepts",
)
SKILL_TEMPLATE_ALLOWED_INPUTS = (
    "agent_pose",
    "target_location",
    "occupancy_grid",
    "direction",
    "object_type",
    "color",
    "adjacency_to_target",
    "passable_positions",
    "grid_size",
    "target_object",
    "execution_contract",
    "percepts",
)
SKILL_TEMPLATE_ALLOWED_OUTPUTS = (
    "execution_report",
    "execution_context",
)
OPERATOR_INTENT_TYPES = (
    "task_instruction",
    "knowledge_update",
    "status_query",
    "primitive_definition",
    "claim_reference",
    "cache_query",
    "concept_teach",
    "concept_recall",
    "concept_forget",
    "procedure_recall",
    "sequence_instruction",
    "motor_command",
    "motor_sequence",
    "conditional_sense_motor",
    "mission_contract",
    "metric_query",
    "steering_directive",
    "reset",
    "quit",
    "accept_proposal",
    "reject_proposal",
    "unsupported",
    "ambiguous",
)
OPERATOR_CLAIM_REFERENCES = ("next_closest", "other_door", "threshold_filter")

# Unified claim scope vocabulary.
# All facts held by the station are Claims. They differ in authority, scope, and
# invalidation policy — not in kind.
#   "grounding" — derived from scene observation; session-scoped; scene-fingerprinted;
#                 invalidated when the scene changes (StationActiveClaims).
#   "operator"  — asserted by the operator; durable across restarts; invalidated only
#                 by explicit retraction (KnowledgeBase, OperationalMemory.knowledge).
CLAIM_SCOPES = ("grounding", "operator", "execution", "episodic", "procedure")
CLAIM_KINDS = (
    "fact",
    "belief",
    "hypothesis",
    "operator_assertion",
    "observation",
    "execution",
    "procedure",
)
CLAIM_STATUSES = (
    "confirmed",
    "asserted",
    "observed",
    "inferred",
    "hypothesis",
    "invalidated",
    "unknown",
)
CLAIM_FRESHNESS = ("current", "unverifiable", "stale", "unknown")
CLAIM_AUTHORITIES = ("operator", "runtime", "system", "compiler", "sense", "spine")
GROUNDING_QUERY_COMPARISONS = ("above", "below", "within", "at_least", "at_most")
OPERATOR_TASK_TYPES = ("go_to_object",)
OPERATOR_COLORS = ("red", "green", "blue", "yellow", "purple", "grey")

# Domain vocabulary registry — populated by the domain adapter at init, never hardcoded here.
_REGISTERED_OBJECT_TYPES: tuple[str, ...] | None = None


def register_domain_vocabulary(object_types: tuple[str, ...]) -> None:
    global _REGISTERED_OBJECT_TYPES
    _REGISTERED_OBJECT_TYPES = object_types


def clear_registered_vocabulary() -> None:
    global _REGISTERED_OBJECT_TYPES
    _REGISTERED_OBJECT_TYPES = None


def get_registered_object_types() -> tuple[str, ...]:
    return _REGISTERED_OBJECT_TYPES if _REGISTERED_OBJECT_TYPES is not None else ()


def _validate_object_type(
    value: str | None,
    label: str,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> None:
    if value is None:
        return
    allowed = (
        tuple(object_types)
        if object_types is not None
        else _REGISTERED_OBJECT_TYPES
    )
    if allowed is not None and value not in allowed:
        raise SchemaValidationError(
            f"{label} object_type '{value}' is not in registered vocabulary: "
            f"{', '.join(allowed)}"
        )
OPERATOR_REFERENCES = ("delivery_target", "last_target", "last_task")
OPERATOR_SELECTOR_RELATIONS = ("closest", "unique")
OPERATOR_DISTANCE_METRICS = ("manhattan", "euclidean")
OPERATOR_DISTANCE_REFERENCES = ("agent",)
GROUNDING_QUERY_OPERATIONS = ("list", "filter", "rank", "select", "answer")
GROUNDING_QUERY_ORDERS = ("ascending", "descending")
GROUNDING_QUERY_TIE_POLICIES = ("clarify", "display")
# Canonical answer-field vocabulary the deterministic executor recognizes. The LLM's output is
# canonicalized to this set BEFORE dispatch (see _ensure_canonical_answer_fields): conservative
# aliases repair near-misses (plural/synonym); ordinal forms `<first..fifth>_<closest|farthest>`
# are matched by pattern; anything else fails closed. Substrate-independent (shared vocabulary).
GROUNDING_QUERY_ANSWER_FIELDS = ("distance", "ranked_doors", "closest", "farthest", "exists", "target")
_ANSWER_FIELD_ALIASES = {
    "distances": "distance",
    "nearest": "closest",
    "furthest": "farthest",
    "ranking": "ranked_doors",
    "ranked_list": "ranked_doors",
}
_ORDINAL_ANSWER_FIELD_RE = re.compile(r"^(first|second|third|fourth|fifth)_(closest|farthest)$")
OPERATOR_STATUS_QUERIES = (
    "status",
    "scene",
    "help",
    "last_run",
    "last_target",
    "delivery_target",
    "ground_target",
    "cache",
    "concepts",
)
OPERATOR_CONTROLS = ("reset", "quit")
OPERATOR_CAPABILITY_STATUSES = (
    "executable",
    "needs_clarification",
    "missing_skills",
    "synthesizable",
    "unsupported",
)
REQUEST_OBJECTIVE_TYPES = (
    "task",
    "query",
    "primitive_definition",
    "knowledge_update",
    "motor_control",
    "control",
    "unsupported",
)
REQUEST_PLAN_LAYERS = (
    "task",
    "grounding",
    "claims",
    "sensing",
    "action",
    "memory",
    "answer",
    "control",
)
REQUEST_PLAN_OPERATIONS = (
    "rank",
    "filter",
    "select",
    "ground",
    "execute",
    "answer",
    "read",
    "update",
    "reset",
    "refuse",
)
REQUEST_EXPECTED_RESPONSES = (
    "execute_task",
    "execute_motor",
    "answer_query",
    "ask_clarification",
    "propose_definition",
    "propose_synthesis",
    "update_memory",
    "refuse",
)
REQUEST_TIE_POLICIES = ("clarify", "display_ties", "fail")
READINESS_NODE_STATUSES = (
    "executable",
    "needs_clarification",
    "needs_evidence",
    "needs_authorization",
    "validation_required",
    "claim_contract_failed",
    "synthesizable",
    "missing_skills",
    "unsupported",
    "stale_claims",
    "blocked_by_dependency",
    "environment_assumption_failed",
)
READINESS_NEXT_ACTIONS = (
    "execute_task",
    "execute_motor",
    "answer_query",
    "ask_clarification",
    "ask_authorization",
    "run_validation",
    "propose_synthesis",
    "refresh_claims",
    "update_memory",
    "refuse",
)
CLARIFICATION_REQUEST_TYPES = (
    "missing_field",
    "candidate_choice",
    "needs_evidence",
)
CLARIFICATION_EVIDENCE_SCOPES = (
    "visible_only",
    "search_allowed",
)
ARBITRATION_DECISION_TYPES = (
    "substitute",
    "clarify",
    "synthesize",
    "refuse",
)
LEGACY_PRIMITIVE_SPEC_TYPES = ("task", "grounding", "sensing", "action", "claims")
ORPI_PRIMITIVE_SPEC_TYPES = ("sense", "actuation", "meta")
PRIMITIVE_SPEC_TYPES = LEGACY_PRIMITIVE_SPEC_TYPES + ORPI_PRIMITIVE_SPEC_TYPES
ORPI_MODES = ("deterministic", "deliberative")
ORPI_CADENCES = ("control", "perception", "deliberation")
ORPI_INVARIANT_LEVELS = ("pose", "contact", "object_state", "intent")
PRIMITIVE_IMPLEMENTATION_STATUSES = (
    "implemented",
    "unsupported",
    "synthesizable",
    "planned",
    "missing",
)
PRIMITIVE_SAFETY_CLASSES = ("query", "memory", "actuation", "hazardous")
PRIMITIVE_AUTHORITY_LEVELS = ("none", "operator", "restricted", "admin")
SELECTION_DIRECTIONS = ("minimum", "maximum")

# Phase 13A steering vocabulary. A SteeringDirective is the typed, separable layer
# the operator uses to shape HOW a task is approached (distinct from the WHAT). Like
# SelectionObjective, the meaning lives in these enums — validation is pure enum logic,
# never vocabulary scanning. Adding a risk tier means changing only these tuples + the
# steering parser's clause patterns, not the readiness/planner logic.
STEERING_SCOPES = ("visible_only", "search_allowed", "full")
STEERING_RISK_LEVELS = ("query_only", "reversible_only", "operator_authorized")
STEERING_STOPPING_RULES = (
    "first_match",
    "exhaustive",
    "on_budget_exhausted",
    "on_ambiguity",
)
# safety_class sets each risk tier authorizes. A step whose primitive falls outside the
# authorized set is an authorization withdrawal (readiness -> needs_authorization).
STEERING_RISK_ALLOWED_SAFETY: dict[str, frozenset[str]] = {
    "query_only": frozenset({"query"}),
    "reversible_only": frozenset({"query", "memory"}),
    "operator_authorized": frozenset({"query", "memory", "actuation", "hazardous"}),
}


def orpi_primitive_type_for(primitive_type: str) -> str:
    mapping = {
        "sensing": "sense",
        "action": "actuation",
        "task": "meta",
        "grounding": "meta",
        "claims": "meta",
        "sense": "sense",
        "actuation": "actuation",
        "meta": "meta",
    }
    try:
        return mapping[primitive_type]
    except KeyError as exc:
        raise SchemaValidationError(
            f"primitive_type must map to ORPI class, got {primitive_type!r}"
        ) from exc


def default_orpi_cadence(*, primitive_type: str, layer: str) -> str:
    orpi_type = orpi_primitive_type_for(primitive_type)
    if orpi_type == "actuation" or layer in {"action", "spine"}:
        return "control"
    if orpi_type == "sense" or layer in {"sensing", "sense"}:
        return "perception"
    return "deliberation"


def default_orpi_invariant_level(primitive_type: str) -> str:
    return "intent" if orpi_primitive_type_for(primitive_type) == "meta" else "object_state"


class SchemaValidationError(ValueError):
    """Raised when compiler output does not satisfy the typed JEENOM schemas."""


def _ensure_mapping(data: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise SchemaValidationError(f"{label} must be an object, got {type(data).__name__}")
    return data


def _ensure_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{label} must be a string")
    return value


def _ensure_optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _ensure_str(value, label)


def _ensure_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{label} must be a boolean")
    return value


def _ensure_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{label} must be a number")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise SchemaValidationError(f"{label} must be between 0.0 and 1.0")
    return result


def _ensure_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{label} must be a list")
    return value


def _ensure_str_list(value: Any, label: str) -> list[str]:
    values = _ensure_list(value, label)
    result: list[str] = []
    for idx, item in enumerate(values):
        result.append(_ensure_str(item, f"{label}[{idx}]"))
    return result


def _ensure_canonical_answer_fields(value: Any, label: str) -> list[str]:
    """Normalize-then-validate the LLM's answer_fields to the canonical vocabulary.

    Conservative aliases repair near-misses (e.g. "distances" -> "distance"); ordinal forms
    (`<first..fifth>_<closest|farthest>`) pass through; an unrecognized value fails CLOSED with
    SchemaValidationError (which the LLM compiler turns into a regex fallback, else an honest
    "I didn't understand"). This is the single chokepoint, so both the LLM and regex paths emit
    canonical answer_fields and the deterministic executor sees one vocabulary.
    """
    result: list[str] = []
    for idx, item in enumerate(_ensure_str_list(value, label)):
        key = item.strip().lower()
        canonical = _ANSWER_FIELD_ALIASES.get(key, key)
        if canonical in GROUNDING_QUERY_ANSWER_FIELDS or _ORDINAL_ANSWER_FIELD_RE.match(canonical):
            result.append(canonical)
        else:
            raise SchemaValidationError(
                f"{label}[{idx}]: unknown answer field {item!r}; canonical values are "
                f"{GROUNDING_QUERY_ANSWER_FIELDS} or <first..fifth>_<closest|farthest>"
            )
    return result


def _ensure_dict(value: Any, label: str) -> dict[str, Any]:
    mapping = _ensure_mapping(value, label)
    return dict(mapping)


def _check_keys(d: dict, required: tuple, label: str, optional: tuple = ()) -> None:
    missing = [key for key in required if key not in d]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(d) - set(required) - set(optional))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")


def _ensure_compiler_params(value: Any, label: str) -> dict[str, Any]:
    params = _ensure_dict(value, label)
    required_keys = ("color", "object_type", "target_location")
    _check_keys(params, required_keys, label)

    for key in ("color", "object_type"):
        field_value = params.get(key)
        if field_value is not None and not isinstance(field_value, str):
            raise SchemaValidationError(f"{label}.{key} must be a string or null")

    target_location = params.get("target_location")
    if target_location is not None:
        if (
            not isinstance(target_location, list)
            or len(target_location) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in target_location)
        ):
            raise SchemaValidationError(
                f"{label}.target_location must be [int, int] or null"
            )
        params["target_location"] = tuple(target_location)

    return params


def _ensure_subset(
    values: list[str],
    allowed: tuple[str, ...],
    label: str,
) -> list[str]:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SchemaValidationError(f"{label} contains unsupported values: {', '.join(unknown)}")
    return values


def _ensure_optional_str_enum(
    value: Any,
    allowed: tuple[str, ...],
    label: str,
) -> str | None:
    if value is None:
        return None
    string_value = _ensure_str(value, label)
    if string_value not in allowed:
        raise SchemaValidationError(
            f"{label} must be one of: {', '.join(allowed)}"
        )
    return string_value


def _ensure_optional_metric_name(value: Any, label: str) -> str | None:
    if value is None:
        return None
    metric = _ensure_str(value, label)
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", metric):
        raise SchemaValidationError(
            f"{label} must be a metric identifier string or null"
        )
    return metric


def _ensure_optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{label} must be an integer or null")
    return value


def _ensure_operator_target(
    value: Any,
    label: str,
    *,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    target = _ensure_dict(value, label)
    _check_keys(target, ("color", "object_type"), label)
    obj_type = _ensure_optional_str(target.get("object_type"), f"{label}.object_type")
    _validate_object_type(obj_type, label, object_types)
    return {
        "color": _ensure_optional_str_enum(target.get("color"), OPERATOR_COLORS, f"{label}.color"),
        "object_type": obj_type,
    }


def _ensure_operator_knowledge_update(
    value: Any,
    label: str,
    *,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    update = _ensure_dict(value, label)
    _check_keys(update, ("delivery_target",), label)
    raw_delivery_target = update.get("delivery_target")
    if raw_delivery_target is None:
        return {"delivery_target": None}
    delivery_target = _ensure_operator_target(
        raw_delivery_target,
        f"{label}.delivery_target",
        object_types=object_types,
    )
    if delivery_target.get("color") is None or delivery_target.get("object_type") is None:
        raise SchemaValidationError(f"{label}.delivery_target must be fully specified")
    return {"delivery_target": delivery_target}


def _ensure_primitive_spec(value: Any, label: str) -> dict[str, Any]:
    spec = _ensure_dict(value, label)
    required_keys = (
        "name",
        "primitive_type",
        "layer",
        "description",
        "inputs",
        "outputs",
        "side_effects",
        "implementation_status",
        "safe_to_synthesize",
        "runtime_binding",
    )
    optional_keys = (
        "preconditions",
        "postconditions",
        "required_claims",
        "produced_claims",
        "units",
        "frame_id",
        "required_frames",
        "safety_class",
        "authority_level",
        "failure_modes",
        "validation_hooks",
        "substrate_fingerprint",
        "postcondition_primitive",
        "mode",
        "cadence",
        "invariant_level",
    )
    _check_keys(spec, required_keys, label, optional_keys)
    primitive_type = _ensure_str(spec["primitive_type"], f"{label}.primitive_type")
    if primitive_type not in PRIMITIVE_SPEC_TYPES:
        raise SchemaValidationError(
            f"{label}.primitive_type must be one of: {', '.join(PRIMITIVE_SPEC_TYPES)}"
        )
    implementation_status = _ensure_str(
        spec["implementation_status"],
        f"{label}.implementation_status",
    )
    if implementation_status not in PRIMITIVE_IMPLEMENTATION_STATUSES:
        raise SchemaValidationError(
            f"{label}.implementation_status must be one of: "
            + ", ".join(PRIMITIVE_IMPLEMENTATION_STATUSES)
        )
    safety_class = _ensure_str(
        spec.get("safety_class", "query"),
        f"{label}.safety_class",
    )
    if safety_class not in PRIMITIVE_SAFETY_CLASSES:
        raise SchemaValidationError(
            f"{label}.safety_class must be one of: "
            + ", ".join(PRIMITIVE_SAFETY_CLASSES)
        )
    authority_level = _ensure_str(
        spec.get("authority_level", "none"),
        f"{label}.authority_level",
    )
    if authority_level not in PRIMITIVE_AUTHORITY_LEVELS:
        raise SchemaValidationError(
            f"{label}.authority_level must be one of: "
            + ", ".join(PRIMITIVE_AUTHORITY_LEVELS)
        )
    mode = _ensure_str(spec.get("mode", "deterministic"), f"{label}.mode")
    if mode not in ORPI_MODES:
        raise SchemaValidationError(
            f"{label}.mode must be one of: " + ", ".join(ORPI_MODES)
        )
    cadence = _ensure_str(
        spec.get(
            "cadence",
            default_orpi_cadence(primitive_type=primitive_type, layer=str(spec["layer"])),
        ),
        f"{label}.cadence",
    )
    if cadence not in ORPI_CADENCES:
        raise SchemaValidationError(
            f"{label}.cadence must be one of: " + ", ".join(ORPI_CADENCES)
        )
    invariant_level = _ensure_str(
        spec.get("invariant_level", default_orpi_invariant_level(primitive_type)),
        f"{label}.invariant_level",
    )
    if invariant_level not in ORPI_INVARIANT_LEVELS:
        raise SchemaValidationError(
            f"{label}.invariant_level must be one of: "
            + ", ".join(ORPI_INVARIANT_LEVELS)
        )
    return {
        "name": _ensure_str(spec["name"], f"{label}.name"),
        "primitive_type": primitive_type,
        "layer": _ensure_str(spec["layer"], f"{label}.layer"),
        "description": _ensure_str(spec["description"], f"{label}.description"),
        "inputs": _ensure_str_list(spec["inputs"], f"{label}.inputs"),
        "outputs": _ensure_str_list(spec["outputs"], f"{label}.outputs"),
        "side_effects": _ensure_str_list(spec["side_effects"], f"{label}.side_effects"),
        "implementation_status": implementation_status,
        "safe_to_synthesize": _ensure_bool(
            spec["safe_to_synthesize"],
            f"{label}.safe_to_synthesize",
        ),
        "runtime_binding": spec["runtime_binding"],
        "preconditions": _ensure_str_list(
            spec.get("preconditions", []),
            f"{label}.preconditions",
        ),
        "postconditions": _ensure_str_list(
            spec.get("postconditions", []),
            f"{label}.postconditions",
        ),
        "required_claims": _ensure_str_list(
            spec.get("required_claims", []),
            f"{label}.required_claims",
        ),
        "produced_claims": _ensure_str_list(
            spec.get("produced_claims", []),
            f"{label}.produced_claims",
        ),
        "units": _ensure_dict(spec.get("units", {}), f"{label}.units"),
        "frame_id": _ensure_optional_str(spec.get("frame_id"), f"{label}.frame_id"),
        "required_frames": _ensure_str_list(
            spec.get("required_frames", []),
            f"{label}.required_frames",
        ),
        "safety_class": safety_class,
        "authority_level": authority_level,
        "failure_modes": _ensure_str_list(
            spec.get("failure_modes", []),
            f"{label}.failure_modes",
        ),
        "validation_hooks": _ensure_str_list(
            spec.get("validation_hooks", []),
            f"{label}.validation_hooks",
        ),
        "substrate_fingerprint": _ensure_optional_str(
            spec.get("substrate_fingerprint"),
            f"{label}.substrate_fingerprint",
        ),
        "postcondition_primitive": _ensure_optional_str(
            spec.get("postcondition_primitive"),
            f"{label}.postcondition_primitive",
        ),
        "mode": mode,
        "cadence": cadence,
        "invariant_level": invariant_level,
    }


def _ensure_primitive_manifest(value: Any, label: str) -> dict[str, Any]:
    manifest = _ensure_dict(value, label)
    required_keys = ("name", "primitives")
    _check_keys(manifest, required_keys, label)
    primitives = _ensure_list(manifest["primitives"], f"{label}.primitives")
    return {
        "name": _ensure_str(manifest["name"], f"{label}.name"),
        "primitives": [
            _ensure_primitive_spec(item, f"{label}.primitives[{idx}]")
            for idx, item in enumerate(primitives)
        ],
    }


def _migrate_exclude_color(selector: dict[str, Any]) -> None:
    """Migrate legacy exclude_color (str) to exclude_colors (list[str]) in-place."""
    if "exclude_color" in selector and "exclude_colors" not in selector:
        val = selector.pop("exclude_color")
        selector["exclude_colors"] = [val] if val else []
    elif "exclude_color" in selector:
        selector.pop("exclude_color")
    if "exclude_colors" not in selector:
        selector["exclude_colors"] = []


def _ensure_target_selector(
    value: Any,
    label: str,
    *,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    selector = _ensure_dict(value, label)
    # Support both legacy exclude_color (single) and new exclude_colors (list)
    _migrate_exclude_color(selector)
    required_keys = (
        "object_type",
        "color",
        "exclude_colors",
        "relation",
        "distance_metric",
        "distance_reference",
    )
    _check_keys(selector, required_keys, label)

    raw_exclude = selector.get("exclude_colors") or []
    if not isinstance(raw_exclude, list):
        raw_exclude = [raw_exclude] if raw_exclude else []
    validated_exclude = []
    for c in raw_exclude:
        validated = _ensure_optional_str_enum(c, OPERATOR_COLORS, f"{label}.exclude_colors[]")
        if validated:
            validated_exclude.append(validated)

    sel_obj_type = _ensure_optional_str(selector.get("object_type"), f"{label}.object_type")
    _validate_object_type(sel_obj_type, label, object_types)
    result = {
        "object_type": sel_obj_type,
        "color": _ensure_optional_str_enum(
            selector.get("color"),
            OPERATOR_COLORS,
            f"{label}.color",
        ),
        "exclude_colors": validated_exclude,
        "relation": _ensure_optional_str_enum(
            selector.get("relation"),
            OPERATOR_SELECTOR_RELATIONS,
            f"{label}.relation",
        ),
        "distance_metric": _ensure_optional_metric_name(
            selector.get("distance_metric"),
            f"{label}.distance_metric",
        ),
        "distance_reference": _ensure_optional_str_enum(
            selector.get("distance_reference"),
            OPERATOR_DISTANCE_REFERENCES,
            f"{label}.distance_reference",
        ),
    }
    _validate_object_type(result["object_type"], label, object_types)
    return result


def _ensure_grounding_query_plan(
    value: Any,
    label: str,
    *,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    plan = _ensure_dict(value, label)
    required_keys = (
        "object_type",
        "operation",
        "primitive_handle",
        "metric",
        "reference",
        "order",
        "ordinal",
        "color",
        "exclude_colors",
        "distance_value",
        "tie_policy",
        "answer_fields",
        "required_capabilities",
        "preserved_constraints",
    )
    optional_keys = ("comparison",)
    _check_keys(plan, required_keys, label, optional_keys)

    raw_exclude = plan.get("exclude_colors") or []
    if not isinstance(raw_exclude, list):
        raise SchemaValidationError(f"{label}.exclude_colors must be a list")
    exclude_colors: list[str] = []
    for idx, color in enumerate(raw_exclude):
        validated = _ensure_optional_str_enum(
            color,
            OPERATOR_COLORS,
            f"{label}.exclude_colors[{idx}]",
        )
        if validated is not None:
            exclude_colors.append(validated)

    primitive_handle = plan.get("primitive_handle")
    if primitive_handle is not None:
        primitive_handle = _ensure_str(primitive_handle, f"{label}.primitive_handle")

    plan_obj_type = _ensure_optional_str(plan.get("object_type"), f"{label}.object_type")
    _validate_object_type(plan_obj_type, label, object_types)
    result = {
        "object_type": plan_obj_type,
        "operation": _ensure_optional_str_enum(
            plan.get("operation"),
            GROUNDING_QUERY_OPERATIONS,
            f"{label}.operation",
        ),
        "primitive_handle": primitive_handle,
        "metric": _ensure_optional_metric_name(
            plan.get("metric"),
            f"{label}.metric",
        ),
        "reference": _ensure_optional_str_enum(
            plan.get("reference"),
            OPERATOR_DISTANCE_REFERENCES,
            f"{label}.reference",
        ),
        "order": _ensure_optional_str_enum(
            plan.get("order"),
            GROUNDING_QUERY_ORDERS,
            f"{label}.order",
        ),
        "ordinal": _ensure_optional_int(plan.get("ordinal"), f"{label}.ordinal"),
        "color": _ensure_optional_str_enum(
            plan.get("color"),
            OPERATOR_COLORS,
            f"{label}.color",
        ),
        "exclude_colors": exclude_colors,
        "distance_value": _ensure_optional_int(
            plan.get("distance_value"),
            f"{label}.distance_value",
        ),
        "comparison": _ensure_optional_str_enum(
            plan.get("comparison"),
            GROUNDING_QUERY_COMPARISONS,
            f"{label}.comparison",
        ),
        "tie_policy": _ensure_optional_str_enum(
            plan.get("tie_policy"),
            GROUNDING_QUERY_TIE_POLICIES,
            f"{label}.tie_policy",
        ),
        "answer_fields": _ensure_canonical_answer_fields(
            plan.get("answer_fields"),
            f"{label}.answer_fields",
        ),
        "required_capabilities": _ensure_str_list(
            plan.get("required_capabilities"),
            f"{label}.required_capabilities",
        ),
        "preserved_constraints": _ensure_str_list(
            plan.get("preserved_constraints"),
            f"{label}.preserved_constraints",
        ),
    }
    _validate_object_type(result["object_type"], label, object_types)
    if result["operation"] is None:
        raise SchemaValidationError(f"{label}.operation must not be null")
    ordinal = result["ordinal"]
    if ordinal is not None and ordinal < 1:
        raise SchemaValidationError(f"{label}.ordinal must be >= 1")
    distance_value = result["distance_value"]
    if distance_value is not None and distance_value < 0:
        raise SchemaValidationError(f"{label}.distance_value must be >= 0")
    if result["metric"] is not None and result["reference"] is None:
        raise SchemaValidationError(f"{label}.reference is required when metric is set")
    if result["operation"] in {"rank", "select"} and result["metric"] is not None:
        handle = result["primitive_handle"]
        if handle is None:
            raise SchemaValidationError(f"{label}.primitive_handle is required for ranked plans")
    return result


@dataclass
class PrimitiveCall:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveCall:
        mapping = _ensure_mapping(data, "PrimitiveCall")
        return cls(
            name=_ensure_str(mapping.get("name"), "PrimitiveCall.name"),
            params=_ensure_compiler_params(mapping.get("params", {}), "PrimitiveCall.params"),
        )


@dataclass
class TaskRequest:
    instruction: str
    task_type: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "compiler"

    @classmethod
    def from_dict(cls, data: Any) -> TaskRequest:
        mapping = _ensure_mapping(data, "TaskRequest")
        return cls(
            instruction=_ensure_str(mapping.get("instruction"), "TaskRequest.instruction"),
            task_type=_ensure_str(mapping.get("task_type"), "TaskRequest.task_type"),
            params=_ensure_compiler_params(mapping.get("params", {}), "TaskRequest.params"),
            source=_ensure_str(mapping.get("source", "compiler"), "TaskRequest.source"),
        )




@dataclass
class TargetSelector:
    object_type: str
    color: str | None = None
    exclude_colors: list[str] = field(default_factory=list)
    relation: str | None = None
    distance_metric: str | None = None
    distance_reference: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        object_types: tuple[str, ...] | list[str] | None = None,
    ) -> TargetSelector:
        selector = _ensure_target_selector(
            data,
            "TargetSelector",
            object_types=object_types,
        )
        if selector is None:
            raise SchemaValidationError("TargetSelector must not be null")
        return cls(**selector)


@dataclass
class GroundingQueryPlan:
    object_type: str
    operation: str
    primitive_handle: str | None = None
    metric: str | None = None
    reference: str | None = None
    order: str | None = None
    ordinal: int | None = None
    color: str | None = None
    exclude_colors: list[str] = field(default_factory=list)
    distance_value: float | None = None  # a metric threshold; float-capable for euclidean
    comparison: str | None = None
    tie_policy: str | None = "clarify"
    answer_fields: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    preserved_constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> GroundingQueryPlan:
        plan = _ensure_grounding_query_plan(data, "GroundingQueryPlan")
        if plan is None:
            raise SchemaValidationError("GroundingQueryPlan must not be null")
        return cls(**plan)


@dataclass
class RequestPlanStep:
    step_id: str
    layer: str
    operation: str
    required_handle: str | None = None
    implementation_status: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    tie_policy: str = "clarify"
    memory_reads: list[str] = field(default_factory=list)
    memory_writes: list[str] = field(default_factory=list)
    scene_fingerprint_required: bool = False
    environment_assumption_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> RequestPlanStep:
        mapping = _ensure_mapping(data, "RequestPlanStep")
        layer = _ensure_optional_str_enum(
            mapping.get("layer"),
            REQUEST_PLAN_LAYERS,
            "RequestPlanStep.layer",
        )
        operation = _ensure_optional_str_enum(
            mapping.get("operation"),
            REQUEST_PLAN_OPERATIONS,
            "RequestPlanStep.operation",
        )
        implementation_status = _ensure_optional_str_enum(
            mapping.get("implementation_status"),
            PRIMITIVE_IMPLEMENTATION_STATUSES,
            "RequestPlanStep.implementation_status",
        )
        tie_policy = _ensure_optional_str_enum(
            mapping.get("tie_policy", "clarify"),
            REQUEST_TIE_POLICIES,
            "RequestPlanStep.tie_policy",
        )
        required_handle = mapping.get("required_handle")
        if required_handle is not None:
            required_handle = _ensure_str(required_handle, "RequestPlanStep.required_handle")
        return cls(
            step_id=_ensure_str(mapping.get("step_id"), "RequestPlanStep.step_id"),
            layer=layer or "control",
            operation=operation or "refuse",
            required_handle=required_handle,
            implementation_status=implementation_status,
            inputs=_ensure_dict(mapping.get("inputs", {}), "RequestPlanStep.inputs"),
            outputs=_ensure_str_list(mapping.get("outputs", []), "RequestPlanStep.outputs"),
            depends_on=_ensure_str_list(
                mapping.get("depends_on", []),
                "RequestPlanStep.depends_on",
            ),
            constraints=_ensure_dict(
                mapping.get("constraints", {}),
                "RequestPlanStep.constraints",
            ),
            tie_policy=tie_policy or "clarify",
            memory_reads=_ensure_str_list(
                mapping.get("memory_reads", []),
                "RequestPlanStep.memory_reads",
            ),
            memory_writes=_ensure_str_list(
                mapping.get("memory_writes", []),
                "RequestPlanStep.memory_writes",
            ),
            scene_fingerprint_required=_ensure_bool(
                mapping.get("scene_fingerprint_required", False),
                "RequestPlanStep.scene_fingerprint_required",
            ),
            environment_assumption_ids=_ensure_str_list(
                mapping.get("environment_assumption_ids", []),
                "RequestPlanStep.environment_assumption_ids",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "layer": self.layer,
            "operation": self.operation,
            "required_handle": self.required_handle,
            "implementation_status": self.implementation_status,
            "inputs": dict(self.inputs),
            "outputs": list(self.outputs),
            "depends_on": list(self.depends_on),
            "constraints": dict(self.constraints),
            "tie_policy": self.tie_policy,
            "memory_reads": list(self.memory_reads),
            "memory_writes": list(self.memory_writes),
            "scene_fingerprint_required": self.scene_fingerprint_required,
            "environment_assumption_ids": list(self.environment_assumption_ids),
        }


@dataclass
class EnvironmentAssumption:
    assumption_id: str
    kind: str
    expected: dict[str, Any] = field(default_factory=dict)
    source: str = "environment_identity"
    required: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "EnvironmentAssumption":
        mapping = _ensure_mapping(data, "EnvironmentAssumption")
        return cls(
            assumption_id=_ensure_str(
                mapping.get("assumption_id"),
                "EnvironmentAssumption.assumption_id",
            ),
            kind=_ensure_str(mapping.get("kind"), "EnvironmentAssumption.kind"),
            expected=_ensure_dict(
                mapping.get("expected", {}),
                "EnvironmentAssumption.expected",
            ),
            source=_ensure_str(
                mapping.get("source", "environment_identity"),
                "EnvironmentAssumption.source",
            ),
            required=_ensure_bool(
                mapping.get("required", True),
                "EnvironmentAssumption.required",
            ),
            description=_ensure_str(
                mapping.get("description", ""),
                "EnvironmentAssumption.description",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "kind": self.kind,
            "expected": dict(self.expected),
            "source": self.source,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class RequestPlan:
    request_id: str
    original_utterance: str
    objective_type: str
    objective_summary: str
    steps: list[RequestPlanStep] = field(default_factory=list)
    preservation_signals: list[str] = field(default_factory=list)
    expected_response: str = "refuse"
    environment_assumptions: list[EnvironmentAssumption] = field(default_factory=list)
    # Phase 13A: the active SteeringDirective (as_dict) that shaped this plan, if any.
    steering: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Any) -> RequestPlan:
        mapping = _ensure_mapping(data, "RequestPlan")
        objective_type = _ensure_optional_str_enum(
            mapping.get("objective_type"),
            REQUEST_OBJECTIVE_TYPES,
            "RequestPlan.objective_type",
        )
        expected_response = _ensure_optional_str_enum(
            mapping.get("expected_response", "refuse"),
            REQUEST_EXPECTED_RESPONSES,
            "RequestPlan.expected_response",
        )
        raw_steps = _ensure_list(mapping.get("steps", []), "RequestPlan.steps")
        raw_assumptions = _ensure_list(
            mapping.get("environment_assumptions", []),
            "RequestPlan.environment_assumptions",
        )
        return cls(
            request_id=_ensure_str(mapping.get("request_id"), "RequestPlan.request_id"),
            original_utterance=_ensure_str(
                mapping.get("original_utterance"),
                "RequestPlan.original_utterance",
            ),
            objective_type=objective_type or "unsupported",
            objective_summary=_ensure_str(
                mapping.get("objective_summary"),
                "RequestPlan.objective_summary",
            ),
            steps=[
                RequestPlanStep.from_dict(item)
                for item in raw_steps
            ],
            preservation_signals=_ensure_str_list(
                mapping.get("preservation_signals", []),
                "RequestPlan.preservation_signals",
            ),
            expected_response=expected_response or "refuse",
            environment_assumptions=[
                EnvironmentAssumption.from_dict(item)
                for item in raw_assumptions
            ],
            steering=mapping.get("steering"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_utterance": self.original_utterance,
            "objective_type": self.objective_type,
            "objective_summary": self.objective_summary,
            "steps": [step.as_dict() for step in self.steps],
            "preservation_signals": list(self.preservation_signals),
            "expected_response": self.expected_response,
            "environment_assumptions": [
                assumption.as_dict()
                for assumption in self.environment_assumptions
            ],
            "steering": dict(self.steering) if self.steering else None,
        }


@dataclass
class ReadinessNode:
    step_id: str
    status: str
    layer: str
    operation: str
    required_handle: str | None = None
    reason: str = ""
    blocking_dependencies: list[str] = field(default_factory=list)
    violated_assumption_ids: list[str] = field(default_factory=list)
    diagnostic_assumption_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> ReadinessNode:
        mapping = _ensure_mapping(data, "ReadinessNode")
        status = _ensure_optional_str_enum(
            mapping.get("status"),
            READINESS_NODE_STATUSES,
            "ReadinessNode.status",
        )
        required_handle = mapping.get("required_handle")
        if required_handle is not None:
            required_handle = _ensure_str(required_handle, "ReadinessNode.required_handle")
        return cls(
            step_id=_ensure_str(mapping.get("step_id"), "ReadinessNode.step_id"),
            status=status or "unsupported",
            layer=_ensure_str(mapping.get("layer"), "ReadinessNode.layer"),
            operation=_ensure_str(mapping.get("operation"), "ReadinessNode.operation"),
            required_handle=required_handle,
            reason=_ensure_str(mapping.get("reason", ""), "ReadinessNode.reason"),
            blocking_dependencies=_ensure_str_list(
                mapping.get("blocking_dependencies", []),
                "ReadinessNode.blocking_dependencies",
            ),
            violated_assumption_ids=_ensure_str_list(
                mapping.get("violated_assumption_ids", []),
                "ReadinessNode.violated_assumption_ids",
            ),
            diagnostic_assumption_ids=_ensure_str_list(
                mapping.get("diagnostic_assumption_ids", []),
                "ReadinessNode.diagnostic_assumption_ids",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "layer": self.layer,
            "operation": self.operation,
            "required_handle": self.required_handle,
            "reason": self.reason,
            "blocking_dependencies": list(self.blocking_dependencies),
            "violated_assumption_ids": list(self.violated_assumption_ids),
            "diagnostic_assumption_ids": list(self.diagnostic_assumption_ids),
        }


@dataclass
class ReadinessGraph:
    request_id: str
    nodes: list[ReadinessNode] = field(default_factory=list)
    graph_status: str = "unsupported"
    next_action: str = "refuse"
    blocking_step_id: str | None = None
    explanation: str = ""
    violated_assumption_ids: list[str] = field(default_factory=list)
    diagnostic_assumption_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> ReadinessGraph:
        mapping = _ensure_mapping(data, "ReadinessGraph")
        graph_status = _ensure_optional_str_enum(
            mapping.get("graph_status", "unsupported"),
            READINESS_NODE_STATUSES,
            "ReadinessGraph.graph_status",
        )
        next_action = _ensure_optional_str_enum(
            mapping.get("next_action", "refuse"),
            READINESS_NEXT_ACTIONS,
            "ReadinessGraph.next_action",
        )
        blocking_step_id = mapping.get("blocking_step_id")
        if blocking_step_id is not None:
            blocking_step_id = _ensure_str(
                blocking_step_id,
                "ReadinessGraph.blocking_step_id",
            )
        raw_nodes = _ensure_list(mapping.get("nodes", []), "ReadinessGraph.nodes")
        return cls(
            request_id=_ensure_str(mapping.get("request_id"), "ReadinessGraph.request_id"),
            nodes=[ReadinessNode.from_dict(item) for item in raw_nodes],
            graph_status=graph_status or "unsupported",
            next_action=next_action or "refuse",
            blocking_step_id=blocking_step_id,
            explanation=_ensure_str(
                mapping.get("explanation", ""),
                "ReadinessGraph.explanation",
            ),
            violated_assumption_ids=_ensure_str_list(
                mapping.get("violated_assumption_ids", []),
                "ReadinessGraph.violated_assumption_ids",
            ),
            diagnostic_assumption_ids=_ensure_str_list(
                mapping.get("diagnostic_assumption_ids", []),
                "ReadinessGraph.diagnostic_assumption_ids",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "graph_status": self.graph_status,
            "next_action": self.next_action,
            "blocking_step_id": self.blocking_step_id,
            "explanation": self.explanation,
            "violated_assumption_ids": list(self.violated_assumption_ids),
            "diagnostic_assumption_ids": list(self.diagnostic_assumption_ids),
        }


@dataclass
class PrimitiveSpec:
    name: str
    primitive_type: str
    layer: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    implementation_status: str = "implemented"
    safe_to_synthesize: bool = False
    runtime_binding: dict[str, Any] | None = None
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    required_claims: list[str] = field(default_factory=list)
    produced_claims: list[str] = field(default_factory=list)
    units: dict[str, Any] = field(default_factory=dict)
    frame_id: str | None = None
    required_frames: list[str] = field(default_factory=list)
    safety_class: str = "query"
    authority_level: str = "none"
    failure_modes: list[str] = field(default_factory=list)
    validation_hooks: list[str] = field(default_factory=list)
    substrate_fingerprint: str | None = None
    postcondition_primitive: str | None = None
    mode: str = "deterministic"
    cadence: str | None = None
    invariant_level: str | None = None

    def __post_init__(self) -> None:
        orpi_type = orpi_primitive_type_for(self.primitive_type)
        if self.cadence is None:
            self.cadence = default_orpi_cadence(
                primitive_type=self.primitive_type,
                layer=self.layer,
            )
        if self.invariant_level is None:
            self.invariant_level = default_orpi_invariant_level(self.primitive_type)
        if self.mode not in ORPI_MODES:
            raise SchemaValidationError(
                "PrimitiveSpec.mode must be one of: " + ", ".join(ORPI_MODES)
            )
        if self.cadence not in ORPI_CADENCES:
            raise SchemaValidationError(
                "PrimitiveSpec.cadence must be one of: " + ", ".join(ORPI_CADENCES)
            )
        if self.invariant_level not in ORPI_INVARIANT_LEVELS:
            raise SchemaValidationError(
                "PrimitiveSpec.invariant_level must be one of: "
                + ", ".join(ORPI_INVARIANT_LEVELS)
            )
        if orpi_type != "meta" and self.mode == "deliberative":
            raise SchemaValidationError(
                "PrimitiveSpec.mode=deliberative is only valid for ORPI meta primitives"
            )

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveSpec:
        return cls(**_ensure_primitive_spec(data, "PrimitiveSpec"))


@dataclass
class PrimitiveManifest:
    name: str
    primitives: list[PrimitiveSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveManifest:
        manifest = _ensure_primitive_manifest(data, "PrimitiveManifest")
        return cls(
            name=manifest["name"],
            primitives=[
                PrimitiveSpec.from_dict(item)
                for item in manifest["primitives"]
            ],
        )


@dataclass
class SelectionObjective:
    """Structured distillation of what the operator wants to select/rank.

    The LLM fills this alongside grounding_query_plan. Validation then checks
    objective fields against plan fields using pure enum logic — no vocabulary
    scanning. Adding 'hottest'/'coldest' to a new primitive means changing only
    the LLM prompt (attribute vocabulary). No code changes to validation.
    """

    attribute: str       # what property to rank by: "distance", "temperature", ...
    direction: str       # "minimum" (closest/coldest) or "maximum" (farthest/hottest)
    ordinal: int         # 1 for "the farthest", 2 for "second farthest"
    metric: str | None   # "manhattan" | "euclidean" | None (use default)

    @classmethod
    def from_dict(cls, data: Any) -> "SelectionObjective | None":
        if data is None:
            return None
        d = _ensure_mapping(data, "SelectionObjective")
        direction = _ensure_str(d.get("direction"), "SelectionObjective.direction")
        if direction not in SELECTION_DIRECTIONS:
            raise SchemaValidationError(
                "SelectionObjective.direction must be one of: "
                + ", ".join(SELECTION_DIRECTIONS)
            )
        raw_ordinal = d.get("ordinal", 1)
        if isinstance(raw_ordinal, bool) or not isinstance(raw_ordinal, int) or raw_ordinal < 1:
            raise SchemaValidationError(
                "SelectionObjective.ordinal must be a positive integer"
            )
        return cls(
            attribute=_ensure_str(d.get("attribute"), "SelectionObjective.attribute"),
            direction=direction,
            ordinal=raw_ordinal,
            metric=_ensure_optional_metric_name(
                d.get("metric"),
                "SelectionObjective.metric",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "direction": self.direction,
            "ordinal": self.ordinal,
            "metric": self.metric,
        }


@dataclass
class SteeringDirective:
    """Typed, separable steering layer (Phase 13A): how the operator constrains the
    approach to a task — budget, scope, risk, stopping rule — distinct from the WHAT.

    Mirrors SelectionObjective: typed fields, enum-validated, no vocabulary scanning.
    On the fully-observable substrate, `risk` (gates actuation via readiness) and
    `budget`/`stopping_rule` (cap execution in the Spine loop) have teeth; `scope` is
    carried + validated but its enforcement is degenerate until partial observability
    (Phase 13B).
    """

    budget: dict[str, Any] | None = None  # {"max_steps": int, "max_clarifications": int}
    scope: str | None = None
    risk: str | None = None
    stopping_rule: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "SteeringDirective | None":
        if data is None:
            return None
        d = _ensure_mapping(data, "SteeringDirective")
        budget = cls._ensure_budget(d.get("budget"), "SteeringDirective.budget")
        directive = cls(
            budget=budget,
            scope=_ensure_optional_str_enum(
                d.get("scope"), STEERING_SCOPES, "SteeringDirective.scope"
            ),
            risk=_ensure_optional_str_enum(
                d.get("risk"), STEERING_RISK_LEVELS, "SteeringDirective.risk"
            ),
            stopping_rule=_ensure_optional_str_enum(
                d.get("stopping_rule"),
                STEERING_STOPPING_RULES,
                "SteeringDirective.stopping_rule",
            ),
        )
        if directive.is_empty():
            raise SchemaValidationError(
                "SteeringDirective must set at least one of budget/scope/risk/stopping_rule"
            )
        return directive

    @staticmethod
    def _ensure_budget(value: Any, label: str) -> dict[str, Any] | None:
        if value is None:
            return None
        mapping = _ensure_mapping(value, label)
        budget: dict[str, Any] = {}
        for key in ("max_steps", "max_clarifications"):
            raw = mapping.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise SchemaValidationError(f"{label}.{key} must be a non-negative integer")
            budget[key] = raw
        return budget or None

    def is_empty(self) -> bool:
        return (
            not self.budget
            and self.scope is None
            and self.risk is None
            and self.stopping_rule is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": dict(self.budget) if self.budget else None,
            "scope": self.scope,
            "risk": self.risk,
            "stopping_rule": self.stopping_rule,
        }


@dataclass
class PrimitiveDefinitionRequest:
    """Typed operator request to define a query-only primitive."""

    definition_type: str
    name: str
    normalized_name: str
    expression: dict[str, Any]
    dependencies: list[str]
    dependency_handles: list[str]
    proposed_handle: str
    safety_class: str = "query"
    authority_level: str = "operator"
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "PrimitiveDefinitionRequest":
        mapping = _ensure_mapping(data, "PrimitiveDefinitionRequest")
        definition_type = _ensure_str(
            mapping.get("definition_type"),
            "PrimitiveDefinitionRequest.definition_type",
        )
        if definition_type not in {"distance_metric"}:
            raise SchemaValidationError(
                "PrimitiveDefinitionRequest.definition_type must be distance_metric"
            )
        normalized_name = _ensure_str(
            mapping.get("normalized_name"),
            "PrimitiveDefinitionRequest.normalized_name",
        )
        if not re.match(r"^[a-z][a-z0-9_]*$", normalized_name):
            raise SchemaValidationError(
                "PrimitiveDefinitionRequest.normalized_name must be registry-safe"
            )
        safety_class = _ensure_str(
            mapping.get("safety_class", "query"),
            "PrimitiveDefinitionRequest.safety_class",
        )
        if safety_class != "query":
            raise SchemaValidationError(
                "PrimitiveDefinitionRequest.safety_class must be query"
            )
        authority_level = _ensure_str(
            mapping.get("authority_level", "operator"),
            "PrimitiveDefinitionRequest.authority_level",
        )
        if authority_level != "operator":
            raise SchemaValidationError(
                "PrimitiveDefinitionRequest.authority_level must be operator"
            )
        proposed_handle = _ensure_str(
            mapping.get("proposed_handle"),
            "PrimitiveDefinitionRequest.proposed_handle",
        )
        return cls(
            definition_type=definition_type,
            name=_ensure_str(mapping.get("name"), "PrimitiveDefinitionRequest.name"),
            normalized_name=normalized_name,
            expression=_ensure_dict(
                mapping.get("expression"),
                "PrimitiveDefinitionRequest.expression",
            ),
            dependencies=_ensure_str_list(
                mapping.get("dependencies", []),
                "PrimitiveDefinitionRequest.dependencies",
            ),
            dependency_handles=_ensure_str_list(
                mapping.get("dependency_handles", []),
                "PrimitiveDefinitionRequest.dependency_handles",
            ),
            proposed_handle=proposed_handle,
            safety_class=safety_class,
            authority_level=authority_level,
            provenance=_ensure_dict(
                mapping.get("provenance", {}),
                "PrimitiveDefinitionRequest.provenance",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "definition_type": self.definition_type,
            "name": self.name,
            "normalized_name": self.normalized_name,
            "expression": dict(self.expression),
            "dependencies": list(self.dependencies),
            "dependency_handles": list(self.dependency_handles),
            "proposed_handle": self.proposed_handle,
            "safety_class": self.safety_class,
            "authority_level": self.authority_level,
            "provenance": dict(self.provenance),
        }


@dataclass
class ClarificationRequest:
    """Typed operator request for missing information or missing evidence."""

    request_type: str
    prompt: str
    reason: str
    resume_kind: str
    evidence_scope: str | None = None
    target: dict[str, Any] = field(default_factory=dict)
    options: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "ClarificationRequest":
        mapping = _ensure_mapping(data, "ClarificationRequest")
        request_type = _ensure_str(
            mapping.get("request_type"),
            "ClarificationRequest.request_type",
        )
        if request_type not in CLARIFICATION_REQUEST_TYPES:
            raise SchemaValidationError(
                "ClarificationRequest.request_type must be one of: "
                + ", ".join(CLARIFICATION_REQUEST_TYPES)
            )
        evidence_scope = _ensure_optional_str_enum(
            mapping.get("evidence_scope"),
            CLARIFICATION_EVIDENCE_SCOPES,
            "ClarificationRequest.evidence_scope",
        )
        return cls(
            request_type=request_type,
            prompt=_ensure_str(mapping.get("prompt"), "ClarificationRequest.prompt"),
            reason=_ensure_str(mapping.get("reason"), "ClarificationRequest.reason"),
            resume_kind=_ensure_str(
                mapping.get("resume_kind"),
                "ClarificationRequest.resume_kind",
            ),
            evidence_scope=evidence_scope,
            target=_ensure_dict(
                mapping.get("target", {}),
                "ClarificationRequest.target",
            ),
            options=_ensure_str_list(
                mapping.get("options", []),
                "ClarificationRequest.options",
            ),
            provenance=_ensure_dict(
                mapping.get("provenance", {}),
                "ClarificationRequest.provenance",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_type": self.request_type,
            "prompt": self.prompt,
            "reason": self.reason,
            "resume_kind": self.resume_kind,
            "evidence_scope": self.evidence_scope,
            "target": dict(self.target),
            "options": list(self.options),
            "provenance": dict(self.provenance),
        }


@dataclass
class OperatorIntent:
    intent_type: str
    canonical_instruction: str | None = None
    task_type: str | None = None
    target: dict[str, Any] | None = None
    knowledge_update: dict[str, Any] | None = None
    reference: str | None = None
    status_query: str | None = None
    claim_reference: str | None = None
    control: str | None = None
    target_selector: dict[str, Any] | None = None
    grounding_query_plan: dict[str, Any] | None = None
    primitive_definition: PrimitiveDefinitionRequest | None = None
    capability_status: str = "executable"
    required_capabilities: list[str] = field(default_factory=list)
    clear_memory: bool = False
    confidence: float = 0.0
    reason: str = ""
    # concept_teach / concept_recall fields
    concept_name: str | None = None
    concept_utterance: str | None = None
    # procedure_recall: ordered list of concept names to execute sequentially
    concept_steps: list[str] | None = None
    # sequence_instruction: ordered list of raw task utterances to execute sequentially
    utterance_steps: list[str] | None = None
    # motor_command: explicit low-level action primitive authorized by RawMotorTicket
    action_name: str | None = None    # key in ACTION_PRIMITIVES (e.g. "move_forward")
    repeat_count: int | None = None   # number of times to execute (>= 1)
    # mission_contract: L4 goal — ordered task sequence with abort-on-failure
    mission_steps: list[str] | None = None  # raw utterance for each task in the mission
    # structured distillation of the selection objective; drives objective-based validation
    selection_objective: SelectionObjective | None = None
    # Phase 13A: typed steering layer attached to an action intent (or standalone turn)
    steering_directive: SteeringDirective | None = None

    _KNOWLEDGE_TYPE_MAP: ClassVar[dict[str, str]] = {
        "status_query": "claim",
        "claim_reference": "claim",
        "cache_query": "claim",
        "concept_teach": "procedure",
        "concept_recall": "procedure",
        "concept_forget": "control",
        "procedure_recall": "procedure",
        "sequence_instruction": "procedure",
        "primitive_definition": "provenance",
        "knowledge_update": "provenance",
        "task_instruction": "action",
        "motor_command": "action",
        "motor_sequence": "action",
        "conditional_sense_motor": "action",
        "mission_contract": "action",
        "metric_query": "claim",
        "steering_directive": "action",
        "reset": "control",
        "quit": "control",
        "accept_proposal": "control",
        "reject_proposal": "control",
        "unsupported": "control",
        "ambiguous": "control",
    }

    @property
    def knowledge_type(self) -> str:
        return self._KNOWLEDGE_TYPE_MAP.get(self.intent_type, "control")

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        object_types: tuple[str, ...] | list[str] | None = None,
    ) -> OperatorIntent:
        mapping = _ensure_mapping(data, "OperatorIntent")
        intent_type = _ensure_str(mapping.get("intent_type"), "OperatorIntent.intent_type")
        if intent_type not in OPERATOR_INTENT_TYPES:
            raise SchemaValidationError(
                "OperatorIntent.intent_type must be one of: "
                + ", ".join(OPERATOR_INTENT_TYPES)
            )

        target = _ensure_operator_target(
            mapping.get("target"),
            "OperatorIntent.target",
            object_types=object_types,
        )
        knowledge_update = _ensure_operator_knowledge_update(
            mapping.get("knowledge_update"),
            "OperatorIntent.knowledge_update",
            object_types=object_types,
        )
        task_type = _ensure_optional_str_enum(
            mapping.get("task_type"),
            OPERATOR_TASK_TYPES,
            "OperatorIntent.task_type",
        )
        reference = _ensure_optional_str_enum(
            mapping.get("reference"),
            OPERATOR_REFERENCES,
            "OperatorIntent.reference",
        )
        status_query = _ensure_optional_str_enum(
            mapping.get("status_query"),
            OPERATOR_STATUS_QUERIES,
            "OperatorIntent.status_query",
        )
        claim_reference = _ensure_optional_str_enum(
            mapping.get("claim_reference"),
            OPERATOR_CLAIM_REFERENCES,
            "OperatorIntent.claim_reference",
        )
        control = _ensure_optional_str_enum(
            mapping.get("control"),
            OPERATOR_CONTROLS,
            "OperatorIntent.control",
        )
        capability_status = _ensure_optional_str_enum(
            mapping.get("capability_status", "executable"),
            OPERATOR_CAPABILITY_STATUSES,
            "OperatorIntent.capability_status",
        )
        target_selector = _ensure_target_selector(
            mapping.get("target_selector"),
            "OperatorIntent.target_selector",
            object_types=object_types,
        )
        grounding_query_plan = _ensure_grounding_query_plan(
            mapping.get("grounding_query_plan"),
            "OperatorIntent.grounding_query_plan",
            object_types=object_types,
        )
        primitive_definition = (
            PrimitiveDefinitionRequest.from_dict(mapping.get("primitive_definition"))
            if mapping.get("primitive_definition") is not None
            else None
        )

        canonical_instruction = mapping.get("canonical_instruction")
        if canonical_instruction is not None:
            canonical_instruction = _ensure_str(
                canonical_instruction,
                "OperatorIntent.canonical_instruction",
            )

        concept_name = _ensure_optional_str(
            mapping.get("concept_name"), "OperatorIntent.concept_name"
        )
        concept_utterance = _ensure_optional_str(
            mapping.get("concept_utterance"), "OperatorIntent.concept_utterance"
        )

        raw_concept_steps = mapping.get("concept_steps")
        if raw_concept_steps is None:
            concept_steps: list[str] | None = None
        elif isinstance(raw_concept_steps, list):
            concept_steps = [str(s) for s in raw_concept_steps if s is not None]
        else:
            concept_steps = None

        raw_utterance_steps = mapping.get("utterance_steps")
        if raw_utterance_steps is None:
            utterance_steps: list[str] | None = None
        elif isinstance(raw_utterance_steps, list):
            utterance_steps = [str(s) for s in raw_utterance_steps if s is not None]
        else:
            utterance_steps = None

        action_name = _ensure_optional_str(
            mapping.get("action_name"), "OperatorIntent.action_name"
        )
        raw_repeat = mapping.get("repeat_count")
        repeat_count: int | None = None
        if raw_repeat is not None:
            try:
                repeat_count = int(raw_repeat)
            except (TypeError, ValueError):
                repeat_count = None

        raw_required = mapping.get("required_capabilities")
        if raw_required is None:
            required_capabilities: list[str] = []
        elif isinstance(raw_required, list):
            required_capabilities = [str(h) for h in raw_required if h is not None]
        else:
            required_capabilities = []

        raw_mission_steps = mapping.get("mission_steps")
        if raw_mission_steps is None:
            mission_steps: list[str] | None = None
        elif isinstance(raw_mission_steps, list):
            mission_steps = [str(s) for s in raw_mission_steps if s is not None]
        else:
            mission_steps = None

        selection_objective = SelectionObjective.from_dict(
            mapping.get("selection_objective")
        )
        steering_directive = SteeringDirective.from_dict(
            mapping.get("steering_directive")
        )

        intent = cls(
            intent_type=intent_type,
            canonical_instruction=canonical_instruction,
            task_type=task_type,
            target=target,
            knowledge_update=knowledge_update,
            reference=reference,
            status_query=status_query,
            claim_reference=claim_reference,
            control=control,
            target_selector=target_selector,
            grounding_query_plan=grounding_query_plan,
            primitive_definition=primitive_definition,
            capability_status=capability_status or "executable",
            required_capabilities=required_capabilities,
            clear_memory=_ensure_bool(
                mapping.get("clear_memory"),
                "OperatorIntent.clear_memory",
            ),
            confidence=_ensure_float(mapping.get("confidence"), "OperatorIntent.confidence"),
            reason=_ensure_str(mapping.get("reason", ""), "OperatorIntent.reason"),
            concept_name=concept_name,
            concept_utterance=concept_utterance,
            concept_steps=concept_steps,
            utterance_steps=utterance_steps,
            action_name=action_name,
            repeat_count=repeat_count,
            mission_steps=mission_steps,
            selection_objective=selection_objective,
            steering_directive=steering_directive,
        )
        intent._validate_supported_shape()
        return intent

    def _validate_supported_shape(self) -> None:
        if self.intent_type == "task_instruction":
            if self.task_type != "go_to_object":
                raise SchemaValidationError("task_instruction requires task_type=go_to_object")
            has_target = (
                isinstance(self.target, dict)
                and self.target.get("color") is not None
                and self.target.get("object_type") is not None
            )
            if (
                not has_target
                and self.reference not in OPERATOR_REFERENCES
                and self.target_selector is None
                and self.grounding_query_plan is None
            ):
                raise SchemaValidationError(
                    "task_instruction requires a supported target, reference, target_selector, or grounding_query_plan"
                )
        elif self.intent_type == "knowledge_update":
            if self.knowledge_update is None:
                raise SchemaValidationError("knowledge_update requires knowledge_update payload")
            if (
                "delivery_target" not in self.knowledge_update
                and self.target_selector is None
                and self.grounding_query_plan is None
            ):
                raise SchemaValidationError(
                    "selector-based knowledge_update requires target_selector or grounding_query_plan"
                )
        elif self.intent_type == "status_query":
            if self.status_query is None:
                raise SchemaValidationError("status_query requires status_query")
        elif self.intent_type == "primitive_definition":
            if self.primitive_definition is None:
                raise SchemaValidationError(
                    "primitive_definition requires primitive_definition payload"
                )
        elif self.intent_type == "cache_query":
            if self.status_query not in {None, "cache"}:
                raise SchemaValidationError("cache_query status_query must be cache or null")
        elif self.intent_type == "reset":
            if self.control not in {None, "reset"}:
                raise SchemaValidationError("reset control must be reset or null")
        elif self.intent_type == "quit":
            if self.control not in {None, "quit"}:
                raise SchemaValidationError("quit control must be quit or null")
        elif self.intent_type == "concept_teach":
            if not self.concept_name:
                raise SchemaValidationError("concept_teach requires concept_name")
            if not self.concept_utterance:
                raise SchemaValidationError("concept_teach requires concept_utterance")
        elif self.intent_type == "concept_recall":
            if not self.concept_name:
                raise SchemaValidationError("concept_recall requires concept_name")
        elif self.intent_type == "procedure_recall":
            if not self.concept_steps or len(self.concept_steps) < 2:
                raise SchemaValidationError(
                    "procedure_recall requires concept_steps with at least 2 elements"
                )
        elif self.intent_type == "sequence_instruction":
            if not self.utterance_steps or len(self.utterance_steps) < 2:
                raise SchemaValidationError(
                    "sequence_instruction requires utterance_steps with at least 2 elements"
                )
        elif self.intent_type == "motor_command":
            if not self.action_name:
                raise SchemaValidationError("motor_command requires action_name")
            if self.repeat_count is not None and self.repeat_count < 1:
                raise SchemaValidationError("motor_command repeat_count must be >= 1")
        elif self.intent_type == "conditional_sense_motor":
            if self.capability_status == "needs_clarification":
                return
            has_target = (
                isinstance(self.target, dict)
                and self.target.get("color") is not None
                and self.target.get("object_type") is not None
            )
            if not has_target:
                raise SchemaValidationError(
                    "conditional_sense_motor requires a color and object_type target"
                )
            if not self.action_name:
                raise SchemaValidationError(
                    "conditional_sense_motor requires action_name"
                )
            if (
                self.steering_directive is None
                or self.steering_directive.stopping_rule != "first_match"
            ):
                raise SchemaValidationError(
                    "conditional_sense_motor requires stopping_rule=first_match"
                )
        elif self.intent_type == "mission_contract":
            if not self.mission_steps or len(self.mission_steps) < 2:
                raise SchemaValidationError(
                    "mission_contract requires mission_steps with at least 2 task utterances"
                )
        elif self.intent_type == "steering_directive":
            if self.steering_directive is None:
                raise SchemaValidationError(
                    "steering_directive requires a steering_directive payload"
                )


@dataclass
class ProcedureRecipe:
    task_type: str
    steps: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> ProcedureRecipe:
        mapping = _ensure_mapping(data, "ProcedureRecipe")
        return cls(
            task_type=_ensure_str(mapping.get("task_type"), "ProcedureRecipe.task_type"),
            steps=_ensure_str_list(mapping.get("steps"), "ProcedureRecipe.steps"),
            source=_ensure_str(mapping.get("source", "compiler"), "ProcedureRecipe.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "ProcedureRecipe.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "ProcedureRecipe.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "ProcedureRecipe.rationale"),
        )




@dataclass
class EvidenceFrame:
    needs: list[str]
    context: dict[str, Any] = field(default_factory=dict)
    active_step: str | None = None
    step_index: int = 0


@dataclass
class SensePlanTemplate:
    primitives: list[str]
    required_inputs: list[str]
    produces: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> SensePlanTemplate:
        mapping = _ensure_mapping(data, "SensePlanTemplate")
        return cls(
            primitives=_ensure_str_list(mapping.get("primitives"), "SensePlanTemplate.primitives"),
            required_inputs=_ensure_subset(
                _ensure_str_list(
                    mapping.get("required_inputs"),
                    "SensePlanTemplate.required_inputs",
                ),
                SENSE_TEMPLATE_ALLOWED_INPUTS,
                "SensePlanTemplate.required_inputs",
            ),
            produces=_ensure_subset(
                _ensure_str_list(mapping.get("produces"), "SensePlanTemplate.produces"),
                SENSE_TEMPLATE_ALLOWED_OUTPUTS,
                "SensePlanTemplate.produces",
            ),
            source=_ensure_str(mapping.get("source", "compiler"), "SensePlanTemplate.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "SensePlanTemplate.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "SensePlanTemplate.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "SensePlanTemplate.rationale"),
        )




@dataclass
class WorldModelSample:
    mission: str | None = None
    direction: int | None = None
    step_count: int = 0
    raw_image: Any | None = None
    grid_size: tuple[int, int] | None = None
    grid_objects: list[dict[str, Any]] = field(default_factory=list)
    occupancy_grid: list[list[bool]] = field(default_factory=list)
    passable_positions: set[tuple[int, int]] = field(default_factory=set)
    observation_model: str = "unknown"
    visible_cells: set[tuple[int, int]] = field(default_factory=set)
    unseen_cells: set[tuple[int, int]] = field(default_factory=set)
    view_to_global: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    agent_pose: dict[str, Any] | None = None
    target_visible: bool = False
    target_location: tuple[int, int] | None = None
    target_object: dict[str, Any] | None = None
    adjacency_to_target: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "mission": self.mission,
            "direction": self.direction,
            "step_count": self.step_count,
            "grid_size": self.grid_size,
            "observation_model": self.observation_model,
            "visible_cells": sorted(self.visible_cells),
            "unseen_cell_count": len(self.unseen_cells),
            "agent_pose": self.agent_pose,
            "target_visible": self.target_visible,
            "target_location": self.target_location,
            "target_object": self.target_object,
            "adjacency_to_target": self.adjacency_to_target,
        }


@dataclass
class SceneObject:
    object_type: str
    color: str | None
    x: float
    y: float
    state: int | None = None
    z: float | None = None  # absent on 2D substrates (MiniGrid); set on 3D ones

    @property
    def coord(self) -> tuple[float, ...]:
        """(x, y) on a 2D substrate, (x, y, z) when a depth axis is present."""
        return (self.x, self.y) if self.z is None else (self.x, self.y, self.z)


@dataclass
class SceneModel:
    """Structured snapshot of the last sensed scene, projected from WorldModelSample."""

    agent_x: float
    agent_y: float
    agent_dir: int
    grid_width: int
    grid_height: int
    objects: list[SceneObject]
    source: str  # "task_sense" | "idle_sense"
    env_id: str | None = None
    seed: int | None = None
    step_count: int = 0
    agent_z: float | None = None  # absent on 2D substrates; set on 3D ones
    observation_model: str = "unknown"
    visible_cells: list[tuple[int, int]] = field(default_factory=list)
    unseen_cells: list[tuple[int, int]] = field(default_factory=list)

    @property
    def agent_coord(self) -> tuple[float, ...]:
        """(x, y) on a 2D substrate, (x, y, z) when a depth axis is present."""
        if self.agent_z is None:
            return (self.agent_x, self.agent_y)
        return (self.agent_x, self.agent_y, self.agent_z)

    def find(
        self,
        *,
        object_type: str | None = None,
        color: str | None = None,
        exclude_colors: list[str] | None = None,
    ) -> list[SceneObject]:
        result: list[SceneObject] = self.objects
        if object_type is not None:
            result = [o for o in result if o.object_type == object_type]
        if color is not None:
            result = [o for o in result if o.color == color]
        if exclude_colors:
            result = [o for o in result if o.color not in exclude_colors]
        return result

    def manhattan_distance_from_agent(self, obj: SceneObject) -> float:
        return geometry.manhattan(obj.coord, self.agent_coord)

    @classmethod
    def from_world_model_sample(
        cls,
        sample: WorldModelSample,
        *,
        source: str,
        env_id: str | None = None,
        seed: int | None = None,
    ) -> SceneModel:
        agent_pose = sample.agent_pose or {}
        grid_w, grid_h = sample.grid_size if sample.grid_size else (0, 0)
        objects = [
            SceneObject(
                object_type=obj["type"],
                color=obj.get("color"),
                x=geometry.as_coord(obj["x"]),
                y=geometry.as_coord(obj["y"]),
                state=obj.get("state"),
                z=geometry.as_coord(obj["z"]) if obj.get("z") is not None else None,
            )
            for obj in (sample.grid_objects or [])
        ]
        agent_z = agent_pose.get("z")
        return cls(
            agent_x=geometry.as_coord(agent_pose.get("x", 0)),
            agent_y=geometry.as_coord(agent_pose.get("y", 0)),
            agent_dir=int(agent_pose.get("dir", 0)),
            agent_z=geometry.as_coord(agent_z) if agent_z is not None else None,
            grid_width=int(grid_w),
            grid_height=int(grid_h),
            objects=objects,
            source=source,
            env_id=env_id,
            seed=seed,
            step_count=sample.step_count,
            observation_model=sample.observation_model,
            visible_cells=sorted(sample.visible_cells),
            unseen_cells=sorted(sample.unseen_cells),
        )


@dataclass
class GroundedObjectEntry:
    color: str | None
    x: float
    y: float
    distance: float  # float for Euclidean and other non-integer metrics
    object_type: str = "unknown"
    metric: str | None = None       # e.g. "manhattan", "euclidean"
    provenance: str | None = None   # primitive handle that produced this entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.object_type,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "distance": self.distance,
            "metric": self.metric,
            "provenance": self.provenance,
        }


@dataclass
class EnvironmentIdentity:
    """Stable identity for the world/task context, separate from scene state."""

    env_id: str | None = None
    seed: int | str | None = None
    grid_width: int | None = None
    grid_height: int | None = None
    mission: str | None = None
    task_family: str | None = None
    scene_fingerprint: str | None = None
    substrate_fingerprint: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "mission": self.mission,
            "task_family": self.task_family,
            "scene_fingerprint": self.scene_fingerprint,
            "substrate_fingerprint": self.substrate_fingerprint,
            "summary": dict(self.summary),
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EnvironmentIdentity":
        mapping = _ensure_mapping(data, "EnvironmentIdentity")
        seed = mapping.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, str))):
            raise SchemaValidationError("EnvironmentIdentity.seed must be an integer, string, or null")
        return cls(
            env_id=_ensure_optional_str(mapping.get("env_id"), "EnvironmentIdentity.env_id"),
            seed=seed,
            grid_width=_ensure_optional_int(mapping.get("grid_width"), "EnvironmentIdentity.grid_width"),
            grid_height=_ensure_optional_int(mapping.get("grid_height"), "EnvironmentIdentity.grid_height"),
            mission=_ensure_optional_str(mapping.get("mission"), "EnvironmentIdentity.mission"),
            task_family=_ensure_optional_str(mapping.get("task_family"), "EnvironmentIdentity.task_family"),
            scene_fingerprint=_ensure_optional_str(
                mapping.get("scene_fingerprint"),
                "EnvironmentIdentity.scene_fingerprint",
            ),
            substrate_fingerprint=_ensure_optional_str(
                mapping.get("substrate_fingerprint"),
                "EnvironmentIdentity.substrate_fingerprint",
            ),
            summary=_ensure_dict(mapping.get("summary", {}), "EnvironmentIdentity.summary"),
        )

    def fingerprint(self) -> str:
        stable = {
            "env_id": self.env_id,
            "seed": self.seed,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "mission": self.mission,
            "task_family": self.task_family,
            "substrate_fingerprint": self.substrate_fingerprint,
            "summary": self.summary,
        }
        return _fp.fingerprint(stable, default=str)


@dataclass
class OperationalContext:
    """Typed situation frame that gives domain meaning to substrate abilities."""

    context_id: str
    substrate_id: str
    version: str = "1"
    object_vocabulary: list[str] = field(default_factory=list)
    attribute_vocabulary: list[str] = field(default_factory=list)
    task_families: list[dict[str, Any]] = field(default_factory=list)
    reference_semantics: dict[str, Any] = field(default_factory=dict)
    grounding_semantics: dict[str, Any] = field(default_factory=dict)
    claim_rules: dict[str, Any] = field(default_factory=dict)
    display_rules: dict[str, Any] = field(default_factory=dict)
    environment_identity_fields: list[str] = field(default_factory=list)
    procedure_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            raise SchemaValidationError("OperationalContext requires context_id")
        if not self.substrate_id:
            raise SchemaValidationError("OperationalContext requires substrate_id")
        if not self.version:
            raise SchemaValidationError("OperationalContext requires version")
        self.object_vocabulary = _ensure_str_list(
            self.object_vocabulary,
            "OperationalContext.object_vocabulary",
        )
        self.attribute_vocabulary = _ensure_str_list(
            self.attribute_vocabulary,
            "OperationalContext.attribute_vocabulary",
        )
        self.environment_identity_fields = _ensure_str_list(
            self.environment_identity_fields,
            "OperationalContext.environment_identity_fields",
        )
        raw_task_families = _ensure_list(
            self.task_families,
            "OperationalContext.task_families",
        )
        self.task_families = [
            _ensure_dict(item, f"OperationalContext.task_families[{idx}]")
            for idx, item in enumerate(raw_task_families)
        ]
        self.reference_semantics = _ensure_dict(
            self.reference_semantics,
            "OperationalContext.reference_semantics",
        )
        self.grounding_semantics = _ensure_dict(
            self.grounding_semantics,
            "OperationalContext.grounding_semantics",
        )
        self.claim_rules = _ensure_dict(
            self.claim_rules,
            "OperationalContext.claim_rules",
        )
        self.display_rules = _ensure_dict(
            self.display_rules,
            "OperationalContext.display_rules",
        )
        self.procedure_hints = _ensure_dict(
            self.procedure_hints,
            "OperationalContext.procedure_hints",
        )
        self.metadata = _ensure_dict(
            self.metadata,
            "OperationalContext.metadata",
        )

    @classmethod
    def from_dict(cls, data: Any) -> "OperationalContext":
        mapping = _ensure_mapping(data, "OperationalContext")
        return cls(
            context_id=_ensure_str(mapping.get("context_id"), "OperationalContext.context_id"),
            substrate_id=_ensure_str(mapping.get("substrate_id"), "OperationalContext.substrate_id"),
            version=_ensure_str(mapping.get("version", "1"), "OperationalContext.version"),
            object_vocabulary=_ensure_str_list(
                mapping.get("object_vocabulary", []),
                "OperationalContext.object_vocabulary",
            ),
            attribute_vocabulary=_ensure_str_list(
                mapping.get("attribute_vocabulary", []),
                "OperationalContext.attribute_vocabulary",
            ),
            task_families=[
                _ensure_dict(item, f"OperationalContext.task_families[{idx}]")
                for idx, item in enumerate(
                    _ensure_list(
                        mapping.get("task_families", []),
                        "OperationalContext.task_families",
                    )
                )
            ],
            reference_semantics=_ensure_dict(
                mapping.get("reference_semantics", {}),
                "OperationalContext.reference_semantics",
            ),
            grounding_semantics=_ensure_dict(
                mapping.get("grounding_semantics", {}),
                "OperationalContext.grounding_semantics",
            ),
            claim_rules=_ensure_dict(
                mapping.get("claim_rules", {}),
                "OperationalContext.claim_rules",
            ),
            display_rules=_ensure_dict(
                mapping.get("display_rules", {}),
                "OperationalContext.display_rules",
            ),
            environment_identity_fields=_ensure_str_list(
                mapping.get("environment_identity_fields", []),
                "OperationalContext.environment_identity_fields",
            ),
            procedure_hints=_ensure_dict(
                mapping.get("procedure_hints", {}),
                "OperationalContext.procedure_hints",
            ),
            metadata=_ensure_dict(
                mapping.get("metadata", {}),
                "OperationalContext.metadata",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "substrate_id": self.substrate_id,
            "version": self.version,
            "object_vocabulary": list(self.object_vocabulary),
            "attribute_vocabulary": list(self.attribute_vocabulary),
            "task_families": [dict(item) for item in self.task_families],
            "reference_semantics": dict(self.reference_semantics),
            "grounding_semantics": dict(self.grounding_semantics),
            "claim_rules": dict(self.claim_rules),
            "display_rules": dict(self.display_rules),
            "environment_identity_fields": list(self.environment_identity_fields),
            "procedure_hints": dict(self.procedure_hints),
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        stable = {
            "context_id": self.context_id,
            "substrate_id": self.substrate_id,
            "version": self.version,
            "object_vocabulary": self.object_vocabulary,
            "attribute_vocabulary": self.attribute_vocabulary,
            "task_families": self.task_families,
            "reference_semantics": self.reference_semantics,
            "grounding_semantics": self.grounding_semantics,
            "claim_rules": self.claim_rules,
            "display_rules": self.display_rules,
            "environment_identity_fields": self.environment_identity_fields,
            "procedure_hints": self.procedure_hints,
            "metadata": self.metadata,
        }
        return _fp.fingerprint(stable, default=str)

    def compact_slice(self, utterance: str | None = None) -> dict[str, Any]:
        task_families = [
            {
                key: value
                for key, value in task.items()
                if key in {"task_type", "canonical_pattern", "object_types", "required_attributes"}
            }
            for task in self.task_families
        ]
        return {
            "context_id": self.context_id,
            "substrate_id": self.substrate_id,
            "version": self.version,
            "fingerprint": self.fingerprint(),
            "utterance": utterance,
            "object_vocabulary": list(self.object_vocabulary),
            "attribute_vocabulary": list(self.attribute_vocabulary),
            "task_families": task_families,
            "reference_semantics": dict(self.reference_semantics),
            "grounding_semantics": dict(self.grounding_semantics),
        }


@dataclass
class StationActiveClaims:
    """Session-scoped claims produced by grounding queries.

    Tied to a SceneModel fingerprint (agent_x, agent_y, step_count).
    Cleared on reset and at task start. Never written to durable memory.
    """

    scene_fingerprint: tuple[int, int, int]  # (agent_x, agent_y, step_count)
    ranked_scene_doors: list[GroundedObjectEntry]
    last_grounded_target: GroundedObjectEntry
    last_grounded_rank: int
    last_grounding_query: dict[str, Any]
    environment_fingerprint: str | None = None
    environment_identity: EnvironmentIdentity | None = None
    confidence: float = 1.0
    frame_id: str | None = None
    source: str = "grounding"
    authority: str = "runtime"

    @property
    def ranked_objects(self) -> list[GroundedObjectEntry]:
        """Object-generic view over the legacy MiniGrid field name."""
        return self.ranked_scene_doors

    def is_valid_for(
        self,
        scene: SceneModel,
        environment_identity: EnvironmentIdentity | None = None,
    ) -> bool:
        scene_valid = self.scene_fingerprint == (scene.agent_x, scene.agent_y, scene.step_count)
        if not scene_valid:
            return False
        if environment_identity is None:
            return True
        if self.environment_fingerprint is None:
            return False
        return self.environment_fingerprint == environment_identity.fingerprint()

    def next_ranked(self) -> tuple[GroundedObjectEntry, int] | tuple[None, None]:
        rank = self.last_grounded_rank + 1
        if rank < len(self.ranked_scene_doors):
            return self.ranked_scene_doors[rank], rank
        return None, None

    def other_objects(self) -> list[GroundedObjectEntry]:
        t = self.last_grounded_target
        return [
            obj for obj in self.ranked_objects
            if not (obj.x == t.x and obj.y == t.y)
        ]

    def other_doors(self) -> list[GroundedObjectEntry]:
        """Backward-compatible alias for older MiniGrid claim consumers."""
        return self.other_objects()

    def compact_summary(self) -> dict[str, Any]:
        ranked_objects = [
            f"{obj.color} {obj.object_type}@{obj.distance}"
            for obj in self.ranked_objects
        ]
        return {
            "object_type": self.last_grounded_target.object_type,
            "last_grounded_target": (
                f"{self.last_grounded_target.color} "
                f"{self.last_grounded_target.object_type} @ distance "
                f"{self.last_grounded_target.distance}"
            ),
            "ranked_objects": ranked_objects,
            "ranked_doors": [
                f"{obj.color}@{obj.distance}" for obj in self.ranked_scene_doors
            ],
            "last_rank": self.last_grounded_rank,
            "environment_fingerprint": self.environment_fingerprint,
            "confidence": self.confidence,
            "frame_id": self.frame_id,
            "source": self.source,
            "authority": self.authority,
        }


@dataclass
class ArbitrationDecision:
    """Typed arbitration decision produced by CapabilityArbitrator when a capability gap is detected.

    decision_type must be one of ARBITRATION_DECISION_TYPES.
    safe_to_execute must be False for refuse and synthesize decision types.
    """

    decision_type: str
    safe_to_execute: bool
    reason: str
    suggested_handle: str | None = None
    clarification_prompt: str | None = None
    operator_message: str = ""
    proposed_handle: str | None = None
    proposed_description: str | None = None
    proposed_condition: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.decision_type not in ARBITRATION_DECISION_TYPES:
            raise SchemaValidationError(
                f"ArbitrationDecision.decision_type must be one of: "
                + ", ".join(ARBITRATION_DECISION_TYPES)
            )
        if self.decision_type in {"refuse", "synthesize"} and self.safe_to_execute:
            raise SchemaValidationError(
                f"ArbitrationDecision with decision_type={self.decision_type} "
                "must have safe_to_execute=False"
            )


@dataclass
class ArbitrationTrace:
    """Provenance record for one arbitration event."""

    utterance: str
    intent_type: str
    required_capabilities: list[str]
    missing_handles: list[str]
    synthesizable_handles: list[str]
    decision: ArbitrationDecision

    def compact(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "intent_type": self.intent_type,
            "required_capabilities": self.required_capabilities,
            "missing_handles": self.missing_handles,
            "synthesizable_handles": self.synthesizable_handles,
            "decision_type": self.decision.decision_type,
            "safe_to_execute": self.decision.safe_to_execute,
            "reason": self.decision.reason,
        }


@dataclass
class ObservationClaim:
    """L1 sensory output stored in Cortex's internal claim store.

    Wraps a raw evidence value with provenance so every fact inside the
    Cortex execution loop has a traceable source and scope.
    """

    key: str                      # evidence name, e.g. "target_location"
    value: Any                    # raw value, e.g. (3, 4) or True
    source: str = "sense"         # component that produced it
    level: str = "command"        # "primitive" | "command"
    confidence: float = 1.0
    scope: str = "grounding"
    freshness: str = "current"    # current | unverifiable | stale | unknown
    last_observed_tick: int | None = None  # step_count when last observed in-view


@dataclass
class ExecutionClaim:
    """L1 motor output — provenance record for a completed motor primitive or command."""

    source_primitive: str         # e.g. "move_forward" / "navigate_to_object"
    level: str                    # "primitive" | "command" | "procedure" | "task"
    scope: str = "motor"
    success: bool = True
    steps_taken: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimRecord:
    """Representation-store claim wrapper.

    Existing specialized claim types remain valid at block boundaries. ClaimRecord
    is the small common shape used by the knowledge surface so facts, beliefs,
    hypotheses, operator assertions, observations, and execution results retain
    authority/provenance/freshness.
    """

    claim_id: str
    key: str
    value: Any
    kind: str
    status: str
    scope: str
    authority: str
    source: str
    confidence: float = 1.0
    valid_until: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    freshness: str = "current"
    invalidation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CLAIM_KINDS:
            raise SchemaValidationError(
                f"ClaimRecord.kind must be one of: {', '.join(CLAIM_KINDS)}"
            )
        if self.status not in CLAIM_STATUSES:
            raise SchemaValidationError(
                f"ClaimRecord.status must be one of: {', '.join(CLAIM_STATUSES)}"
            )
        if self.scope not in CLAIM_SCOPES:
            raise SchemaValidationError(
                f"ClaimRecord.scope must be one of: {', '.join(CLAIM_SCOPES)}"
            )
        if self.authority not in CLAIM_AUTHORITIES:
            raise SchemaValidationError(
                f"ClaimRecord.authority must be one of: {', '.join(CLAIM_AUTHORITIES)}"
            )
        if self.freshness not in CLAIM_FRESHNESS:
            raise SchemaValidationError(
                f"ClaimRecord.freshness must be one of: {', '.join(CLAIM_FRESHNESS)}"
            )
        if self.valid_until is not None:
            if isinstance(self.valid_until, bool) or not isinstance(
                self.valid_until, (int, float)
            ):
                raise SchemaValidationError("ClaimRecord.valid_until must be numeric or null")
            self.valid_until = float(self.valid_until)
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise SchemaValidationError("ClaimRecord.confidence must be numeric")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SchemaValidationError("ClaimRecord.confidence must be between 0 and 1")
        self.confidence = float(self.confidence)

    @classmethod
    def from_dict(cls, data: Any) -> "ClaimRecord":
        mapping = _ensure_mapping(data, "ClaimRecord")
        valid_until_raw = mapping.get("valid_until")
        if valid_until_raw is not None:
            if isinstance(valid_until_raw, bool) or not isinstance(
                valid_until_raw, (int, float)
            ):
                raise SchemaValidationError("ClaimRecord.valid_until must be numeric or null")
            valid_until = float(valid_until_raw)
        else:
            valid_until = None
        return cls(
            claim_id=_ensure_str(mapping.get("claim_id"), "ClaimRecord.claim_id"),
            key=_ensure_str(mapping.get("key"), "ClaimRecord.key"),
            value=mapping.get("value"),
            kind=_ensure_str(mapping.get("kind"), "ClaimRecord.kind"),
            status=_ensure_str(mapping.get("status"), "ClaimRecord.status"),
            scope=_ensure_str(mapping.get("scope"), "ClaimRecord.scope"),
            authority=_ensure_str(mapping.get("authority"), "ClaimRecord.authority"),
            source=_ensure_str(mapping.get("source"), "ClaimRecord.source"),
            confidence=float(mapping.get("confidence", 1.0)),
            valid_until=valid_until,
            provenance=_ensure_dict(mapping.get("provenance", {}), "ClaimRecord.provenance"),
            freshness=_ensure_str(mapping.get("freshness", "current"), "ClaimRecord.freshness"),
            invalidation=_ensure_dict(mapping.get("invalidation", {}), "ClaimRecord.invalidation"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "key": self.key,
            "value": self.value,
            "kind": self.kind,
            "status": self.status,
            "scope": self.scope,
            "authority": self.authority,
            "source": self.source,
            "confidence": self.confidence,
            "valid_until": self.valid_until,
            "provenance": dict(self.provenance),
            "freshness": self.freshness,
            "invalidation": dict(self.invalidation),
        }


@dataclass
class KnowledgeSnapshot:
    """Typed snapshot consumed by planning/readiness instead of station fields."""

    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    procedures: dict[str, dict[str, Any]] = field(default_factory=dict)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    active_claims: StationActiveClaims | None = None
    scene_model: SceneModel | None = None
    environment_identity: EnvironmentIdentity | None = None
    claims_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": {key: claim.as_dict() for key, claim in self.claims.items()},
            "procedures": {key: dict(value) for key, value in self.procedures.items()},
            "provenance": [dict(item) for item in self.provenance],
            "active_claims": (
                self.active_claims.compact_summary()
                if self.active_claims is not None
                else None
            ),
            "scene_model": (
                {
                    "agent_x": self.scene_model.agent_x,
                    "agent_y": self.scene_model.agent_y,
                    "agent_dir": self.scene_model.agent_dir,
                    "step_count": self.scene_model.step_count,
                    "source": self.scene_model.source,
                }
                if self.scene_model is not None
                else None
            ),
            "environment_identity": (
                self.environment_identity.as_dict()
                if self.environment_identity is not None
                else None
            ),
            "claims_valid": self.claims_valid,
            "metadata": dict(self.metadata),
        }


@dataclass
class OperationalEvidence:
    claims: dict[str, Any]
    confidence: float = 1.0
    source: str = "sense"


@dataclass
class Percepts:
    cues: dict[str, Any] = field(default_factory=dict)
    source: str = "sense"


@dataclass
class ExecutionContract:
    skill: str
    params: dict[str, Any] = field(default_factory=dict)
    stop_conditions: list[str] = field(default_factory=list)
    source: str = "cortex"


MotorSkillRequest = ExecutionContract  # L3 task motor request — hierarchy alias


@dataclass
class SkillPlanTemplate:
    primitives: list[str]
    required_inputs: list[str]
    produces: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> SkillPlanTemplate:
        mapping = _ensure_mapping(data, "SkillPlanTemplate")
        return cls(
            primitives=_ensure_str_list(mapping.get("primitives"), "SkillPlanTemplate.primitives"),
            required_inputs=_ensure_subset(
                _ensure_str_list(
                    mapping.get("required_inputs"),
                    "SkillPlanTemplate.required_inputs",
                ),
                SKILL_TEMPLATE_ALLOWED_INPUTS,
                "SkillPlanTemplate.required_inputs",
            ),
            produces=_ensure_subset(
                _ensure_str_list(mapping.get("produces"), "SkillPlanTemplate.produces"),
                SKILL_TEMPLATE_ALLOWED_OUTPUTS,
                "SkillPlanTemplate.produces",
            ),
            source=_ensure_str(mapping.get("source", "compiler"), "SkillPlanTemplate.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "SkillPlanTemplate.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "SkillPlanTemplate.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "SkillPlanTemplate.rationale"),
        )


MotorCommandTemplate = SkillPlanTemplate  # L1 motor command template — hierarchy alias


@dataclass
class ExecutionReport:
    status: str
    progress: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    source: str = "spine"


@dataclass
class ExecutionContext:
    active_skill: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionContract:
    """L4 goal — approved tasks/procedure with explicit success conditions.

    Distinct from sequence_instruction (L2 procedure): MissionContract carries
    abort-on-failure semantics and may carry the validated ProcedureRecipe and
    parameters that an ExecutionTicket authorizes.
    """

    mission_id: str
    description: str
    task_sequence: list[str]          # ordered raw task utterances; one for a conditional mission
    success_condition: str = "all_complete"
    abort_on_failure: bool = True
    risk_tier: str = "low"
    cadence: str | None = None
    procedure: ProcedureRecipe | None = None
    params: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mission_id:
            raise SchemaValidationError("MissionContract requires mission_id")
        if not self.task_sequence:
            raise SchemaValidationError("MissionContract requires at least one task")
        if self.procedure is not None and not self.procedure.steps:
            raise SchemaValidationError("MissionContract procedure requires at least one step")
        if self.procedure is not None and not self.procedure.validated:
            raise SchemaValidationError("MissionContract procedure must be validated")


@dataclass
class CorticalEnvelope:
    """Typed record for one operator turn through the cortical control plane."""

    envelope_id: str
    utterance: str
    intent: OperatorIntent | None = None
    request_plan: RequestPlan | None = None
    readiness_graph: ReadinessGraph | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    pending_context: dict[str, Any] = field(default_factory=dict)


@dataclass(init=False)
class ApprovedCommand:
    """Typed station command authorized by a RequestPlan + ReadinessGraph."""

    command_type: str
    request_id: str = ""
    source: str = "station"
    utterance: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    request_plan: RequestPlan | None = None
    readiness_graph: ReadinessGraph | None = None
    capability_match: Any = None
    ticket: Any = None

    def __init__(
        self,
        command_type: str | None = None,
        *,
        kind: str | None = None,
        request_id: str = "",
        source: str = "station",
        utterance: str = "",
        payload: dict[str, Any] | None = None,
        request_plan: RequestPlan | None = None,
        readiness_graph: ReadinessGraph | None = None,
        capability_match: Any = None,
        ticket: Any = None,
    ) -> None:
        self.command_type = command_type if command_type is not None else kind or ""
        self.request_id = request_id
        self.source = source
        self.utterance = utterance
        self.payload = dict(payload or {})
        self.request_plan = request_plan
        self.readiness_graph = readiness_graph
        self.capability_match = capability_match
        self.ticket = ticket
        self.__post_init__()

    @property
    def kind(self) -> str:
        return self.command_type

    @kind.setter
    def kind(self, value: str) -> None:
        self.command_type = value

    def __post_init__(self) -> None:
        if not self.command_type:
            raise SchemaValidationError("ApprovedCommand requires command_type")
        if self.readiness_graph is not None and self.request_id:
            if self.readiness_graph.request_id != self.request_id:
                raise SchemaValidationError("ApprovedCommand request_id must match ReadinessGraph")
        if self.request_plan is not None and self.request_id:
            if self.request_plan.request_id != self.request_id:
                raise SchemaValidationError("ApprovedCommand request_id must match RequestPlan")


@dataclass
class ExecutionTicket:
    """Authority token required before a task can enter the runtime loop."""

    request_id: str
    instruction: str
    task_type: str
    params: dict[str, Any]
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    source: str = "station"
    mission_id: str | None = None
    parent_request_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    mission_contract: MissionContract | None = None

    def __post_init__(self) -> None:
        if self.readiness_graph.request_id != self.request_id:
            raise SchemaValidationError("ExecutionTicket request_id must match ReadinessGraph")
        if self.request_plan.request_id != self.request_id:
            raise SchemaValidationError("ExecutionTicket request_id must match RequestPlan")
        if self.readiness_graph.graph_status != "executable":
            raise SchemaValidationError("ExecutionTicket requires executable readiness graph")
        if self.readiness_graph.next_action != "execute_task":
            raise SchemaValidationError("ExecutionTicket requires next_action=execute_task")
        if self.mission_contract is not None:
            contract = self.mission_contract
            if self.mission_id != contract.mission_id:
                raise SchemaValidationError(
                    "ExecutionTicket mission_id must match MissionContract"
                )
            if contract.procedure is None:
                raise SchemaValidationError(
                    "ExecutionTicket MissionContract requires an approved procedure"
                )
            if self.task_type != contract.procedure.task_type:
                raise SchemaValidationError(
                    "ExecutionTicket task_type must match MissionContract procedure"
                )
            if self.params != contract.params:
                raise SchemaValidationError(
                    "ExecutionTicket params must match MissionContract params"
                )


@dataclass
class MemoryWriteTicket:
    """Authority token for durable operator-claim or memory mutation."""

    request_id: str
    writes: list[MemoryUpdate]
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    source: str = "station"

    def __post_init__(self) -> None:
        if self.readiness_graph.request_id != self.request_id:
            raise SchemaValidationError("MemoryWriteTicket request_id must match ReadinessGraph")
        if self.request_plan.request_id != self.request_id:
            raise SchemaValidationError("MemoryWriteTicket request_id must match RequestPlan")
        if self.readiness_graph.graph_status != "executable":
            raise SchemaValidationError("MemoryWriteTicket requires executable readiness graph")
        if self.readiness_graph.next_action != "update_memory":
            raise SchemaValidationError("MemoryWriteTicket requires next_action=update_memory")
        if not self.writes:
            raise SchemaValidationError("MemoryWriteTicket requires at least one write")


@dataclass
class RawMotorTicket:
    """Authority token for an explicit low-level motor command."""

    request_id: str
    action_name: str
    repeat_count: int
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    source: str = "station"

    def __post_init__(self) -> None:
        if self.readiness_graph.request_id != self.request_id:
            raise SchemaValidationError("RawMotorTicket request_id must match ReadinessGraph")
        if self.request_plan.request_id != self.request_id:
            raise SchemaValidationError("RawMotorTicket request_id must match RequestPlan")
        if self.readiness_graph.graph_status != "executable":
            raise SchemaValidationError("RawMotorTicket requires executable readiness graph")
        if self.readiness_graph.next_action != "execute_motor":
            raise SchemaValidationError("RawMotorTicket requires next_action=execute_motor")
        if not self.action_name:
            raise SchemaValidationError("RawMotorTicket requires action_name")
        if self.repeat_count < 1:
            raise SchemaValidationError("RawMotorTicket repeat_count must be >= 1")


@dataclass
class SenseTicket:
    """Authority token required before invoking a grounding/ranking primitive.

    Issued on a cache miss in _ensure_ranked_door_claims once the readiness
    graph confirms the full request plan is executable.  Sense is an
    intermediate step, so no next_action check is applied — only the graph
    status gate matters.
    """

    request_id: str
    primitive_handle: str
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    source: str = "station"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.readiness_graph.request_id != self.request_id:
            raise SchemaValidationError("SenseTicket request_id must match ReadinessGraph")
        if self.request_plan.request_id != self.request_id:
            raise SchemaValidationError("SenseTicket request_id must match RequestPlan")
        if not self.primitive_handle:
            raise SchemaValidationError("SenseTicket requires a non-empty primitive_handle")
        # Refuse/unsupported intents must never trigger sense (the graph itself rejected the request).
        if self.readiness_graph.graph_status in {"refuse", "unsupported"}:
            raise SchemaValidationError(
                f"SenseTicket: graph_status='{self.readiness_graph.graph_status}' "
                "— sense not authorized for refused/unsupported intents"
            )
        # When the full plan is executable, all steps are authorized — sense is fine.
        if self.readiness_graph.graph_status == "executable":
            return
        # Partial plans (e.g. synthesizable filter step, but executable ranking step):
        # allow sense if the specific grounding node for this primitive is executable.
        grounding_node_ready = any(
            node.required_handle == self.primitive_handle and node.status == "executable"
            for node in self.readiness_graph.nodes
        )
        if not grounding_node_ready:
            raise SchemaValidationError(
                f"SenseTicket: primitive '{self.primitive_handle}' is not authorized — "
                f"no executable grounding node found in readiness graph "
                f"(graph_status={self.readiness_graph.graph_status})"
            )


@dataclass
class MissionExecutionPlan:
    """L4 mission plan plus child execution tickets."""

    mission_id: str
    description: str
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    mission_contract: MissionContract | None = None
    primitive_definition: PrimitiveDefinitionRequest | None = None
    continuation_intent: OperatorIntent | None = None
    continuation_request_plan: RequestPlan | None = None
    continuation_readiness_graph: ReadinessGraph | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    child_tickets: list[ExecutionTicket] = field(default_factory=list)


@dataclass
class FailureOutcome:
    """Typed failure descriptor attached to CommandResult when a command fails.

    Makes traces trainable: a string label does not carry enough structure for
    downstream learning systems to distinguish stuck/progress/blocking/timeout.
    """

    category: str  # "stuck" | "progress" | "blocking_claim" | "timeout" | "budget_exhausted"
    detail: str | None = None
    blocking_claim_handle: str | None = None


class CommandResult(str):
    """Typed user-visible result from a station command."""

    message: str
    envelope: CorticalEnvelope | None
    command: ApprovedCommand | None
    ticket: ExecutionTicket | MemoryWriteTicket | RawMotorTicket | None
    result: dict[str, Any]
    failure_outcome: FailureOutcome | None
    labelled_episode: Any | None

    def __new__(
        cls,
        message: str,
        *,
        envelope: CorticalEnvelope | None = None,
        command: ApprovedCommand | None = None,
        ticket: ExecutionTicket | MemoryWriteTicket | RawMotorTicket | None = None,
        result: dict[str, Any] | None = None,
        failure_outcome: FailureOutcome | None = None,
    ) -> CommandResult:
        obj = str.__new__(cls, message)
        obj.message = message
        obj.envelope = envelope
        obj.command = command
        obj.ticket = ticket
        obj.result = dict(result or {})
        obj.failure_outcome = failure_outcome
        obj.labelled_episode = None
        return obj


def _decode_memory_value(value: Any) -> Any:
    """Decode a memory-update value off the wire.

    The strict LLM schema transports `value` as a JSON-encoded string (strict mode
    cannot carry free-form objects). Parse it back to native here. A bare scalar
    string that is not valid JSON (e.g. 'red') is kept verbatim. Non-string values
    — the deterministic path constructs MemoryUpdate directly with native values —
    pass through untouched.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


@dataclass
class MemoryUpdate:
    scope: str
    key: str
    value: Any
    reason: str

    @classmethod
    def from_dict(cls, data: Any) -> MemoryUpdate:
        mapping = _ensure_mapping(data, "MemoryUpdate")
        return cls(
            scope=_ensure_str(mapping.get("scope"), "MemoryUpdate.scope"),
            key=_ensure_str(mapping.get("key"), "MemoryUpdate.key"),
            value=_decode_memory_value(mapping.get("value")),
            reason=_ensure_str(mapping.get("reason", ""), "MemoryUpdate.reason"),
        )


@dataclass
class ReadinessReport:
    status: str
    task_type: str
    missing_task_primitives: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    missing_actions: list[str] = field(default_factory=list)
    recipe_steps: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    loop_index: int | None = None
    step_name: str | None = None


@dataclass
class PlanCacheEntry:
    key: str
    template_type: Literal["procedure", "sense", "skill"]
    template: ProcedureRecipe | SensePlanTemplate | SkillPlanTemplate
    compiler_backend: str
    source: str
    created_at_loop: int
    hit_count: int = 0


@dataclass
class PlanCacheStats:
    hits: int = 0
    misses: int = 0
    llm_calls_saved: int = 0


SensePrimitivePlan = SensePlanTemplate
SpineSkillPlan = SkillPlanTemplate


def task_request_json_schema(task_types: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "task_type": {"type": "string", "enum": task_types},
            "params": primitive_params_json_schema(),
            "source": {"type": "string"},
        },
        "required": ["instruction", "task_type", "params", "source"],
        "additionalProperties": False,
    }


def procedure_recipe_json_schema(primitive_names: list[str], task_types: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "enum": task_types},
            "steps": {
                "type": "array",
                "items": {"type": "string", "enum": primitive_names},
            },
            "source": {"type": "string"},
            "compiler_backend": {"type": "string"},
            "validated": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "task_type",
            "steps",
            "source",
            "compiler_backend",
            "validated",
            "rationale",
        ],
        "additionalProperties": False,
    }


def sense_plan_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return template_json_schema(
        primitive_names=primitive_names,
        allowed_inputs=list(SENSE_TEMPLATE_ALLOWED_INPUTS),
        allowed_outputs=list(SENSE_TEMPLATE_ALLOWED_OUTPUTS),
    )


def skill_plan_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return template_json_schema(
        primitive_names=primitive_names,
        allowed_inputs=list(SKILL_TEMPLATE_ALLOWED_INPUTS),
        allowed_outputs=list(SKILL_TEMPLATE_ALLOWED_OUTPUTS),
    )


def template_json_schema(
    primitive_names: list[str],
    allowed_inputs: list[str],
    allowed_outputs: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primitives": {
                "type": "array",
                "items": {"type": "string", "enum": primitive_names},
            },
            "required_inputs": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_inputs},
            },
            "produces": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_outputs},
            },
            "source": {"type": "string"},
            "compiler_backend": {"type": "string"},
            "validated": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "primitives",
            "required_inputs",
            "produces",
            "source",
            "compiler_backend",
            "validated",
            "rationale",
        ],
        "additionalProperties": False,
    }


def memory_updates_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["knowledge", "episodic_memory"],
                        },
                        "key": {"type": "string"},
                        "value": {
                            "type": "string",
                            "description": (
                                "The value to store, JSON-encoded as a string. Scalars may be "
                                "given verbatim (e.g. 'red', '5'); objects and arrays MUST be "
                                "valid JSON text (e.g. '[3, 4]', '{\"x\": 1}'). Strict mode "
                                "cannot carry free-form objects, so everything is transported "
                                "as a string and decoded on receipt."
                            ),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["scope", "key", "value", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["updates"],
        "additionalProperties": False,
    }


def operator_intent_json_schema(
    *,
    object_types: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    schema_object_types = list(
        object_types if object_types is not None else get_registered_object_types()
    )
    example_object_type = schema_object_types[0] if schema_object_types else "object"
    target_schema = {
        "type": ["object", "null"],
        "properties": {
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "object_type": {"type": ["string", "null"], "enum": [*schema_object_types, None]},
        },
        "required": ["color", "object_type"],
        "additionalProperties": False,
    }
    target_selector_schema = {
        "type": ["object", "null"],
        "properties": {
            "object_type": {"type": ["string", "null"], "enum": [*schema_object_types, None]},
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "exclude_colors": {
                "type": "array",
                "items": {"type": "string", "enum": list(OPERATOR_COLORS)},
                "description": "Colors to exclude. Use [] when no exclusion. Supports multiple: ['purple', 'yellow'].",
            },
            "relation": {"type": ["string", "null"], "enum": [*OPERATOR_SELECTOR_RELATIONS, None]},
            "distance_metric": {
                "type": ["string", "null"],
                "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
            },
            "distance_reference": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_DISTANCE_REFERENCES, None],
            },
        },
        "required": [
            "object_type",
            "color",
            "exclude_colors",
            "relation",
            "distance_metric",
            "distance_reference",
        ],
        "additionalProperties": False,
    }
    grounding_query_plan_schema = {
        "type": ["object", "null"],
        "properties": {
            "object_type": {"type": ["string", "null"], "enum": [*schema_object_types, None]},
            "operation": {"type": ["string", "null"], "enum": [*GROUNDING_QUERY_OPERATIONS, None]},
            "primitive_handle": {"type": ["string", "null"]},
            "metric": {
                "type": ["string", "null"],
                "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
            },
            "reference": {"type": ["string", "null"], "enum": [*OPERATOR_DISTANCE_REFERENCES, None]},
            "order": {"type": ["string", "null"], "enum": [*GROUNDING_QUERY_ORDERS, None]},
            "ordinal": {"type": ["integer", "null"]},
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "exclude_colors": {
                "type": "array",
                "items": {"type": "string", "enum": list(OPERATOR_COLORS)},
            },
            "distance_value": {"type": ["integer", "null"]},
            "comparison": {
                "type": ["string", "null"],
                "enum": [*GROUNDING_QUERY_COMPARISONS, None],
            },
            "tie_policy": {
                "type": ["string", "null"],
                "enum": [*GROUNDING_QUERY_TIE_POLICIES, None],
            },
            "answer_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "preserved_constraints": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "object_type",
            "operation",
            "primitive_handle",
            "metric",
            "reference",
            "order",
            "ordinal",
            "color",
            "exclude_colors",
            "distance_value",
            "comparison",
            "tie_policy",
            "answer_fields",
            "required_capabilities",
            "preserved_constraints",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "intent_type": {"type": "string", "enum": list(OPERATOR_INTENT_TYPES)},
            "canonical_instruction": {"type": ["string", "null"]},
            "task_type": {"type": ["string", "null"], "enum": [*OPERATOR_TASK_TYPES, None]},
            "target": target_schema,
            "knowledge_update": {
                "type": ["object", "null"],
                "properties": {
                    "delivery_target": target_schema,
                },
                "required": ["delivery_target"],
                "additionalProperties": False,
            },
            "reference": {"type": ["string", "null"], "enum": [*OPERATOR_REFERENCES, None]},
            "status_query": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_STATUS_QUERIES, None],
            },
            "claim_reference": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_CLAIM_REFERENCES, None],
            },
            "control": {"type": ["string", "null"], "enum": [*OPERATOR_CONTROLS, None]},
            "target_selector": target_selector_schema,
            "grounding_query_plan": grounding_query_plan_schema,
            "primitive_definition": {
                "type": ["object", "null"],
                "properties": {
                    "definition_type": {
                        "type": "string",
                        "enum": ["distance_metric"],
                    },
                    "name": {"type": "string"},
                    "normalized_name": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_]*$",
                    },
                    "expression": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "alias", "min", "max", "sum",
                                    "mod", "add", "subtract",
                                    "abs_diff", "abs_diff_plus",
                                ],
                            },
                            "metric": {"type": ["string", "null"]},
                            "metrics": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "constant": {"type": ["number", "null"]},
                            "reason": {"type": ["string", "null"]},
                        },
                        "required": ["op", "metric", "metrics", "constant", "reason"],
                        "additionalProperties": False,
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "dependency_handles": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "proposed_handle": {"type": "string"},
                    "safety_class": {"type": "string", "enum": ["query"]},
                    "authority_level": {"type": "string", "enum": ["operator"]},
                    "provenance": {
                        "type": "object",
                        "properties": {
                            "operator_utterance": {"type": ["string", "null"]},
                            "formula": {"type": ["string", "null"]},
                            "approval_utterance": {"type": ["string", "null"]},
                            "registered_handle": {"type": ["string", "null"]},
                        },
                        "required": [
                            "operator_utterance",
                            "formula",
                            "approval_utterance",
                            "registered_handle",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "definition_type",
                    "name",
                    "normalized_name",
                    "expression",
                    "dependencies",
                    "dependency_handles",
                    "proposed_handle",
                    "safety_class",
                    "authority_level",
                    "provenance",
                ],
                "additionalProperties": False,
            },
            "capability_status": {
                "type": "string",
                "enum": list(OPERATOR_CAPABILITY_STATUSES),
            },
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact capability handles this intent requires. List every handle "
                    "needed, including any not yet in the registry — the station will "
                    "classify missing ones as missing_skills. No weakening: do not "
                    "substitute a broader capability for a specific one."
                ),
            },
            "selection_objective": {
                "type": ["object", "null"],
                "properties": {
                    "attribute": {
                        "type": "string",
                        "description": (
                            "The property being ranked/selected (e.g. 'distance', 'temperature'). "
                            f"Use 'distance' for {example_object_type}-distance queries."
                        ),
                    },
                    "direction": {
                        "type": "string",
                        "enum": list(SELECTION_DIRECTIONS),
                        "description": (
                            "'maximum' for superlatives like farthest/hottest/largest. "
                            "'minimum' for superlatives like closest/coldest/smallest. "
                            "Never use 'ascending' or 'descending' here."
                        ),
                    },
                    "ordinal": {
                        "type": "integer",
                        "description": "1 for 'the farthest', 2 for 'second farthest', etc.",
                    },
                    "metric": {
                        "type": ["string", "null"],
                        "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                    },
                },
                "required": ["attribute", "direction", "ordinal", "metric"],
                "additionalProperties": False,
                "description": (
                    "Structured distillation of the operator's selection intent. "
                    "Set this whenever the request involves picking by a ranked attribute "
                    f"(farthest {example_object_type}, second closest, hottest room, etc.). "
                    "Null for non-ranking intents."
                ),
            },
            "clear_memory": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
            "concept_name": {
                "type": ["string", "null"],
                "description": (
                    "For concept_teach: the operator-defined shorthand label. "
                    "For concept_recall: the name of the concept to execute. "
                    "Must be a clean identifier — lowercase, no trailing punctuation or commas. "
                    "Examples: 'bingo', 'patrol', 'home_base'. "
                    "Do not include surrounding punctuation from the operator's sentence."
                ),
            },
            "concept_utterance": {
                "type": ["string", "null"],
                "description": (
                    "For concept_teach: the full instruction the label expands to "
                    f"(e.g. 'go to the red {example_object_type}'). "
                    "Null for concept_recall."
                ),
            },
            "concept_steps": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "For procedure_recall: ordered list of concept names to execute "
                    "sequentially (e.g. ['bingo', 'scout']). Null for all other intents."
                ),
            },
            "utterance_steps": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "For sequence_instruction: ordered list of raw task utterances to "
                    "execute sequentially (e.g. "
                    f"['go to the red {example_object_type}', "
                    f"'go to the green {example_object_type}']). "
                    "Null for all other intents."
                ),
            },
            "action_name": {
                "type": ["string", "null"],
                "description": (
                    "For motor_command or conditional_sense_motor: the low-level action "
                    "primitive key to authorize "
                    "(e.g. 'move_forward', 'turn_right', 'turn_left'). "
                    "Null for all other intents."
                ),
            },
            "repeat_count": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": (
                    "For motor_command: how many times to execute the action. "
                    "Defaults to 1 if omitted. Null for all other intents."
                ),
            },
            "mission_steps": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "minItems": 2,
                "description": (
                    "For mission_contract: ordered list of raw task utterances constituting "
                    "the mission (e.g. "
                    f"['go to the red {example_object_type}', "
                    f"'go to the green {example_object_type}']). "
                    "Requires at least 2 steps. Null for all other intents."
                ),
            },
            "steering_directive": {
                "type": ["object", "null"],
                "properties": {
                    "budget": {
                        "type": ["object", "null"],
                        "properties": {
                            "max_steps": {"type": ["integer", "null"], "minimum": 0},
                            "max_clarifications": {"type": ["integer", "null"], "minimum": 0},
                        },
                        "required": ["max_steps", "max_clarifications"],
                        "additionalProperties": False,
                    },
                    "scope": {"type": ["string", "null"], "enum": [*STEERING_SCOPES, None]},
                    "risk": {"type": ["string", "null"], "enum": [*STEERING_RISK_LEVELS, None]},
                    "stopping_rule": {
                        "type": ["string", "null"],
                        "enum": [*STEERING_STOPPING_RULES, None],
                    },
                },
                "required": ["budget", "scope", "risk", "stopping_rule"],
                "additionalProperties": False,
                "description": (
                    "Typed steering layer: HOW to approach the task, separate from WHAT. "
                    "Set budget.max_steps to cap execution, risk to gate side effects "
                    "(query_only forbids actuation), scope for search bounds, stopping_rule "
                    "for when to stop. Null when the operator gives no steering."
                ),
            },
        },
        "required": [
            "intent_type",
            "canonical_instruction",
            "task_type",
            "target",
            "knowledge_update",
            "reference",
            "status_query",
            "claim_reference",
            "control",
            "target_selector",
            "grounding_query_plan",
            "primitive_definition",
            "capability_status",
            "required_capabilities",
            "selection_objective",
            "clear_memory",
            "confidence",
            "reason",
            "concept_name",
            "concept_utterance",
            "concept_steps",
            "utterance_steps",
            "action_name",
            "repeat_count",
            "mission_steps",
            "steering_directive",
        ],
        "additionalProperties": False,
    }


def primitive_call_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": primitive_names},
            "params": primitive_params_json_schema(),
        },
        "required": ["name", "params"],
        "additionalProperties": False,
    }


def primitive_params_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "color": {"type": ["string", "null"]},
            "object_type": {"type": ["string", "null"]},
            "target_location": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
            },
        },
        "required": ["color", "object_type", "target_location"],
        "additionalProperties": False,
    }
