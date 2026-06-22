from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Callable

from .planning_semantics import PlanningSemantics
from .primitive_library import library_payload, primitive_names
from .semantic_normalizer import normalize_distance_ordinal, get_semantic_constraints
from .schemas import (
    EvidenceFrame,
    ExecutionContract,
    ExecutionContext,
    MemoryUpdate,
    OPERATOR_COLORS,
    OPERATOR_INTENT_TYPES,
    OPERATOR_REFERENCES,
    OPERATOR_STATUS_QUERIES,
    OPERATOR_TASK_TYPES,
    OperatorIntent,
    ProcedureRecipe,
    SchemaValidationError,
    SelectionObjective,
    SensePlanTemplate,
    SkillPlanTemplate,
    SteeringDirective,
    TaskRequest,
    memory_updates_json_schema,
    get_registered_object_types,
    operator_intent_json_schema,
    procedure_recipe_json_schema,
    sense_plan_json_schema,
    skill_plan_json_schema,
    task_request_json_schema,
)


class CompilerBackend(ABC):
    name = "compiler"

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.call_history: list[dict[str, Any]] = []
        self.planning_semantics: PlanningSemantics | None = None

    def bind_planning_semantics(self, planning_semantics: PlanningSemantics) -> None:
        self.planning_semantics = planning_semantics
        fallback = getattr(self, "fallback", None)
        if isinstance(fallback, CompilerBackend):
            fallback.planning_semantics = planning_semantics

    def object_types(self) -> tuple[str, ...]:
        if self.planning_semantics is not None:
            return self.planning_semantics.object_types
        return get_registered_object_types()

    def default_object_type(self) -> str | None:
        if self.planning_semantics is not None:
            return self.planning_semantics.default_object_type
        object_types = self.object_types()
        return object_types[0] if object_types else None

    def object_type_pattern(self) -> str:
        return "|".join(re.escape(object_type) for object_type in self.object_types())

    def object_type_from_text(self, text: str) -> str | None:
        if self.planning_semantics is not None:
            return self.planning_semantics.object_type_from_text(text)
        normalized = text.strip().lower()
        for object_type in self.object_types():
            if re.search(rf"\b{re.escape(object_type)}s?\b", normalized):
                return object_type
        return None

    def task_handle(self, task_type: str, object_type: str) -> str:
        if self.planning_semantics is not None:
            handle = self.planning_semantics.task_handle(task_type, object_type)
            if handle is not None:
                return handle
        return f"task.{task_type}.{object_type}"

    def pluralize_object_type(self, object_type: str) -> str:
        if self.planning_semantics is not None:
            return self.planning_semantics.pluralize(object_type)
        return object_type if object_type.endswith("s") else f"{object_type}s"

    def ranked_handle(self, metric: str, object_type: str) -> str:
        if self.planning_semantics is not None:
            handle = self.planning_semantics.capability_handle(
                "ranked",
                metric=metric,
                object_type=object_type,
            )
            if handle is not None:
                return handle
        plural = self.pluralize_object_type(object_type)
        return f"grounding.all_{plural}.ranked.{metric}.agent"

    def closest_handle(
        self,
        metric: str,
        object_type: str,
        reference: str = "agent",
    ) -> str:
        if self.planning_semantics is not None:
            handle = self.planning_semantics.capability_handle(
                "closest",
                metric=metric,
                reference=reference,
                object_type=object_type,
            )
            if handle is not None:
                return handle
        return f"grounding.closest_{object_type}.{metric}.{reference}"

    def unique_handle(self, object_type: str) -> str:
        if self.planning_semantics is not None:
            handle = self.planning_semantics.capability_handle(
                "unique",
                object_type=object_type,
            )
            if handle is not None:
                return handle
        return f"grounding.unique_{object_type}.color_filter"

    def motor_object_terms(self) -> tuple[str, ...]:
        if self.planning_semantics is not None:
            return self.planning_semantics.motor_object_terms
        return _MOTOR_TASK_OBJECT_TERMS

    @property
    def active_backend(self) -> str:
        return self.name

    def log(self, message: str) -> None:
        self.logs.append(message)

    def record_call(
        self,
        method_name: str,
        backend: str,
        success: bool,
        used_fallback: bool = False,
        reason: str | None = None,
        requested_max_tokens: int | None = None,
    ) -> None:
        self.call_history.append(
            {
                "method_name": method_name,
                "backend": backend,
                "success": success,
                "used_fallback": used_fallback,
                "reason": reason,
                "requested_max_tokens": requested_max_tokens,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        total_requested_max_tokens = sum(
            call["requested_max_tokens"] or 0 for call in self.call_history
        )
        return {
            "backend": self.active_backend,
            "llm_used": False,
            "llm_success_count": 0,
            "fallback_count": 0,
            "total_requested_max_tokens": total_requested_max_tokens,
            "call_history": list(self.call_history),
        }

    @abstractmethod
    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        raise NotImplementedError

    @abstractmethod
    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        raise NotImplementedError

    @abstractmethod
    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        raise NotImplementedError

    @abstractmethod
    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        raise NotImplementedError

    @abstractmethod
    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        raise NotImplementedError

    @abstractmethod
    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        raise NotImplementedError


def is_dataclass_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def make_json_safe(value: Any) -> Any:
    if is_dataclass_instance(value):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return [make_json_safe(item) for item in sorted(value)]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def canonical_task_params(
    color: str | None = None,
    object_type: str | None = None,
    target_location: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "color": color,
        "object_type": object_type,
        "target_location": target_location,
    }


_MOTOR_ACTION_ALIASES: dict[str, re.Pattern[str]] = {
    "move_forward": re.compile(
        r"\b(?:go\s+(?:straight|forward)|move\s+forward|step\s+forward|walk\s+forward"
        r"|advance(?!\s+to\b)|step\s+ahead)\b"
    ),
    "turn_right": re.compile(r"\b(?:turn|rotate|face)\s+right\b"),
    "turn_left":  re.compile(r"\b(?:turn|rotate|face)\s+left\b"),
    "pickup":     re.compile(r"\b(?:pick\s+up|pickup|grab)\b"),
    "toggle":     re.compile(r"\btoggle\b"),
}

_MOTOR_COUNT_WORDS: dict[str, int] = {
    "once": 1, "one time": 1, "one step": 1,
    "twice": 2, "two times": 2, "two steps": 2, "two step": 2,
    "thrice": 3, "three times": 3, "three steps": 3, "three step": 3,
    "four times": 4, "four steps": 4,
    "five times": 5, "five steps": 5,
    "six times": 6, "six steps": 6,
    "seven times": 7, "seven steps": 7,
    "eight times": 8, "eight steps": 8,
    "nine times": 9, "nine steps": 9,
    "ten times": 10, "ten steps": 10,
}

_WORD_TO_NUM: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Fallback object vocabulary used only when no domain semantics are bound; the
# bound path sources these from PlanningSemantics.motor_object_terms (config).
_MOTOR_TASK_OBJECT_TERMS = (
    "ball",
    "box",
    "door",
    "goal",
    "key",
    "target",
)


def _looks_like_object_task(
    normalized: str,
    object_terms: tuple[str, ...] = _MOTOR_TASK_OBJECT_TERMS,
) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in object_terms)


def _parse_motor_command(
    normalized: str,
    object_terms: tuple[str, ...] = _MOTOR_TASK_OBJECT_TERMS,
) -> tuple[str, int] | None:
    """Return (action_name, count) if the utterance is a direct motor command, else None."""
    for action_name, pattern in _MOTOR_ACTION_ALIASES.items():
        if pattern.search(normalized):
            if action_name in {"pickup", "toggle"} and _looks_like_object_task(normalized, object_terms):
                return None
            m = re.search(r"\b(\d+)\b", normalized)
            if m:
                return action_name, int(m.group(1))
            for phrase, val in _MOTOR_COUNT_WORDS.items():
                if phrase in normalized:
                    return action_name, val
            # Fallback: bare word number anywhere in the segment
            for word, val in _WORD_TO_NUM.items():
                if re.search(rf"\b{word}\b", normalized):
                    return action_name, val
            return action_name, 1
    return None


def _parse_motor_sequence(
    normalized: str,
    object_terms: tuple[str, ...] = _MOTOR_TASK_OBJECT_TERMS,
) -> list[tuple[str, int]] | None:
    """Return ordered (action, count) pairs when the utterance chains multiple motor actions.

    Splits on 'and', 'then', 'and then'. Returns None if any segment is not a
    recognised motor command, so non-motor utterances are never mis-routed.
    """
    parts = re.split(r"\band\s+then\b|\bthen\b|\band\b", normalized)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return None
    results: list[tuple[str, int]] = []
    for part in parts:
        cmd = _parse_motor_command(part, object_terms)
        if cmd is None:
            return None
        results.append(cmd)
    if len(results) < 2:
        return None
    return results


class SmokeTestCompiler(CompilerBackend):
    name = "smoke_test_compiler"

    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        self.record_call(
            method_name="compile_task",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        normalized = instruction.strip().lower()

        object_type_pattern = self.object_type_pattern()
        object_match = (
            re.search(
                rf"go to the (?P<color>\w+) (?P<object_type>{object_type_pattern})\b",
                normalized,
            )
            if object_type_pattern
            else None
        )
        if object_match:
            return TaskRequest(
                instruction=instruction,
                task_type="go_to_object",
                params=canonical_task_params(
                    color=object_match.group("color"),
                    object_type=object_match.group("object_type"),
                ),
                source=self.name,
            )

        goal_match = re.search(
            r"(?:get to|go to|reach|find) the (?:(?P<color>\w+) )?(?P<object_type>goal)",
            normalized,
        )
        if goal_match:
            return TaskRequest(
                instruction=instruction,
                task_type="go_to_object",
                params=canonical_task_params(
                    color=goal_match.group("color"),
                    object_type=goal_match.group("object_type"),
                ),
                source=self.name,
            )

        return TaskRequest(
            instruction=instruction,
            task_type="search_for_object",
            params=canonical_task_params(object_type="goal"),
            source=self.name,
        )

    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        self.record_call(
            method_name="compile_procedure",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        recipe = memory.understanding.get(task.task_type, {}).get("default_recipe")
        if not recipe:
            raise ValueError(f"No default recipe for task type: {task.task_type}")

        self._validate_primitive_names(recipe, available_task_primitives, "task primitive")
        return ProcedureRecipe(
            task_type=task.task_type,
            steps=list(recipe),
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic smoke-test recipe from operational memory.",
        )

    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        self.record_call(
            method_name="compile_sense_plan",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        needs = set(evidence_frame.needs)
        primitives: list[str] = []
        if needs & {"object_location", "agent_pose", "adjacency_to_target", "occupancy_grid"}:
            primitives.append("parse_grid_objects")
            primitives.append("build_occupancy_grid")
        if "object_location" in needs:
            primitives.append("find_object_by_color_type")
        if "agent_pose" in needs:
            primitives.append("get_agent_pose")
        if "adjacency_to_target" in needs:
            primitives.append("check_adjacency")

        self._validate_primitive_names(primitives, available_sensing_primitives, "sensing primitive")

        required_inputs = ["observation"]
        merged_context = dict(execution_context.params)
        merged_context.update(evidence_frame.context)
        if merged_context.get("color") is not None:
            required_inputs.append("color")
        if merged_context.get("object_type") is not None:
            required_inputs.append("object_type")
        return SensePlanTemplate(
            primitives=primitives,
            required_inputs=required_inputs,
            produces=["world_sample", "operational_evidence", "percepts"],
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic sensing template based on evidence needs.",
        )

    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        self.record_call(
            method_name="compile_skill_plan",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        if execution_contract.skill == "navigate_to_object":
            primitives = ["plan_grid_path", "execute_next_path_action"]
            required_inputs = ["agent_pose", "target_location", "occupancy_grid", "direction"]
        elif execution_contract.skill == "done":
            primitives = ["done"]
            required_inputs = ["adjacency_to_target"]
        else:
            primitives = [execution_contract.skill]
            required_inputs = ["execution_contract"]

        self._validate_primitive_names(primitives, available_action_primitives, "action primitive")
        return SkillPlanTemplate(
            primitives=primitives,
            required_inputs=required_inputs,
            produces=["execution_report", "execution_context"],
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic skill template from the execution contract.",
        )

    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        self.record_call(
            method_name="compile_memory_updates",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        task_request = final_state.get("task_request", {})
        resolved_task_params = final_state.get("resolved_task_params", {})
        updates = [
            MemoryUpdate(
                scope="knowledge",
                key="last_instruction",
                value=task_request.get("instruction"),
                reason="Remember the latest operator instruction.",
            ),
            MemoryUpdate(
                scope="knowledge",
                key="last_task_type",
                value=task_request.get("task_type"),
                reason="Remember the latest task type.",
            ),
        ]

        if resolved_task_params.get("color") is not None:
            updates.append(
                MemoryUpdate(
                    scope="knowledge",
                    key="target_color",
                    value=resolved_task_params.get("color"),
                    reason="Persist the active target color as durable knowledge.",
                )
            )
        if resolved_task_params.get("object_type") is not None:
            updates.append(
                MemoryUpdate(
                    scope="knowledge",
                    key="target_type",
                    value=resolved_task_params.get("object_type"),
                    reason="Persist the active target object type as durable knowledge.",
                )
            )
        if final_claims.get("target_location") is not None:
            updates.append(
                MemoryUpdate(
                    scope="episodic_memory",
                    key="known_target_location",
                    value=final_claims.get("target_location"),
                    reason="Store the last observed target location for this episode only.",
                )
            )
        if final_state.get("last_world_sample") is not None:
            updates.append(
                MemoryUpdate(
                    scope="episodic_memory",
                    key="last_world_sample",
                    value=final_state["last_world_sample"],
                    reason="Keep the last world model sample for debugging this run.",
                )
            )
        updates.append(
            MemoryUpdate(
                scope="episodic_memory",
                key="last_trace_length",
                value=len(trace),
                reason="Track how many trace events were emitted in the episode.",
            )
        )
        return updates

    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        self.record_call(
            method_name="compile_operator_intent",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        # When a synthesis proposal is pending, classify acceptance/rejection first.
        if pending_proposal:
            normalized_quick = " ".join(utterance.lower().strip().split())
            _ACCEPT = {"yes", "ok", "okay", "sure", "go ahead", "do it", "yep", "yeah",
                       "please", "sounds good", "go for it", "correct", "that works"}
            _REJECT = {"no", "cancel", "stop", "nope", "don't", "skip", "never mind",
                       "nevermind", "abort"}
            if normalized_quick in _ACCEPT:
                return OperatorIntent(intent_type="accept_proposal", confidence=1.0, reason="")
            if normalized_quick in _REJECT:
                return OperatorIntent(intent_type="reject_proposal", confidence=1.0, reason="")
        normalized = " ".join(utterance.lower().strip().split())

        # Concept teach — check BEFORE door_match so "when i say X go to Y" isn't hijacked.
        # Try comma-delimited first (multi-word label: "when I say fing fam foom, ...").
        _teach_m_early = re.match(
            r"^(?:when|if) (?:i )?say (.+?),\s*"
            r"(?:you need to\s+|please\s+|i want you to\s+|just\s+|automatically\s+)?(.+?)$",
            normalized,
            re.IGNORECASE,
        ) or re.match(
            r"^(?:when|if) (?:i )?say (\S+)\s+"
            r"(?:you need to\s+|please\s+|i want you to\s+|just\s+|automatically\s+)?(.+?)$",
            normalized,
            re.IGNORECASE,
        )
        if _teach_m_early:
            cname = _teach_m_early.group(1).strip().strip("'\"")
            cutterance = _teach_m_early.group(2).strip()
            cutterance = re.sub(
                r"[.!]?\s*(?:can you\s+|please\s+|ok\s+|alright\s+)?"
                r"(?:remember|keep|store|save)(?:\s+(?:that|this))?\s*[?.!]*\s*$",
                "",
                cutterance,
                flags=re.IGNORECASE,
            ).strip().strip("'\"")
            if cname and cutterance:
                return OperatorIntent(
                    intent_type="concept_teach",
                    concept_name=cname,
                    concept_utterance=cutterance,
                    confidence=0.85,
                    reason="Deterministic fallback matched natural-language concept-teach pattern.",
                )

        # Bare "X means Y" concept teach — check before door_match.
        _bare_means_early = re.match(r"^(\S+) means (.+)$", normalized)
        if _bare_means_early:
            cname = _bare_means_early.group(1).strip().strip("'\"")
            cutterance = _bare_means_early.group(2).strip().strip("'\"")
            _reserved = {
                "manhattan", "euclidean", "distance", "closest", "red",
                "blue", "green", "yellow", "purple", "grey", "gray",
                *self.object_types(),
            }
            if cname.lower() not in _reserved and cutterance:
                return OperatorIntent(
                    intent_type="concept_teach",
                    concept_name=cname,
                    concept_utterance=cutterance,
                    confidence=0.8,
                    reason="Deterministic fallback matched bare 'X means Y' concept-teach pattern.",
                )

        # Sequential pattern detected BEFORE navigation/grounding checks.
        # Structural shape (multi-step) must be determined before content-specific
        # patterns run — door_match etc. would otherwise grab only the first step.
        _seq_split_e = re.compile(r"\b(?:and\s+then|then|followed\s+by)\b")
        _seq_pre_e = re.compile(
            r"^\s*(?:(?:do|execute|run|perform|also)\s+)?(?:a\s+|the\s+|an\s+)?(?:first\s+)?"
        )
        _seq_suf_e = re.compile(r"\s+(?:first|next|also|too)\s*$")
        _seq_parts_e = _seq_split_e.split(normalized)
        if len(_seq_parts_e) >= 2:
            _cleaned_e = [
                _seq_suf_e.sub("", _seq_pre_e.sub("", p)).strip()
                for p in _seq_parts_e
            ]
            _cleaned_e = [c for c in _cleaned_e if c]
            if len(_cleaned_e) >= 2:
                if all(re.match(r"^[a-z_][a-z0-9_]*$", c) for c in _cleaned_e):
                    return OperatorIntent(
                        intent_type="procedure_recall",
                        concept_steps=_cleaned_e,
                        confidence=0.8,
                        reason="Deterministic operator-intent fallback detected sequential concept pattern.",
                    )
                else:
                    return OperatorIntent(
                        intent_type="sequence_instruction",
                        utterance_steps=_cleaned_e,
                        confidence=0.75,
                        reason="Deterministic operator-intent fallback detected sequential utterance pattern.",
                    )

        # mission_contract: "mission: step1; step2[; step3...]" — explicit prefix, before door_match
        _mission_m = re.match(r"^mission\s*:\s*(.+)$", normalized)
        if _mission_m:
            raw_steps = [s.strip() for s in _mission_m.group(1).split(";") if s.strip()]
            if len(raw_steps) >= 2:
                return OperatorIntent(
                    intent_type="mission_contract",
                    mission_steps=raw_steps,
                    confidence=0.95,
                    reason=f"Explicit mission contract with {len(raw_steps)} tasks.",
                )

        _color_idx = (capability_manifest or {}).get("symbol_mappings", {}).get("color_index") or {}
        if _color_idx:
            _color_names = list(_color_idx.values())
            if "grey" in _color_names and "gray" not in _color_names:
                _color_names.append("gray")
            color_pattern = "|".join(re.escape(c) for c in _color_names)
        else:
            color_pattern = r"red|green|blue|yellow|purple|grey|gray"
        object_type_pattern = self.object_type_pattern()
        mentioned_object_type = self.object_type_from_text(normalized)
        claims_object_type = (
            active_claims_summary.get("object_type")
            if active_claims_summary is not None
            else None
        )
        if claims_object_type not in self.object_types():
            claims_object_type = None
        default_object_type = self.default_object_type()
        object_type = (
            mentioned_object_type
            or claims_object_type
            or default_object_type
        )
        task_handle = (
            self.task_handle("go_to_object", object_type)
            if object_type is not None
            else None
        )
        object_type_mentioned = mentioned_object_type is not None
        object_type_plural = (
            self.pluralize_object_type(object_type)
            if object_type is not None
            else None
        )
        object_match = (
            re.search(
                rf"\b(?:go to|go the|reach|find|get to|head to|navigate to)\s+"
                rf"(?:the )?(?P<color>{color_pattern}) "
                rf"(?P<object_type>{object_type_pattern})\b",
                normalized,
            )
            if object_type_pattern
            else None
        )
        _SUPERLATIVE_TERMS = frozenset([
            "farthest", "furthest", "most distant", "most far",
            "farthest away", "furthest away", "maximum distance",
            "highest", "largest", "greatest", "maximum", "max distance",
        ])
        metric = (
            "euclidean" if "euclidean" in normalized
            else "manhattan"
        )
        ranked_handle = (
            self.ranked_handle(metric, object_type)
            if object_type is not None
            else None
        )
        is_navigation = re.search(
            r"\b(go to|go the|reach|find|get to|head to|navigate to)\b",
            normalized,
        ) is not None
        threshold_match = re.search(
            r"\b(?:distance\s+)?"
            r"(?P<comparison>above|greater than|more than|over|exceeds|at least|"
            r"below|less than|under|at most|within)\s+"
            r"(?P<value>\d+(?:\.\d+)?)\b",
            normalized,
        )
        if threshold_match and (
            object_type_mentioned or active_claims_summary is not None
        ):
            comparison_text = threshold_match.group("comparison")
            comparison = {
                "above": "above",
                "greater than": "above",
                "more than": "above",
                "over": "above",
                "exceeds": "above",
                "at least": "at_least",
                "below": "below",
                "less than": "below",
                "under": "below",
                "at most": "at_most",
                "within": "within",
            }[comparison_text]
            threshold = float(threshold_match.group("value"))
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else None
            )
            if metric is None and active_claims_summary is not None:
                ranked = (
                    active_claims_summary.get("ranked_objects")
                    or active_claims_summary.get("ranked_doors")
                    or []
                )
                # compact summaries are strings such as "red@7.62"; metric is not
                # always present, so default to manhattan if the operator omitted it.
                metric = "manhattan"
            metric = metric or "manhattan"
            claims_handle = f"claims.filter.threshold.{metric}"
            wants_highest = any(
                term in normalized
                for term in ("highest", "largest", "maximum", "farthest", "furthest")
            )
            wants_lowest = any(
                term in normalized
                for term in ("lowest", "smallest", "minimum", "closest", "nearest")
            )
            order = "descending" if wants_highest else "ascending" if wants_lowest else None
            ordinal = 1 if order is not None else None
            required = [claims_handle]
            if is_navigation and task_handle is not None:
                required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "claim_reference",
                status_query=None,
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                claim_reference="threshold_filter",
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "filter",
                    "primitive_handle": claims_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": order,
                    "ordinal": ordinal,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": (
                        int(threshold) if threshold.is_integer() else threshold
                    ),
                    "comparison": comparison,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": required,
                    "preserved_constraints": [
                        str(object_type),
                        metric,
                        comparison,
                        str(threshold),
                        *(["highest"] if wants_highest else []),
                        *(["lowest"] if wants_lowest else []),
                    ],
                },
                capability_status="synthesizable",
                required_capabilities=required,
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a threshold claims-filter plan.",
            )
        ordinal_match = re.search(
            r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+"
            r"(closest|nearest|farthest|furthest)\b",
            normalized,
        )
        ordinal_semantics = normalize_distance_ordinal(normalized)
        if ordinal_semantics is not None:
            semantic_ranked_handle = (
                self.ranked_handle(ordinal_semantics.metric, object_type)
                if object_type is not None
                else None
            )
            semantic_required = [semantic_ranked_handle]
            if is_navigation and task_handle is not None:
                semantic_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": semantic_ranked_handle,
                    "metric": ordinal_semantics.metric,
                    "reference": "agent",
                    "order": ordinal_semantics.order,
                    "ordinal": ordinal_semantics.ordinal,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": semantic_required,
                    "preserved_constraints": [
                        *ordinal_semantics.preserved_constraints,
                        str(object_type),
                    ],
                },
                capability_status=(
                    "executable"
                    if ordinal_semantics.metric == "manhattan"
                    else "synthesizable"
                ),
                required_capabilities=semantic_required,
                selection_objective=SelectionObjective(
                    attribute="distance",
                    direction="maximum" if ordinal_semantics.order == "descending" else "minimum",
                    ordinal=ordinal_semantics.ordinal,
                    metric=ordinal_semantics.metric,
                ),
                confidence=0.9,
                reason="Deterministic semantic normalizer emitted a ranked ordinal distance plan.",
            )
        if (
            not is_navigation
            and
            ("closest" in normalized or "nearest" in normalized)
            and re.search(r"\b(second|2nd)\s+(closest|nearest)\b", normalized)
            and object_type_mentioned
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["closest", "second_closest"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": [
                        "closest",
                        "second",
                        str(object_type),
                        metric,
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a closest/second-closest answer plan.",
            )
        if ordinal_match and object_type_mentioned:
            ordinal_map = {
                "second": 2,
                "2nd": 2,
                "third": 3,
                "3rd": 3,
                "fourth": 4,
                "4th": 4,
                "fifth": 5,
                "5th": 5,
            }
            ordinal = ordinal_map[ordinal_match.group(1)]
            direction = ordinal_match.group(2)
            order = "descending" if direction in {"farthest", "furthest"} else "ascending"
            ordinal_required = [ranked_handle]
            if is_navigation and task_handle is not None:
                ordinal_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": order,
                    "ordinal": ordinal,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": ordinal_required,
                    "preserved_constraints": [
                        ordinal_match.group(1),
                        direction,
                        str(object_type),
                        metric,
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=ordinal_required,
                selection_objective=SelectionObjective(
                    attribute="distance",
                    direction="maximum" if order == "descending" else "minimum",
                    ordinal=ordinal,
                    metric=metric if metric != "manhattan" else None,
                ),
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a ranked ordinal query plan.",
            )

        distance_match = re.search(
            r"\b(?:distance\s+(?:of\s+)?|with\s+(?:a\s+)?distance\s+(?:of\s+)?)(\d+)\b",
            normalized,
        )
        if distance_match and object_type_mentioned:
            distance_value = int(distance_match.group(1))
            distance_required = [ranked_handle]
            if is_navigation and task_handle is not None:
                distance_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": distance_value,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": distance_required,
                    "preserved_constraints": [
                        "distance",
                        str(distance_value),
                        str(object_type),
                        metric,
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=distance_required,
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a distance-value query plan.",
            )

        color_mention = (
            re.search(
                rf"\b(?P<color>{color_pattern})\s+"
                rf"(?P<object_type>{object_type_pattern})\b",
                normalized,
            )
            if object_type_pattern
            else None
        )
        if color_mention and (
            "how far" in normalized
            or "distance" in normalized
            or normalized.startswith("is there")
            or normalized.startswith("do you see")
        ):
            color = "grey" if color_mention.group("color") == "gray" else color_mention.group("color")
            wants_distance = "how far" in normalized or "distance" in normalized
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": color_mention.group("object_type"),
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": color,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["distance"] if wants_distance else ["exists"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": [
                        color,
                        color_mention.group("object_type"),
                        "distance" if wants_distance else "exists",
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a color-specific answer query plan.",
            )

        if (
            ("closest" in normalized or "nearest" in normalized)
            and ("farthest" in normalized or "furthest" in normalized)
            and object_type_mentioned
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["closest", "farthest"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": [
                        "closest",
                        "farthest",
                        str(object_type),
                        metric,
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a closest/farthest answer plan.",
            )

        if is_navigation and re.search(r"\b(that|it|that one|the one)\b", normalized):
            claim_handle = "grounding.claims.last_grounded_target"
            claim_required = [claim_handle]
            if task_handle is not None:
                claim_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "select",
                    "primitive_handle": claim_handle,
                    "metric": None,
                    "reference": None,
                    "order": None,
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target"],
                    "required_capabilities": claim_required,
                    "preserved_constraints": ["that"],
                },
                capability_status="executable",
                required_capabilities=claim_required,
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted an active-claim reference plan.",
            )

        # Block random-walk actuation requests before they degrade to a grounding answer.
        if re.search(r"\brandom(?:ly)?\b", normalized) and re.search(
            r"\b(?:walk|move\s+to|go\s+to|navigate)\b", normalized
        ):
            return OperatorIntent(
                intent_type="unsupported",
                required_capabilities=["policy.random_walk"],
                capability_status="unsupported",
                confidence=0.9,
                reason="Random walk policy is not a supported navigation strategy.",
            )

        if any(t in normalized for t in _SUPERLATIVE_TERMS) and object_type_mentioned:
            superlative_required = [ranked_handle]
            if is_navigation and task_handle is not None:
                superlative_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "descending",
                    "ordinal": 1,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify" if is_navigation else "display",
                    "answer_fields": ["farthest", "distance"],
                    "required_capabilities": superlative_required,
                    "preserved_constraints": [
                        "farthest",
                        str(object_type),
                        metric,
                    ],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=superlative_required,
                selection_objective=SelectionObjective(
                    attribute="distance",
                    direction="maximum",
                    ordinal=1,
                    metric=metric if metric != "manhattan" else None,
                ),
                confidence=0.9,
                reason=(
                    "Deterministic operator-intent fallback emitted a farthest-object "
                    "query plan."
                ),
            )

        next_reference_phrases = ["next closest", "next one"]
        other_reference_phrases = ["other one", "the other"]
        if object_type is not None:
            next_reference_phrases.extend(
                [f"next {object_type}", f"another {object_type}"]
            )
            other_reference_phrases.extend(
                [f"other {object_type}", f"remaining {object_type}"]
            )
        if any(
            phrase in normalized
            for phrase in next_reference_phrases
        ):
            return OperatorIntent(
                intent_type="claim_reference",
                claim_reference="next_closest",
                target_selector=None,
                required_capabilities=[],
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed a next-closest claim reference.",
            )

        if any(
            phrase in normalized
            for phrase in other_reference_phrases
        ):
            return OperatorIntent(
                intent_type="claim_reference",
                claim_reference="other_door",
                target_selector=None,
                required_capabilities=[],
                confidence=0.85,
                reason=(
                    "Deterministic operator-intent fallback parsed an other-object "
                    "claim reference."
                ),
            )

        is_ranked_query = any(
            phrase in normalized
            for phrase in (
                "in order", "in descending order", "in ascending order",
                "ranked", "ranking", "rank them", "rank the", "rank distances",
                "rank the distances", "list them",
                "all of them", "list all",
            )
        )
        if (
            not is_ranked_query
            and object_type is not None
            and object_type_plural is not None
        ):
            is_ranked_query = bool(
                re.search(
                    rf"\b(?:all|each|every)\s+(?:of\s+these\s+)?"
                    rf"(?:{re.escape(object_type)}|{re.escape(object_type_plural)})\b",
                    normalized,
                )
            )
        # Paraphrase coverage for plural object-distance questions. Requiring the
        # plural form or "distances" avoids swallowing singular clarification cases.
        if not is_ranked_query and not is_navigation and object_type_mentioned:
            is_ranked_query = (
                "distances" in normalized
                or (
                    object_type_plural is not None
                    and "how far" in normalized
                    and object_type_plural in normalized
                )
                or (
                    object_type_plural is not None
                    and "far away" in normalized
                    and object_type_plural in normalized
                )
                or (
                    object_type_plural is not None
                    and "distance" in normalized
                    and object_type_plural in normalized
                )
            )

        # Stateful pronoun reference: when active claims already have ranked
        # objects, resolve "their"/"them" against those visible objects.
        _has_ranked_claims = (
            active_claims_summary is not None
            and bool(
                active_claims_summary.get("ranked_objects")
                or active_claims_summary.get("ranked_doors")
                or active_claims_summary.get("ranked_scene_doors")
            )
        )
        implicit_terms = ["their", "them", "these", "those", "the ones", "the objects"]
        if object_type_plural is not None:
            implicit_terms.append(f"the {object_type_plural}")
        _has_implicit_object_ref = _has_ranked_claims and any(
            term in normalized for term in implicit_terms
        )
        if (
            not is_ranked_query
            and not is_navigation
            and _has_implicit_object_ref
            and ("distances" in normalized or "distance" in normalized or "how far" in normalized or "far" in normalized)
        ):
            is_ranked_query = True

        if is_ranked_query and (
            object_type_mentioned or _has_implicit_object_ref
        ):
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else "manhattan"
            )
            ranked_handle = self.ranked_handle(metric, object_type)
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": object_type,
                    "operation": "rank",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["ranked_doors", "distance"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": ["rank", str(object_type), metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a ranked-object query plan.",
            )

        if (
            "closest" in normalized
            or "nearest" in normalized
            or "shortest" in normalized
        ) and object_type_mentioned:
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else "manhattan"
            )
            is_closest_knowledge_update = (
                "delivery target" in normalized or "make" in normalized
            )
            if not is_closest_knowledge_update:
                closest_ranked_handle = self.ranked_handle(metric, object_type)
                closest_required = [closest_ranked_handle]
                if is_navigation and task_handle is not None:
                    closest_required.append(task_handle)
                return OperatorIntent(
                    intent_type="task_instruction" if is_navigation else "status_query",
                    status_query=None if is_navigation else "ground_target",
                    task_type="go_to_object" if is_navigation else None,
                    target_selector=None,
                    grounding_query_plan={
                        "object_type": object_type,
                        "operation": "select" if is_navigation else "answer",
                        "primitive_handle": closest_ranked_handle,
                        "metric": metric,
                        "reference": "agent",
                        "order": "ascending",
                        "ordinal": 1,
                        "color": None,
                        "exclude_colors": [],
                        "distance_value": None,
                        "tie_policy": "clarify" if is_navigation else "display",
                        "answer_fields": ["closest", "distance"],
                        "required_capabilities": closest_required,
                        "preserved_constraints": [
                            "closest",
                            str(object_type),
                            metric,
                        ],
                    },
                    capability_status="executable" if metric == "manhattan" else "synthesizable",
                    required_capabilities=closest_required,
                    selection_objective=SelectionObjective(
                        attribute="distance",
                        direction="minimum",
                        ordinal=1,
                        metric=metric if metric != "manhattan" else None,
                    ),
                    confidence=0.9,
                    reason="Deterministic operator-intent fallback emitted a closest-object query plan.",
                )
            if is_ranked_query:
                metric_suffix = metric or "manhattan"
                return OperatorIntent(
                    intent_type="status_query",
                    status_query="ground_target",
                    target_selector={
                        "object_type": object_type,
                        "color": None,
                        "exclude_colors": [],
                        "relation": "closest",
                        "distance_metric": metric,
                        "distance_reference": "agent" if metric is not None else None,
                    },
                    capability_status="missing_skills",
                    required_capabilities=[
                        self.ranked_handle(metric_suffix, object_type)
                    ],
                    confidence=0.85,
                    reason=(
                        "Ranked listing requires the context-declared ranked "
                        "grounding capability, not the single closest capability."
                    ),
                )
            return OperatorIntent(
                intent_type=(
                    "knowledge_update"
                    if "delivery target" in normalized or "make" in normalized
                    else "task_instruction"
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    else "status_query"
                ),
                task_type=(
                    "go_to_object"
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    else None
                ),
                knowledge_update=(
                    {"delivery_target": None}
                    if "delivery target" in normalized or "make" in normalized
                    else None
                ),
                status_query=(
                    None
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    or "delivery target" in normalized
                    or "make" in normalized
                    else "ground_target"
                ),
                target_selector={
                    "object_type": object_type,
                    "color": None,
                    "exclude_colors": [],
                    "relation": "closest",
                    "distance_metric": metric,
                    "distance_reference": "agent" if metric is not None else None,
                },
                capability_status=(
                    "synthesizable"
                    if metric == "euclidean"
                    else "needs_clarification"
                    if metric is None
                    else "executable"
                ),
                required_capabilities=(
                    [self.closest_handle("euclidean", object_type)]
                    if metric == "euclidean"
                    else [self.closest_handle("manhattan", object_type)]
                    if metric == "manhattan"
                    else []
                ),
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed a closest-object selector.",
            )

        not_color_matches = re.findall(
            rf"\b(?:not|other than|except|neither|nor)\s+(?:the )?({color_pattern})\b",
            normalized,
        )
        if not_color_matches:
            # Also capture colors joined with "or"/"nor" (e.g. "not purple or yellow")
            extra = re.findall(
                rf"\b(?:or|nor)\s+(?:the )?({color_pattern})\b",
                normalized,
            )
            not_color_matches.extend(extra)
        if object_type_mentioned and not_color_matches:
            exclude_colors = [
                "grey" if c == "gray" else c
                for c in not_color_matches
            ]
            unique_handle = self.unique_handle(object_type)
            excluded_required = [unique_handle]
            if task_handle is not None:
                excluded_required.append(task_handle)
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                target_selector={
                    "object_type": object_type,
                    "color": None,
                    "exclude_colors": exclude_colors,
                    "relation": "unique",
                    "distance_metric": None,
                    "distance_reference": None,
                },
                capability_status="executable",
                required_capabilities=excluded_required,
                confidence=0.85,
                reason=(
                    "Deterministic operator-intent fallback parsed an excluded-color "
                    "object selector."
                ),
            )

        if object_match:
            color = (
                "grey"
                if object_match.group("color") == "gray"
                else object_match.group("color")
            )
            matched_object_type = object_match.group("object_type")
            matched_task_handle = self.task_handle(
                "go_to_object",
                matched_object_type,
            )
            return OperatorIntent(
                intent_type="task_instruction",
                canonical_instruction=f"go to the {color} {matched_object_type}",
                task_type="go_to_object",
                target={"color": color, "object_type": matched_object_type},
                target_selector=None,
                required_capabilities=[matched_task_handle],
                confidence=1.0,
                reason=(
                    "Deterministic operator-intent fallback parsed an object "
                    "navigation task."
                ),
            )

        delivery_match = re.search(
            rf"\b(?P<color>{color_pattern}) "
            rf"(?P<object_type>{object_type_pattern})\b.*\bdelivery target\b",
            normalized,
        ) if object_type_pattern else None
        if delivery_match is not None:
            color = (
                "grey"
                if delivery_match.group("color") == "gray"
                else delivery_match.group("color")
            )
            delivery_object_type = delivery_match.group("object_type")
            return OperatorIntent(
                intent_type="knowledge_update",
                knowledge_update={
                    "delivery_target": {
                        "color": color,
                        "object_type": delivery_object_type,
                    }
                },
                target_selector=None,
                required_capabilities=[],
                confidence=1.0,
                reason=(
                    "Deterministic operator-intent fallback parsed delivery "
                    "target knowledge."
                ),
            )

        if (
            re.search(r"\b(clear|delete|forget|remove)\b", normalized)
            and re.search(r"\b(target|delivery target)\b", normalized)
        ):
            return OperatorIntent(
                intent_type="knowledge_update",
                knowledge_update={"delivery_target": None},
                target_selector=None,
                required_capabilities=[],
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed target clearing.",
            )

        if "same" in normalized or "again" in normalized:
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                reference="last_target",
                target_selector=None,
                required_capabilities=[task_handle] if task_handle is not None else [],
                confidence=0.8,
                reason="Deterministic operator-intent fallback parsed a last-target reference.",
            )

        if "delivery target" in normalized and self._looks_like_question(normalized):
            return OperatorIntent(
                intent_type="status_query",
                status_query="delivery_target",
                target_selector=None,
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed a delivery-target query.",
            )

        if (
            "what do you see" in normalized
            or "what can you see" in normalized
            or "look around" in normalized
            or "around you" in normalized
            or (
                object_type_mentioned
                and any(
                    term in normalized
                    for term in ("see", "there", "visible", "around", "available")
                )
                and any(
                    prefix in normalized
                    for prefix in ("what", "which", "do you see", "can you see")
                )
            )
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="scene",
                target_selector=None,
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed a scene query.",
            )

        if (
            "what can you do" in normalized
            or "capabilit" in normalized
            or "what are you able" in normalized
            or "overview" in normalized
            or "skill" in normalized
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="help",
                target_selector=None,
                confidence=0.8,
                reason="Deterministic operator-intent fallback parsed a capability query.",
            )

        if "last" in normalized or "previous" in normalized:
            return OperatorIntent(
                intent_type="status_query",
                status_query="last_run",
                target_selector=None,
                confidence=0.7,
                reason="Deterministic operator-intent fallback parsed a last-run query.",
            )

        if re.search(r"\b(?:pick\s+up|pickup|grab)\b", normalized) and re.search(
            r"\bkey\b", normalized
        ):
            return OperatorIntent(
                intent_type="unsupported",
                required_capabilities=["task.pickup.key"],
                capability_status="unsupported",
                confidence=0.9,
                reason="Pickup key task is unsupported.",
            )

        if re.search(r"\b(?:toggle|open|unlock)\b", normalized) and re.search(
            r"\bdoor\b", normalized
        ):
            return OperatorIntent(
                intent_type="unsupported",
                required_capabilities=["task.open_or_unlock.door"],
                capability_status="unsupported",
                confidence=0.9,
                reason="Door toggle/open task is unsupported.",
            )

        # Bounded conditional evidence mission: preserve the stop clause instead of
        # truncating this to an ordinary motor command.
        until_target = re.search(
            rf"\b(?:until|till)\b.*\b(?:see|spot|observe|detect)\b.*"
            rf"\b(?P<color>{color_pattern})\s+"
            rf"(?P<object_type>{object_type_pattern})\b",
            normalized,
        ) if object_type_pattern else None
        if until_target is not None and re.search(
            r"\b(?:go\s+straight|go\s+forward|move\s+forward|advance|step\s+ahead)\b",
            normalized,
        ):
            color = until_target.group("color")
            color = "grey" if color == "gray" else color
            conditional_object_type = until_target.group("object_type")
            return OperatorIntent(
                intent_type="conditional_sense_motor",
                target={
                    "color": color,
                    "object_type": conditional_object_type,
                },
                action_name="move_forward",
                capability_status="executable",
                required_capabilities=[
                    "sensing.find_object_by_color_type",
                    "action.move_forward",
                    "task.act_until_evidence",
                ],
                steering_directive=SteeringDirective(
                    budget={"max_steps": 32},
                    scope="visible_only",
                    risk="operator_authorized",
                    stopping_rule="first_match",
                ),
                confidence=0.95,
                reason=(
                    "Deterministic fallback preserved an until-visible condition as "
                    "a bounded conditional evidence mission."
                ),
            )

        # Ambiguous navigation to a typed object without a color specifier.
        if (
            is_navigation
            and mentioned_object_type is not None
            and not re.search(
                rf"\b(?:{color_pattern})\s+{re.escape(mentioned_object_type)}\b",
                normalized,
            )
            and not re.search(r"\b(?:closest|nearest|farthest|furthest|highest|lowest|second|third|first)\b", normalized)
        ):
            color_list = "red, blue, green, yellow, purple, grey"
            return OperatorIntent(
                intent_type="ambiguous",
                confidence=0.85,
                reason=(
                    f"Ambiguous target: no color specified. Please clarify which "
                    f"{mentioned_object_type}. Supported: {color_list}"
                ),
            )

        # Ambiguous distance query for one typed object without a color.
        if (
            ("how far" in normalized or "distance" in normalized)
            and mentioned_object_type is not None
            and not re.search(
                rf"\b(?:{color_pattern})\s+{re.escape(mentioned_object_type)}\b",
                normalized,
            )
            and f"{mentioned_object_type}s" not in normalized
        ):
            color_list = "red, blue, green, yellow, purple, grey"
            return OperatorIntent(
                intent_type="ambiguous",
                confidence=0.85,
                reason=(
                    f"Ambiguous query: no color specified. Please clarify which "
                    f"{mentioned_object_type}. Supported: {color_list}"
                ),
            )

        # Front-cell sense query paraphrases → status_query (no motion)
        _SENSE_FRONT_PATTERNS = [
            r"\bwhat\s+is\s+in\s+front\b",
            r"\bwhat\s+(?:am\s+i|are\s+you)\s+facing\b",
            r"\bwhat\s+(?:object|thing)\s+is\s+(?:ahead|in\s+front)\b",
            r"\bsense\s+the\s+cell\b",
            r"\blook\s+forward\b",
        ]
        if any(re.search(p, normalized) for p in _SENSE_FRONT_PATTERNS):
            return OperatorIntent(
                intent_type="status_query",
                status_query="scene",
                target_selector=None,
                confidence=0.9,
                reason="Deterministic operator-intent fallback matched a front-cell sense query.",
            )

        # Conditional motor: "if X, go forward" / "move only if X" — Sense evidence before Spine.
        _is_conditional = re.match(r"^(?:if|when)\b", normalized) or re.search(
            r"\b(?:only\s+if|only\s+when)\b", normalized
        )
        if _is_conditional and re.search(
            r"\b(?:go\s+forward|step\s+forward|move\s+forward|advance|step\s+ahead|move)\b",
            normalized,
        ):
            return OperatorIntent(
                intent_type="conditional_sense_motor",
                confidence=0.9,
                reason="Conditional motor command requires Sense evidence before Spine actuation.",
            )

        # Expand implicit motor commands into explicit sequences
        motor_text = normalized
        motor_text = re.sub(r"\b(?:go|head|take a)\s+left\b", "turn left and go forward", motor_text)
        motor_text = re.sub(r"\b(?:go|head|take a)\s+right\b", "turn right and go forward", motor_text)
        motor_text = re.sub(r"\b(?:go|head)\s+back(?:wards?)?\b", "turn right twice and go forward", motor_text)
        motor_text = re.sub(r"\bturn\s+around\b", "turn right twice", motor_text)
        motor_text = re.sub(r"\b(?:turn|rotate)\b(?!\s+(?:left|right|around))", "turn right ", motor_text)

        # Multi-motor sequence: "turn right once, and go straight two times"
        # Must be checked BEFORE single motor so the compound isn't truncated to one action.
        _motor_seq = _parse_motor_sequence(motor_text, self.motor_object_terms())
        if _motor_seq is not None:
            # Encode each step as "action_name:count" in utterance_steps for lossless transport.
            seq_steps = [f"{a}:{c}" for a, c in _motor_seq]
            return OperatorIntent(
                intent_type="motor_sequence",
                utterance_steps=seq_steps,
                confidence=0.92,
                reason=f"Multi-motor sequence: {len(_motor_seq)} actions.",
            )

        # Motor-command pattern: "go straight for N steps", "turn right twice", etc.
        _motor = _parse_motor_command(motor_text, self.motor_object_terms())
        if _motor is not None:
            action_name, count = _motor
            return OperatorIntent(
                intent_type="motor_command",
                action_name=action_name,
                repeat_count=count,
                confidence=0.9,
                reason=f"Deterministic motor command: {action_name} × {count}.",
            )

        return OperatorIntent(
            intent_type="unsupported",
            target_selector=None,
            confidence=0.0,
            reason="Deterministic operator-intent fallback could not resolve utterance.",
        )

    def _looks_like_question(self, normalized_utterance: str) -> bool:
        return (
            normalized_utterance.endswith("?")
            or normalized_utterance.startswith(
                (
                    "what",
                    "whats",
                    "what's",
                    "which",
                    "who",
                    "where",
                    "when",
                    "why",
                    "how",
                    "and",
                )
            )
        )

    def _validate_primitive_names(self, names, available_primitives, label: str) -> None:
        allowed = set(available_primitives)
        for name in names:
            if name not in allowed:
                raise ValueError(f"Unknown {label}: {name}")


class LLMCompiler(CompilerBackend):
    name = "llm_compiler"
    DEFAULT_METHOD_MAX_TOKENS = {
        # Each method's strict json_schema emits its full object, so caps that were too small
        # truncated the response mid-JSON → parse error → silent smoke/regex fallback (observed
        # on compile_operator_intent at 256; the others shared the same risk). Sized to fit the
        # full schema. max_tokens is an upper bound — well-formed output is bounded by the
        # schema, so headroom does not increase cost; it only stops the truncation.
        "compile_operator_intent": 1024,
        "compile_task": 768,
        "compile_procedure": 1024,
        "compile_sense_plan": 768,
        "compile_skill_plan": 768,
        "compile_memory_updates": 768,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        fallback: SmokeTestCompiler | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout = timeout
        self.transport = transport or self._chat_completions_transport
        self.fallback = fallback or SmokeTestCompiler()
        self.fallback_reason: str | None = None
        self.default_max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "0") or "0")

        if not self.api_key:
            self.fallback_reason = "OPENROUTER_API_KEY not set; falling back to smoke_test_compiler."
            self.log(self.fallback_reason)

    @property
    def active_backend(self) -> str:
        if self.fallback_reason:
            return f"{self.name} -> {self.fallback.name}"
        return self.name

    def usage_summary(self) -> dict[str, Any]:
        llm_success_count = sum(
            1 for call in self.call_history if call["backend"] == self.name and call["success"]
        )
        fallback_count = sum(1 for call in self.call_history if call["used_fallback"])
        total_requested_max_tokens = sum(
            call["requested_max_tokens"] or 0 for call in self.call_history
        )
        return {
            "backend": self.active_backend,
            "llm_used": llm_success_count > 0,
            "llm_success_count": llm_success_count,
            "fallback_count": fallback_count,
            "total_requested_max_tokens": total_requested_max_tokens,
            "call_history": list(self.call_history),
        }

    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        task_types = sorted(memory.understanding)
        payload = {
            "instruction": instruction,
            "available_task_primitives": library_payload(available_task_primitives),
            "known_knowledge": memory.knowledge,
            "known_task_types": task_types,
        }
        return self._compile_or_fallback(
            method_name="compile_task",
            schema_name="jeenom_task_request",
            schema=task_request_json_schema(task_types),
            parser=TaskRequest.from_dict,
            system_prompt=(
                "You are the JEENOM cortical compiler. Parse the operator instruction into a "
                "typed TaskRequest. Choose only from the provided task types and never invent "
                "runtime actions or call the environment. Always include color, object_type, "
                "and target_location keys in params, using null when unknown."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_task(
                instruction, available_task_primitives, memory
            ),
        )

    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        object_types = list(self.object_types())
        payload = {
            "utterance": utterance,
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
            "scene_summary": scene_summary,
            "capability_manifest": capability_manifest,
            "active_claims_summary": active_claims_summary,
            "pending_synthesis_proposal": pending_proposal,
            "supported": {
                "intent_types": list(OPERATOR_INTENT_TYPES),
                "task_types": list(OPERATOR_TASK_TYPES),
                "object_types": object_types,
                "colors": list(OPERATOR_COLORS),
                "references": list(OPERATOR_REFERENCES),
                "status_queries": list(OPERATOR_STATUS_QUERIES),
            },
        }
        semantic_bounds = get_semantic_constraints()
        advertised_constraints = (
            " STRICT VOCABULARY CONSTRAINTS: You must map ordinals and distance concepts using "
            "ONLY the exact terms supported by the semantic normalizer. Supported ordinals: "
            + ", ".join(semantic_bounds["ordinals"]) + ". "
            "Supported descending terms: " + ", ".join(semantic_bounds["descending_distance_terms"]) + ". "
            "Supported ascending terms: " + ", ".join(semantic_bounds["ascending_distance_terms"]) + ". "
            "Do not invent new ordinals or distance terms."
        )
        return self._compile_or_fallback(
            method_name="compile_operator_intent",
            schema_name="jeenom_operator_intent",
            schema=operator_intent_json_schema(object_types=object_types),
            parser=lambda data: OperatorIntent.from_dict(
                data,
                object_types=object_types,
            ),
            system_prompt=(
                "You are the JEENOM operator intent compiler. Convert the operator utterance "
                "into one typed OperatorIntent. You only describe intent; you do not update "
                "memory, execute tasks, or call tools. The payload's supported.object_types "
                "and capability_manifest are authoritative. An object type being registered "
                "does not make a task executable; require the exact task/grounding capability "
                "handle declared by the manifest. Examples below use MiniGrid doors only as "
                "illustrations: substitute the payload's exact object type and exact handles. "
                "For unsupported, ambiguous, pickup, open, unlock, exploration, correction, "
                "or replan requests, emit unsupported or ambiguous and do not fabricate a "
                "supported intent. Knowledge updates are "
                "narrowly limited to delivery_target. Question-shaped utterances must be "
                "status_query intents, not knowledge_update intents. Scene questions such as "
                "'what do you see around you' map to status_query=scene. Delivery-target "
                "questions map to status_query=delivery_target only when the utterance "
                "explicitly mentions delivery target. Capability or help questions such as "
                "'what can you do', 'what are your capabilities', 'what are your skills', "
                "or 'give me an overview of your capabilities' map to status_query=help "
                "and should be answered from the capability_manifest by the station. "
                "Relational target questions such as "
                "'what is the closest door to you', 'which door is nearest', or 'shortest "
                "distance to a door' must map to status_query=ground_target with a "
                "target_selector, never status_query=delivery_target. For relational target "
                "requests such as closest or not-yellow doors, emit a target_selector. Do not "
                "choose the closest object yourself; the station will ground the selector "
                "using current scene data. "
                "Use the capability_manifest to decide whether a request is executable, "
                "needs clarification, unsupported, or missing a primitive. Set "
                "capability_status explicitly: executable when the manifest says the needed "
                "primitive is implemented; needs_clarification when the request is supported "
                "but a slot like distance_metric is missing; synthesizable when the manifest "
                "says a missing primitive is safe_to_synthesize; missing_skills when a "
                "required primitive is missing and not synthesizable; unsupported when the "
                "request is outside the manifest. Closest-door Manhattan grounding is "
                "implemented through grounding.all_doors.ranked.manhattan.agent. Closest-door "
                "Euclidean grounding is missing but marked safe to synthesize later through "
                "grounding.all_doors.ranked.euclidean.agent; emit capability_status=synthesizable "
                "with metric=euclidean rather than unsupported. If closest is requested without "
                "a metric, default to metric=manhattan so closest and farthest use the same "
                "ranked-distance grounding path. Example: utterance='I see. What is the closest "
                "door to you' -> intent_type=status_query, capability_status=executable, "
                "status_query=ground_target, grounding_query_plan={object_type: door, "
                "operation: answer, primitive_handle: grounding.all_doors.ranked.manhattan.agent, "
                "metric: manhattan, order: ascending, ordinal: 1}. "
                "If active_claims_summary is provided in the payload, the operator may be "
                "referring to a prior grounding result. Phrases like 'the next closest', "
                "'next one', 'the next door', 'the other door', 'the remaining door', or "
                "'another door' should be classified as intent_type=claim_reference. Set "
                "claim_reference=next_closest for sequential references (next one, next "
                "closest, next door) and claim_reference=other_door for residual references "
                "(the other door, the remaining door). Do NOT choose the object yourself; "
                "the station resolves claim references deterministically from active_claims. "
                "If active_claims_summary is provided and the utterance asks for a distance "
                "threshold over the prior ranked results (e.g. 'the door where euclidean "
                "distance is above 6', 'doors below 10', 'go to one with distance at least "
                "7'), emit intent_type=claim_reference, claim_reference=threshold_filter, "
                "and a grounding_query_plan with operation='filter', primitive_handle="
                "'claims.filter.threshold.<metric>', metric set to the named metric or the "
                "active claim metric, distance_value set to the numeric threshold, and "
                "comparison in ['above','below','within','at_least','at_most']. Include the "
                "claims.filter.threshold.<metric> handle in required_capabilities. Do not "
                "bake the threshold into the handle; the primitive is parametric. If the "
                "utterance asks for the highest/largest/farthest item inside the filtered "
                "set, set order='descending' and ordinal=1. If it asks for the lowest/"
                "smallest/closest item inside the filtered set, set order='ascending' and "
                "ordinal=1. "
                "REQUIRED_CAPABILITIES: Every intent must include a required_capabilities "
                "field listing the exact capability handles needed. Use the handle names "
                "exactly as they appear in the capability_manifest (e.g. "
                "'grounding.closest_door.manhattan.agent', 'task.go_to_object.door'). "
                "Include any handle the intent needs even if it is NOT in the manifest — the "
                "station's CapabilityMatcher will classify it as missing_skills. "
                "No weakening: grounding.closest_door does NOT satisfy legacy grounding.ranked_doors; "
                "grounding.closest_door does NOT satisfy grounding.nth_closest_door; "
                "task.go_to_object does NOT satisfy task.pickup. "
                "A request for ranked or ordered listing of objects (e.g. 'all doors in "
                "order', 'list doors by distance', 'doors closest to me ranked') requires "
                "'grounding.all_doors.ranked.manhattan.agent' — this is NOT the same as "
                "grounding.closest_door.manhattan.agent and must be listed separately. "
                "SELECTION_OBJECTIVE: Whenever the request involves selecting or navigating "
                "to an object chosen by a ranked attribute (farthest door, second closest, "
                "highest-distance room, etc.), set selection_objective with: "
                "attribute='distance' for door-distance queries; "
                "direction='maximum' for superlatives meaning 'the most' (farthest, furthest, "
                "most distant, highest, largest, greatest, maximum — any domain); "
                "direction='minimum' for superlatives meaning 'the least' (closest, nearest, "
                "shortest, lowest, smallest, minimum — any domain); "
                "ordinal=1 for 'the farthest', ordinal=2 for 'second farthest', etc.; "
                "metric='manhattan' or 'euclidean' if specified, else null. "
                "Never use 'ascending' or 'descending' in selection_objective.direction — "
                "those are grounding_query_plan fields. "
                "Set selection_objective=null for non-ranking intents (status queries, "
                "scene questions, knowledge updates, ranked listings without a specific pick). "
                "Example: 'go to the farthest door' -> selection_objective={attribute:'distance', "
                "direction:'maximum', ordinal:1, metric:null}. "
                "Example: 'navigate to the second closest door' -> selection_objective={"
                "attribute:'distance', direction:'minimum', ordinal:2, metric:null}. "
                "GROUNDING_QUERY_PLAN: For any operator request that asks about visible "
                "doors, distances, ordering, closest/farthest, ordinal ranks, color-specific "
                "existence, or a target selected from scene grounding, emit a "
                "grounding_query_plan. For non-grounding intents, set grounding_query_plan=null. "
                "Do not answer the question yourself and do not choose an object yourself. "
                "Describe the query. Use primitive_handle='grounding.all_doors.ranked.manhattan.agent' "
                "for Manhattan ranked-door plans. Use operation='rank' for ranked/list displays, "
                "operation='answer' for questions, and operation='select' for tasks that choose "
                "a grounded target. Use order='ascending' for closest/nearest and order='descending' "
                "for farthest/furthest/least-close. Convert ordinal words to integers: second=2, "
                "third=3. Example: 'can you navigate to the second farthest door' -> "
                "intent_type=task_instruction, task_type=go_to_object, grounding_query_plan={"
                "object_type:'door', operation:'select', primitive_handle:"
                "'grounding.all_doors.ranked.manhattan.agent', metric:'manhattan', "
                "reference:'agent', order:'descending', ordinal:2, color:null, "
                "exclude_colors:[], distance_value:null, tie_policy:'clarify', "
                "answer_fields:['target','distance'], required_capabilities:["
                "'grounding.all_doors.ranked.manhattan.agent','task.go_to_object.door'], "
                "preserved_constraints:['second','farthest','door','manhattan']}. "
                "Example: 'how far is the red door' -> status_query=ground_target and a "
                "grounding_query_plan with color='red', answer_fields=['distance']. "
                "Example: 'is there a green door' -> status_query=ground_target and a "
                "grounding_query_plan with color='green', answer_fields=['exists']. "
                "Example: 'which door is closest and which is farthest' -> answer_fields="
                "['closest','farthest']. "
                "Example: 'which door is closest and second closest' -> operation='answer', "
                "order='ascending', ordinal=null, answer_fields=['closest','second_closest'], "
                "preserved_constraints=['closest','second','door','manhattan']. "
                "Use answer fields first_closest, second_closest, third_closest, "
                "first_farthest, second_farthest, third_farthest for multi-answer "
                "ranked questions; do not collapse 'second closest' to ordinal=1. "
                "Reference requests such as 'go to that', 'go to it', or 'take me there' "
                "after a grounding answer must also use grounding_query_plan, not a plain "
                "target guess. Emit primitive_handle='grounding.claims.last_grounded_target', "
                "operation='select', metric=null, reference=null, order=null, ordinal=null, "
                "answer_fields=['target'], required_capabilities=["
                "'grounding.claims.last_grounded_target','task.go_to_object.door'], and "
                "preserved_constraints=['that'] or the actual pronoun used. "
                "Requests to clear/delete/forget only the current delivery target should be "
                "knowledge_update with knowledge_update={delivery_target:null}; this is not "
                "a motion task. "
                "CONCEPT INTENT TYPES: The station supports named operator concepts — "
                "shorthand labels that expand to full instructions. "
                "When the operator teaches a new shorthand — e.g. 'when I say bingo, go to "
                "the red door', 'if I say scout you need to go to the red door, can you "
                "remember that?', 'define bingo as go to the blue door', 'remember patrol "
                "means go to the closest door by manhattan distance' — emit "
                "intent_type='concept_teach' with concept_name set to the shorthand label "
                "(e.g. 'bingo') and concept_utterance set to the full instruction it expands "
                "to (e.g. 'go to the red door'). Strip trailing memory confirmation phrases "
                "like 'can you remember that', 'please remember this', 'remember that' from "
                "concept_utterance. Strip leading modifiers like 'automatically', 'always', "
                "'just', 'please' from concept_utterance. Set required_capabilities=[]. "
                "When the operator invokes a single known concept by name — e.g. 'bingo', "
                "'run bingo', 'execute scout', 'do patrol' — emit intent_type='concept_recall' "
                "with concept_name set to the label. The station will look up and execute "
                "the stored concept. Set required_capabilities=[]. "
                "When the operator requests multiple known concepts executed in sequence — "
                "e.g. 'do bingo then scout', 'execute scout first and then bingo', "
                "'run bingo followed by scout' — emit intent_type='procedure_recall' with "
                "concept_steps set to the ordered list of short concept labels "
                "(e.g. ['bingo', 'scout']). Use procedure_recall when each step is a "
                "single-word shorthand label. Do NOT collapse this to a single "
                "concept_recall. Do NOT try to execute the concepts yourself. "
                "Set required_capabilities=[]. "
                "When the operator requests a sequence of FULL TASK INSTRUCTIONS (not "
                "named shorthand labels) — e.g. 'go to the red door then go to the green "
                "door', 'navigate to the closest door and then go to the farthest one' — "
                "emit intent_type='sequence_instruction' with utterance_steps set to the "
                "ordered list of full task descriptions "
                "(e.g. ['go to the red door', 'go to the green door']). "
                "Use sequence_instruction when steps are multi-word task phrases, not "
                "single-word labels. Set required_capabilities=[]. "
                "When the operator issues a direct motor action — e.g. 'go straight for 3 steps', "
                "'turn right twice', 'move forward once', 'turn left 4 times' — emit "
                "intent_type='motor_command' with action_name set to the primitive key "
                "('move_forward', 'turn_right', 'turn_left', 'pickup', 'toggle') and "
                "repeat_count set to the integer count (default 1). Motor commands are "
                "low-level control requests that require station authorization with a "
                "RawMotorTicket before execution. "
                "When the operator issues a sequence made only of direct motor actions — e.g. "
                "'turn left twice and go forward once', 'turn right then move forward two "
                "steps' — emit intent_type='motor_sequence' and encode utterance_steps as "
                "['action_name:count', ...], for example ['turn_left:2','move_forward:1']. "
                "Do not emit sequence_instruction for all-motor sequences. "
                "Set required_capabilities=[]. "
                "When the operator says to repeat one motor action UNTIL or TILL a target "
                "becomes visible — e.g. 'go straight until you see a blue door' — emit "
                "intent_type='conditional_sense_motor', target={color:'blue',object_type:'door'}, "
                "action_name='move_forward', repeat_count=null, required_capabilities=["
                "'sensing.find_object_by_color_type','action.move_forward',"
                "'task.act_until_evidence'], and steering_directive={budget:{max_steps:32,"
                "max_clarifications:null},scope:'visible_only',risk:'operator_authorized',"
                "stopping_rule:'first_match'}. Do not emit motor_command or motor_sequence "
                "for an until/till request, because that would discard the stop condition. "
                "Do NOT classify concept-teach utterances as task_instruction. "
                "Do NOT try to execute the concept's underlying instruction yourself. "
                "Concepts listing query 'list concepts', 'what concepts do you know', "
                "'what do you remember' maps to status_query=concepts. "
                "For status queries and claim references with no grounding or task "
                "requirements, emit required_capabilities=[]. "
                + (
                    " PENDING SYNTHESIS PROPOSAL: The station has proposed building a new "
                    "primitive and is awaiting operator approval. The pending proposal is in "
                    "pending_synthesis_proposal in the payload. If the operator's utterance "
                    "expresses agreement, confirmation, or approval of the proposal "
                    "(e.g. 'yes', 'yes build it', 'go ahead', 'do it', 'sure', 'yes please', "
                    "'I mean yes', 'build it', 'ok yes', 'yes that one'), emit "
                    "intent_type='accept_proposal' with required_capabilities=[]. "
                    "If the operator declines or cancels (e.g. 'no', 'cancel', 'stop', "
                    "'don't build it', 'no thanks'), emit intent_type='reject_proposal' "
                    "with required_capabilities=[]. "
                    "Only fall back to other intent types if the utterance is clearly a "
                    "new, unrelated instruction."
                    if pending_proposal else ""
                )
                + advertised_constraints
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_operator_intent(
                utterance,
                memory,
                scene_summary=scene_summary,
                capability_manifest=capability_manifest,
                active_claims_summary=active_claims_summary,
                pending_proposal=pending_proposal,
            ),
        )

    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        task_types = sorted(memory.understanding)
        primitive_list = primitive_names(available_task_primitives)
        payload = {
            "task_request": asdict(task),
            "available_task_primitives": library_payload(available_task_primitives),
            "understanding": memory.understanding,
            "knowledge": memory.knowledge,
        }
        return self._compile_or_fallback(
            method_name="compile_procedure",
            schema_name="jeenom_procedure_recipe",
            schema=procedure_recipe_json_schema(primitive_list, task_types),
            parser=ProcedureRecipe.from_dict,
            system_prompt=(
                "You are the JEENOM cortical compiler. Compose a typed high-level ProcedureRecipe "
                "using only task primitive names from the provided library. Output a reusable task "
                "recipe, not per-tick executable steps."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_procedure(
                task, available_task_primitives, memory
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        primitive_list = primitive_names(available_sensing_primitives)
        payload = {
            "evidence_frame": asdict(evidence_frame),
            "execution_context": asdict(execution_context),
            "available_sensing_primitives": library_payload(available_sensing_primitives),
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
        }
        return self._compile_or_fallback(
            method_name="compile_sense_plan",
            schema_name="jeenom_sense_plan",
            schema=sense_plan_json_schema(primitive_list),
            parser=SensePlanTemplate.from_dict,
            system_prompt=(
                "You are the JEENOM Sense compiler. Compose a reusable SensePlanTemplate using "
                "only the provided sensing primitive names. Do not emit per-tick observations or "
                "instant sensor results. Emit required_inputs and produces for a reusable template."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_sense_plan(
                evidence_frame,
                execution_context,
                available_sensing_primitives,
                memory,
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        primitive_list = primitive_names(available_action_primitives)
        payload = {
            "execution_contract": asdict(execution_contract),
            "percepts": asdict(percepts),
            "available_action_primitives": library_payload(available_action_primitives),
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
        }
        return self._compile_or_fallback(
            method_name="compile_skill_plan",
            schema_name="jeenom_skill_plan",
            schema=skill_plan_json_schema(primitive_list),
            parser=SkillPlanTemplate.from_dict,
            system_prompt=(
                "You are the JEENOM Spine compiler. Compose a reusable SkillPlanTemplate using "
                "only the provided action primitive names. Output a reusable template, not a "
                "per-tick action decision. Never execute MiniGrid actions directly."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_skill_plan(
                execution_contract,
                percepts,
                available_action_primitives,
                memory,
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        payload = {
            "final_state": final_state,
            "final_claims": final_claims,
            "trace_events": trace,
            "current_memory": memory.serializable_snapshot(),
        }
        result = self._compile_or_fallback(
            method_name="compile_memory_updates",
            schema_name="jeenom_memory_updates",
            schema=memory_updates_json_schema(),
            parser=lambda data: [MemoryUpdate.from_dict(item) for item in data.get("updates", [])],
            system_prompt=(
                "You are the JEENOM memory compiler. Propose only typed MemoryUpdate records "
                "that separate durable knowledge from episodic run facts. Do not invent new "
                "memory scopes."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_memory_updates(
                final_state, final_claims, trace, memory
            ),
        )
        return result

    def _compile_or_fallback(
        self,
        method_name: str,
        schema_name: str,
        schema: dict[str, Any],
        parser: Callable[[Any], Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        fallback_call: Callable[[], Any],
        allowed_primitive_names: set[str] | None = None,
    ):
        if self.fallback_reason:
            self.record_call(
                method_name=method_name,
                backend=self.fallback.name,
                success=True,
                used_fallback=True,
                reason=self.fallback_reason,
            )
            return fallback_call()

        requested_max_tokens = self._max_tokens_for_method(method_name)
        request_payload = {
            "method_name": method_name,
            "system_prompt": system_prompt,
            "user_payload": make_json_safe(user_payload),
            "schema_name": schema_name,
            "schema": schema,
            "max_tokens": requested_max_tokens,
        }
        self.log(
            f"{method_name} requested max_tokens={requested_max_tokens} on model {self.model}"
        )

        try:
            raw_data = self.transport(request_payload)
            parsed = parser(raw_data)
            if allowed_primitive_names is not None:
                self._validate_compiler_primitives(parsed, allowed_primitive_names)
            parsed = self._normalize_compiler_output(parsed)
            self.record_call(
                method_name=method_name,
                backend=self.name,
                success=True,
                used_fallback=False,
                requested_max_tokens=requested_max_tokens,
            )
            return parsed
        except Exception as exc:  # noqa: BLE001
            message = f"{method_name} failed in llm_compiler, falling back: {exc}"
            self.log(message)
            self.record_call(
                method_name=method_name,
                backend=self.fallback.name,
                success=True,
                used_fallback=True,
                reason=str(exc),
                requested_max_tokens=requested_max_tokens,
            )
            return fallback_call()

    def _normalize_compiler_output(self, parsed):
        if isinstance(parsed, TaskRequest):
            parsed.source = self.name
            return parsed
        if isinstance(parsed, (ProcedureRecipe, SensePlanTemplate, SkillPlanTemplate)):
            parsed.source = self.name
            parsed.compiler_backend = self.name
            parsed.validated = True
            return parsed
        return parsed

    def _validate_compiler_primitives(self, parsed, allowed: set[str]) -> None:
        if isinstance(parsed, ProcedureRecipe):
            primitives = parsed.steps
        elif isinstance(parsed, SensePlanTemplate):
            primitives = parsed.primitives
        elif isinstance(parsed, SkillPlanTemplate):
            primitives = parsed.primitives
        else:
            return

        for primitive in primitives:
            if primitive not in allowed:
                raise SchemaValidationError(
                    f"Unknown primitive emitted by compiler: {primitive}"
                )

    def _max_tokens_for_method(self, method_name: str) -> int:
        if self.default_max_tokens > 0:
            return self.default_max_tokens
        return self.DEFAULT_METHOD_MAX_TOKENS.get(method_name, 384)

    def _chat_completions_transport(self, request_payload: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request_payload["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(request_payload["user_payload"], indent=2),
                },
            ],
            "max_tokens": request_payload["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request_payload["schema_name"],
                    "schema": request_payload["schema"],
                    "strict": True,
                },
            },
        }

        req = urllib.request.Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc

        message = raw_response["choices"][0]["message"]
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(f"Model refusal: {refusal}")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Expected string JSON content from OpenRouter response")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Model did not return valid JSON: {exc}") from exc


def build_compiler(name: str) -> CompilerBackend:
    if name == "smoke_test":
        return SmokeTestCompiler()
    if name == "llm":
        return LLMCompiler()
    raise ValueError(f"Unknown compiler backend: {name}")
