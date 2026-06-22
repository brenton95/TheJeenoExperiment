from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .capability_matcher import default_matcher
from .knowledge_base import KnowledgeBase, NamedConcept, derive_scope
from .schemas import (
    ApprovedCommand,
    ClarificationRequest,
    CorticalEnvelope,
    MissionExecutionPlan,
    PrimitiveDefinitionRequest,
    ReadinessGraph,
    RequestPlan,
    RequestPlanStep,
)


def _approved(
    command_type: str,
    utterance: str = "",
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ApprovedCommand:
    p = payload if payload is not None else ({"message": message} if message is not None else {})
    return ApprovedCommand(command_type=command_type, utterance=utterance, payload=p, **kwargs)


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


_NEW_INTENT_COMMAND_KINDS = {
    "ambiguous",
    "claim_reference",
    "conditional_mission_execute",
    "concept_forget",
    "concept_teach",
    "ground_target_query",
    "knowledge_update",
    "metric_query",
    "mission_execute",
    "motor_execute",
    "motor_sequence_execute",
    "primitive_definition",
    "procedure_execute",
    "sequence_execute",
    "status_query",
    "steering_directive",
    "task_instruction",
    "task_selector",
    "unsupported",
}


@dataclass
class PendingClarification:
    clarification_type: str
    original_utterance: str
    resume_kind: str
    partial_selector: dict[str, Any]
    missing_field: str
    supported_values: list[str]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    request_plan: RequestPlan | None = None
    readiness_graph: ReadinessGraph | None = None
    pending_envelope: CorticalEnvelope | None = None
    clarification_request: ClarificationRequest | None = None


@dataclass
class PendingSynthesisProposal:
    handle: str
    original_utterance: str
    intent: Any
    cap_match: Any
    similar_handles: list[str]
    proposed_description: str | None = None
    proposed_condition: dict[str, Any] | None = None


@dataclass
class PendingPrimitiveDefinition:
    request: PrimitiveDefinitionRequest
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    mission_plan: MissionExecutionPlan | None = None


class KnowledgeChannel:
    """Scoped, writer-gated access surface for durable named concepts."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self._write_events: list[dict[str, Any]] = []
        self._reuse_counters: dict[str, int] = {
            "episodic": 0,
            "site": 0,
            "embodiment": 0,
            "universal": 0,
        }

    def _check_write_policy(
        self,
        *,
        scope: str,
        writer: str,
        record: Any,
        manifest: Any = None,
    ) -> None:
        if scope == "episodic":
            raise ValueError("episodic records live in the claim store, not KnowledgeBase")
        if scope == "site" and writer != "operator":
            raise PermissionError("site knowledge writes require writer='operator'")
        if scope == "embodiment" and writer not in {"manifest", "synthesizer"}:
            raise PermissionError(
                "embodiment knowledge writes require manifest registration or synthesizer"
            )
        if scope == "universal":
            if writer != "synthesizer":
                raise PermissionError("universal knowledge writes require writer='synthesizer'")
            if derive_scope(record, manifest) != "universal":
                raise PermissionError("universal write failed derive_scope verification")

    def teach(
        self,
        name: str,
        utterance: str,
        *,
        plan: Any | None = None,
        writer: str = "operator",
        scope: str | None = "site",
        manifest: Any = None,
    ) -> NamedConcept:
        probe = NamedConcept(name=name, utterance=utterance, plan=plan, scope=scope or "site")
        resolved_scope = scope or derive_scope(probe, manifest)
        self._check_write_policy(
            scope=resolved_scope,
            writer=writer,
            record=probe,
            manifest=manifest,
        )
        concept = self.knowledge_base.teach(
            name,
            utterance,
            plan=plan,
            scope=resolved_scope,
            manifest=manifest,
        )
        self._write_events.append(
            {
                "event": "knowledge_write",
                "name": concept.name,
                "scope": concept.scope,
                "writer": writer,
                "concept_type": concept.concept_type,
            }
        )
        return concept

    def forget(self, name: str, *, writer: str = "operator") -> bool:
        if writer != "operator":
            raise PermissionError("forget requires writer='operator'")
        removed = self.knowledge_base.forget(name)
        if removed:
            self._write_events.append(
                {
                    "event": "knowledge_forget",
                    "name": KnowledgeBase._normalize_name(name),
                    "writer": writer,
                }
            )
        return removed

    def recall(self, name: str) -> NamedConcept | None:
        concept = self.knowledge_base.recall(name)
        if concept is not None:
            self._reuse_counters[concept.scope] = self._reuse_counters.get(concept.scope, 0) + 1
        return concept

    def search(self, query: str) -> list[NamedConcept]:
        concepts = self.knowledge_base.search(query)
        for concept in concepts:
            self._reuse_counters[concept.scope] = self._reuse_counters.get(concept.scope, 0) + 1
        return concepts

    def all_concepts(self) -> list[NamedConcept]:
        return self.knowledge_base.all_concepts()

    def is_sequence(self, utterance: str) -> list[str] | None:
        return self.knowledge_base._is_sequence(utterance)

    def invalidate_site(self, *, reason: str = "site_change") -> int:
        removed = self.knowledge_base.purge_scope("site")
        if removed:
            self._write_events.append(
                {"event": "knowledge_scope_invalidated", "scope": "site", "count": removed, "reason": reason}
            )
        return removed

    def invalidate_substrate(self, *, reason: str = "substrate_fingerprint_change") -> int:
        removed = self.knowledge_base.purge_scope("embodiment")
        if removed:
            self._write_events.append(
                {
                    "event": "knowledge_scope_invalidated",
                    "scope": "embodiment",
                    "count": removed,
                    "reason": reason,
                }
            )
        return removed

    def consume_write_events(self) -> list[dict[str, Any]]:
        events = [dict(event) for event in self._write_events]
        self._write_events.clear()
        return events

    def consume_reuse_counters(self) -> dict[str, int]:
        counters = dict(self._reuse_counters)
        self._reuse_counters = {key: 0 for key in self._reuse_counters}
        return counters


@dataclass
class TurnOrchestrator:
    """Top-level operator turn router for an OperatorStationSession facade.

    Owns the multi-turn pending state (clarification, synthesis, primitive definition)
    so that the station facade remains stateless with respect to turn flow.
    """

    classify_utterance: Callable[[str], ApprovedCommand]
    normalize_utterance: Callable[[str], str]
    looks_like_bare_label: Callable[[str], bool]
    cortex_session: Any = None
    mission_cortex: Any = None
    intent_cache: Any = None
    pending_clarification: PendingClarification | None = None
    pending_synthesis_proposal: PendingSynthesisProposal | None = None
    pending_primitive_definition: PendingPrimitiveDefinition | None = None

    def dispatch(self, station: Any, intent: Any, utterance: str) -> ApprovedCommand:
        # Reset per-turn so stale values from previous turns are never observed.
        station.last_readiness_graph = None
        station.last_repair_events = []
        station.last_operational_mismatches = []

        # ── Proactive Intent Signal Verification (Phase 7.595) ───────────────
        # Runs unconditionally on ALL LLM outputs before any routing.
        # Blueprint Rule 9: deterministic gate between compiler output and
        # CapabilityMatcher — regardless of what the LLM declared.
        parsed_steering = getattr(station, "_parsed_steering_directive", None)
        station._parsed_steering_directive = None
        intent, verif_result = station.intent_verifier.enrich(
            utterance, intent, steering_directive=parsed_steering
        )
        promoted_intent = station._promote_verified_query_intent(
            utterance,
            intent,
            verif_result,
        )
        if promoted_intent is not intent:
            station.log(
                "intent verifier promoted unresolved utterance to "
                f"{promoted_intent.intent_type}"
            )
            intent = promoted_intent
        station.last_operator_intent = intent
        if intent.steering_directive is not None:
            # Verified + attached: persist for this turn so the instruction-based plan
            # rebuild (and the execution ticket) carry the same directive.
            station.active_steering_directive = intent.steering_directive
        if verif_result.injected_handles:
            station.log(f"intent verifier injected: {verif_result.summary()}")
        if verif_result.inversion_detected:
            station.log(f"intent verifier blocked inversion: {verif_result.inversion_reason}")
            return _approved("ambiguous", utterance, (
                    "Semantic inversion detected in compiled plan: "
                    f"{verif_result.inversion_reason} "
                    "Please rephrase."
                ))

        # ── Request plan recording (all non-control intents) ─────────────────
        request_plan_recorded = False
        if intent.grounding_query_plan is not None:
            station._record_request_plan(utterance, intent)
            request_plan_recorded = True
            plan_command = station._command_from_grounding_query_plan(utterance, intent)
            if plan_command is not None:
                return plan_command

        if not request_plan_recorded:
            station._record_request_plan(utterance, intent)

        # ── Intent Readiness Requirement Matching (Phase 7.59) ────────────────
        # Runs for claim/provenance/action; skipped for procedure/control which own
        # their own readiness semantics (knowledge ops need no capability gate).
        knowledge_type = intent.knowledge_type
        cap_match = default_matcher.match(intent, station.capability_registry)
        if knowledge_type not in {"procedure", "control"}:
            composition_command = station._try_compose_grounding_result(
                utterance,
                intent,
                cap_match,
                verif_result,
            )
            if composition_command is not None:
                return composition_command
            if cap_match.verdict in {"missing_skills", "synthesizable", "unsupported"}:
                return station._arbitrate_gap(utterance, intent, cap_match)

            if intent.target_selector is not None and intent.capability_status in {
                "needs_clarification",
                "missing_skills",
                "synthesizable",
            }:
                if intent.capability_status == "needs_clarification":
                    grounded = station.ground_target_selector(intent.target_selector)
                    # resume_kind derived from knowledge_type — no intent_type comparison needed
                    resume_kind = (
                        "task_instruction" if knowledge_type == "action"
                        else "knowledge_update" if knowledge_type == "provenance"
                        else "ground_target_query"
                    )
                    clarification = station.maybe_start_selector_clarification(
                        utterance=utterance,
                        resume_kind=resume_kind,
                        grounded=grounded,
                    )
                    if clarification is not None:
                        return clarification
                readiness_command = station.command_from_selector_readiness(
                    intent.target_selector,
                    utterance,
                )
                if readiness_command is not None:
                    return readiness_command

        # ── Knowledge-type dispatch (5 paths, no intent_type comparisons) ────
        if knowledge_type == "claim":
            return self._handle_claim(station, intent, utterance, cap_match)
        if knowledge_type == "procedure":
            return self._handle_procedure(station, intent, utterance)
        if knowledge_type == "provenance":
            return self._handle_provenance(station, intent, utterance)
        if knowledge_type == "action":
            return self._handle_action(station, intent, utterance, cap_match)
        return self._handle_control(station, intent, utterance, cap_match)

    # ── Knowledge-type sub-handlers ──────────────────────────────────────────

    def _handle_claim(
        self, station: Any, intent: Any, utterance: str, cap_match: Any
    ) -> ApprovedCommand:
        if intent.intent_type == "cache_query":
            return ApprovedCommand(command_type="cache_query", utterance=utterance)

        if intent.intent_type == "status_query":
            if intent.status_query == "ground_target" and intent.target_selector is not None:
                return _approved("ground_target_query", utterance, payload={"target_selector": intent.target_selector})
            return _approved("status_query", utterance, payload={"query": intent.status_query or "status"})

        if intent.intent_type == "claim_reference":
            if intent.claim_reference == "threshold_filter":
                # Route through capability matching → arbitration → synthesis pipeline.
                if cap_match.verdict in {"synthesizable", "missing_skills"}:
                    return station._arbitrate_gap(utterance, intent, cap_match)
                return station._dispatch_claims_filter(utterance, intent, cap_match)
            return _approved("claim_reference", utterance, payload={"ref_type": intent.claim_reference or ""})

        if intent.intent_type == "metric_query":
            metric = (intent.status_query or "").strip()
            if not metric:
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            return _approved("metric_query", utterance, payload={"metric": metric})

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def _handle_procedure(
        self, station: Any, intent: Any, utterance: str
    ) -> ApprovedCommand:
        if intent.intent_type == "concept_teach":
            name = (intent.concept_name or "").strip()
            expansion = (intent.concept_utterance or "").strip()
            if not name or not expansion:
                return _approved("clarification", utterance, "Please specify both a name and an instruction for the concept.")
            return _approved("concept_teach", utterance, payload={"name": name, "utterance": expansion})

        if intent.intent_type == "concept_recall":
            name = (intent.concept_name or "").strip()
            if not name:
                return _approved("clarification", utterance, "Please specify the concept name to recall.")
            concept = station.knowledge_channel.recall(name)
            if concept is None:
                return ApprovedCommand(
                    kind="clarification",
                    utterance=utterance,
                    payload={
                        "message": (
                            f"I don't know a concept named '{name}'.\n"
                            f"To teach it, say: remember {name} means <full instruction>"
                        )
                    },
                )
            if concept.concept_type == "procedure":
                return _approved("procedure_execute", utterance, payload={"steps": list(concept.steps)})
            expanded = concept.utterance
            station.log(f"concept_recall: '{name}' → '{expanded}'")
            return ApprovedCommand(command_type="task_instruction", utterance=expanded)

        if intent.intent_type == "procedure_recall":
            steps = list(intent.concept_steps or [])
            if not steps:
                return _approved("clarification", utterance, "Please specify the concept names to execute in sequence.")
            station.log(f"procedure_recall: steps={steps}")
            return _approved("procedure_execute", utterance, payload={"steps": steps})

        if intent.intent_type == "sequence_instruction":
            usteps = list(intent.utterance_steps or [])
            if not usteps:
                return _approved("clarification", utterance, "Please specify the task steps to execute in sequence.")
            from .llm_compiler import _parse_motor_command as _pmc
            motor_steps = [_pmc(step) for step in usteps]
            if all(step is not None for step in motor_steps):
                sequence = [
                    {"action": str(action), "count": int(count)}
                    for action, count in motor_steps
                    if action is not None
                ]
                station.log(f"sequence_instruction resolved as motor_sequence: {len(sequence)} actions")
                return _approved("motor_sequence_execute", utterance, payload={"sequence": sequence})
            station.log(f"sequence_instruction: steps={usteps}")
            return _approved("sequence_execute", utterance, payload={"steps": usteps})

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def _handle_provenance(
        self, station: Any, intent: Any, utterance: str
    ) -> ApprovedCommand:
        if intent.intent_type == "knowledge_update":
            question_command = station._question_override_command(utterance)
            if question_command is not None:
                station.log("question-shaped utterance overrode knowledge_update intent")
                return question_command
            update = intent.knowledge_update or {}
            delivery_target = update.get("delivery_target")
            if delivery_target is None and intent.target_selector is not None:
                readiness_command = station.command_from_selector_readiness(
                    intent.target_selector,
                    utterance,
                )
                if readiness_command is not None:
                    return readiness_command
                grounded = station.ground_target_selector(intent.target_selector)
                if not grounded["ok"]:
                    clarification = station.maybe_start_selector_clarification(
                        utterance=utterance,
                        resume_kind="knowledge_update",
                        grounded=grounded,
                    )
                    if clarification is not None:
                        return clarification
                    return _approved("ambiguous", utterance, grounded["message"])
                target = grounded["target"]
                return ApprovedCommand(
                    kind="knowledge_update",
                    utterance=utterance,
                    payload={
                        "target_color": target["color"],
                        "target_type": target["type"],
                        "delivery_target": {
                            "color": target["color"],
                            "object_type": target["type"],
                        },
                    },
                )
            if delivery_target is None and intent.knowledge_update is not None:
                return ApprovedCommand(
                    kind="knowledge_update",
                    utterance=utterance,
                    payload={
                        "target_color": None,
                        "target_type": None,
                        "delivery_target": None,
                    },
                )
            if not isinstance(delivery_target, dict):
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            return ApprovedCommand(
                kind="knowledge_update",
                utterance=utterance,
                payload={
                    "target_color": delivery_target["color"],
                    "target_type": delivery_target["object_type"],
                    "delivery_target": {
                        "color": delivery_target["color"],
                        "object_type": delivery_target["object_type"],
                    },
                },
            )

        if intent.intent_type == "primitive_definition":
            if intent.primitive_definition is None:
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            return _approved(
                "primitive_definition",
                utterance,
                payload={"definition": intent.primitive_definition.as_dict()},
            )

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def _handle_action(
        self, station: Any, intent: Any, utterance: str, cap_match: Any
    ) -> ApprovedCommand:
        if intent.intent_type == "steering_directive":
            # Safety net for an LLM-emitted standalone steering intent (the deterministic
            # path handles it earlier). Acknowledge without planning a task.
            directive = intent.steering_directive
            active = ", ".join(
                f"{k}={v}"
                for k, v in (directive.as_dict() if directive else {}).items()
                if v is not None
            )
            return _approved(
                "steering_directive",
                utterance,
                payload={"message": f"STEERING NOTED: {active}."},
            )

        if intent.intent_type == "motor_command":
            action = (intent.action_name or "").strip()
            count = max(1, intent.repeat_count or 1)
            if not action:
                return _approved("clarification", utterance, "Please specify which motor action to perform.")
            station.log(f"motor_command: action={action} count={count}")
            return _approved("motor_execute", utterance, payload={"action": action, "count": count})

        if intent.intent_type == "motor_sequence":
            sequence: list[dict[str, Any]] = []
            for step in (intent.utterance_steps or []):
                parts = step.split(":", 1)
                if len(parts) != 2:
                    continue
                action_name, count_str = parts
                try:
                    action_count = max(1, int(count_str))
                except ValueError:
                    continue
                sequence.append({"action": action_name, "count": action_count})
            # Execute any sequence with at least one parseable step — a single repeated action
            # ("go forward twice" -> ["move_forward:2"]) is valid. Only zero parseable steps is
            # genuinely unparseable. (The sequence_instruction path applies the same rule.)
            if not sequence:
                return _approved("clarification", utterance, "Could not parse motor sequence steps.")
            station.log(f"motor_sequence: {len(sequence)} actions")
            return _approved("motor_sequence_execute", utterance, payload={"sequence": sequence})

        if intent.intent_type == "conditional_sense_motor":
            target = intent.target or {}
            if (
                not target.get("color")
                or not target.get("object_type")
                or not intent.action_name
                or intent.steering_directive is None
            ):
                return _approved(
                    "clarification",
                    utterance,
                    "Conditional motor command requires Sense evidence before actuation. "
                    "Please specify a visible target condition and the action to repeat.",
                )
            mission_plan = self.mission_cortex.plan_conditional_evidence_action(
                intent,
                utterance=utterance,
                active_claims=station.active_claims,
                claims_valid=station._claims_valid_for_current_environment(),
                environment_identity=station.current_environment_identity,
            )
            station.last_request_plan = mission_plan.request_plan
            station.last_readiness_graph = mission_plan.readiness_graph
            station._record_request_state(
                request_plan=mission_plan.request_plan,
                readiness_graph=mission_plan.readiness_graph,
                reason="conditional_mission_planned",
            )
            if mission_plan.readiness_graph.graph_status != "executable":
                return _approved(
                    "clarification",
                    utterance,
                    (
                        "Conditional mission is not executable: "
                        f"{mission_plan.readiness_graph.explanation}"
                    ),
                )
            return _approved(
                "conditional_mission_execute",
                utterance,
                payload={"mission_plan": mission_plan},
                request_id=mission_plan.request_plan.request_id,
                request_plan=mission_plan.request_plan,
                readiness_graph=mission_plan.readiness_graph,
            )

        if intent.intent_type == "mission_contract":
            steps = list(intent.mission_steps or [])
            if len(steps) < 2:
                return _approved("clarification", utterance, "A mission requires at least 2 task steps.")
            station.log(f"mission_contract: {len(steps)} steps")
            return _approved("mission_execute", utterance, payload={"steps": steps})

        if intent.intent_type == "task_instruction":
            if intent.reference == "delivery_target":
                instruction = "go to the delivery target"
            elif intent.reference == "last_target":
                instruction = "go there again"
            elif intent.reference == "last_task":
                instruction = "repeat the last task"
            elif intent.target_selector is not None:
                readiness_command = station.command_from_selector_readiness(
                    intent.target_selector,
                    utterance,
                )
                if readiness_command is not None:
                    return readiness_command
                evidence_command = station._maybe_start_needs_evidence_clarification(
                    utterance,
                    intent,
                )
                if evidence_command is not None:
                    return evidence_command
                grounded = station.ground_target_selector(intent.target_selector)
                if not grounded["ok"]:
                    if cap_match.verdict in {"missing_skills", "synthesizable", "unsupported"}:
                        arb_command = station._arbitrate_gap(utterance, intent, cap_match)
                        if arb_command.kind in {
                            "synthesis_proposal", "missing_skills", "synthesizable",
                        }:
                            return arb_command
                    clarification = station.maybe_start_selector_clarification(
                        utterance=utterance,
                        resume_kind="task_instruction",
                        grounded=grounded,
                    )
                    if clarification is not None:
                        return clarification
                    return _approved("ambiguous", utterance, grounded["message"])
                target = grounded["target"]
                instruction = f"go to the {target['color']} {target['type']}"
            elif isinstance(intent.target, dict):
                if "closest" in _normalize_utterance(utterance):
                    return ApprovedCommand(
                        kind="ambiguous",
                        utterance=utterance,
                        payload={
                            "message": (
                                "I need a valid target selector to ground closest. "
                                "I did not execute."
                            )
                        },
                    )
                color = intent.target.get("color")
                object_type = intent.target.get("object_type")
                supported_object_types = set(station.planning_semantics.object_types)
                if (
                    not color
                    or object_type not in supported_object_types
                    or intent.task_type != "go_to_object"
                ):
                    return ApprovedCommand(command_type="unsupported", utterance=utterance)
                instruction = intent.canonical_instruction or f"go to the {color} {object_type}"
            else:
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            # Phase 13A: honor a steering risk withdrawal — the readiness graph (recorded
            # during dispatch from the steering-bearing intent) is non-executable, so do
            # not attempt to issue an ExecutionTicket. Surface the block instead.
            if station.active_steering_directive is not None:
                graph = station.last_readiness_graph
                if graph is not None and graph.next_action == "ask_authorization":
                    return _approved(
                        "unsupported",
                        utterance,
                        payload={
                            "message": (
                                f"STEERING BLOCKED: {graph.explanation} "
                                "No execution ticket was issued."
                            )
                        },
                    )
            return ApprovedCommand(command_type="task_instruction", utterance=instruction)

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def _handle_control(
        self, station: Any, intent: Any, utterance: str, cap_match: Any
    ) -> ApprovedCommand:
        if intent.intent_type in {"accept_proposal", "reject_proposal"}:
            return ApprovedCommand(command_type=intent.intent_type, utterance=utterance)

        if intent.intent_type == "quit":
            return ApprovedCommand(command_type="quit", utterance=utterance)

        if intent.intent_type == "reset":
            return _approved("reset", utterance, payload={"clear_memory": bool(intent.clear_memory)})

        if intent.intent_type == "concept_forget":
            cname = (intent.concept_name or "").strip().strip("'\"")
            if not cname:
                return _approved("clarification", utterance, "Please specify the concept name to forget.")
            return _approved("concept_forget", utterance, payload={"name": cname})

        if intent.intent_type in {"unsupported", "ambiguous"}:
            # Pure semantic/schema failure (no declared capabilities) → plain error.
            # Reserve arbitration for intents where the LLM identified a capability but
            # the registry says it is missing or synthesizable.
            if not intent.required_capabilities and not cap_match.missing:
                reason = intent.reason or "I could not understand that request."
                # Decide on STRUCTURED intent fields (the LLM's tool-call), not a substring
                # search of its prose. A genuine capability gap carries
                # capability_status="unsupported" → kind="unsupported"; a parse failure
                # (compiler couldn't resolve the utterance) leaves the default status → it is
                # an "I didn't understand" clarification, as is an `ambiguous` intent. The
                # reason text may be the LLM's (helper text), but it never steers the routing.
                if (
                    intent.intent_type == "unsupported"
                    and intent.capability_status == "unsupported"
                ):
                    return ApprovedCommand(
                        kind="unsupported",
                        utterance=utterance,
                        payload={
                            "message": f"I cannot safely execute that capability yet. {reason}"
                        },
                    )
                return _approved("clarification", utterance, f"I didn't understand that: {reason}")
            return station._arbitrate_gap(utterance, intent, cap_match)

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def handle_utterance_text(self, station: Any, utterance: str) -> str:
        from .schemas import OperatorIntent as _OI
        from .steering_parser import parse_steering_clauses

        station.last_repair_events = []
        station.last_operational_mismatches = []

        # Phase 13A: split the steering HOW off the WHAT before any compilation, so the
        # core parser sees clean text and a misparse cannot ride inline shadow regex into
        # the plan. The directive is stashed for IntentVerifier (the gate) to validate and
        # attach; the residual drives the normal pipeline.
        steering_directive, residual = parse_steering_clauses(utterance)
        station._parsed_steering_directive = steering_directive
        if steering_directive is not None:
            if not residual:
                # Standalone steering turn (no WHAT) has nothing to reshape yet —
                # steering a *pending* plan is a later phase. Acknowledge the parsed,
                # typed directive (recorded as last_operator_intent so the turn still
                # traces) without routing into the task planner.
                station._parsed_steering_directive = None
                station.last_operator_intent = _OI(
                    intent_type="steering_directive",
                    steering_directive=steering_directive,
                    confidence=1.0,
                    reason="standalone steering directive",
                )
                active = ", ".join(
                    f"{k}={v}"
                    for k, v in steering_directive.as_dict().items()
                    if v is not None
                )
                return (
                    f"STEERING NOTED: {active}. "
                    "Issue a task to steer (mid-plan steering arrives in a later phase)."
                )
            # Embedded steering: route the residual WHAT through the compiler + dispatch
            # path so IntentVerifier (the gate) validates and attaches the directive —
            # never via the deterministic fast path, which skips verification.
            command = station.command_from_llm_intent(residual)
            if station.pending_clarification is not None:
                pending_response = self.handle_pending_clarification(
                    station,
                    utterance,
                    command,
                )
                if pending_response is not None:
                    return pending_response
            return self.execute_command(station, command)

        # IntentCache fast path: regex patterns that produce OperatorIntent and route
        # through dispatch (IntentVerifier + knowledge-type routing) just like LLM intents.
        if self.intent_cache is not None:
            cached = self.intent_cache.lookup(utterance)
            if cached is not None:
                if isinstance(cached, _OI):
                    station.log(f"intent cache hit: {cached.intent_type}")
                    command = self.dispatch(station, cached, utterance)
                else:
                    # ApprovedCommand from cache (error cases, e.g. unsafe formula)
                    command = cached
                if station.pending_clarification is not None:
                    pending_response = self.handle_pending_clarification(
                        station,
                        utterance,
                        command,
                    )
                    if pending_response is not None:
                        return pending_response
                return self.execute_command(station, command)

        command = self.classify_utterance(utterance)
        if getattr(station, "pending_primitive_definition", None) is not None:
            return station.handle_pending_primitive_definition(utterance, command)
        if station.pending_synthesis_proposal is not None:
            return station.handle_pending_synthesis_proposal(utterance, command)
        if station.pending_clarification is not None:
            pending_response = self.handle_pending_clarification(station, utterance, command)
            if pending_response is not None:
                return pending_response

        if command.kind == "unresolved":
            command = station._command_from_active_claim_text(utterance) or command
            if command.kind == "unresolved":
                grounding_followup = station._command_from_grounding_followup(utterance)
                if grounding_followup is not None:
                    command = grounding_followup
            if command.kind == "unresolved":
                concept_command = station._command_from_concept(utterance)
                if concept_command is not None:
                    command = concept_command
            if command.kind == "unresolved" and self.looks_like_bare_label(
                self.normalize_utterance(utterance.strip())
            ):
                label = self.normalize_utterance(utterance.strip())
                return (
                    f"I don't recognise '{label}' as a command or a known concept.\n"
                    f"To teach it as a shorthand, say:\n"
                    f"  remember {label} means <full instruction>\n"
                    f"Example: remember {label} means go to the red door"
                )
            if command.kind == "unresolved":
                station.log("deterministic fast path unresolved; compiling operator intent")
                command = station.command_from_llm_intent(utterance)

        return self.execute_command(station, command)

    def handle_pending_clarification(
        self,
        station: Any,
        utterance: str,
        command: ApprovedCommand,
    ) -> str | None:
        normalized = self.normalize_utterance(utterance)
        if command.kind == "quit":
            station.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            station.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return station.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "cache_query":
            return station.cache_summary()
        if command.kind == "status_query":
            return station.status_summary(query=command.payload.get("query", "status"))

        pending = station.pending_clarification
        if pending is not None and pending.clarification_type == "arbitrator_offer":
            if station._is_acceptance(normalized):
                station.pending_clarification = None
                return station.resume_arbitration_offer(pending.resume_kind)

        if pending is not None and pending.clarification_type == "semantic_query_missing_field":
            if station.domain_helper.is_metric_answer(normalized, "manhattan"):
                station.pending_clarification = None
                clarified = f"{pending.original_utterance} using manhattan distance"
                station.log("resuming semantic query plan with distance_metric=manhattan")
                resumed = station.command_from_llm_intent(clarified)
                if resumed.kind == "clarification":
                    return resumed.payload["message"]
                return self.execute_command(station, resumed)
            if station.domain_helper.is_metric_answer(normalized, "euclidean"):
                station.pending_clarification = None
                return "I cannot use Euclidean distance yet. Supported: manhattan."

        if station.domain_helper.is_metric_answer(normalized, "manhattan"):
            return station.resume_pending_clarification("manhattan")
        if station.domain_helper.is_metric_answer(normalized, "euclidean"):
            station.pending_clarification = None
            return "I cannot use Euclidean distance yet. Supported: manhattan."

        color_answer = station.domain_helper.color_answer(normalized)
        if color_answer is not None:
            candidate_response = station.resume_candidate_clarification(color_answer)
            if candidate_response is not None:
                return candidate_response

        if command.kind == "unresolved":
            station.log("deterministic fast path unresolved; compiling operator intent")
            command = station.command_from_llm_intent(utterance)
        if command.kind in _NEW_INTENT_COMMAND_KINDS:
            station.pending_clarification = None
            station.log("new operator intent cancelled pending clarification")
            return self.execute_command(station, command)
        if command.kind == "synthesis_proposal":
            station.pending_clarification = None
            return command.payload["message"]
        if command.kind == "clarification":
            return command.payload["message"]
        if command.kind in {"missing_skills", "synthesizable"}:
            station.pending_clarification = None
            return command.payload.get("message", "I cannot fulfil that request.")

        pending = station.pending_clarification
        if pending is None:
            return None
        return station.clarification_prompt(
            pending.missing_field,
            pending.supported_values,
            candidates=pending.candidates,
        )

    def execute_command(self, station: Any, command: ApprovedCommand) -> str:
        station.log(f"classified utterance as {command.kind}")
        if command.kind == "quit":
            station.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            station.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return station.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "clarification":
            return command.payload.get("message", "")
        if command.kind == "concept_teach":
            return station.teach_concept(command.payload["name"], command.payload["utterance"])
        if command.kind == "concept_forget":
            return station.forget_concept(command.payload["name"])
        if command.kind == "procedure_execute":
            return station._run_procedure(command.payload["steps"], command.utterance)
        if command.kind == "sequence_execute":
            return station._run_sequence(command.payload["steps"], command.utterance)
        if command.kind == "motor_execute":
            return station._run_motor_command(
                command.payload["action"],
                command.payload["count"],
                command.utterance,
            )
        if command.kind == "motor_sequence_execute":
            return station._run_motor_sequence_command(
                command.payload["sequence"],
                command.utterance,
            )
        if command.kind == "conditional_mission_execute":
            return station._run_conditional_mission(command.payload["mission_plan"])
        if command.kind == "mission_execute":
            return station._run_mission(command.payload["steps"], command.utterance)
        if command.kind == "cache_query":
            return station.cache_summary()
        if command.kind == "status_query":
            return station.status_summary(query=command.payload.get("query", "status"))
        if command.kind == "ground_target_query":
            return station.grounded_target_summary(command.payload)
        if command.kind == "metric_query":
            return station.metric_query_summary(command)
        if command.kind == "task_selector":
            return station.task_selector_summary(command)
        if command.kind == "primitive_definition":
            return station.propose_primitive_definition(
                command.payload["definition"],
                mission_request=command.payload.get("mission_request"),
            )
        if command.kind == "knowledge_update":
            return station._apply_knowledge_update_from_payload(
                command.utterance,
                command.payload,
                source="operator",
            )
        if command.kind == "claim_reference":
            return station.claim_reference_summary(command.payload.get("ref_type", ""))
        if command.kind == "synthesis_proposal":
            return command.payload["message"]
        if command.kind in {"accept_proposal", "reject_proposal"}:
            return station.status_summary(query="help")
        if command.kind == "missing_skills":
            return command.payload.get("message", "I do not have the required capabilities.")
        if command.kind == "synthesizable":
            return command.payload.get("message", "That capability is not yet implemented.")
        if command.kind == "task_instruction":
            instruction = station.resolve_task_instruction(command.utterance)
            if instruction is None:
                return station.missing_reference_summary(command.utterance)
            if instruction != command.utterance:
                station.log(f"resolved task instruction to: {instruction}")
            result = station._run_task_from_instruction(
                command.utterance,
                instruction,
                source="operator",
                record_plan=True,
            )
            return station.result_summary(result)
        if command.kind == "steering_directive":
            return command.payload.get("message", "STEERING NOTED.")
        if command.kind == "unsupported":
            return command.payload.get(
                "message",
                "I cannot safely execute that capability yet.",
            )
        if command.kind == "ambiguous":
            return command.payload.get(
                "message",
                "I could not safely resolve that instruction yet.",
            )
        return station.status_summary(query="help")
