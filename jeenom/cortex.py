from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .claim_freshness import (
    FRESHNESS_CURRENT,
    FRESHNESS_UNVERIFIABLE,
    OBSERVATION_KIND,
    next_freshness,
)
from .command_registry import DIRECT_ACTION_SKILLS, evidence_needs_for_step
from .schemas import (
    EvidenceFrame,
    ExecutionContract,
    ObservationClaim,
    ReadinessReport,
    TraceEvent,
)


# A claim is actionable on the hot path while it is either observed now
# (``current``) or believed-but-unconfirmable (``unverifiable``, e.g. out of
# view). Once it goes ``stale`` (world changed) or ``unknown`` (decayed past
# belief) it is treated as absent.
_USABLE_FRESHNESS = frozenset({FRESHNESS_CURRENT, FRESHNESS_UNVERIFIABLE})


class Cortex:
    def __init__(self, memory, compiler, plan_cache=None):
        self.memory = memory
        self.compiler = compiler
        self.plan_cache = plan_cache
        self._claims: dict[str, ObservationClaim] = {}
        self.trace: list[TraceEvent] = []
        self.task_request = None
        self.procedure = None
        self.resolved_task_params = {}
        self.last_world_sample = None
        self.execution_state = {
            "step_index": 0,
            "task_complete": False,
            "current_skill": None,
            "last_report": None,
            "knowledge_override_active": False,
        }

    # ── Claim accessors ────────────────────────────────────────────────────────

    @staticmethod
    def _is_usable(claim: ObservationClaim | None) -> bool:
        """A claim contributes to the hot path only while its freshness is usable."""
        return claim is not None and claim.freshness in _USABLE_FRESHNESS

    @property
    def claims(self) -> dict[str, Any]:
        """Raw-value view of the internal claim store (backward-compatible dict).

        Only usable (current/unverifiable) claims are exposed; a belief that has
        gone stale or decayed to unknown reads as absent.
        """
        return {k: v.value for k, v in self._claims.items() if self._is_usable(v)}

    def get_claim(self, key: str) -> Any:
        """Return the raw value of claim `key`, or None if absent or not usable."""
        claim = self._claims.get(key)
        return claim.value if self._is_usable(claim) else None

    def set_claim(
        self,
        key: str,
        value: Any,
        source: str = "sense",
        level: str = "command",
        freshness: str = FRESHNESS_CURRENT,
        last_observed_tick: int | None = None,
    ) -> None:
        """Store a raw value as a typed ObservationClaim carrying a freshness state."""
        self._claims[key] = ObservationClaim(
            key=key,
            value=value,
            source=source,
            level=level,
            freshness=freshness,
            last_observed_tick=last_observed_tick,
        )

    def has_claim(self, key: str) -> bool:
        """Return True if claim `key` is present, usable, and its value is truthy."""
        claim = self._claims.get(key)
        return self._is_usable(claim) and bool(claim.value)

    # ── Task lifecycle ─────────────────────────────────────────────────────────

    def onboard_task(self, task_request, procedure):
        self.task_request = task_request
        self.procedure = procedure
        self.resolved_task_params = self.memory.resolve_target_params(task_request.params)
        self.memory.reset_episode(clear_reference_context=False)
        self.execution_state.update(
            {
                "step_index": 0,
                "task_complete": False,
                "task_failed": False,
                "failure_reason": None,
                "current_skill": None,
                "last_report": None,
                "conditional_last_pose": None,
                "conditional_action_pending": False,
                "knowledge_override_active": False,
            }
        )

        color_override = (
            task_request.params.get("color") is not None
            and self.resolved_task_params.get("color") != task_request.params.get("color")
        )
        type_override = (
            task_request.params.get("object_type") is not None
            and self.resolved_task_params.get("object_type") != task_request.params.get("object_type")
        )
        self.execution_state["knowledge_override_active"] = color_override or type_override

        readiness = ReadinessReport(
            status="executable",
            task_type=procedure.task_type,
            recipe_steps=list(procedure.steps),
        )
        self.record_trace(
            "task_onboarded",
            {
                "task_request": asdict(task_request),
                "resolved_task_params": dict(self.resolved_task_params),
                "procedure": list(procedure.steps),
                "readiness": asdict(readiness),
            },
        )
        return readiness

    def make_evidence_frame(self):
        self._advance_completed_steps()
        if self.execution_state.get("task_failed"):
            return EvidenceFrame(
                needs=[],
                context=dict(self.resolved_task_params),
                active_step=self._current_step_name(),
                step_index=self.execution_state["step_index"],
            )
        active_step = self._current_step_name()

        if active_step is None:
            return EvidenceFrame(
                needs=[],
                context=dict(self.resolved_task_params),
                step_index=self.execution_state["step_index"],
            )
        needs = evidence_needs_for_step(active_step)

        frame = EvidenceFrame(
            needs=needs,
            context=dict(self.resolved_task_params),
            active_step=active_step,
            step_index=self.execution_state["step_index"],
        )
        self.record_trace(
            "evidence_frame_created",
            {"evidence_needs": list(needs), "context": dict(frame.context)},
            step_name=active_step,
        )
        return frame

    def update_from_evidence(self, evidence, world_sample=None):
        active_step = self._current_step_name()
        if active_step == "act_until_evidence":
            pose = evidence.claims.get("agent_pose")
            prior_pose = self.execution_state.get("conditional_last_pose")
            if self.execution_state.get("conditional_action_pending"):
                stop_claim = str(self.resolved_task_params.get("stop_claim") or "")
                stop_value = self.resolved_task_params.get("stop_value", True)
                condition_met = evidence.claims.get(stop_claim) == stop_value
                if not condition_met and pose == prior_pose:
                    self.execution_state["task_failed"] = True
                    self.execution_state["failure_reason"] = "no_progress"
                    self.record_trace(
                        "conditional_mission_failed",
                        {
                            "reason": "no_progress",
                            "action_name": self.resolved_task_params.get("action_name"),
                            "agent_pose": pose,
                        },
                        step_name=active_step,
                    )
                self.execution_state["conditional_action_pending"] = False
            self.execution_state["conditional_last_pose"] = pose

        # TECH-DEBT(mission-clock-rests-on-skip-reset): step_count is the decay clock,
        # but it only spans a whole mission because the station reuses the adapter with
        # skip_reset=True — that reuse is not yet a guaranteed contract. Intra-task
        # decay (the only thing wired here) does not depend on it.
        tick = getattr(world_sample, "step_count", None) if world_sample is not None else None
        observed_keys = set(evidence.claims.keys())

        if tick is not None:
            for key, claim in self._claims.items():
                if key in observed_keys:
                    continue  # re-observed this tick; refreshed below
                last_tick = claim.last_observed_tick
                steps_unseen = max(0, tick - last_tick) if last_tick is not None else 0
                # TECH-DEBT(uniform-decay): every carried claim ages as an observation
                # at the single UNVERIFIABLE_DECAY_STEPS rate; per-kind rates
                # (ttl_for_kind) stay dormant until a substrate with a changing world
                # (AI2-THOR) can falsify them. MiniGrid cannot.
                # TECH-DEBT(intra-task-decay): cortex._claims is rebuilt per task, so
                # this decay is intra-task only; mission-scope decay waits until belief
                # moves into memory (mission-contract phase).
                claim.freshness = next_freshness(
                    claim.freshness,
                    kind=OBSERVATION_KIND,
                    in_view=False,
                    steps_unseen=steps_unseen,
                )

        for k, v in evidence.claims.items():
            self.set_claim(k, v, source=evidence.source, last_observed_tick=tick)
        if world_sample is not None:
            self.last_world_sample = world_sample
        self.record_trace(
            "evidence_update",
            {"claims": evidence.claims},
            step_name=self._current_step_name(),
        )
        if not self.execution_state.get("task_failed"):
            self._advance_completed_steps()

    def choose_execution_contract(self):
        self._advance_completed_steps()
        if self.execution_state.get("task_failed"):
            return None
        active_step = self._current_step_name()
        if active_step is None:
            self.execution_state["task_complete"] = True
            return None

        if active_step == "done" and self.execution_state.get("knowledge_override_active"):
            self.execution_state["task_complete"] = True
            self.record_trace(
                "done_skipped_for_knowledge_override",
                {
                    "resolved_task_params": dict(self.resolved_task_params),
                    "reason": "MiniGrid mission reward disagrees with durable knowledge target.",
                },
                step_name=active_step,
            )
            return None

        params = dict(self.resolved_task_params)
        params["target_location"] = self.get_claim("target_location")

        if active_step == "locate_object":
            skill = "turn_right"
        elif active_step in {"navigate_to_object", "verify_adjacent"}:
            skill = "navigate_to_object" if self.get_claim("target_location") else "turn_right"
        elif active_step == "act_until_evidence":
            skill = str(self.resolved_task_params.get("action_name") or "")
            if skill not in DIRECT_ACTION_SKILLS or skill in {"done"}:
                self.execution_state["task_failed"] = True
                self.execution_state["failure_reason"] = "invalid_conditional_action"
                self.record_trace(
                    "conditional_mission_failed",
                    {"reason": "invalid_conditional_action", "action_name": skill},
                    step_name=active_step,
                )
                return None
        elif active_step == "done":
            skill = "done"
        else:
            skill = "abort"

        stop_conditions = ["adjacent_to_target", "task_complete"]
        if active_step == "act_until_evidence":
            stop_conditions = [
                str(self.resolved_task_params.get("stop_claim") or "target_visible")
            ]
        contract = ExecutionContract(
            skill=skill,
            params=params,
            stop_conditions=stop_conditions,
            source="cortex",
        )
        self.execution_state["current_skill"] = skill
        self.record_trace(
            "execution_contract_issued",
            asdict(contract),
            step_name=active_step,
        )
        return contract

    def update_from_report(self, report):
        self.execution_state["last_report"] = asdict(report)
        if self._current_step_name() == "act_until_evidence":
            if report.status == "failed":
                self.execution_state["task_failed"] = True
                self.execution_state["failure_reason"] = report.reason or "execution_failed"
            elif report.progress.get("executed_action") is not None:
                self.execution_state["conditional_action_pending"] = True
        if self.execution_state["current_skill"] == "done" and report.status == "succeeded":
            self.execution_state["task_complete"] = True

        self.record_trace(
            "execution_report",
            asdict(report),
            step_name=self._current_step_name(),
        )

    def finalize(self):
        final_state = {
            "task_request": asdict(self.task_request) if self.task_request else None,
            "resolved_task_params": dict(self.resolved_task_params),
            "execution_state": dict(self.execution_state),
            "last_world_sample": self.last_world_sample.summary() if self.last_world_sample else None,
        }
        trace_payload = [asdict(event) for event in self.trace]
        updates = self.compiler.compile_memory_updates(
            final_state=final_state,
            final_claims=dict(self.claims),
            trace=trace_payload,
            memory=self.memory,
        )
        self.memory.apply_memory_updates(updates)
        self.record_trace(
            "memory_updates_applied",
            {"updates": [asdict(update) for update in updates]},
        )
        return updates

    def record_trace(self, event: str, payload: dict, step_name: str | None = None) -> None:
        self.trace.append(
            TraceEvent(
                event=event,
                payload=payload,
                loop_index=len(self.trace),
                step_name=step_name,
            )
        )

    def _advance_completed_steps(self):
        if self.execution_state.get("task_failed"):
            return
        while True:
            active_step = self._current_step_name()
            if active_step is None:
                self.execution_state["task_complete"] = True
                return
            if not self._step_complete(active_step):
                return
            self.execution_state["step_index"] += 1

    def _current_step_name(self):
        if self.procedure is None:
            return None
        idx = self.execution_state["step_index"]
        if idx >= len(self.procedure.steps):
            return None
        return self.procedure.steps[idx]

    def _step_complete(self, step_name):
        if step_name == "act_until_evidence":
            stop_claim = str(
                self.resolved_task_params.get("stop_claim") or "target_visible"
            )
            stop_value = self.resolved_task_params.get("stop_value", True)
            condition_met = self.get_claim(stop_claim) == stop_value
            if condition_met:
                self.record_trace(
                    "conditional_stop_condition_satisfied",
                    {"claim": stop_claim, "value": stop_value},
                    step_name=step_name,
                )
            return condition_met
        if step_name == "locate_object":
            return self.has_claim("target_location")
        if step_name in {"navigate_to_object", "verify_adjacent"}:
            return self.has_claim("adjacency_to_target")
        if step_name == "done":
            return bool(self.execution_state.get("task_complete"))
        return False
