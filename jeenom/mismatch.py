"""Operational Mismatch Detection — Phase 8.4.

A MismatchDetector inspects a compiled RequestPlan against the current
environment state (scene model, active claims, capability registry) and
produces typed OperationalMismatch records describing structural divergences
between what the plan assumed and what is actually present.

Detection is purely diagnostic: it does not gate execution.  ReadinessGraph
remains the execution gate.  The station records last_operational_mismatches
after every plan evaluation; Phase 8.5 repair loop consumes them.

Six mismatch types:

  STALE_CLAIMS
    Active claims exist but their scene fingerprint does not match the current
    scene (agent moved or step count changed since grounding).

  REQUIRED_ENTITY_ABSENT
    A required entity, target, or condition assumed by the plan is absent from
    current claims or scene evidence.  MiniGrid red-door absence is one
    implementation; the definition is substrate-independent.

  GROUNDING_RELATION_INVALIDATED
    A grounding predicate (distance ranking, tie resolution, candidate set
    membership, or selected-target predicate) no longer holds against the
    current scene.  The claims fingerprint may still match but the grounding
    result would differ if recomputed now.

  UNSUPPORTED_GROUNDING
    A plan step requires a grounding primitive whose implementation_status in
    the registry is "unsupported" or "synthesizable".  The plan cannot be
    correctly grounded with the available substrate.

  MISSING_PRIMITIVE_IN_REGISTRY
    A plan step references a required_handle that does not exist in the
    capability registry at all.

  CONSTRAINT_WEAKENING
    A plan step carries a semantic constraint (exclude_colors, ordinal,
    threshold, comparison) for which no enforcing primitive is present or
    implemented in the registry.  Silent execution would drop the constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .capability_registry import CapabilityRegistry
    from .schemas import (
        EnvironmentIdentity,
        RequestPlan,
        SceneModel,
        StationActiveClaims,
    )

MISMATCH_TYPES = (
    "STALE_CLAIMS",
    "REQUIRED_ENTITY_ABSENT",
    "GROUNDING_RELATION_INVALIDATED",
    "UNSUPPORTED_GROUNDING",
    "MISSING_PRIMITIVE_IN_REGISTRY",
    "CONSTRAINT_WEAKENING",
)

MISMATCH_SEVERITIES = ("critical", "warning", "info")

MISMATCH_RECOMMENDED_REPAIRS = (
    "refresh_claims",
    "reground_target",
    "recompile",
    "clarify_operator",
    "none",
)

# Plan-step constraint fields that require an enforcing primitive.
# Maps constraint key → registry handle prefix that must be present and implemented.
_CONSTRAINT_ENFORCEMENT_MAP: dict[str, str] = {
    "exclude_colors": "grounding.unique_door.color_filter",
    "comparison": "grounding.closest_door",
    "ordinal": "grounding.closest_door",
}


@dataclass
class OperationalMismatch:
    """A typed divergence between plan assumptions and current environment state."""

    mismatch_type: str
    step_id: str | None
    description: str
    severity: str                            # "critical" | "warning" | "info"
    affected_assumption_ids: list[str] = field(default_factory=list)
    recommended_repair: str = "none"         # one of MISMATCH_RECOMMENDED_REPAIRS

    def as_dict(self) -> dict[str, Any]:
        return {
            "mismatch_type": self.mismatch_type,
            "step_id": self.step_id,
            "description": self.description,
            "severity": self.severity,
            "affected_assumption_ids": list(self.affected_assumption_ids),
            "recommended_repair": self.recommended_repair,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationalMismatch:
        return cls(
            mismatch_type=data["mismatch_type"],
            step_id=data.get("step_id"),
            description=data["description"],
            severity=data.get("severity", "warning"),
            affected_assumption_ids=list(data.get("affected_assumption_ids", [])),
            recommended_repair=data.get("recommended_repair", "none"),
        )


class MismatchDetector:
    """Detects structural mismatches between a RequestPlan and current state.

    All checks are pure — no I/O, no env resets, no LLM calls.  Pass None for
    any context component that is unavailable; that check is skipped.
    """

    def detect(
        self,
        plan: RequestPlan,
        *,
        registry: CapabilityRegistry | None = None,
        scene_model: SceneModel | None = None,
        active_claims: StationActiveClaims | None = None,
        environment_identity: EnvironmentIdentity | None = None,
    ) -> list[OperationalMismatch]:
        results: list[OperationalMismatch] = []
        results.extend(self._check_stale_claims(active_claims, scene_model))
        results.extend(self._check_required_entity_absent(plan, scene_model))
        results.extend(self._check_grounding_relation_invalidated(
            active_claims, scene_model,
        ))
        if registry is not None:
            results.extend(self._check_unsupported_grounding(plan, registry))
            results.extend(self._check_missing_primitive(plan, registry))
            results.extend(self._check_constraint_weakening(plan, registry))
        return results

    # ── individual checks ─────────────────────────────────────────────────────

    def _check_stale_claims(
        self,
        active_claims: StationActiveClaims | None,
        scene_model: SceneModel | None,
    ) -> list[OperationalMismatch]:
        if active_claims is None or scene_model is None:
            return []
        if not active_claims.is_valid_for(scene_model):
            return [OperationalMismatch(
                mismatch_type="STALE_CLAIMS",
                step_id=None,
                description=(
                    "Active claims were computed at a different scene state "
                    f"(claims fingerprint={active_claims.scene_fingerprint}, "
                    f"current=(agent_x={scene_model.agent_x}, "
                    f"agent_y={scene_model.agent_y}, "
                    f"step_count={scene_model.step_count})). "
                    "Grounding results may not reflect the current environment."
                ),
                severity="warning",
                recommended_repair="refresh_claims",
            )]
        return []

    def _check_required_entity_absent(
        self,
        plan: RequestPlan,
        scene_model: SceneModel | None,
    ) -> list[OperationalMismatch]:
        if scene_model is None:
            return []
        results: list[OperationalMismatch] = []
        for step in plan.steps:
            color = step.constraints.get("color")
            if not color:
                continue
            object_type = step.constraints.get("object_type")
            if not object_type:
                continue
            matches = scene_model.find(color=color, object_type=object_type)
            if not matches:
                results.append(OperationalMismatch(
                    mismatch_type="REQUIRED_ENTITY_ABSENT",
                    step_id=step.step_id,
                    description=(
                        f"Plan step '{step.step_id}' requires a {color} {object_type} "
                        f"but none is present in the current scene."
                    ),
                    severity="critical",
                    affected_assumption_ids=list(step.environment_assumption_ids),
                    recommended_repair="reground_target",
                ))
        return results

    def _check_grounding_relation_invalidated(
        self,
        active_claims: StationActiveClaims | None,
        scene_model: SceneModel | None,
    ) -> list[OperationalMismatch]:
        """Fire when claims fingerprint matches but grounding predicate result differs.

        Concretely: the ranked-object ordering in active_claims used distances
        computed at the time of grounding. If the scene now has objects at
        positions that would produce a different ranking (e.g. because agent
        moved between grounding and now, captured in the claims' recorded
        distances vs. freshly computed ones), fire this mismatch.
        """
        if active_claims is None or scene_model is None:
            return []
        # Only fires when claims ARE scene-valid; stale claims are covered by STALE_CLAIMS.
        if not active_claims.is_valid_for(scene_model):
            return []
        ranked = active_claims.ranked_objects
        if not ranked:
            return []

        from .schemas import SceneObject

        # Recompute distances from current agent pose for the same object positions.
        try:
            recomputed: list[tuple[float, str | None]] = []
            for entry in ranked:
                obj = SceneObject(
                    object_type=entry.object_type,
                    color=entry.color,
                    x=entry.x,
                    y=entry.y,
                )
                dist = scene_model.manhattan_distance_from_agent(obj)
                recomputed.append((dist, entry.color))

            fresh_order = sorted(range(len(recomputed)), key=lambda i: recomputed[i][0])
            original_order = list(range(len(ranked)))

            if fresh_order != original_order:
                top_fresh = recomputed[fresh_order[0]]
                top_original = ranked[0]
                object_type = top_original.object_type
                return [OperationalMismatch(
                    mismatch_type="GROUNDING_RELATION_INVALIDATED",
                    step_id=None,
                    description=(
                        f"Ranked-{object_type} ordering has changed since grounding. "
                        f"Claims ranked '{top_original.color}' {object_type} first "
                        f"(distance={top_original.distance}), but current scene "
                        f"would rank '{top_fresh[1]}' {object_type} first "
                        f"(distance={top_fresh[0]})."
                    ),
                    severity="warning",
                    recommended_repair="refresh_claims",
                )]
        except Exception:
            pass
        return []

    def _check_unsupported_grounding(
        self,
        plan: RequestPlan,
        registry: CapabilityRegistry,
    ) -> list[OperationalMismatch]:
        results: list[OperationalMismatch] = []
        for step in plan.steps:
            handle = step.required_handle
            if handle is None or step.layer not in {"grounding", "claims"}:
                continue
            spec = registry.lookup(handle)
            if spec is None:
                continue  # covered by MISSING_PRIMITIVE_IN_REGISTRY
            if spec.implementation_status in {"unsupported", "synthesizable"}:
                results.append(OperationalMismatch(
                    mismatch_type="UNSUPPORTED_GROUNDING",
                    step_id=step.step_id,
                    description=(
                        f"Plan step '{step.step_id}' requires grounding primitive "
                        f"'{handle}' which has implementation_status="
                        f"'{spec.implementation_status}' and cannot be executed."
                    ),
                    severity="warning",
                    affected_assumption_ids=list(step.environment_assumption_ids),
                    recommended_repair="clarify_operator",
                ))
        return results

    def _check_missing_primitive(
        self,
        plan: RequestPlan,
        registry: CapabilityRegistry,
    ) -> list[OperationalMismatch]:
        results: list[OperationalMismatch] = []
        for step in plan.steps:
            handle = step.required_handle
            if handle is None:
                continue
            if registry.lookup(handle) is None:
                results.append(OperationalMismatch(
                    mismatch_type="MISSING_PRIMITIVE_IN_REGISTRY",
                    step_id=step.step_id,
                    description=(
                        f"Plan step '{step.step_id}' requires primitive '{handle}' "
                        f"which is not registered in the capability registry."
                    ),
                    severity="critical",
                    affected_assumption_ids=list(step.environment_assumption_ids),
                    recommended_repair="recompile",
                ))
        return results

    def _check_constraint_weakening(
        self,
        plan: RequestPlan,
        registry: CapabilityRegistry,
    ) -> list[OperationalMismatch]:
        results: list[OperationalMismatch] = []
        for step in plan.steps:
            constraints = step.constraints
            if not constraints:
                continue
            for constraint_key, required_handle_prefix in _CONSTRAINT_ENFORCEMENT_MAP.items():
                value = constraints.get(constraint_key)
                if not value:
                    continue
                # exclude_colors must be a non-empty list to trigger the check
                if constraint_key == "exclude_colors" and (
                    not isinstance(value, list) or len(value) == 0
                ):
                    continue
                # Check if an implementing primitive exists and is operational
                implementing = [
                    registry.lookup(name)
                    for name in registry.primitive_names()
                    if name.startswith(required_handle_prefix)
                    and registry.lookup(name) is not None
                    and registry.lookup(name).implementation_status == "implemented"
                ]
                if not implementing:
                    results.append(OperationalMismatch(
                        mismatch_type="CONSTRAINT_WEAKENING",
                        step_id=step.step_id,
                        description=(
                            f"Plan step '{step.step_id}' carries constraint "
                            f"'{constraint_key}={value}' but no implemented primitive "
                            f"matching '{required_handle_prefix}' exists in the registry. "
                            f"The constraint would be silently dropped during execution."
                        ),
                        severity="warning",
                        affected_assumption_ids=list(step.environment_assumption_ids),
                        recommended_repair="recompile",
                    ))
        return results


default_detector = MismatchDetector()
