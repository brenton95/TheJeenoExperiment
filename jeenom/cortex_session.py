from __future__ import annotations

from typing import Any

from .capability_registry import CapabilityRegistry
from .orpi import assert_no_deliberative_meta_plan_references
from .planning_semantics import PlanningSemantics
from .readiness_graph import evaluate_request_plan
from .request_planner import build_request_plan
from .schemas import (
    EnvironmentIdentity,
    OperatorIntent,
    ReadinessGraph,
    RequestPlan,
    StationActiveClaims,
)


class CortexSession:
    """Substrate-neutral planning kernel: intent → (RequestPlan, ReadinessGraph).

    Centralises all calls to build_request_plan and evaluate_request_plan so the
    station never invokes the planning pipeline directly.  Stateless between calls:
    the caller passes the current session state each time.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        planning_semantics: PlanningSemantics,
        risk_policy: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.planning_semantics = planning_semantics
        self.risk_policy = risk_policy

    def plan(
        self,
        utterance: str,
        intent: OperatorIntent,
        *,
        active_claims: StationActiveClaims | None = None,
        claims_valid: bool = False,
        environment_identity: EnvironmentIdentity | None = None,
        evidence_state: dict[str, Any] | None = None,
    ) -> tuple[RequestPlan, ReadinessGraph]:
        """Build and evaluate a request plan for an intent."""
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
        assert_no_deliberative_meta_plan_references(plan, self.registry)
        graph = evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
            risk_policy=self.risk_policy,
            evidence_state=evidence_state,
        )
        return plan, graph

    def evaluate(
        self,
        plan: RequestPlan,
        *,
        active_claims: StationActiveClaims | None = None,
        claims_valid: bool = False,
        environment_identity: EnvironmentIdentity | None = None,
        evidence_state: dict[str, Any] | None = None,
    ) -> ReadinessGraph:
        """Evaluate an already-built plan (e.g. after repair, cache reuse, or manual construction)."""
        return evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
            risk_policy=self.risk_policy,
            evidence_state=evidence_state,
        )
