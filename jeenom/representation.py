from __future__ import annotations

from typing import Any

from .knowledge_base import NamedConcept
from .turn_orchestrator import KnowledgeChannel
from .memory import OperationalMemory
from .schemas import (
    ClaimRecord,
    KnowledgeSnapshot,
    MemoryUpdate,
    MemoryWriteTicket,
    ReadinessGraph,
    RequestPlan,
    SceneModel,
    StationActiveClaims,
)


class RepresentationStore:
    """Thin boundary over current memory pockets.

    This is deliberately small. It does not replace OperationalMemory or
    KnowledgeBase yet; it prevents architecture blocks from treating those
    internals as cross-block authority.
    """

    def __init__(
        self,
        *,
        memory: OperationalMemory,
        knowledge_channel: KnowledgeChannel,
    ) -> None:
        self.memory = memory
        self.knowledge_channel = knowledge_channel
        self._provenance: list[dict[str, Any]] = []
        self._active_claims: StationActiveClaims | None = None

    @property
    def _claims(self) -> dict[str, ClaimRecord]:
        """The single mission-scoped claim store, owned by OperationalMemory.

        Read through a property (not a captured reference) so a typed reset that
        reassigns ``memory.claims`` is always reflected here. The cortex belief
        loop writes the same dict, so this is genuinely one store.
        """
        return self.memory.claims

    # -- claims -----------------------------------------------------------------

    def put_claim(self, claim: ClaimRecord) -> ClaimRecord:
        if not isinstance(claim, ClaimRecord):
            raise TypeError("put_claim requires a ClaimRecord")
        self._claims[claim.key] = claim
        return claim

    def get_claim(self, key: str, *, scope: str | None = None) -> ClaimRecord | None:
        claim = self._claims.get(key)
        if claim is None:
            return None
        if scope is not None and claim.scope != scope:
            return None
        return claim

    def query_claims(
        self,
        *,
        kind: str | None = None,
        scope: str | None = None,
        status: str | None = None,
    ) -> list[ClaimRecord]:
        claims = list(self._claims.values())
        if kind is not None:
            claims = [claim for claim in claims if claim.kind == kind]
        if scope is not None:
            claims = [claim for claim in claims if claim.scope == scope]
        if status is not None:
            claims = [claim for claim in claims if claim.status == status]
        return claims

    def invalidate_claims(
        self,
        *,
        scope: str | None = None,
        source: str | None = None,
        reason: str = "",
    ) -> int:
        count = 0
        for claim in self._claims.values():
            if scope is not None and claim.scope != scope:
                continue
            if source is not None and claim.source != source:
                continue
            claim.status = "invalidated"
            claim.freshness = "stale"
            claim.invalidation = {"reason": reason}
            count += 1
        if scope in {None, "grounding"}:
            self._active_claims = None
        return count

    # -- procedures -------------------------------------------------------------

    def put_procedure(
        self,
        name: str,
        utterance: str,
        plan: Any | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> NamedConcept:
        concept = self.knowledge_channel.teach(
            name,
            utterance,
            plan=plan,
            writer="operator",
            scope="site",
        )
        self.put_claim(
            ClaimRecord(
                claim_id=f"procedure:{concept.name}",
                key=f"procedure:{concept.name}",
                value=concept.as_dict(),
                kind="procedure",
                status="asserted",
                scope="procedure",
                authority="operator",
                source="knowledge_base",
                provenance=dict(provenance or {}),
            )
        )
        return concept

    def get_procedure(self, name: str) -> NamedConcept | None:
        return self.knowledge_channel.recall(name)

    # -- provenance -------------------------------------------------------------

    def record_provenance(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise TypeError("record_provenance requires a dict event")
        stored = dict(event)
        self._provenance.append(stored)
        return stored

    def record_turn_graph(
        self,
        *,
        request_plan: RequestPlan | None = None,
        readiness_graph: ReadinessGraph | None = None,
        reason: str = "request_state",
    ) -> None:
        if request_plan is not None:
            self.memory.update_episodic_memory("last_request_plan", request_plan.as_dict())
        if readiness_graph is not None:
            self.memory.update_episodic_memory("last_readiness_graph", readiness_graph.as_dict())
        self.record_provenance(
            {
                "event": reason,
                "request_id": (
                    readiness_graph.request_id
                    if readiness_graph is not None
                    else request_plan.request_id
                    if request_plan is not None
                    else None
                ),
            }
        )

    # -- existing memory pocket wrappers ---------------------------------------

    def apply_memory_write_ticket(self, ticket: MemoryWriteTicket) -> None:
        if not isinstance(ticket, MemoryWriteTicket):
            raise TypeError("apply_memory_write_ticket requires a MemoryWriteTicket")
        for write in ticket.writes:
            self._apply_memory_update(write, ticket=ticket)

    def _apply_memory_update(
        self,
        update: MemoryUpdate,
        *,
        ticket: MemoryWriteTicket,
    ) -> None:
        if update.scope == "knowledge":
            self.memory.update_knowledge(update.key, update.value)
            self.put_claim(
                ClaimRecord(
                    claim_id=f"operator:{update.key}",
                    key=update.key,
                    value=update.value,
                    kind="operator_assertion",
                    status="asserted",
                    scope="operator",
                    authority="operator",
                    source=ticket.source,
                    provenance={
                        "request_id": ticket.request_id,
                        "reason": update.reason,
                    },
                )
            )
            return
        if update.scope == "episodic_memory":
            self.memory.update_episodic_memory(update.key, update.value)
            self.put_claim(
                ClaimRecord(
                    claim_id=f"episodic:{update.key}",
                    key=update.key,
                    value=update.value,
                    kind="fact",
                    status="confirmed",
                    scope="episodic",
                    authority="runtime",
                    source=ticket.source,
                    provenance={
                        "request_id": ticket.request_id,
                        "reason": update.reason,
                    },
                )
            )
            return
        raise ValueError(f"Unknown memory update scope: {update.scope}")

    def clear_operator_knowledge(self) -> None:
        for key in (
            "target_color",
            "target_type",
            "delivery_target",
            "last_task_type",
            "last_instruction",
        ):
            self.memory.update_knowledge(key, None, persist=(key == "last_instruction"))
            self._claims.pop(key, None)
        for concept in list(self.knowledge_channel.all_concepts()):
            self.knowledge_channel.forget(concept.name)
            self._claims.pop(f"procedure:{concept.name}", None)
        self.record_provenance({"event": "operator_knowledge_cleared"})

    def update_scene_model(self, model: SceneModel) -> None:
        self.memory.update_scene_model(model)

    def clear_scene_model(self) -> None:
        self.memory.scene_model = None

    def scene_model(self) -> SceneModel | None:
        return self.memory.scene_model

    # -- active grounding claims ------------------------------------------------

    def set_active_claims(self, claims: StationActiveClaims) -> None:
        if not isinstance(claims, StationActiveClaims):
            raise TypeError("set_active_claims requires StationActiveClaims")
        self._active_claims = claims
        self.put_claim(
            ClaimRecord(
                claim_id="grounding:active_claims",
                key="active_claims",
                value=claims.compact_summary(),
                kind="observation",
                status="observed",
                scope="grounding",
                authority=claims.authority,
                source=claims.source,
                confidence=claims.confidence,
                provenance={
                    "frame_id": claims.frame_id,
                    "environment_fingerprint": claims.environment_fingerprint,
                },
            )
        )

    def get_active_claims(self) -> StationActiveClaims | None:
        return self._active_claims

    def clear_active_claims(self, *, reason: str = "") -> None:
        self._active_claims = None
        claim = self._claims.get("active_claims")
        if claim is not None:
            claim.status = "invalidated"
            claim.freshness = "stale"
            claim.invalidation = {"reason": reason}

    # -- snapshots --------------------------------------------------------------

    def snapshot(
        self,
        *,
        claims_valid: bool = False,
        environment_identity: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeSnapshot:
        return KnowledgeSnapshot(
            claims=dict(self._claims),
            procedures={
                concept.name: concept.as_dict()
                for concept in self.knowledge_channel.all_concepts()
            },
            provenance=[dict(item) for item in self._provenance],
            active_claims=self._active_claims,
            scene_model=self.memory.scene_model,
            environment_identity=environment_identity,
            claims_valid=claims_valid,
            metadata=dict(metadata or {}),
        )
