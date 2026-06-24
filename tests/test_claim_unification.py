"""Claim unification: one claim type (ClaimRecord), one mission-scoped store.

This pins the systemic merge of the two claim stores onto a single
``OperationalMemory``-owned ``dict[str, ClaimRecord]``:

- (a) the cortex hot-path store holds ``ClaimRecord`` objects authored as
  ``kind="observation"``;
- (b) belief is mission-scoped — it survives a fresh ``Cortex`` over the same
  ``memory`` (as a per-task ``run_episode`` would build) and is cleared only on
  the typed-reset path (``clear_reference_context=True``);
- (c) **safety gate** — durable claims (operator_assertion / procedure / fact)
  sharing the store must NOT decay when the cortex loop ages observations;
- (d) one store — a cortex write is visible through the same dict the
  ``RepresentationStore`` reads, and vice versa.

If (c) ever goes red, the merge is unsafe and must not ship regardless of the
rest of the suite (durable operator truth would silently rot).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from jeenom.claim_freshness import UNVERIFIABLE_DECAY_STEPS
from jeenom.cortex import Cortex
from jeenom.knowledge_base import KnowledgeBase
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.representation import RepresentationStore
from jeenom.schemas import ClaimRecord, OperationalEvidence, TaskRequest
from jeenom.turn_orchestrator import KnowledgeChannel


def _memory() -> OperationalMemory:
    return OperationalMemory(root=Path(tempfile.mkdtemp()))


def _cortex(memory: OperationalMemory) -> Cortex:
    return Cortex(memory=memory, compiler=SmokeTestCompiler())


def _representation(memory: OperationalMemory) -> RepresentationStore:
    return RepresentationStore(
        memory=memory,
        knowledge_channel=KnowledgeChannel(KnowledgeBase(storage_path=None)),
    )


def _ev(claims: dict) -> OperationalEvidence:
    return OperationalEvidence(claims=claims, confidence=1.0, source="sense")


def _ws(step_count: int) -> SimpleNamespace:
    return SimpleNamespace(step_count=step_count)


def _durable(key: str, kind: str, scope: str, authority: str) -> ClaimRecord:
    return ClaimRecord(
        claim_id=f"{kind}:{key}",
        key=key,
        value=True,
        kind=kind,
        status="asserted" if kind != "fact" else "confirmed",
        scope=scope,
        authority=authority,
        source="test",
    )


# ── (a) cortex store holds ClaimRecord(kind=observation) ──────────────────────


def test_cortex_store_holds_claim_record_observations():
    memory = _memory()
    cortex = _cortex(memory)
    cortex.set_claim("target_location", (3, 4))

    claim = memory.claims["target_location"]
    assert isinstance(claim, ClaimRecord)
    assert claim.kind == "observation"
    assert claim.scope == "grounding"
    assert claim.authority == "sense"
    assert claim.status == "observed"
    assert claim.claim_id == "observation:target_location"
    assert claim.value == (3, 4)
    # raw-value accessor unchanged
    assert cortex.get_claim("target_location") == (3, 4)


def test_cortex_private_claims_view_is_the_memory_store():
    memory = _memory()
    cortex = _cortex(memory)
    # The legacy ``cortex._claims`` name must keep working and BE the shared dict.
    assert cortex._claims is memory.claims


# ── (b) mission-scoped: survives a fresh Cortex, cleared only by typed reset ───


def test_belief_survives_a_fresh_cortex_over_same_memory():
    memory = _memory()
    first = _cortex(memory)
    first.set_claim("target_location", (5, 6))

    # A new task run builds a fresh Cortex (as run_episode does) on the same memory.
    second = _cortex(memory)
    assert second.get_claim("target_location") == (5, 6)


def test_task_admission_reset_does_not_clear_belief():
    memory = _memory()
    cortex = _cortex(memory)
    cortex.set_claim("target_location", (5, 6))

    # onboard_task uses clear_reference_context=False — belief must persist.
    memory.reset_episode(clear_reference_context=False)
    assert cortex.get_claim("target_location") == (5, 6)


def test_typed_reset_clears_belief():
    memory = _memory()
    cortex = _cortex(memory)
    cortex.set_claim("target_location", (5, 6))

    # Typed reset (the episode boundary) clears the mission store.
    memory.reset_episode(clear_reference_context=True)
    assert cortex.get_claim("target_location") is None
    assert memory.claims == {}


# ── (c) SAFETY GATE: durable claims must not decay in the shared store ─────────


def test_durable_claims_do_not_decay_when_observations_age():
    memory = _memory()
    cortex = _cortex(memory)

    durable = {
        "op_fact": _durable("op_fact", "operator_assertion", "operator", "operator"),
        "proc": _durable("proc", "procedure", "procedure", "operator"),
        "known_fact": _durable("known_fact", "fact", "episodic", "runtime"),
    }
    for claim in durable.values():
        memory.claims[claim.key] = claim

    # An observation is seen once, then the loop runs well past the decay TTL
    # without re-observing anything (current -> unverifiable -> unknown).
    cortex.update_from_evidence(_ev({"target_location": (0, 2)}), world_sample=_ws(0))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(1))
    cortex.update_from_evidence(_ev({}), world_sample=_ws(UNVERIFIABLE_DECAY_STEPS + 1))

    # The observation decayed...
    assert memory.claims["target_location"].freshness == "unknown"
    # ...but every durable claim stays current and present. This is the property
    # the whole merge rests on.
    for key in durable:
        assert memory.claims[key].freshness == "current"
        assert cortex.get_claim(key) is True


# ── (d) one store: cortex and representation share the same dict ──────────────


def test_cortex_write_is_visible_through_representation():
    memory = _memory()
    cortex = _cortex(memory)
    representation = _representation(memory)

    cortex.set_claim("target_location", (7, 8))
    seen = representation.get_claim("target_location")
    assert isinstance(seen, ClaimRecord)
    assert seen.value == (7, 8)


def test_representation_write_is_visible_through_cortex():
    memory = _memory()
    cortex = _cortex(memory)
    representation = _representation(memory)

    representation.put_claim(
        ClaimRecord(
            claim_id="fact:door_count",
            key="door_count",
            value=4,
            kind="fact",
            status="confirmed",
            scope="grounding",
            authority="runtime",
            source="test",
        )
    )
    assert cortex.get_claim("door_count") == 4


# ── grounding observations stale when the target context changes ──────────────


def _procedure():
    return SimpleNamespace(
        task_type="go_to_object",
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"],
    )


def _task(color: str, object_type: str = "door") -> TaskRequest:
    return TaskRequest(
        instruction=f"go to the {color} {object_type}",
        task_type="go_to_object",
        params={"color": color, "object_type": object_type},
        source="test",
    )


def test_grounding_observation_staled_on_target_context_change():
    memory = _memory()
    cortex = _cortex(memory)
    cortex.onboard_task(_task("red"), _procedure())
    cortex.set_claim("target_location", (2, 2))
    assert cortex.get_claim("target_location") == (2, 2)

    # A new task with a different target: prior grounding belief is no longer
    # valid for this grounding frame and must not contaminate the new target.
    cortex.onboard_task(_task("blue"), _procedure())
    assert cortex.get_claim("target_location") is None


def test_grounding_observation_persists_on_same_target_context():
    memory = _memory()
    cortex = _cortex(memory)
    cortex.onboard_task(_task("red"), _procedure())
    cortex.set_claim("target_location", (2, 2))

    # Same target (e.g. "repeat the last task"): belief survives the re-admission.
    cortex.onboard_task(_task("red"), _procedure())
    assert cortex.get_claim("target_location") == (2, 2)


def test_durable_claims_survive_target_context_change():
    """Context-change staling touches only grounding observations, not durable claims."""
    memory = _memory()
    cortex = _cortex(memory)
    representation = _representation(memory)
    cortex.onboard_task(_task("red"), _procedure())

    representation.put_claim(
        ClaimRecord(
            claim_id="operator:delivery_target",
            key="delivery_target",
            value={"color": "green"},
            kind="operator_assertion",
            status="asserted",
            scope="operator",
            authority="operator",
            source="test",
        )
    )
    cortex.onboard_task(_task("blue"), _procedure())
    assert cortex.get_claim("delivery_target") == {"color": "green"}
