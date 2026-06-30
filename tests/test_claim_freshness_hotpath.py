"""Step 1 of the partial-observability fix: freshness on the hot path.

The cortex execution loop stores sensory facts as `ObservationClaim` objects.
Today those claims are timeless — a belief is either present or absent. This step
makes the hot-path claim type *carry* a freshness state and makes the cortex
accessors *respect* it, without yet introducing any decay machine (Step 2) or
spatial map (Step 3).

Freshness semantics on read:
- ``current``      — observed now; usable.
- ``unverifiable`` — believed but not currently confirmable (e.g. out of view);
                     still our best belief, so still usable/actionable.
- ``stale``        — known to be outdated (world changed); NOT usable.
- ``unknown``      — decayed past belief; NOT usable (treated as absent).

This is the step the rest of the plan rests on: once the hot path honours
freshness, an out-of-view spatial belief is just an ``unverifiable`` claim, not a
special case.
"""
from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.schemas import ObservationClaim


def _cortex() -> Cortex:
    return Cortex(
        memory=OperationalMemory(root=Path(tempfile.mkdtemp())),
        compiler=SmokeTestCompiler(),
    )


def test_observation_claim_carries_freshness_defaulting_current():
    field_names = {f.name for f in fields(ObservationClaim)}
    assert "freshness" in field_names

    claim = ObservationClaim(key="target_location", value=(3, 4))
    assert claim.freshness == "current"

    explicit = ObservationClaim(key="k", value=1, freshness="unverifiable")
    assert explicit.freshness == "unverifiable"


def test_set_claim_accepts_and_stores_freshness():
    cortex = _cortex()
    cortex.set_claim("target_location", (3, 4), freshness="unverifiable")
    assert cortex._claims["target_location"].freshness == "unverifiable"


def test_current_and_unverifiable_claims_are_usable():
    cortex = _cortex()

    cortex.set_claim("a", (1, 1))  # default current
    assert cortex.get_claim("a") == (1, 1)
    assert cortex.has_claim("a")

    cortex.set_claim("b", (2, 2), freshness="unverifiable")
    assert cortex.get_claim("b") == (2, 2)
    assert cortex.has_claim("b")


def test_stale_and_unknown_claims_are_treated_as_absent():
    cortex = _cortex()

    cortex.set_claim("gone", (9, 9), freshness="unknown")
    assert cortex.get_claim("gone") is None
    assert not cortex.has_claim("gone")

    cortex.set_claim("outdated", (8, 8), freshness="stale")
    assert cortex.get_claim("outdated") is None
    assert not cortex.has_claim("outdated")


def _seat_cortex_on_navigate_step(cortex: Cortex) -> None:
    """Position the cortex mid-procedure on the navigate_to_object step."""
    cortex.procedure = SimpleNamespace(
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"]
    )
    cortex.execution_state["step_index"] = 1  # navigate_to_object
    cortex.resolved_task_params = {"color": "red", "object_type": "door"}


def test_hot_path_navigates_on_unverifiable_target_belief():
    """An out-of-view (unverifiable) target is still acted upon — we navigate."""
    cortex = _cortex()
    _seat_cortex_on_navigate_step(cortex)
    cortex.set_claim("target_location", (0, 2), freshness="unverifiable")

    contract = cortex.choose_execution_contract()
    assert contract.skill == "navigate_to_object"


def test_hot_path_reacquires_when_target_belief_is_unknown():
    """A decayed (unknown) target is treated as absent — we turn to re-acquire,
    instead of navigating on a dead belief."""
    cortex = _cortex()
    _seat_cortex_on_navigate_step(cortex)
    cortex.set_claim("target_location", (0, 2), freshness="unknown")

    contract = cortex.choose_execution_contract()
    assert contract.skill == "turn_right"
