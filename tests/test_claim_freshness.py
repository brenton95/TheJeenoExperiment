from __future__ import annotations

import pytest

from jeenom.claim_freshness import (
    FRESHNESS_CURRENT,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    FRESHNESS_UNVERIFIABLE,
    UNVERIFIABLE_DECAY_STEPS,
    ClaimTTL,
    framing_satisfiable,
    next_freshness,
)
from jeenom.schemas import CLAIM_FRESHNESS, CLAIM_STATUSES, ClaimRecord, SchemaValidationError


def test_unverifiable_is_freshness_not_status() -> None:
    assert CLAIM_FRESHNESS == ("current", "unverifiable", "stale", "unknown")
    assert "unverifiable" not in CLAIM_STATUSES


def test_observation_freshness_state_machine() -> None:
    assert (
        next_freshness(FRESHNESS_CURRENT, kind="observation", in_view=False)
        == FRESHNESS_UNVERIFIABLE
    )
    assert (
        next_freshness(FRESHNESS_CURRENT, kind="observation", in_view=True)
        == FRESHNESS_CURRENT
    )
    assert (
        next_freshness(FRESHNESS_CURRENT, kind="observation", in_view=True, env_changed=True)
        == FRESHNESS_STALE
    )
    assert (
        next_freshness(FRESHNESS_CURRENT, kind="observation", in_view=True, world_changed=True)
        == FRESHNESS_STALE
    )
    assert (
        next_freshness(
            FRESHNESS_UNVERIFIABLE,
            kind="observation",
            in_view=True,
            steps_unseen=3,
        )
        == FRESHNESS_CURRENT
    )
    assert (
        next_freshness(
            FRESHNESS_UNVERIFIABLE,
            kind="observation",
            in_view=False,
            steps_unseen=UNVERIFIABLE_DECAY_STEPS - 1,
        )
        == FRESHNESS_UNVERIFIABLE
    )
    assert (
        next_freshness(
            FRESHNESS_UNVERIFIABLE,
            kind="observation",
            in_view=False,
            steps_unseen=UNVERIFIABLE_DECAY_STEPS,
        )
        == FRESHNESS_UNKNOWN
    )
    assert (
        next_freshness(FRESHNESS_UNKNOWN, kind="observation", in_view=True)
        == FRESHNESS_UNKNOWN
    )


def test_only_observations_depend_on_view_framing() -> None:
    assert framing_satisfiable(kind="observation", in_view=False) is False
    for kind in ("operator_assertion", "fact", "procedure"):
        assert framing_satisfiable(kind=kind, in_view=False) is True
        assert (
            next_freshness(FRESHNESS_CURRENT, kind=kind, in_view=False)
            != FRESHNESS_UNVERIFIABLE
        )


def test_claim_ttl_validates_decay_inputs() -> None:
    ttl = ClaimTTL.timed(UNVERIFIABLE_DECAY_STEPS)
    assert ttl.expired(steps_unseen=UNVERIFIABLE_DECAY_STEPS)
    assert not ClaimTTL.eternal().expired(steps_unseen=UNVERIFIABLE_DECAY_STEPS * 10)
    with pytest.raises(ValueError):
        ClaimTTL.timed(-1)
    with pytest.raises(ValueError):
        ttl.expired(steps_unseen=-1)


def test_claim_record_valid_until_round_trips() -> None:
    claim = ClaimRecord(
        claim_id="grounding:door",
        key="door",
        value={"x": 1, "y": 2},
        kind="observation",
        status="observed",
        scope="grounding",
        authority="sense",
        source="sense",
        freshness="unverifiable",
        valid_until=12,
    )

    payload = claim.as_dict()
    assert payload["valid_until"] == 12.0
    restored = ClaimRecord.from_dict(payload)
    assert restored.valid_until == 12.0
    assert restored.freshness == "unverifiable"

    payload["valid_until"] = "later"
    with pytest.raises(SchemaValidationError):
        ClaimRecord.from_dict(payload)
