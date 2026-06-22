from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


UNVERIFIABLE_DECAY_STEPS = 16

FRESHNESS_CURRENT = "current"
FRESHNESS_UNVERIFIABLE = "unverifiable"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"

CLAIM_FRESHNESS_VALUES = (
    FRESHNESS_CURRENT,
    FRESHNESS_UNVERIFIABLE,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
)

OBSERVATION_KIND = "observation"
NON_LINE_OF_SIGHT_KINDS = frozenset({"operator_assertion", "fact", "procedure"})


@dataclass(frozen=True)
class ClaimTTL:
    """Small value type for the only timed freshness edge in 13B.

    `valid_until` remains the serialized storage field on ClaimRecord. This helper
    keeps the policy explicit: only an out-of-view observation claim gets a timed
    decay toward unknown; non-line-of-sight claim kinds are eternal here.
    """

    mode: Literal["eternal", "timed", "conditional"]
    n_steps: int | None = None
    predicate: str | None = None

    @classmethod
    def eternal(cls) -> "ClaimTTL":
        return cls(mode="eternal")

    @classmethod
    def timed(cls, n_steps: int) -> "ClaimTTL":
        if (
            isinstance(n_steps, bool)
            or not isinstance(n_steps, int)
            or n_steps < 0
        ):
            raise ValueError("ClaimTTL.timed requires a non-negative step count")
        return cls(mode="timed", n_steps=int(n_steps))

    @classmethod
    def conditional(cls, predicate: str) -> "ClaimTTL":
        if not predicate:
            raise ValueError("ClaimTTL.conditional requires a predicate label")
        return cls(mode="conditional", predicate=predicate)

    def expired(
        self,
        *,
        steps_unseen: int,
        predicate_satisfied: Callable[[str], bool] | None = None,
    ) -> bool:
        if (
            isinstance(steps_unseen, bool)
            or not isinstance(steps_unseen, int)
            or steps_unseen < 0
        ):
            raise ValueError("steps_unseen must be a non-negative integer")
        if self.mode == "eternal":
            return False
        if self.mode == "timed":
            return steps_unseen >= int(self.n_steps or 0)
        if predicate_satisfied is None or self.predicate is None:
            return False
        return bool(predicate_satisfied(self.predicate))


def ttl_for_kind(kind: str, *, freshness: str = FRESHNESS_CURRENT) -> ClaimTTL:
    if kind == OBSERVATION_KIND and freshness == FRESHNESS_UNVERIFIABLE:
        return ClaimTTL.timed(UNVERIFIABLE_DECAY_STEPS)
    return ClaimTTL.eternal()


def framing_satisfiable(*, kind: str, in_view: bool) -> bool:
    """Return whether the claim kind's grounding frame is currently satisfiable.

    Only observation claims have a line-of-sight assumption. Operator assertions,
    facts, and procedures are not made unverifiable just because the agent looks
    away from a cell.
    """

    if kind == OBSERVATION_KIND:
        return bool(in_view)
    return True


def next_freshness(
    current: str,
    *,
    kind: str,
    in_view: bool,
    world_changed: bool = False,
    env_changed: bool = False,
    steps_unseen: int = 0,
) -> str:
    """Evaluate the Phase 13B freshness state machine for one claim.

    The caller is responsible for resolving the geometric `in_view` predicate.
    This function keeps the freshness and status axes separate: it never returns
    or consumes claim statuses such as `hypothesis` or `inferred`.
    """

    if current not in CLAIM_FRESHNESS_VALUES:
        raise ValueError(f"Unknown claim freshness: {current!r}")
    if isinstance(steps_unseen, bool) or steps_unseen < 0:
        raise ValueError("steps_unseen must be a non-negative integer")

    if kind == OBSERVATION_KIND:
        if env_changed or world_changed:
            return FRESHNESS_STALE
        if current == FRESHNESS_CURRENT:
            return (
                FRESHNESS_CURRENT
                if framing_satisfiable(kind=kind, in_view=in_view)
                else FRESHNESS_UNVERIFIABLE
            )
        if current == FRESHNESS_UNVERIFIABLE:
            if framing_satisfiable(kind=kind, in_view=in_view):
                return FRESHNESS_CURRENT
            ttl = ttl_for_kind(kind, freshness=FRESHNESS_UNVERIFIABLE)
            return (
                FRESHNESS_UNKNOWN
                if ttl.expired(steps_unseen=steps_unseen)
                else FRESHNESS_UNVERIFIABLE
            )
        # Unknown is dead until a fresh observation overwrites the claim; stale
        # stays stale until a new grounding writes a replacement.
        return current

    if current in {FRESHNESS_UNKNOWN, FRESHNESS_STALE}:
        return current
    if kind == "fact" and env_changed:
        return FRESHNESS_STALE
    return FRESHNESS_CURRENT
