"""Phase 13B: claim freshness under partial observability.

Spike-first eval-first artifact. Asserts the freshness contract designed in
task_plan.md Phase 13B, entirely SYNTHETICALLY (constructed claims/poses; no
FullyObsWrapper change). It started in the `expected_fail` suite and graduated into the
main suite when the 13B freshness kernel landed.

Target contract:
  - schemas.CLAIM_FRESHNESS gains "unverifiable" -> current | unverifiable | stale | unknown
  - jeenom/claim_freshness.py exposes:
      UNVERIFIABLE_DECAY_STEPS: int
      framing_satisfiable(*, kind, in_view) -> bool   # only observation consults in_view
      next_freshness(current, *, kind, in_view, world_changed, env_changed, steps_unseen) -> str

The freshness STATE MACHINE (given an already-resolved `in_view`) is what this probe
pins. The FOV geometry that COMPUTES in_view from pose+cell is separate 13B work and is
deliberately not asserted here.
"""
from __future__ import annotations

from typing import Any

from harness import emit_result

METRIC_KEYS = (
    "freshness_enum_has_unverifiable",
    "unverifiable_is_freshness_not_status",
    "lookaway_drops_observation_to_unverifiable",
    "env_change_is_stale_separate_from_world_mutation",
    "nonspatial_kinds_never_unverifiable_on_lookaway",
    "unverifiable_snaps_back_to_current_for_free",
    "unknown_does_not_snap_back_requires_fresh_observation",
    "decay_to_unknown_parameterized_on_constant",
    "eternal_kinds_never_timed_decay",
)
_NONSPATIAL_KINDS = ("operator_assertion", "fact", "procedure")


def _run(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    from jeenom.schemas import CLAIM_FRESHNESS, CLAIM_STATUSES

    # --- The one axis change: a new freshness value, not a new status. ---
    metrics["freshness_enum_has_unverifiable"] = "unverifiable" in CLAIM_FRESHNESS
    metrics["unverifiable_is_freshness_not_status"] = (
        "unverifiable" in CLAIM_FRESHNESS and "unverifiable" not in CLAIM_STATUSES
    )
    details["claim_freshness"] = list(CLAIM_FRESHNESS)

    # --- The freshness state machine (synthetic; in_view already resolved). ---
    from jeenom.claim_freshness import (
        UNVERIFIABLE_DECAY_STEPS,
        framing_satisfiable,
        next_freshness,
    )

    details["decay_steps"] = UNVERIFIABLE_DECAY_STEPS

    def step(current, *, kind, in_view, world_changed=False, env_changed=False, steps_unseen=0):
        return next_freshness(
            current,
            kind=kind,
            in_view=in_view,
            world_changed=world_changed,
            env_changed=env_changed,
            steps_unseen=steps_unseen,
        )

    # Look-away on an observation claim, world unchanged -> unverifiable (NOT stale).
    metrics["lookaway_drops_observation_to_unverifiable"] = (
        step("current", kind="observation", in_view=False) == "unverifiable"
    )

    # env-identity change -> stale; world-mutation is a SEPARATE input that also -> stale
    # (carry: kept distinct even though static GoToDoor collapses them today).
    metrics["env_change_is_stale_separate_from_world_mutation"] = (
        step("current", kind="observation", in_view=True, env_changed=True) == "stale"
        and step("current", kind="observation", in_view=True, world_changed=True) == "stale"
    )

    # The table guard: non-spatial kinds have no in-view assumption and never go
    # unverifiable on look-away (a blanket "looked away -> unverifiable" rule would be a bug).
    metrics["nonspatial_kinds_never_unverifiable_on_lookaway"] = all(
        framing_satisfiable(kind=k, in_view=False) is True
        and step("current", kind=k, in_view=False) != "unverifiable"
        for k in _NONSPATIAL_KINDS
    ) and framing_satisfiable(kind="observation", in_view=False) is False

    # Free snap-back: unverifiable -> current when the cell re-enters view, same world.
    metrics["unverifiable_snaps_back_to_current_for_free"] = (
        step("unverifiable", kind="observation", in_view=True, steps_unseen=3) == "current"
    )

    # Dead claim: unknown does NOT snap back from a passive transition — re-grounding
    # (a fresh observation that overwrites) is the only way back, never restoration.
    metrics["unknown_does_not_snap_back_requires_fresh_observation"] = (
        step("unknown", kind="observation", in_view=True, steps_unseen=0) != "current"
    )

    # Decay asserted parameterized on the imported constant, never the literal.
    metrics["decay_to_unknown_parameterized_on_constant"] = (
        step("unverifiable", kind="observation", in_view=False,
             steps_unseen=UNVERIFIABLE_DECAY_STEPS) == "unknown"
        and step("unverifiable", kind="observation", in_view=False,
                 steps_unseen=UNVERIFIABLE_DECAY_STEPS - 1) == "unverifiable"
    )

    # Eternal kinds never decay via the timer no matter how long unseen.
    metrics["eternal_kinds_never_timed_decay"] = all(
        step("current", kind=k, in_view=False,
             steps_unseen=UNVERIFIABLE_DECAY_STEPS * 4) != "unknown"
        for k in _NONSPATIAL_KINDS
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        _run(metrics, details)
    except Exception as exc:  # pragma: no cover - red-bar until 13B lands
        details["error"] = f"{type(exc).__name__}: {exc}"
    for key in METRIC_KEYS:
        metrics.setdefault(key, False)
    metrics["unverifiable_freshness_holds"] = all(metrics[key] for key in METRIC_KEYS)
    return emit_result(metrics, details, pass_metric="unverifiable_freshness_holds")


if __name__ == "__main__":
    raise SystemExit(main())
