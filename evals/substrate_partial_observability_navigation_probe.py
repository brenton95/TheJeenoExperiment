"""End-to-end navigation under partial observability.

Sibling to `substrate_partial_observability_needs_evidence_probe.py`, but for a
*different* meaning of partial observability. That probe checks the query/answer
path stays epistemically honest ("don't answer what you can't see"). This one
checks the *task-execution* path: once the agent has located a target, it must
keep its route to that target even as the target leaves the field of view while
navigating.

Symptom this guards against (observed in a live GoToDoor run): the agent locates
a door, starts walking, turns, the door leaves the FOV, and the path planner —
which only ever sees the *current* field of view — loses the route and reports
`no_path_found`, so the agent spins in place / exhausts its step budget instead
of walking to the door it just saw.

This is NOT a target-memory bug: `known_target_location` is remembered. The
defect is that the occupancy / passability map is rebuilt from the current FOV
every tick (jeenom/sense.py:_build_occupancy_grid) instead of accumulating
observed free space, which is inconsistent with the claim_freshness belief-state
policy (out-of-view observations persist and decay, they are not deleted).

Deterministic repro (smoke compiler, no LLM): MiniGrid-GoToDoor-16x16-v0, seed 2,
"go to the blue door". The agent locates the blue door, then navigation reports
no_path_found and the task never completes.
"""
from __future__ import annotations

from typing import Any

from harness import make_session


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"


def _run_navigation(seed: int, color: str, *, max_loops: int = 80) -> dict[str, Any]:
    session = make_session(
        env_id=ENV_ID,
        seed=seed,
        observability="partial",
        max_loops=max_loops,
    )
    session.handle_utterance(f"go to the {color} door")

    result = session.last_result or {}
    loop_records = result.get("loop_records", [])
    final_state = result.get("final_state", {})

    located = False
    target_location: tuple[int, int] | None = None
    no_path_after_located = 0
    failure_reasons: list[dict[str, Any]] = []

    for record in loop_records:
        evidence = record.get("operational_evidence") or {}
        if evidence.get("target_location") is not None:
            located = True
            target_location = tuple(evidence["target_location"])

        report = record.get("report") or {}
        reason = report.get("reason")
        if reason:
            skill = (record.get("contract") or {}).get("skill")
            failure_reasons.append({"loop": record.get("loop"), "skill": skill, "reason": reason})
            if located and reason == "no_path_found":
                no_path_after_located += 1

    return {
        "task_complete": bool(final_state.get("task_complete")),
        "located": located,
        "target_location": target_location,
        "loops": len(loop_records),
        "max_loops": max_loops,
        "no_path_after_located": no_path_after_located,
        "failure_reasons": failure_reasons,
    }


def _navigation_keeps_route_to_located_target(
    metrics: dict[str, bool], details: dict[str, Any]
) -> None:
    run = _run_navigation(seed=2, color="blue")
    details["navigate_blue_door_seed2"] = run

    # Precondition: the agent does see and locate the door (otherwise the test
    # would be vacuous — it could "pass" simply by never finding the target).
    metrics["target_is_located"] = run["located"]

    # The route to an already-located target must never be lost. A no_path_found
    # *after* locating the target is the precise signature of the discarded map.
    metrics["route_to_located_target_never_lost"] = run["no_path_after_located"] == 0

    # The user-visible capability: the agent actually reaches the door it saw,
    # within a sane step budget (a spinning agent exhausts the budget instead).
    metrics["navigation_completes_within_budget"] = (
        run["task_complete"] and run["loops"] < run["max_loops"]
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    _navigation_keeps_route_to_located_target(metrics, details)
    metrics["partial_observability_navigation_holds"] = all(metrics.values())

    from harness import emit_result

    return emit_result(
        metrics, details, pass_metric="partial_observability_navigation_holds"
    )


if __name__ == "__main__":
    raise SystemExit(main())
