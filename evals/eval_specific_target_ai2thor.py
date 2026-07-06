"""A4 (plan 013) — F13 live close-out eval: specific-target disambiguation.

Proves on LIVE AI2-THOR that "go to the apple" navigates to the SPECIFIC apple the
kernel chose (nearest / farthest), not the first description match in scan order.
This is the F13 acceptance test that mock-green cannot substitute for: only a live
continuous-3D substrate with two description-identical, colourless objects exercises
the identity path end-to-end (adapter object_id -> kernel target_ref stamp -> sense
resolution -> nav).

COLAB ONLY. Uses the verified Linux64 + Xvfb recipe (see .agent/SESSION_STATE.md
"WORKING COLAB RECIPE"). Do NOT run native Windows (no 5.x build) and do NOT use
CloudRendering on Colab (Vulkan segfault). Setup in a Colab cell BEFORE this:

    import os, subprocess, time
    subprocess.Popen(["Xvfb", ":0", "-screen", "0", "1024x768x24"]); time.sleep(3)
    os.environ["DISPLAY"] = ":0"          # MUST be in-process; !export is a no-op
    !python evals/eval_specific_target_ai2thor.py

What it does:
  1. Spawns FloorPlan1 (Linux64), InitialRandomSpawn's TWO identical apples, then
     PlaceObjectAtPoint's them onto two distinct GetReachablePositions cells at
     clearly different distances from the agent (near cell A, far cell B).
     (RemoveFromScene/CreateObject hang the backend on Colab software render — see
     _place_two_apples; only PlaceObjectAtPoint + InitialRandomSpawn are used.)
  2. Two-turn flow through the REAL station:
       turn 1: "which apple is closest?"   -> grounding ranks, stamps last_grounded_target
       turn 2: "go to the apple"           -> _stamp_target_ref -> sense resolves by object_id
     Then repeats with "which apple is farthest?" to prove the OTHER apple wins.
  3. Asserts the kernel resolved to the CHOSEN apple by object_id — read from the
     resolved target_object.object_id (which carries AI2-THOR's native objectId),
     surfaced via result["last_world_sample"]. No new field, no matched_by flag:
     the resolved object_id already rides the channel the kernel produces.

PASS criteria (F13 CLOSED):
  - nearest run: resolved_target_id == near apple's object_id (the "closest" one)
  - farthest run: resolved_target_id == far apple's object_id (the "farthest" one)
  - discriminated: the two queries resolve to DIFFERENT apples
  - all runs: runtime_llm_calls_during_render == 0, cache_miss_during_render == 0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


def _place_two_apples(controller: Any) -> dict[str, dict[str, Any]]:
    """Spawn two identical apples on reachable cells at different distances.

    Returns {"near": {"cell": (x,z), "object_id": id}, "far": {...}}. y is
    irrelevant to floor nav; (x,z) is what matters (see SESSION_STATE
    apple-on-counter finding). object_id is re-read AFTER PlaceObjectAtPoint
    because AI2-THOR re-derives objectId from the placed position — the id at
    spawn is not the id after placement. The eval asserts the kernel resolves to
    THIS specific id per run (near for "closest", far for "farthest"), which is
    the F13 close-out: description-identical apples, identity the sole discriminator.

    Uses ONLY InitialRandomSpawn + PlaceObjectAtPoint — both verified live on the
    Colab Linux64+Xvfb software renderer (2026-07-06). The prior RemoveFromScene +
    CreateObject path HUNG the backend for the full server_timeout on Colab
    (physics-heavy scene mutation under software render); those actions were never
    exercised by the green 012 mission run, which only ever PlaceObjectAtPoint'd.
    Two apples of the same type are description-identical, so object_id stays the
    sole discriminator (the F13 acceptance condition).
    """
    controller.step(action="Initialize", renderImage=False)
    reach = controller.step(action="GetReachablePositions", renderImage=False)
    positions = reach.metadata["actionReturn"]
    agent = controller.last_event.metadata["agent"]["position"]

    def dist(p: dict[str, float]) -> float:
        return ((p["x"] - agent["x"]) ** 2 + (p["z"] - agent["z"]) ** 2) ** 0.5

    ranked = sorted(positions, key=dist)
    near, far = ranked[1], ranked[-1]  # skip the agent's own cell at index 0

    # Ensure exactly two Apples exist (FloorPlan1 ships one native). This is the
    # non-hanging duplicate route: no RemoveFromScene / CreateObject.
    controller.step(
        action="InitialRandomSpawn", randomSeed=42, forceVisible=True,
        numDuplicatesOfType=[{"objectType": "Apple", "count": 2}],
        renderImage=False,
    )
    apple_ids = [o["objectId"] for o in controller.last_event.metadata["objects"]
                 if o["objectType"] == "Apple"]
    if len(apple_ids) < 2:
        raise RuntimeError(
            f"expected >=2 apples after InitialRandomSpawn, got {len(apple_ids)}")

    placed: dict[str, dict[str, Any]] = {}
    for (label, cell), apple_id in zip((("near", near), ("far", far)), apple_ids):
        controller.step(
            action="PlaceObjectAtPoint", objectId=apple_id,
            position={"x": cell["x"], "y": cell["y"] + 0.05, "z": cell["z"]},
            renderImage=False,
        )
        # Re-read the id after placement: AI2-THOR re-derives objectId from position,
        # so match the apple now nearest this cell to recover its post-placement id.
        apples_now = [o for o in controller.last_event.metadata["objects"]
                      if o["objectType"] == "Apple"]
        placed_id = min(
            apples_now,
            key=lambda o: (o["position"]["x"] - cell["x"]) ** 2
            + (o["position"]["z"] - cell["z"]) ** 2,
        )["objectId"]
        placed[label] = {"cell": (cell["x"], cell["z"]), "object_id": placed_id}
    return placed


def _run_turn_pair(controller: Any, rank_utterance: str) -> dict[str, Any]:
    """turn 1 = ranking query, turn 2 = 'go to the apple'. Returns nav + identity."""
    rp = build_ai2thor_runtime_package(controller=controller)
    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke_test",
        env_id="FloorPlan1",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        runtime_package=rp,
    )
    session.handle_utterance(rank_utterance)             # turn 1: stamps grounding claim
    session.handle_utterance("go to the apple")          # turn 2: F13 identity path
    lr = session.last_result or {}
    fs = lr.get("final_state", {})
    # The resolved target + agent pose ride in last_world_sample (the adapter already
    # surfaces WorldModelSample.summary() at result["last_world_sample"]); they are NOT
    # in final_state (which is dict(cortex.execution_state)). target_object is the
    # resolved grid_obj carrying the native objectId — no new field, no matched_by flag.
    sample = lr.get("last_world_sample") or {}
    target_object = sample.get("target_object")
    # No agent_pose check: task_complete already requires the spine's adjacency
    # postcondition (agent reached the target), so asserting pose here would be
    # redundant. The load-bearing proof is resolved_target_id == the expected apple.
    return {
        "task_complete": fs.get("task_complete", False),
        "cache_miss_during_render": lr.get("cache_miss_during_render", -1),
        "runtime_llm_calls_during_render": lr.get("runtime_llm_calls_during_render", -1),
        "resolved_target_id": target_object.get("object_id")
        if isinstance(target_object, dict) else None,
    }


def main() -> int:
    if os.environ.get("DISPLAY") is None:
        print("REFUSE: DISPLAY unset. Run the Xvfb setup cell first (see docstring).")
        return 2
    import ai2thor.platform  # type: ignore[import-untyped]
    from ai2thor.controller import Controller  # type: ignore[import-untyped]

    controller = Controller(
        scene="FloorPlan1",
        platform=ai2thor.platform.Linux64,
        renderImage=False,
        server_timeout=600,   # slow software-render step (SESSION_STATE TimeoutError fix)
    )

    placed = _place_two_apples(controller)
    print(f"placed apples: near={placed['near']} far={placed['far']}\n")

    # Which apple id the kernel MUST resolve to per ranking query. "closest?" ->
    # the near apple; "farthest?" -> the far apple. Different expected ids from the
    # SAME "go to the apple" instruction is the whole F13 proof.
    expected_id = {"nearest": placed["near"]["object_id"],
                   "farthest": placed["far"]["object_id"]}
    results = {
        "nearest": _run_turn_pair(controller, "which apple is closest?"),
        "farthest": _run_turn_pair(controller, "which apple is farthest?"),
    }

    checks: dict[str, bool] = {}
    for which, r in results.items():
        print(f"--- {which} --- (expected id: {expected_id[which]})\n{r}\n")
        checks[f"{which}_task_complete"] = r["task_complete"] is True
        checks[f"{which}_cache_miss_zero"] = r["cache_miss_during_render"] == 0
        checks[f"{which}_llm_calls_zero"] = r["runtime_llm_calls_during_render"] == 0
        # The load-bearing F13 check: resolved to the CORRECT apple by object_id, not
        # merely to *an* apple. non-None alone would pass even on scan-order fallback.
        checks[f"{which}_resolved_correct_id"] = (
            r["resolved_target_id"] == expected_id[which]
        )

    # Behavioural cross-check: the two runs must NOT resolve to the same apple.
    checks["discriminated"] = (
        results["nearest"]["resolved_target_id"] is not None
        and results["nearest"]["resolved_target_id"]
        != results["farthest"]["resolved_target_id"]
    )

    # Diagnostic: if 'discriminated' PASSES but both 'resolved_correct_id' FAIL, the
    # kernel IS discriminating (two different apples) but the stored expected_id drifted
    # from the id seen during nav — AI2-THOR re-derives position-based objectIds and
    # physics may have re-settled the apple after placement. That is an eval-harness id
    # issue, NOT a dead wire. Distinguish it explicitly so the live run isn't misread.
    if (checks.get("discriminated")
            and not checks.get("nearest_resolved_correct_id")
            and not checks.get("farthest_resolved_correct_id")):
        print("DIAGNOSTIC: discrimination works but expected ids drifted (objectId "
              "re-derivation / physics resettle) — F13 wire likely OK; fix expected-id "
              "capture, not the adapter.\n")

    print("CHECKS")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    n = sum(checks.values())
    print(f"\n{n}/{len(checks)} passed")
    print("\nNOTE: resolved_correct_id proves identity match (near apple for 'closest', "
          "far for 'farthest'); 'discriminated' proves the two queries reach DIFFERENT "
          "apples. Both green = F13 closed on live AI2-THOR.")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
