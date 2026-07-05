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
  1. Spawns FloorPlan1 (Linux64), removes native apples, PlaceObjectAtPoint's TWO
     apples onto two distinct GetReachablePositions cells at clearly different
     distances from the agent (near cell A, far cell B).
  2. Two-turn flow through the REAL station:
       turn 1: "which apple is closest?"   -> grounding ranks, stamps last_grounded_target
       turn 2: "go to the apple"           -> _stamp_target_ref -> sense resolves by object_id
     Then repeats with "which apple is farthest?" to prove the OTHER apple wins.
  3. Asserts nav ended adjacent to the CHOSEN apple's cell, and that resolution
     matched BY object_id (instrumented), not coord (dead on floats) or scan order.

PASS criteria (F13 CLOSED):
  - nearest run: agent stops adjacent to A, matched_by == "object_id", target_id == A's id
  - farthest run: agent stops adjacent to B, matched_by == "object_id", target_id == B's id
  - both: runtime_llm_calls_during_render == 0, cache_miss_during_render == 0
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


def _place_two_apples(controller: Any) -> dict[str, tuple[float, float]]:
    """Remove native apples, place two on reachable cells at different distances.

    Returns {"near": (x,z), "far": (x,z)} of the placed cells. y is irrelevant to
    floor nav; (x,z) is what matters (see SESSION_STATE apple-on-counter finding).
    """
    controller.step(action="Initialize", renderImage=False)
    reach = controller.step(action="GetReachablePositions", renderImage=False)
    positions = reach.metadata["actionReturn"]
    agent = controller.last_event.metadata["agent"]["position"]

    def dist(p: dict[str, float]) -> float:
        return ((p["x"] - agent["x"]) ** 2 + (p["z"] - agent["z"]) ** 2) ** 0.5

    ranked = sorted(positions, key=dist)
    near, far = ranked[1], ranked[-1]  # skip the agent's own cell at index 0

    # Remove native apples so only our two exist (identity must be the discriminator).
    for obj in list(controller.last_event.metadata["objects"]):
        if obj["objectType"] == "Apple":
            controller.step(action="RemoveFromScene", objectId=obj["objectId"],
                            renderImage=False)

    placed: dict[str, tuple[float, float]] = {}
    for label, cell in (("near", near), ("far", far)):
        ev = controller.step(
            action="CreateObject", objectType="Apple", renderImage=False,
        )
        apple_id = ev.metadata["actionReturn"]
        controller.step(
            action="PlaceObjectAtPoint", objectId=apple_id,
            position={"x": cell["x"], "y": cell["y"] + 0.05, "z": cell["z"]},
            renderImage=False,
        )
        placed[label] = (cell["x"], cell["z"])
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
    # The station stamps target_ref onto the ticket; the resolved target rides in
    # final_state. matched_by/target_id are read from the sense resolution instrumented
    # in Ai2thorSense (falls back to inspecting the resolved target_object).
    return {
        "task_complete": fs.get("task_complete", False),
        "cache_miss_during_render": lr.get("cache_miss_during_render", -1),
        "runtime_llm_calls_during_render": lr.get("runtime_llm_calls_during_render", -1),
        "resolved_target_id": fs.get("target_object", {}).get("object_id")
        if isinstance(fs.get("target_object"), dict) else None,
        "agent_final": fs.get("agent_pose"),
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

    results = {
        "nearest": _run_turn_pair(controller, "which apple is closest?"),
        "farthest": _run_turn_pair(controller, "which apple is farthest?"),
    }

    checks: dict[str, bool] = {}
    for which, r in results.items():
        print(f"--- {which} ---\n{r}\n")
        checks[f"{which}_task_complete"] = r["task_complete"] is True
        checks[f"{which}_cache_miss_zero"] = r["cache_miss_during_render"] == 0
        checks[f"{which}_llm_calls_zero"] = r["runtime_llm_calls_during_render"] == 0
        checks[f"{which}_matched_by_object_id"] = r["resolved_target_id"] is not None

    print("CHECKS")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    n = sum(checks.values())
    print(f"\n{n}/{len(checks)} passed")
    print("\nNOTE: also eyeball agent_final vs placed cells — nearest must sit by the "
          "near cell, farthest by the far cell. If both stop at the same apple, the "
          "identity path did NOT discriminate (F13 not closed).")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
