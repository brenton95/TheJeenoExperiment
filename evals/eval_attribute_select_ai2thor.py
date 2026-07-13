"""Plan 014 (Task 2) — LIVE attribute-select close-out: one-turn state selection.

Proves on LIVE AI2-THOR that "go to the open {type}" navigates to the object whose
substrate-defined STATE matches (isOpen==True), NOT the first description match in
scan order. This is the acceptance test mock-green cannot substitute for: only a live
continuous-3D scene with two description-identical, same-type objects differing ONLY
in isOpen exercises the attribute-carrier path end-to-end (adapter parser packs
target_ref={"attributes":{...}} -> kernel forwards it opaquely (compose_known_task,
352a806) -> sense filters grid_objects by state -> nav).

This is a ONE-TURN flow — no ranking turn. The attribute is known at PARSE TIME, so
_stamp_target_ref never fires (no prior grounding). Do NOT copy the F13 two-turn
structure of eval_specific_target_ai2thor.py.

COLAB ONLY. Uses the verified Linux64 + Xvfb recipe (see .agent/SESSION_STATE.md
"WORKING COLAB RECIPE" and .agent/LIVE_RESULTS_LEDGER.md). Do NOT run native Windows
(no 5.x build) and do NOT use CloudRendering on Colab (Vulkan segfault). Setup cell
BEFORE this:

    import os, subprocess, time
    subprocess.Popen(["Xvfb", ":0", "-screen", "0", "1024x768x24"]); time.sleep(3)
    os.environ["DISPLAY"] = ":0"          # MUST be in-process; !export is a no-op
    !python evals/eval_attribute_select_ai2thor.py --object-type cabinet

TWO live-unknowns this eval refuses to hardcode (per the ledger — untested scene
actions have SILENTLY HUNG before; see RemoveFromScene/CreateObject):
  1. WHICH openable type ships a same-type pair AND is floor-navigable. The
     apple-on-counter/F6 trap means a wall cabinet resolves CORRECTLY but nav fails.
     Discover it live FIRST:
         !python evals/probe_floor_targets_ai2thor.py --live
     Note a type whose row has goals>0 (floor-reachable) and that ships >=2 instances
     exposing isOpen (Cabinet/Drawer usually do in a kitchen). Pass it as --object-type.
  2. WHETHER OpenObject is hang-safe on this software renderer. OpenObject+forceAction
     is far lighter than SliceObject, but was NEVER verified live here. This eval calls
     it with a wall-clock guard note: if the run goes silent for ~server_timeout after
     "force-opening", that is the OpenObject hang — fall back to a type that ships one
     natively-open instance (--no-force), don't fight it.

PASS criteria (attribute-select proven — resolution-first, per plan 014 owner flag #2):
  PRIMARY (load-bearing; sense computes this BEFORE nav, so it holds even if the
  openable is not navigable — decouples the attribute proof from the F6 nav trap):
    - resolved_isOpen: the resolved target's state.isOpen is True
    - resolved_correct_id: resolved object_id == the id we force-opened (LOAD-BEARING).
      The staging guarantees the OPEN target is NOT last in scan order and the last
      same-type object IS closed, so a broken/disabled attribute filter falls to the
      last-wins type-only fallback and resolves the CLOSED distractor — flipping this
      check to FAIL. That is what makes the eval able to fail (see _openable_pair).
    - discriminated: resolved id is NOT a closed sibling's id (implied by
      resolved_correct_id; kept as an explicit cross-check, not the load-bearing one)
  SECONDARY (nav; non-blocking for the thesis, reported separately):
    - task_complete, cache_miss_during_render==0, runtime_llm_calls_during_render==0

  A run with PRIMARY green + SECONDARY nav-fail means: attribute-select WORKS, the
  chosen openable just isn't floor-navigable (F6) — re-pick --object-type via the
  probe. That is NOT an attribute-carrier failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jeenom.ai2thor_substrate_adapter import build_ai2thor_runtime_package
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


def _openable_pair(controller: Any, object_type: str, force: bool) -> dict[str, Any]:
    """Find two native same-type openables and ensure exactly one is open.

    Returns {"open_id": <forced/native-open id>, "closed_ids": [...], "type": <AThorType>}.
    NO scene mutation beyond OpenObject (ledger rule: InitialRandomSpawn/
    PlaceObjectAtPoint/OpenObject only — never RemoveFromScene/CreateObject).

    Strategy: prefer a scene that already ships the type in mixed isOpen states (zero
    mutation, safest). Only if all instances share one state do we OpenObject one
    (guarded by --force; the ledger flags OpenObject as unverified on this renderer).
    """
    athor_type = {"cabinet": "Cabinet", "drawer": "Drawer", "fridge": "Fridge"}.get(
        object_type.lower(), object_type.capitalize()
    )
    controller.step(action="Initialize", renderImage=False)
    objs = [o for o in controller.last_event.metadata["objects"]
            if o.get("objectType") == athor_type and "isOpen" in o]
    if len(objs) < 2:
        raise RuntimeError(
            f"scene ships < 2 openable {athor_type} (found {len(objs)}); pick another "
            f"--object-type via probe_floor_targets_ai2thor.py")

    already_open = [o for o in objs if o.get("isOpen") is True]
    closed = [o for o in objs if o.get("isOpen") is False]

    if already_open and closed:
        # Best case: mixed native states, zero mutation. Open the FIRST open one.
        open_obj = already_open[0]
    else:
        # All same state — must force one open. Force-open the FIRST in scan order.
        # forceAction bypasses reachability so we don't have to navigate to it.
        if not force:
            raise RuntimeError(
                f"all {athor_type} share isOpen state and --force not set; rerun with "
                f"--force to OpenObject one (ledger: OpenObject unverified on this "
                f"renderer — if the run then hangs ~server_timeout, that is the cause)")
        target = objs[0]
        print(f"force-opening {target['objectId']} (OpenObject, forceAction=True) ...",
              flush=True)
        controller.step(action="OpenObject", objectId=target["objectId"],
                        forceAction=True, renderImage=False)
        if not controller.last_event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                f"OpenObject failed: "
                f"{controller.last_event.metadata.get('errorMessage', '?')}")
        # Re-read post-open state (objectId is stable for OpenObject — no re-derivation).
        objs = [o for o in controller.last_event.metadata["objects"]
                if o.get("objectType") == athor_type and "isOpen" in o]
        open_obj = next(o for o in objs if o["objectId"] == target["objectId"])
        closed = [o for o in objs if o.get("isOpen") is False]

    if not closed:
        raise RuntimeError(
            f"no CLOSED {athor_type} sibling remains — the test needs a closed "
            f"distractor to prove attribute (not type) discrimination")
    # THE BITE INVARIANT (learned from the mock, plan 014 STATUS): Ai2thorSense's
    # type-only fallback is LAST-WINS — with the attribute filter broken/disabled it
    # resolves the LAST same-type object in scan order. So the last object MUST be a
    # CLOSED distractor, else a broken filter would still land on an open object and
    # the eval could not fail. Refuse rather than ship a run that cannot discriminate.
    if objs[-1].get("isOpen") is not False:
        raise RuntimeError(
            f"last {athor_type} in scan order (objectId={objs[-1]['objectId']}) is not "
            f"CLOSED — the last-wins fallback would resolve it, so a broken attribute "
            f"filter could not be caught. Pick a --object-type/scene where the matching "
            f"(open) object is NOT last in scan order.")
    if open_obj["objectId"] == objs[-1]["objectId"]:
        raise RuntimeError(
            f"the OPEN target is last in scan order — last-wins fallback would resolve "
            f"it even with the filter disabled; the eval could not fail. Re-stage so the "
            f"open object is not last.")
    return {
        "open_id": open_obj["objectId"],
        "closed_ids": [o["objectId"] for o in closed],
        "type": athor_type,
    }


def _run_one_turn(controller: Any, object_type: str, attribute_word: str) -> dict[str, Any]:
    """One-turn 'go to the {attribute} {type}'. Returns resolution + nav facts."""
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
    session.handle_utterance(f"go to the {attribute_word} {object_type}")
    lr = session.last_result or {}
    fs = lr.get("final_state", {})
    # Resolved target rides in last_world_sample (NOT final_state — ledger rule 2).
    # target_object is the resolved grid_obj; its "state" dict carries isOpen (the
    # attribute the adapter sense populated, ai2thor_sense.py Step 1). No new field.
    sample = lr.get("last_world_sample") or {}
    target_object = sample.get("target_object")
    state = target_object.get("state") if isinstance(target_object, dict) else None
    return {
        "task_complete": fs.get("task_complete", False),
        "cache_miss_during_render": lr.get("cache_miss_during_render", -1),
        "runtime_llm_calls_during_render": lr.get("runtime_llm_calls_during_render", -1),
        "resolved_target_id": target_object.get("object_id")
        if isinstance(target_object, dict) else None,
        "resolved_isOpen": state.get("isOpen") if isinstance(state, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AI2-THOR one-turn attribute-select eval.")
    parser.add_argument("--object-type", default="cabinet",
                        help="openable go-to type (must be in AI2THOR_GO_TO_OBJECT_TYPES "
                             "and ship a same-type pair; discover via probe_floor_targets)")
    parser.add_argument("--attribute", default="open",
                        help="attribute word (declared in AI2THOR_ATTRIBUTE_STATE_FIELDS)")
    parser.add_argument("--force", action="store_true",
                        help="OpenObject one instance if the scene ships none open "
                             "(ledger: OpenObject unverified on this renderer)")
    parser.add_argument("--scene", default="FloorPlan1")
    args = parser.parse_args()

    if os.environ.get("DISPLAY") is None:
        print("REFUSE: DISPLAY unset. Run the Xvfb setup cell first (see docstring).")
        return 2

    import ai2thor.platform  # type: ignore[import-untyped]
    from ai2thor.controller import Controller  # type: ignore[import-untyped]

    controller = Controller(
        scene=args.scene,
        platform=ai2thor.platform.Linux64,
        renderImage=False,
        server_timeout=600,   # slow software-render step (ledger env note)
    )

    pair = _openable_pair(controller, args.object_type, args.force)
    print(f"pair: open={pair['open_id']} closed={pair['closed_ids']} "
          f"type={pair['type']}\n")

    r = _run_one_turn(controller, args.object_type, args.attribute)
    print(f"result: {r}\n")

    checks: dict[str, bool] = {}
    # PRIMARY — attribute-select proven (holds even if nav fails; sense resolves
    # target_object BEFORE the spine navigates).
    checks["resolved_isOpen_true"] = r["resolved_isOpen"] is True
    checks["resolved_correct_id"] = r["resolved_target_id"] == pair["open_id"]
    checks["discriminated"] = (
        r["resolved_target_id"] is not None
        and r["resolved_target_id"] not in pair["closed_ids"]
    )
    # SECONDARY — nav (F6-sensitive; reported but does NOT gate the attribute thesis).
    checks["task_complete"] = r["task_complete"] is True
    checks["cache_miss_zero"] = r["cache_miss_during_render"] == 0
    checks["llm_calls_zero"] = r["runtime_llm_calls_during_render"] == 0

    primary = ["resolved_isOpen_true", "resolved_correct_id", "discriminated"]
    secondary = ["task_complete", "cache_miss_zero", "llm_calls_zero"]

    print("PRIMARY (attribute-select thesis):")
    for name in primary:
        print(f"  {'PASS' if checks[name] else 'FAIL'} {name}")
    print("SECONDARY (nav; non-blocking for the thesis):")
    for name in secondary:
        print(f"  {'PASS' if checks[name] else 'FAIL'} {name}")

    primary_ok = all(checks[n] for n in primary)
    secondary_ok = all(checks[n] for n in secondary)
    n = sum(checks.values())
    print(f"\n{n}/{len(checks)} checks passed  "
          f"(primary={'GREEN' if primary_ok else 'RED'}, "
          f"secondary={'GREEN' if secondary_ok else 'RED'})")

    if primary_ok and not secondary_ok:
        print("\nDIAGNOSTIC: attribute-select WORKS (resolved the open "
              f"{pair['type']} by state, not scan order) but nav failed — the chosen "
              "openable is likely not floor-navigable (F6). Re-pick --object-type via "
              "probe_floor_targets_ai2thor.py. This is NOT an attribute-carrier failure.")
    print("\nNOTE: PRIMARY green = Task 2 attribute-select proven on live AI2-THOR "
          "(one-turn, adapter-only carrier through the reused target_ref bag). "
          "PRIMARY+SECONDARY green = fully closed.")
    # Gate on PRIMARY (the thesis). Secondary nav failure is diagnosable, not a
    # carrier failure — the user re-picks the type rather than the code being wrong.
    return 0 if primary_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
