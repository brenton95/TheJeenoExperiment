"""F13 spike — thread the chosen target's IDENTITY from Cortex to Sense.

Pressure: when two objects are indistinguishable by description (same type, same
colour — e.g. two apples, or two red doors), the kernel ranks them and picks one
(say the nearest), then hands Sense only a *re-description* ("the apple"). Sense
re-grounds by description and returns the FIRST match in scan order, which need not
be the one the kernel chose. On MiniGrid GoToDoor every object has a unique colour,
so the re-description round-trips losslessly and this never surfaced.

Fix under test: every object carries an adapter-minted ``object_id`` — an opaque
disambiguation handle the kernel never interprets. MiniGrid mints its own (position
is its only stable identity); a substrate like AI2-THOR supplies native object ids.
The kernel stamps the chosen object's id into ``target_ref`` and threads it through
task params into the EvidenceFrame context; Sense matches id-to-id, not by
attributes. Identity and attributes are separated: attributes ("closest", "at this
point") are *selection* queries that resolve to an id; the id is what execution
carries.

Red bar: with two description-identical objects, Sense must ground the one the
kernel designated by ``object_id`` — not whichever it scans first.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jeenom.minigrid_operational_context  # noqa: F401  (registers MiniGrid index maps)
from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import Observation
from jeenom.schemas import PrimitiveCall
from jeenom.sense import MiniGridSense


GRID = (6, 6)
DOOR_A = (1, 1)
DOOR_B = (4, 1)  # same type + colour as DOOR_A: indistinguishable by description
AGENT_XY = (3, 3)
_EMPTY, _DOOR, _YELLOW, _DOOR_OPEN = 1, 4, 4, 0


def _image() -> np.ndarray:
    img = np.zeros((GRID[0], GRID[1], 3), dtype=int)
    img[:, :, 0] = _EMPTY
    for (dx, dy) in (DOOR_A, DOOR_B):
        img[dx, dy] = [_DOOR, _YELLOW, _DOOR_OPEN]
    return img


def _obs(*, step_count: int = 0) -> Observation:
    visible = [AGENT_XY, DOOR_A, DOOR_B, (2, 2), (3, 2)]
    raw = {
        "image": _image(),
        "mission": "go to the yellow door",
        "direction": 0,
        "_jeenom_grid_size": GRID,
        "_jeenom_agent_pos": AGENT_XY,
        "_jeenom_agent_dir": 0,
        "_jeenom_observation_model": "agent_fov",
        "_jeenom_visible_cells": [
            {"view_x": x, "view_y": y, "x": x, "y": y} for (x, y) in visible
        ],
    }
    return Observation(raw=raw, step_count=step_count)


def _sense_plan(target_ref: dict | None) -> list[PrimitiveCall]:
    find_params: dict = {"color": "yellow", "object_type": "door"}
    if target_ref is not None:
        find_params["target_ref"] = target_ref
    return [
        PrimitiveCall(name="parse_grid_objects"),
        PrimitiveCall(name="get_agent_pose"),
        PrimitiveCall(name="find_object_by_color_type", params=find_params),
        PrimitiveCall(name="build_occupancy_grid"),
        PrimitiveCall(name="check_adjacency", params={"object_type": "door"}),
    ]


def _sense() -> MiniGridSense:
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    return MiniGridSense(memory, SmokeTestCompiler())


def _object_id_at(sense: MiniGridSense, coord: tuple[int, int]) -> str:
    """Discover the adapter-minted object_id for the object at a coordinate.

    The id scheme is the adapter's business — tests never hardcode it, they read it
    back from a parse, so this stays valid when MiniGrid's minting changes or a
    different substrate supplies native ids.
    """
    sample = sense.execute_plan(_obs(), _sense_plan(target_ref=None))
    for obj in sample.grid_objects:
        if (obj["x"], obj["y"]) == coord:
            assert obj.get("object_id") is not None, "adapter did not mint an object_id"
            return obj["object_id"]
    raise AssertionError(f"no object parsed at {coord}")


def test_every_parsed_object_gets_a_unique_adapter_minted_id():
    """The MiniGrid adapter stamps every object with an object_id, and two
    description-identical doors get distinct ids (that is the whole point)."""
    sense = _sense()
    sample = sense.execute_plan(_obs(), _sense_plan(target_ref=None))
    ids = [obj.get("object_id") for obj in sample.grid_objects]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == len(ids)  # unique


def test_two_identical_objects_default_scan_is_ambiguous():
    """Sanity: with no identity hint the two doors are indistinguishable by
    description, so Sense returns whichever it scans first — establishing that the
    scenario is genuinely ambiguous."""
    sense = _sense()
    sample = sense.execute_plan(_obs(), _sense_plan(target_ref=None))
    assert sample.target_location in (DOOR_A, DOOR_B)


def test_target_ref_object_id_disambiguates_to_the_chosen_object():
    """The kernel's choice must win: pointing target_ref at each door's id in turn
    must ground THAT door. Today Sense matches by description and returns
    first-in-scan for both, so one direction fails — the F13 red bar."""
    for chosen in (DOOR_A, DOOR_B):
        sense = _sense()
        chosen_id = _object_id_at(sense, chosen)
        sample = sense.execute_plan(
            _obs(), _sense_plan(target_ref={"object_id": chosen_id})
        )
        assert sample.target_location == chosen, (
            f"Sense grounded {sample.target_location}, ignoring the kernel's "
            f"chosen object_id {chosen_id} at {chosen}"
        )


def test_target_ref_threads_through_evidence_frame_context_to_sense():
    """End-to-end thread: target_ref sits in resolved_task_params, rides the
    EvidenceFrame context into the sense plan, and Sense grounds the chosen door by
    id. Proves identity survives the Cortex->Sense handoff, not just a direct
    Sense call."""
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    sense = MiniGridSense(memory, SmokeTestCompiler())
    cortex = Cortex(memory, SmokeTestCompiler())
    cortex.procedure = SimpleNamespace(
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"]
    )
    cortex.execution_state["step_index"] = 1
    chosen_id = _object_id_at(sense, DOOR_B)
    cortex.resolved_task_params = {
        "color": "yellow",
        "object_type": "door",
        "target_ref": {"object_id": chosen_id},
    }

    frame = cortex.make_evidence_frame()
    assert frame.context.get("target_ref") == {"object_id": chosen_id}

    sample = sense.execute_plan(_obs(), _sense_plan(frame.context.get("target_ref")))
    evidence = sense.project_to_cortex(sample)
    cortex.update_from_evidence(evidence, world_sample=sample)
    assert cortex.get_claim("target_location") == DOOR_B
