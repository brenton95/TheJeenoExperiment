"""Step 3, Gate A: target_location decays on the LIVE Sense->Cortex loop.

This is the first claim to actually decay on the real loop (Step 2 only proved the
machine fires with synthetic evidence). Sense becomes in_view-aware: when the
target is in the field of view it re-emits ``target_location`` fresh; when the
target leaves the FOV, Sense stops re-emitting it, so the cortex's prior belief
ages via the Step 2 decay machine instead of being blanket-refreshed from the
``known_target_location`` memory fallback.

The proof is Step 1's decision contract, now exercised through the real Sense
projection: an out-of-view (unverifiable) target is still navigated to; a decayed
(unknown) target is treated as absent and we turn to re-acquire.

No occupancy yet (Gate B). The belief persists-with-decay — it is not unioned.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jeenom.minigrid_operational_context  # noqa: F401  (registers MiniGrid index maps)
from jeenom.claim_freshness import UNVERIFIABLE_DECAY_STEPS
from jeenom.cortex import Cortex
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import Observation
from jeenom.schemas import PrimitiveCall
from jeenom.sense import MiniGridSense


GRID = (6, 6)
DOOR_XY = (1, 1)
AGENT_XY = (3, 3)
_EMPTY, _DOOR, _YELLOW, _DOOR_OPEN = 1, 4, 4, 0


def _image() -> np.ndarray:
    img = np.zeros((GRID[0], GRID[1], 3), dtype=int)
    img[:, :, 0] = _EMPTY
    img[DOOR_XY[0], DOOR_XY[1]] = [_DOOR, _YELLOW, _DOOR_OPEN]
    return img


def _obs(*, step_count: int, visible: list[tuple[int, int]]) -> Observation:
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


def _sense_plan() -> list[PrimitiveCall]:
    return [
        PrimitiveCall(name="parse_grid_objects"),
        PrimitiveCall(name="get_agent_pose"),
        PrimitiveCall(
            name="find_object_by_color_type",
            params={"color": "yellow", "object_type": "door"},
        ),
        PrimitiveCall(name="build_occupancy_grid"),
        PrimitiveCall(name="check_adjacency", params={"object_type": "door"}),
    ]


def _seat_on_navigate(cortex: Cortex) -> None:
    cortex.procedure = SimpleNamespace(
        steps=["locate_object", "navigate_to_object", "verify_adjacent", "done"]
    )
    cortex.execution_state["step_index"] = 1
    cortex.resolved_task_params = {"color": "yellow", "object_type": "door"}


def _tick(sense: MiniGridSense, cortex: Cortex, observation: Observation):
    sample = sense.execute_plan(observation, _sense_plan())
    evidence = sense.project_to_cortex(sample)
    cortex.update_from_evidence(evidence, world_sample=sample)
    return cortex.choose_execution_contract()


def _setup():
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    sense = MiniGridSense(memory, SmokeTestCompiler())
    cortex = Cortex(memory, SmokeTestCompiler())
    _seat_on_navigate(cortex)
    return sense, cortex


_IN_VIEW = [AGENT_XY, (2, 2), DOOR_XY]
_OUT_OF_VIEW = [AGENT_XY, (3, 2)]  # door not among visible cells


def test_in_view_target_is_current_and_navigated():
    sense, cortex = _setup()
    contract = _tick(sense, cortex, _obs(step_count=0, visible=_IN_VIEW))

    assert cortex._claims["target_location"].freshness == "current"
    assert cortex.get_claim("target_location") == DOOR_XY
    assert contract.skill == "navigate_to_object"


def test_target_leaving_view_ages_to_unverifiable_but_still_navigates():
    sense, cortex = _setup()
    _tick(sense, cortex, _obs(step_count=0, visible=_IN_VIEW))
    contract = _tick(sense, cortex, _obs(step_count=1, visible=_OUT_OF_VIEW))

    claim = cortex._claims["target_location"]
    assert claim.freshness == "unverifiable"
    assert cortex.get_claim("target_location") == DOOR_XY  # belief survives
    assert contract.skill == "navigate_to_object"


def test_target_decays_to_unknown_and_we_turn_to_reacquire():
    sense, cortex = _setup()
    _tick(sense, cortex, _obs(step_count=0, visible=_IN_VIEW))
    _tick(sense, cortex, _obs(step_count=1, visible=_OUT_OF_VIEW))  # -> unverifiable
    contract = _tick(
        sense, cortex, _obs(step_count=UNVERIFIABLE_DECAY_STEPS, visible=_OUT_OF_VIEW)
    )

    assert cortex._claims["target_location"].freshness == "unknown"
    assert cortex.get_claim("target_location") is None
    assert contract.skill == "turn_right"


def test_reacquiring_the_target_snaps_belief_back_to_current():
    sense, cortex = _setup()
    _tick(sense, cortex, _obs(step_count=0, visible=_IN_VIEW))
    _tick(sense, cortex, _obs(step_count=1, visible=_OUT_OF_VIEW))
    contract = _tick(sense, cortex, _obs(step_count=2, visible=_IN_VIEW))

    assert cortex._claims["target_location"].freshness == "current"
    assert contract.skill == "navigate_to_object"
