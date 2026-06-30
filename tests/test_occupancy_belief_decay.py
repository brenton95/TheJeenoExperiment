"""Step 3, Gate B: the occupancy/passable belief persists WITH decay.

The map-persistence red bars (tests/test_partial_observability_map_persistence.py)
prove an observed passable cell survives the agent looking away. This file proves
the other half of the frame: the belief is not unioned forever — a cell unseen for
``UNVERIFIABLE_DECAY_STEPS`` ticks decays out of the passable set, exactly the
freshness model's ``unverifiable -> unknown`` edge applied per cell.

If this test could be made to pass by accumulating cells without expiry, that would
be the band-aid we rejected; the expiry assertion is what forbids it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import jeenom.minigrid_operational_context  # noqa: F401  (registers MiniGrid index maps)
from jeenom.claim_freshness import UNVERIFIABLE_DECAY_STEPS
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import Observation
from jeenom.schemas import PrimitiveCall
from jeenom.sense import MiniGridSense

GRID = (6, 6)
CELL = (1, 1)        # a passable corridor cell we observe once
AGENT_XY = (3, 3)
_EMPTY = 1


def _image() -> np.ndarray:
    img = np.zeros((GRID[0], GRID[1], 3), dtype=int)
    img[:, :, 0] = _EMPTY
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


def _plan() -> list[PrimitiveCall]:
    return [
        PrimitiveCall(name="parse_grid_objects"),
        PrimitiveCall(name="get_agent_pose"),
        PrimitiveCall(name="build_occupancy_grid"),
    ]


def _sense() -> MiniGridSense:
    return MiniGridSense(OperationalMemory(root=Path(tempfile.mkdtemp())), SmokeTestCompiler())


def _passable_after(sense: MiniGridSense, observation: Observation) -> set:
    sample = sense.execute_plan(observation, _plan())
    return set(sense.project_to_spine(sample).cues["passable_positions"])


def test_unseen_cell_survives_until_one_step_before_the_constant():
    sense = _sense()
    _passable_after(sense, _obs(step_count=0, visible=[AGENT_XY, CELL]))
    # CELL not observed again; just before the decay threshold it is still believed.
    passable = _passable_after(
        sense, _obs(step_count=UNVERIFIABLE_DECAY_STEPS - 1, visible=[AGENT_XY])
    )
    assert CELL in passable


def test_unseen_cell_decays_out_of_the_belief_at_the_constant():
    sense = _sense()
    _passable_after(sense, _obs(step_count=0, visible=[AGENT_XY, CELL]))
    # At the threshold the belief decays to unknown and drops out — not unioned forever.
    passable = _passable_after(
        sense, _obs(step_count=UNVERIFIABLE_DECAY_STEPS, visible=[AGENT_XY])
    )
    assert CELL not in passable


def test_reobserving_a_cell_refreshes_it_and_resets_decay():
    sense = _sense()
    _passable_after(sense, _obs(step_count=0, visible=[AGENT_XY, CELL]))
    _passable_after(sense, _obs(step_count=5, visible=[AGENT_XY, CELL]))  # refresh at tick 5
    # 15 ticks after the refresh (tick 20) it is still believed; expiry is from last sight.
    passable = _passable_after(
        sense, _obs(step_count=5 + UNVERIFIABLE_DECAY_STEPS - 1, visible=[AGENT_XY])
    )
    assert CELL in passable
