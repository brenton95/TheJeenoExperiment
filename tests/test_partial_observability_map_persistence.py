"""Red-bar test for the partial-observability navigation spin.

Symptom (observed in a live GoToDoor run): the agent locates the target door,
starts navigating, turns, the door leaves its field of view, and it then spins
in place instead of walking to the door it just saw.

Root cause under test: the occupancy / passability map is rebuilt from scratch
every sense tick using only the *currently observed* cells
(`MiniGridSense._build_occupancy_grid`). Previously-observed free space is
discarded the instant it leaves the field of view, so the path planner loses the
route to the (still-remembered) target location and returns no path.

This is inconsistent with the belief-state policy the system already commits to
in `claim_freshness` (out-of-view observations persist and *decay*, they are not
deleted). These tests pin the desired behaviour: observed free space must
persist across sense ticks, so navigation to a remembered target survives the
target leaving view.

NOTE: this target is NOT a memory bug. `known_target_location` is correctly
remembered (asserted below). The defect is purely the spatial map being
ephemeral.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

import jeenom.minigrid_operational_context  # noqa: F401  (registers MiniGrid index maps)
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_adapter import Observation
from jeenom.schemas import PrimitiveCall
from jeenom.sense import MiniGridSense
from jeenom.spine import MiniGridSpine


GRID = (6, 6)
DOOR_XY = (3, 1)          # yellow door, due north of the agent
CORRIDOR_NEIGHBOR = (3, 2)  # passable cell adjacent to the door (the nav goal)
AGENT_XY = (3, 3)

# MiniGrid encoding indices (resolved from the registered domain maps):
_EMPTY, _DOOR = 1, 4
_YELLOW = 4
_DOOR_OPEN = 0


def _full_empty_image() -> np.ndarray:
    img = np.zeros((GRID[0], GRID[1], 3), dtype=int)
    img[:, :, 0] = _EMPTY
    img[DOOR_XY[0], DOOR_XY[1]] = [_DOOR, _YELLOW, _DOOR_OPEN]
    return img


def _observation(*, agent_dir: int, visible_global_cells: list[tuple[int, int]]) -> Observation:
    """Build a synthetic agent-FOV observation that only *reveals* the given cells."""
    raw = {
        "image": _full_empty_image(),
        "mission": "go to the yellow door",
        "direction": agent_dir,
        "_jeenom_grid_size": GRID,
        "_jeenom_agent_pos": AGENT_XY,
        "_jeenom_agent_dir": agent_dir,
        "_jeenom_observation_model": "agent_fov",
        "_jeenom_visible_cells": [
            {"view_x": x, "view_y": y, "x": x, "y": y}
            for (x, y) in visible_global_cells
        ],
    }
    return Observation(raw=raw, step_count=0)


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


def _make_sense() -> MiniGridSense:
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    return MiniGridSense(memory=memory, compiler=SmokeTestCompiler())


def test_observed_free_space_persists_after_target_leaves_view():
    sense = _make_sense()

    # Tick 1: the door and the corridor leading to it are in view.
    obs_seen = _observation(
        agent_dir=3,  # facing up, toward the door
        visible_global_cells=[AGENT_XY, CORRIDOR_NEIGHBOR, DOOR_XY],
    )
    sample1 = sense.execute_plan(obs_seen, _sense_plan())
    passable1 = sense.project_to_spine(sample1).cues["passable_positions"]

    # Precondition: while visible, the route cell and door are known-passable.
    assert CORRIDOR_NEIGHBOR in passable1
    assert DOOR_XY in passable1

    # Tick 2: the agent has turned; the door and corridor are no longer in view.
    obs_turned = _observation(
        agent_dir=2,  # turned to face left; door no longer visible
        visible_global_cells=[AGENT_XY, (2, 3)],
    )
    sample2 = sense.execute_plan(obs_turned, _sense_plan())

    # The target location itself is still remembered — this is NOT a memory bug.
    assert sense.memory.episodic_memory["known_target_location"] == DOOR_XY

    passable2 = sense.project_to_spine(sample2).cues["passable_positions"]

    # The architectural property: free space the agent has already observed must
    # remain part of its belief, so it can still route to the remembered door.
    # Currently FAILS — the map is rebuilt from the current FOV only.
    assert CORRIDOR_NEIGHBOR in passable2, (
        "previously-observed passable cell was discarded when it left the field "
        "of view; the agent's spatial belief does not persist"
    )


def test_can_still_plan_route_to_remembered_door_after_it_leaves_view():
    sense = _make_sense()

    sense.execute_plan(
        _observation(agent_dir=3, visible_global_cells=[AGENT_XY, CORRIDOR_NEIGHBOR, DOOR_XY]),
        _sense_plan(),
    )
    sample2 = sense.execute_plan(
        _observation(agent_dir=2, visible_global_cells=[AGENT_XY, (2, 3)]),
        _sense_plan(),
    )

    percepts = sense.project_to_spine(sample2)
    spine = MiniGridSpine(memory=sense.memory, adapter=None, compiler=SmokeTestCompiler())
    plan = spine._plan_grid_path(
        {
            "target_location": sense.memory.episodic_memory["known_target_location"],
            "object_type": "door",
        },
        percepts,
    )

    # Direct consequence of the discarded map: no route to the door it just saw,
    # which is exactly what makes the agent spin in place. Currently FAILS.
    assert plan["path"], (
        "planner found no route to the remembered door after it left view — the "
        "agent cannot make progress and spins"
    )
