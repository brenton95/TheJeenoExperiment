from __future__ import annotations

from typing import Any

import pytest

from jeenom.ai2thor_domain_helper import Ai2thorDomainHelper
from jeenom.ai2thor_operational_context import Ai2thorOperationalContext
from jeenom.ai2thor_sense import Ai2thorSense
from jeenom.ai2thor_spine import AI2THOR_ACTIONS, Ai2thorSpine, GRID_SIZE, _quantize_point
from jeenom.ai2thor_substrate_adapter import (
    Ai2thorSubstrateAdapter,
    build_ai2thor_runtime_package,
)
from jeenom.capability_registry import CapabilityRegistry
from jeenom.orpi import OrpiManifest
from jeenom.runtime_package import RuntimePackage
from jeenom.schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    ExecutionReport,
    OperationalEvidence,
    Percepts,
    WorldModelSample,
)
from jeenom.substrate_adapter import SubstrateAdapter


class _FakeEvent:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


class _MockController:
    """Stub exposing .step(**kwargs) -> object with .metadata, no Unity needed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _FakeEvent:
        self.calls.append(kwargs)
        return _FakeEvent(metadata={"objects": [], "agent": {"position": {"x": 0, "z": 0}}})


@pytest.fixture
def adapter() -> Ai2thorSubstrateAdapter:
    return Ai2thorSubstrateAdapter(controller=_MockController(), scene_id="FloorPlan1")


def test_adapter_satisfies_substrate_adapter_protocol(adapter: Ai2thorSubstrateAdapter) -> None:
    protocol_methods = [
        name
        for name in vars(SubstrateAdapter)
        if not name.startswith("_")
    ]
    assert protocol_methods, "expected SubstrateAdapter Protocol to define methods"
    for method_name in protocol_methods:
        assert hasattr(adapter, method_name), f"adapter missing protocol method: {method_name}"
        assert callable(getattr(adapter, method_name))


def test_controller_is_injected_not_constructed() -> None:
    bare = Ai2thorSubstrateAdapter()
    assert bare.controller is None

    mock = _MockController()
    wired = Ai2thorSubstrateAdapter(controller=mock)
    assert wired.controller is mock


def test_capability_registry_and_orpi_manifest(adapter: Ai2thorSubstrateAdapter) -> None:
    registry = adapter.capability_registry()
    assert isinstance(registry, CapabilityRegistry)

    manifest = adapter.orpi_manifest()
    assert isinstance(manifest, OrpiManifest)


def test_known_action_names_and_is_action_known(adapter: Ai2thorSubstrateAdapter) -> None:
    known = adapter.known_action_names()
    assert isinstance(known, list)
    for name in known:
        assert adapter.is_action_known(name)
    assert not adapter.is_action_known("definitely_not_a_real_action")


def test_adapter_wires_into_valid_runtime_package() -> None:
    operational_context = Ai2thorOperationalContext.default(scene_id="FloorPlan1")
    adapter = Ai2thorSubstrateAdapter(
        controller=_MockController(),
        scene_id="FloorPlan1",
        operational_context=operational_context,
    )
    domain_helper = Ai2thorDomainHelper(operational_context=operational_context)

    package = RuntimePackage(
        substrate=adapter,
        operational_context=operational_context,
        domain_helper=domain_helper,
    )

    assert package.domain_helper.operational_context is package.operational_context

    registry = package.resolve_capability_registry()
    assert isinstance(registry, CapabilityRegistry)

    manifest = package.resolve_orpi_manifest()
    assert isinstance(manifest, OrpiManifest)


def test_build_ai2thor_runtime_package_convenience_helper() -> None:
    mock = _MockController()
    package = build_ai2thor_runtime_package(controller=mock, scene_id="FloorPlan1")
    assert isinstance(package, RuntimePackage)
    assert package.substrate.controller is mock
    assert package.domain_helper.operational_context is package.operational_context


def test_capability_registry_has_apple_go_to_object_handle(adapter: Ai2thorSubstrateAdapter) -> None:
    registry = adapter.capability_registry()
    spec = registry._by_name.get("task.go_to_object.apple")
    assert spec is not None
    assert spec.layer == "task"


def test_run_task_episode_and_run_motor_actions_do_not_run_live_episode(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    episode_result = adapter.run_task_episode(instruction="go to the apple")
    assert episode_result["task_complete"] is False
    assert episode_result["success"] is False

    motor_result = adapter.run_motor_actions(seed=0, actions=["move_forward"])
    assert motor_result["task_complete"] is False
    assert motor_result["success"] is False


def test_create_sense_returns_ai2thor_sense(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    sense = adapter.create_sense(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]
    assert isinstance(sense, Ai2thorSense)


def test_create_spine_returns_ai2thor_spine(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    spine = adapter.create_spine(memory=None, compiler=None, plan_cache=None)  # type: ignore[arg-type]
    assert isinstance(spine, Ai2thorSpine)
    assert spine.controller is adapter.controller


def test_preview_and_task_window_lifecycle(adapter: Ai2thorSubstrateAdapter) -> None:
    assert adapter.has_preview_window() is False
    adapter.open_preview(seed=0)
    assert adapter.has_preview_window() is True
    adapter.close_preview()
    assert adapter.has_preview_window() is False

    assert adapter.has_task_window() is False
    adapter.close_task_window()
    assert adapter.has_task_window() is False

    adapter.close()
    assert adapter.has_preview_window() is False
    assert adapter.has_task_window() is False


def test_mock_controller_step_returns_metadata() -> None:
    mock = _MockController()
    event = mock.step(action="Done")
    assert hasattr(event, "metadata")
    assert "objects" in event.metadata
    assert mock.calls == [{"action": "Done"}]


def test_known_action_names_is_non_empty_and_all_pass_is_action_known(
    adapter: Ai2thorSubstrateAdapter,
) -> None:
    known = adapter.known_action_names()
    assert len(known) > 0, "known_action_names must be non-empty now that spine is wired"
    for name in known:
        assert adapter.is_action_known(name), f"{name} should be known"


def test_spine_motor_dispatch_move_forward() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="move_forward")

    report, context, plan, cache_meta = spine.tick(contract)

    assert isinstance(report, ExecutionReport)
    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "MoveAhead"
    assert mock.calls[0]["renderImage"] is False


def test_spine_motor_dispatch_turn_right() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="turn_right")

    report, context, plan, cache_meta = spine.tick(contract)

    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "RotateRight"
    assert mock.calls[0]["renderImage"] is False


def test_spine_motor_dispatch_turn_left() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="turn_left")

    report, context, plan, cache_meta = spine.tick(contract)

    assert len(mock.calls) == 1
    assert mock.calls[0]["action"] == "RotateLeft"
    assert mock.calls[0]["renderImage"] is False


def test_spine_unknown_skill_fails() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="teleport_to_moon")

    report, context, plan, cache_meta = spine.tick(contract)

    assert report.status == "failed"
    assert "unknown_action_primitive" in (report.reason or "")
    assert len(mock.calls) == 0


def test_spine_tick_returns_4_tuple() -> None:
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)
    contract = ExecutionContract(skill="move_forward")

    result = spine.tick(contract)

    assert len(result) == 4
    report, context, plan, cache_meta = result
    assert isinstance(report, ExecutionReport)
    assert isinstance(context, ExecutionContext)
    assert isinstance(plan, list)
    assert isinstance(cache_meta, dict)


# ── Sense tests ──────────────────────────────────────────────────────────


class _FakeObservation:
    """Mimics AI2-THOR event with .metadata for sense tests."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


def _apple_metadata(
    apple_x: float = 1.5,
    apple_z: float = 3.7,
    apple_y: float = 0.9,
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    agent_rot_y: float = 90.0,
) -> dict[str, Any]:
    return {
        "objects": [
            {
                "objectType": "Apple",
                "name": "Apple_1",
                "position": {"x": apple_x, "y": apple_y, "z": apple_z},
            },
        ],
        "agent": {
            "position": {"x": agent_x, "y": 0.0, "z": agent_z},
            "rotation": {"x": 0.0, "y": agent_rot_y, "z": 0.0},
        },
    }


def test_sense_tick_returns_5_tuple() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    result = sense.tick(obs, ef, ec)

    assert len(result) == 5
    evidence, percepts, sample, plan, cache_meta = result
    assert isinstance(evidence, OperationalEvidence)
    assert isinstance(percepts, Percepts)
    assert isinstance(sample, WorldModelSample)
    assert isinstance(plan, list)
    assert isinstance(cache_meta, dict)
    assert cache_meta["runtime_compiler_call"] is False


def test_sense_parses_apple_from_metadata() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(apple_x=1.5, apple_z=3.7))
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    evidence, percepts, sample, plan, cache_meta = sense.tick(obs, ef, ec)

    assert sample.target_visible is True
    assert sample.target_object is not None
    assert sample.target_object["type"] == "apple"
    assert sample.target_location is not None


def test_sense_project_to_spine_cues() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    _, percepts, _, _, _ = sense.tick(obs, ef, ec)

    assert "agent_pose" in percepts.cues
    assert "target_location" in percepts.cues
    assert percepts.cues["agent_pose"] is not None


def test_sense_agent_pose_from_metadata() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(agent_x=2.0, agent_z=5.0, agent_rot_y=180.0))
    ef = EvidenceFrame(needs=[])
    ec = ExecutionContext(active_skill="go_to_object", params={})

    _, percepts, sample, _, _ = sense.tick(obs, ef, ec)

    pose = sample.agent_pose
    assert pose is not None
    assert pose["x"] == 2.0
    assert pose["y"] == 5.0
    assert pose["z"] == 0.0
    assert pose["dir"] == 180


def test_sense_no_target_when_type_missing() -> None:
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata())
    ef = EvidenceFrame(needs=[])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "banana"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    assert sample.target_visible is False
    assert sample.target_location is None


def test_sense_coord_preserves_float_precision() -> None:
    """Three distinct AI2-THOR axes -> correct JEENO fields, float-preserved."""
    sense = Ai2thorSense(memory=None, compiler=None)
    obs = _FakeObservation(_apple_metadata(apple_x=1.5, apple_z=3.7, apple_y=0.9))
    ef = EvidenceFrame(needs=["target_location"])
    ec = ExecutionContext(active_skill="go_to_object", params={"object_type": "apple"})

    _, _, sample, _, _ = sense.tick(obs, ef, ec)

    obj = sample.target_object
    assert obj is not None
    assert isinstance(obj["x"], float), f"expected float x, got {type(obj['x'])}"
    assert abs(obj["x"] - 1.5) < 0.01
    assert abs(obj["y"] - 3.7) < 0.01
    assert abs(obj["z"] - 0.9) < 0.01

    assert sample.target_location is not None
    tx, ty = sample.target_location
    assert isinstance(tx, float), f"expected float tx, got {type(tx)}"
    assert abs(tx - 1.5) < 0.01
    assert abs(ty - 3.7) < 0.01
    assert len(sample.target_location) == 2


# ── Navigation tests (plan 005) ─────────────────────────────────────────


def _make_reachable_grid(
    x_range: range,
    z_range: range,
    grid_size: float = GRID_SIZE,
) -> list[dict[str, float]]:
    """Build a canned GetReachablePositions return in AI2-THOR coords."""
    return [
        {"x": x * grid_size, "y": 0.0, "z": z * grid_size}
        for x in x_range
        for z in z_range
    ]


class _NavMockController:
    """Mock controller that tracks agent position and returns canned reachable positions."""

    def __init__(
        self,
        reachable_points: list[dict[str, float]],
        agent_x: float = 0.0,
        agent_z: float = 0.0,
        agent_yaw: float = 0.0,
    ) -> None:
        self.reachable_points = reachable_points
        self.agent_x = agent_x
        self.agent_z = agent_z
        self.agent_yaw = agent_yaw
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> _FakeEvent:
        self.calls.append(kwargs)
        action = kwargs.get("action", "")

        if action == "GetReachablePositions":
            return _FakeEvent(metadata={"actionReturn": self.reachable_points})

        if action == "MoveAhead":
            yaw = int(self.agent_yaw) % 360
            if yaw == 0:
                self.agent_z += GRID_SIZE
            elif yaw == 90:
                self.agent_x += GRID_SIZE
            elif yaw == 180:
                self.agent_z -= GRID_SIZE
            elif yaw == 270:
                self.agent_x -= GRID_SIZE
        elif action == "RotateRight":
            self.agent_yaw = (self.agent_yaw + 90) % 360
        elif action == "RotateLeft":
            self.agent_yaw = (self.agent_yaw - 90) % 360

        return _FakeEvent(metadata={
            "objects": [],
            "agent": {
                "position": {"x": self.agent_x, "y": 0.0, "z": self.agent_z},
                "rotation": {"x": 0.0, "y": self.agent_yaw, "z": 0.0},
            },
        })


def test_nav_plans_path_and_succeeds() -> None:
    """Agent at (0,0) facing +jy (yaw=0), apple at (0.5, 0.5).
    Reachable grid 0..3 x 0..3 (quantized). Agent should navigate to a cell
    adjacent to the apple and report succeeded."""
    reachable = _make_reachable_grid(range(4), range(4))
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.5, 0.5) — quantizes to (2, 2) with gridSize=0.25
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.5, 0.5),
    })
    contract = ExecutionContract(skill="navigate_to_object", params={"object_type": "apple"})

    report, context, plan, cache_meta = spine.tick(contract, percepts)

    assert report.status == "succeeded", f"expected succeeded, got {report.status}: {report.reason}"
    assert report.progress.get("actions") is not None
    actions = report.progress["actions"]
    assert all(a in ("move_forward", "turn_right", "turn_left") for a in actions)


def test_nav_success_detection_already_adjacent() -> None:
    """Agent already within reach of target → succeeded immediately, no movement."""
    reachable = _make_reachable_grid(range(4), range(4))
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # agent at (0,0) in JEENO, target at (0.25, 0) — within REACH_THRESHOLD
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.25, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    assert report.progress.get("already_adjacent") is True
    # No movement actions should have been dispatched
    move_calls = [c for c in ctrl.calls if c.get("action") in ("MoveAhead", "RotateRight", "RotateLeft")]
    assert len(move_calls) == 0


def test_nav_unreachable_target_fails_honestly() -> None:
    """Target surrounded by no reachable cells → no_reachable_goal_adjacent_to_target."""
    # Only agent's position is reachable — nothing near the target
    reachable = [{"x": 0.0, "y": 0.0, "z": 0.0}]
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (5.0, 5.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert "no_reachable_goal" in (report.reason or "")


def test_nav_no_path_found_fails_honestly() -> None:
    """Goal cells exist but are disconnected from agent → no_path_found."""
    # Agent at (0,0), target at (2.0, 0.0) with a gap: reachable = {(0,0), (8,0)}
    reachable = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 2.0, "y": 0.0, "z": 0.0},     # adjacent to target but disconnected
    ]
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (2.25, 0.0) → quantized (9, 0); adjacent cell (8, 0) is reachable but disconnected
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (2.25, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "failed"
    assert report.reason == "no_path_found"


def test_nav_action_sequence_straight_line() -> None:
    """Agent faces +jy, target is directly ahead — should be pure move_forward actions."""
    reachable = _make_reachable_grid(range(1), range(5))  # a column x=0, z=0..4
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.0, 0.75) → quantized (0, 3); agent at (0, 0) facing yaw 0 (+jy)
    # goal = adjacent to (0, 3) = (0, 2) (reachable). path = (0,0)->(0,1)->(0,2).
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.0, 0.75),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    actions = report.progress["actions"]
    # Already facing the right way, so no turns
    assert all(a == "move_forward" for a in actions)
    assert len(actions) == 2  # two steps: (0,0)->(0,1)->(0,2)


def test_nav_action_sequence_requires_turn() -> None:
    """Agent faces +jy (yaw 0) but target is to the right (+jx) — needs RotateRight first."""
    reachable = _make_reachable_grid(range(5), range(1))  # a row z=0, x=0..4
    ctrl = _NavMockController(reachable, agent_x=0.0, agent_z=0.0, agent_yaw=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)

    # target at JEENO (0.75, 0.0) → quantized (3, 0); agent at (0, 0).
    # goal = adjacent to (3, 0) = (2, 0). path = (0,0)->(1,0)->(2,0).
    # Agent faces yaw 0 (+jy), needs to face yaw 90 (+jx) → one turn_right.
    percepts = Percepts(cues={
        "agent_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "dir": 0},
        "target_location": (0.75, 0.0),
    })
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts)

    assert report.status == "succeeded"
    actions = report.progress["actions"]
    assert actions[0] == "turn_right"
    assert actions.count("move_forward") == 2


def test_nav_existing_motor_dispatch_unaffected() -> None:
    """Existing single-action dispatch (move_forward, turn_right, etc.) still works."""
    mock = _MockController()
    spine = Ai2thorSpine(memory=None, controller=mock, compiler=None)

    for skill, expected_action in AI2THOR_ACTIONS.items():
        mock.calls.clear()
        contract = ExecutionContract(skill=skill)
        report, _, _, _ = spine.tick(contract)
        assert report.status == "running"
        assert len(mock.calls) == 1
        assert mock.calls[0]["action"] == expected_action


def test_nav_missing_percepts_fails() -> None:
    """navigate_to_object without percepts → failed, not a crash."""
    ctrl = _NavMockController([], agent_x=0.0, agent_z=0.0)
    spine = Ai2thorSpine(memory=None, controller=ctrl, compiler=None)
    contract = ExecutionContract(skill="navigate_to_object")

    report, _, _, _ = spine.tick(contract, percepts=None)

    assert report.status == "failed"
    assert report.reason == "no_percepts"
